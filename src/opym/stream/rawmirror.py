# Ruff style: Compliant
"""
Raw OME-Zarr mirror writer for the real-time stream receiver.

Writes each channel's raw, un-rotated `(T, Z, Y, X)` volumes directly into a
`<base_name>_<channel_name>.ome.zarr` store on GPFS -- the same directory
name, zarr v2 layout, and per-plane chunking that a completed Globus
transfer already leaves behind for the newer pymmcore-based MDA writer (see
`opym.discovery`'s `KIND_ZARR_PRECROPPED`). Writing to that exact shape
means `opym.discovery.is_zarr_leaf_dataset_dir` /
`group_channel_zarr_stores`, `bioimaging.backfill.pipeline._zarr_store_is_ready`,
and `opym.metadata.parse_zarr_z_step_from_store` all see a normal completed
acquisition once `SESSION_END` finalizes it -- nothing downstream of this
module needs to know the data arrived over the network instead of Globus.

See `opym_local/docs/STREAMING_PROTOCOL.md` for the wire protocol this
feeds from.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path

import numpy as np
import zarr

from .protocol import epoch_s_or_none

logger = logging.getLogger(__name__)

_ZATTRS = {
    "multiscales": [
        {
            "axes": [
                {"name": "t", "type": "time"},
                {"name": "z", "type": "space"},
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"},
            ],
            "datasets": [
                {
                    "coordinateTransformations": [
                        {"scale": [1, 1, 1, 1], "type": "scale"}
                    ],
                    "path": "p0",
                }
            ],
            "name": "p0",
            "version": "0.4",
        }
    ]
}

# Final-output formats a client may request for the processed (DSR) result.
# Recorded on the raw store as `.zattrs["opym"]["output_format"]` so the
# choice travels with the data through the drain to GPFS, where
# `bioimaging.backfill`'s viewer export reads it back.
OUTPUT_FORMATS = ("tiff", "ome-zarr", "both")


def _read_opym_attrs(store_path: Path) -> dict:
    try:
        attrs = json.loads((Path(store_path) / ".zattrs").read_text())
    except (OSError, ValueError):
        return {}
    opym_attrs = attrs.get("opym") if isinstance(attrs, dict) else None
    return opym_attrs if isinstance(opym_attrs, dict) else {}


def read_output_format(store_path: Path) -> str | None:
    """The `output_format` a stream client recorded on this raw store, or
    None if none was (e.g. a Globus-landed acquisition)."""
    fmt = _read_opym_attrs(store_path).get("output_format")
    return fmt if fmt in OUTPUT_FORMATS else None


def read_session_id(store_path: Path) -> str | None:
    """The stream session that created this raw store (see
    `create_channel_store`), or None for a store with no such tag -- a
    Globus-landed acquisition, one written before tagging existed, or no
    store at all."""
    sid = _read_opym_attrs(store_path).get("session_id")
    return sid if isinstance(sid, str) else None


def create_channel_store(
    store_path: Path,
    *,
    num_timepoints: int,
    shape_zyx: tuple[int, int, int],
    dtype: str,
    z_step_um: float,
    output_format: str | None = None,
    session_id: str | None = None,
) -> zarr.Array:
    """Creates (or reopens) one channel's raw zarr store at `store_path` and
    returns its `p0` pixel array, shaped `(num_timepoints, *shape_zyx)`.

    Idempotent: calling this again for the same `store_path` with the same
    shape/dtype (e.g. because the client resent `SESSION_START` after a
    reconnect) reopens the existing array rather than recreating it, same
    as `zarr.open(..., mode='a')`'s own contract. A DIFFERENT shape or dtype
    raises instead: `zarr.open(mode='a')` ignores the requested shape of an
    existing array, which on 2026-09-24 silently reopened a 1-timepoint test
    store for a 100-timepoint acquisition of the same name and rejected
    every one of its frames. The receiver now never reuses another
    session's name (`StreamReceiver._resolve_base_name`); this check makes
    any future regression fail with a clear message.

    `session_id` is recorded as `.zattrs["opym"]["session_id"]` so the
    receiver can tell its own store (a SESSION_START resent after a restart)
    from an earlier acquisition's (see `read_session_id`).

    Chunked one z-plane at a time (`[1, 1, ny, nx]`, `dimension_separator=
    '/'`) to match what the real acquisition writer produces -- not for
    write performance here (writing a whole timepoint in one call still
    touches every z-chunk), but so nothing downstream that assumes this
    chunk shape (e.g. `channel_store_timepoints`'s "count numbered
    subdirectories" logic) needs a special case for stream-written stores.

    The `p0` array is created (and reopened) before `.zattrs` is written,
    and `.zattrs` is written last -- `_zarr_store_is_ready` treats a store
    with `.zgroup` but no `.zattrs`/pixel array as still-arriving and
    defers rather than failing, so this ordering keeps a mid-creation store
    looking exactly like a mid-Globus-transfer one to any concurrent
    backfill scan.
    """
    store_path = Path(store_path)
    store_path.mkdir(parents=True, exist_ok=True)
    (store_path / ".zgroup").write_text(json.dumps({"zarr_format": 2}))

    nz, ny, nx = shape_zyx
    arr = zarr.open(
        str(store_path / "p0"),
        mode="a",
        shape=(num_timepoints, nz, ny, nx),
        chunks=(1, 1, ny, nx),
        dtype=np.dtype(dtype),
        compressor=None,
        dimension_separator="/",
    )
    expected_shape = (num_timepoints, nz, ny, nx)
    if tuple(arr.shape) != expected_shape or arr.dtype != np.dtype(dtype):
        raise ValueError(
            f"{store_path} already holds a {tuple(arr.shape)} {arr.dtype} array "
            f"(tagged session {read_session_id(store_path)!r}); this session "
            f"needs {expected_shape} {np.dtype(dtype)} -- refusing to write into "
            "another acquisition's store"
        )

    _write_z_coordinate(store_path / "z", nz, z_step_um)
    # when each volume was acquired; see write_epoch
    _epoch_array(store_path, num_timepoints)

    # Written last -- see docstring.
    attrs = dict(_ZATTRS)
    opym_attrs = {}
    if output_format is not None:
        opym_attrs["output_format"] = output_format
    if session_id is not None:
        opym_attrs["session_id"] = session_id
    if opym_attrs:
        attrs["opym"] = opym_attrs
    (store_path / ".zattrs").write_text(json.dumps(attrs))
    return arr


def _write_z_coordinate(z_dir: Path, nz: int, z_step_um: float) -> None:
    """Writes the `z` 1-D coordinate array `parse_zarr_z_step_from_store`
    reads z step back out of -- the only real geometry the acquisition
    writer's own stores record (NGFF `coordinateTransformations` is always
    the placeholder `[1,1,1,1]`; see that function's docstring). No-ops if
    already written, since every channel of a session shares one z step and
    only the first channel to arrive needs to write it.
    """
    if (z_dir / ".zarray").exists():
        return
    z_vals = np.arange(nz, dtype=np.float64) * float(z_step_um)
    z_arr = zarr.open(str(z_dir), mode="a", shape=(nz,), dtype="float64")
    z_arr[:] = z_vals
    (z_dir / ".zattrs").write_text(json.dumps({"units": "um"}))


EPOCH_ARRAY = "t_epoch_s"


def _epoch_array(store_path: Path, n_t: int | None = None) -> zarr.Array:
    """The per-volume acquisition-time array, created (NaN-filled) if missing.

    One chunk per timepoint: the two channels' frames of a volume are written from
    different threads, and a shared chunk would let one overwrite the other's value.
    """
    path = Path(store_path) / EPOCH_ARRAY
    if (path / ".zarray").exists():
        return zarr.open(str(path), mode="a")
    if n_t is None:
        n_t = zarr.open(str(Path(store_path) / "p0"), mode="r").shape[0]
    arr = zarr.open(
        str(path),
        mode="a",
        shape=(int(n_t),),
        chunks=(1,),
        dtype="float64",
        fill_value=float("nan"),
        compressor=None,
    )
    arr.attrs.update(
        {
            "_ARRAY_DIMENSIONS": ["t"],
            "units": "s",
            "origin": "unix epoch, rig clock: the FRAME timestamp of each volume",
        }
    )
    return arr


def write_epoch(store_path: Path, t: int, timestamp) -> None:
    """Record when volume `t` was acquired (epoch seconds, rig clock).

    Callers pass `protocol.frame_epoch_s(header)`, not the raw FRAME `timestamp`: an
    old rig sends milliseconds since the run started, which is not a time of day.
    Anything outside the epoch-seconds window (or None) is not written here either,
    so the array only ever holds real times and the volume stays "unknown" (NaN).
    `opym.timeline` later turns this into real elapsed time for the processed store.
    It never raises: a time that cannot be written must not cost its volume.
    """
    try:
        ts = epoch_s_or_none(timestamp)
        if ts is None:
            return
        _epoch_array(store_path)[int(t)] = ts
    except Exception:  # noqa: BLE001 - a missing time is not worth a dropped frame
        logger.debug(
            "could not record the acquisition time of t=%s in %s",
            t,
            store_path,
            exc_info=True,
        )


def write_timepoint(arr: zarr.Array, t: int, volume_zyx: np.ndarray) -> None:
    """Writes one full raw `(Z, Y, X)` volume into timepoint `t` of a
    channel array created by `create_channel_store`.

    Not atomic across z: this single assignment still writes one chunk
    file per z-plane, so a crash mid-write can leave timepoint `t` with
    some z-chunks present and others not. `channel_store_timepoints` (the
    batch path's aborted-acquisition detector) already treats a
    partially-written trailing timepoint this way for Globus-landed data,
    so this is an existing, already-handled failure mode, not a new one.
    """
    arr[t] = volume_zyx


def write_planes(arr: zarr.Array, t: int, z0: int, planes_zyx: np.ndarray) -> None:
    """Writes z-planes `[z0, z0 + n)` of timepoint `t` (a slab of a volume
    still being acquired). With one chunk per plane, each plane is its own
    chunk file, so slabs of one volume never touch each other's chunks."""
    arr[t, z0 : z0 + planes_zyx.shape[0]] = planes_zyx


def discard_timepoint(store_path: Path, t: int) -> bool:
    """Remove every plane written for timepoint `t` of the channel store at
    `store_path` (its `p0/<t>` chunk directory); True if there was any.

    For a volume that never completed (session ended mid-volume): zarr reads a
    missing plane as zeros and `channel_store_timepoints` counts the numbered
    directory as a written timepoint, so leaving the partial planes makes the
    backfill process a volume with holes as if it were whole (ING-F2). The
    array itself stays; a later resume of the same store starts that volume
    again from nothing.
    """
    tdir = Path(store_path) / "p0" / str(int(t))
    if not tdir.is_dir():
        return False
    shutil.rmtree(tdir)
    return True


# Names a client sends go into paths. Whatever the trust model (loopback on a
# shared host, or the direct endpoint), none of them may leave the folder they
# are meant for (ING-F5).
NAME_MAX_CHARS = 100
SESSION_ID_MAX_CHARS = 128
_SESSION_ID_RE = re.compile(r"[A-Za-z0-9._#-]+")


def check_path_component(value: object, what: str = "name") -> str:
    """`value` as ONE safe path component, or ValueError: a non-empty string of
    at most `NAME_MAX_CHARS` with no `/`, no `\\`, no NUL or other control
    character, and not `.` or `..`."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{what} must be a non-empty string, got {value!r}")
    if len(value) > NAME_MAX_CHARS:
        raise ValueError(f"{what} is longer than {NAME_MAX_CHARS} characters")
    if value in (".", ".."):
        raise ValueError(f"{what} may not be {value!r}")
    if any(ch in "/\\" or ord(ch) < 32 or ord(ch) == 127 for ch in value):
        raise ValueError(
            f"{what} {value!r} may not hold a path separator or control character"
        )
    return value


def check_session_id(value: object) -> str:
    """`value` as a session id: `[A-Za-z0-9._#-]`, at most `SESSION_ID_MAX_CHARS`
    long, not made of dots only (it names drain-manifest files)."""
    if (
        not isinstance(value, str)
        or not value
        or len(value) > SESSION_ID_MAX_CHARS
        or _SESSION_ID_RE.fullmatch(value) is None
        or not value.strip(".")
    ):
        raise ValueError(
            f"session_id must be 1..{SESSION_ID_MAX_CHARS} characters of "
            f"[A-Za-z0-9._#-], got {value!r}"
        )
    return value


def check_raw_root(value: object) -> Path:
    """`value` as an absolute POSIX directory path with no NUL and no `..`
    part, or ValueError. A relative path would land wherever this service's
    working directory is; a Windows path (`D:\\data`) is not one on this host."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"raw_root must be a non-empty string, got {value!r}")
    if "\0" in value:
        raise ValueError("raw_root may not hold a NUL")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"raw_root must be an absolute path, got {value!r}")
    if ".." in path.parts:
        raise ValueError(f"raw_root may not hold '..', got {value!r}")
    return path


def store_path_for_channel(raw_root: Path, base_name: str, channel_name: str) -> Path:
    """The canonical per-channel store path for `channel_name` (e.g.
    "GFP_488") under `raw_root`, matching
    `opym.discovery._ZARR_CHANNEL_SUFFIX_RE` / `parse_zarr_group_prefix` so
    the store this module writes is grouped back under `base_name` exactly
    the way a Globus-landed one would be. `base_name` and `channel_name` must
    each be one safe path component (`check_path_component`).
    """
    check_path_component(base_name, "base_name")
    check_path_component(channel_name, "channel name")
    return Path(raw_root) / f"{base_name}_{channel_name}.ome.zarr"
