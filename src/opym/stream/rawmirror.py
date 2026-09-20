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
from pathlib import Path

import numpy as np
import zarr

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


def create_channel_store(
    store_path: Path,
    *,
    num_timepoints: int,
    shape_zyx: tuple[int, int, int],
    dtype: str,
    z_step_um: float,
) -> zarr.Array:
    """Creates (or reopens) one channel's raw zarr store at `store_path` and
    returns its `p0` pixel array, shaped `(num_timepoints, *shape_zyx)`.

    Idempotent: calling this again for the same `store_path` with the same
    shape/dtype (e.g. because the client resent `SESSION_START` after a
    reconnect) reopens the existing array rather than recreating it, same
    as `zarr.open(..., mode='a')`'s own contract.

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

    _write_z_coordinate(store_path / "z", nz, z_step_um)

    # Written last -- see docstring.
    (store_path / ".zattrs").write_text(json.dumps(_ZATTRS))
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


def store_path_for_channel(raw_root: Path, base_name: str, channel_name: str) -> Path:
    """The canonical per-channel store path for `channel_name` (e.g.
    "GFP_488") under `raw_root`, matching
    `opym.discovery._ZARR_CHANNEL_SUFFIX_RE` / `parse_zarr_group_prefix` so
    the store this module writes is grouped back under `base_name` exactly
    the way a Globus-landed one would be.
    """
    return Path(raw_root) / f"{base_name}_{channel_name}.ome.zarr"
