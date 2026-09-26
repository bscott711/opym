# Ruff style: Compliant
"""Pyramidal (t, c, z, y, x) OME-Zarr (NGFF 0.4) for deskewed/rotated output,
written one timepoint at a time.

Used by the live lane (opym.stream.live), which writes each timepoint the
moment it's deconvolved and deskewed so napari can follow the acquisition
(opym.live_view), and by the backfill's viewer export
(bioimaging.backfill.viewer_export), which builds the same store from
finished DSR frames. One writer means one layout, so a store the live lane
completed is the final export as-is.

A store is allocated at full size up front; timepoints not yet written read
as zeros. `.opym_live.json` next to the arrays records which (t, c) are
written, so a viewer knows what's real and the export knows the store is
complete.

Two layouts:

- The processed store (`create_processed_store`), written by the one-format
  live path: bioformats2raw layout 3. Series "0" is the DSR multiscale
  (t, c, z, y, x), series "1" its Z-MIP, and `OME/METADATA.ome.xml`
  (opym.ome_xml) is the OME-XML for both. The GPU server writes the pixels
  directly (run_live_zarr.m -> opymWriteZarrBlock); Python only creates it.
- The legacy viewer store (`create_store`): multiscales at the root, no
  OME-XML, as the TIFF live path and the backfill export wrote it.

Read through `image_group(store, series)`, which handles both.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# DSR resamples onto an isotropic grid whose spacing is the lateral pixel size.
DSR_VOXEL_UM = 0.136
# napari opens a pyramid lazily; a single full-resolution level per timepoint
# is slow to pan. Three levels cost ~14% extra storage.
PYRAMID_LEVELS = 3
# Distinct, colour-blind-safe-ish emission colours; index = channel number.
# The second is magenta, not red: two-colour composites are green/magenta.
CHANNEL_COLORS = ("00FF00", "FF00FF", "00B3FF", "FFC400")
PROGRESS_NAME = ".opym_live.json"
# The processed store (bioformats2raw layout 3); see the module docstring.
BF2RAW_LAYOUT = 3
DSR_SERIES = "0"
MIP_SERIES = "1"
CHUNKS_ZYX = (64, 256, 256)


def downsample2(vol: np.ndarray, threads: int = 4) -> np.ndarray:
    """2x block mean on each axis, trimming any odd trailing plane/row/column,
    truncated back to the input dtype.

    For unsigned <=16-bit data (every DSR volume) the 8-voxel sum is taken in
    uint32 and floor-divided by 8, split across threads along z (numpy's adds
    release the GIL). That is bit-identical to the float32 mean-then-truncate
    it replaces (the float sum of eight uint16 values is exact, and /8 is
    exact) and ~9x faster on a 419x1458x649 frame (5.1 s -> ~0.6 s).
    """
    z, y, x = (s - (s % 2) for s in vol.shape)
    v = vol[:z, :y, :x]
    if not (np.issubdtype(vol.dtype, np.unsignedinteger) and vol.dtype.itemsize <= 2):
        f = v.astype(np.float32).reshape(z // 2, 2, y // 2, 2, x // 2, 2)
        return f.mean(axis=(1, 3, 5)).astype(vol.dtype)
    out = np.empty((z // 2, y // 2, x // 2), dtype=vol.dtype)

    def slab(z0: int, z1: int) -> None:
        w = v[2 * z0 : 2 * z1]
        acc = np.zeros((z1 - z0, y // 2, x // 2), np.uint32)
        for a in (0, 1):
            for b in (0, 1):
                for c in (0, 1):
                    acc += w[a::2, b::2, c::2]
        out[z0:z1] = acc // 8

    bounds = np.linspace(0, z // 2, max(1, min(threads, z // 2)) + 1).astype(int)
    pairs = [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
    if len(pairs) <= 1:
        for a, b in pairs:
            slab(a, b)
        return out
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(len(pairs)) as ex:
        list(ex.map(lambda ab: slab(*ab), pairs))
    return out


def level_shapes(n_t: int, n_c: int, shape_zyx, levels: int = PYRAMID_LEVELS):
    nz, ny, nx = shape_zyx
    return [
        (n_t, n_c, max(1, nz // 2**lvl), max(1, ny // 2**lvl), max(1, nx // 2**lvl))
        for lvl in range(levels)
    ]


def _chunks(shape_tczyx) -> tuple[int, ...]:
    return (1, 1, *(min(c, n) for c, n in zip(CHUNKS_ZYX, shape_tczyx[2:])))


def _multiscales(name: str, levels: int, voxel_um: float, time_interval_s: float):
    return [
        {
            "version": "0.4",
            "name": name,
            "axes": [
                {"name": "t", "type": "time", "unit": "second"},
                {"name": "c", "type": "channel"},
                {"name": "z", "type": "space", "unit": "micrometer"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"},
            ],
            "datasets": [
                {
                    "path": str(lvl),
                    "coordinateTransformations": [
                        {
                            "type": "scale",
                            "scale": [time_interval_s, 1.0] + [voxel_um * 2**lvl] * 3,
                        }
                    ],
                }
                for lvl in range(levels)
            ],
        }
    ]


def _omero(name: str, n_c: int, labels: list[str]):
    return {
        "name": name,
        "channels": [
            {
                "label": labels[c] if c < len(labels) else f"C{c}",
                "color": CHANNEL_COLORS[c % len(CHANNEL_COLORS)],
                "active": True,
                "window": {"start": 0, "end": 300, "min": 0, "max": 65535},
            }
            for c in range(n_c)
        ],
    }


def create_store(
    out_path: Path,
    *,
    n_t: int,
    n_c: int,
    shape_zyx: tuple[int, int, int],
    dtype,
    channel_labels: list[str] | None = None,
    voxel_um: float = DSR_VOXEL_UM,
    levels: int = PYRAMID_LEVELS,
    time_interval_s: float = 1.0,
):
    """Allocate an empty legacy-layout store (all zeros, nothing written)."""
    import zarr

    out_path = Path(out_path)
    root = zarr.open_group(str(out_path), mode="w")
    for lvl, shp in enumerate(level_shapes(n_t, n_c, shape_zyx, levels)):
        root.create_dataset(
            str(lvl),
            shape=shp,
            dtype=dtype,
            chunks=_chunks(shp),
            dimension_separator="/",
        )
    labels = channel_labels or [f"C{c}" for c in range(n_c)]
    name = out_path.name.removesuffix(".ome.zarr")
    root.attrs["multiscales"] = _multiscales(name, levels, voxel_um, time_interval_s)
    root.attrs["omero"] = _omero(out_path.name, n_c, labels)
    write_progress(out_path, n_t=n_t, n_c=n_c, done=[], state="running")
    return root


@dataclass(frozen=True)
class ProcessedArrays:
    """Where the GPU server writes one (t, c): every DSR pyramid level
    (full resolution first) and the Z-MIP."""

    levels: list[Path]
    mip: Path


def processed_arrays(out_path: Path, levels: int = PYRAMID_LEVELS) -> ProcessedArrays:
    out_path = Path(out_path)
    return ProcessedArrays(
        levels=[out_path / DSR_SERIES / str(lvl) for lvl in range(levels)],
        mip=out_path / MIP_SERIES / "0",
    )


def create_processed_store(
    out_path: Path,
    *,
    n_t: int,
    n_c: int,
    shape_zyx: tuple[int, int, int],
    channel_labels: list[str] | None = None,
    voxel_um: float = DSR_VOXEL_UM,
    levels: int = PYRAMID_LEVELS,
    time_interval_s: float | None = None,
    acquisition_date=None,
) -> ProcessedArrays:
    """Allocate an empty processed store (bioformats2raw layout 3, see the
    module docstring) for `n_t` x `n_c` DSR volumes of `shape_zyx`.

    uint16, blosc-lz4, `/` separators, chunk size 1 on t and c -- the layout
    opymWriteZarrBlock requires -- and fill value 0, so a chunk the writer
    skipped as all-zero reads back as zeros.
    """
    import zarr
    from numcodecs import Blosc

    from opym.ome_xml import processed_ome_xml

    out_path = Path(out_path)
    name = out_path.name.removesuffix(".ome.zarr").removesuffix("_dsr")
    labels = channel_labels or [f"C{c}" for c in range(n_c)]
    compressor = Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE)
    root = zarr.open_group(str(out_path), mode="w")
    root.attrs["bioformats2raw.layout"] = BF2RAW_LAYOUT

    def array(group, path, shape):
        group.create_dataset(
            path,
            shape=shape,
            chunks=_chunks(shape),
            dtype="uint16",
            compressor=compressor,
            dimension_separator="/",
            fill_value=0,
        )

    dsr = root.create_group(DSR_SERIES)
    for lvl, shp in enumerate(level_shapes(n_t, n_c, shape_zyx, levels)):
        array(dsr, str(lvl), shp)
    dsr.attrs["multiscales"] = _multiscales(
        name, levels, voxel_um, time_interval_s or 1.0
    )
    dsr.attrs["omero"] = _omero(name, n_c, labels)
    # The same image on the root, pointing into series "0": NGFF readers that
    # don't walk bioformats2raw series (ome-zarr-py 0.11, so napari-ome-zarr)
    # then open the store itself. Bio-Formats reads the layout + OME-XML.
    root_ms = _multiscales(name, levels, voxel_um, time_interval_s or 1.0)
    for ds in root_ms[0]["datasets"]:
        ds["path"] = f"{DSR_SERIES}/{ds['path']}"
    root.attrs["multiscales"] = root_ms
    root.attrs["omero"] = _omero(name, n_c, labels)

    mip = root.create_group(MIP_SERIES)
    array(mip, "0", (n_t, n_c, 1, shape_zyx[1], shape_zyx[2]))
    mip.attrs["multiscales"] = _multiscales(
        f"{name} Z-MIP", 1, voxel_um, time_interval_s or 1.0
    )
    mip.attrs["omero"] = _omero(f"{name} Z-MIP", n_c, labels)

    ome = root.create_group("OME")
    ome.attrs["series"] = [DSR_SERIES, MIP_SERIES]
    (out_path / "OME" / "METADATA.ome.xml").write_text(
        processed_ome_xml(
            name=name,
            n_t=n_t,
            n_c=n_c,
            shape_zyx=tuple(shape_zyx),
            channel_labels=labels,
            voxel_um=voxel_um,
            time_interval_s=time_interval_s,
            colors=CHANNEL_COLORS,
            acquisition_date=acquisition_date,
        )
    )
    write_progress(out_path, n_t=n_t, n_c=n_c, done=[], state="running")
    return processed_arrays(out_path, levels)


def image_group(store: Path, series: str = DSR_SERIES, mode: str = "r"):
    """The zarr group holding `series`' multiscales: the numbered series
    group of a processed (bioformats2raw) store, or the root of a legacy
    one, which only has the DSR image."""
    import zarr

    root = zarr.open_group(str(store), mode=mode)
    if "bioformats2raw.layout" in root.attrs:
        return root[series]
    if series != DSR_SERIES:
        raise KeyError(f"{store} is a legacy store; it has no series {series!r}")
    return root


def write_timepoint(out_path: Path, t: int, c: int, vol: np.ndarray) -> None:
    """Write one (t, c) volume into every pyramid level. Safe to call from
    several threads for different (t, c): each writes its own chunks."""
    root = image_group(out_path, mode="r+")
    lvl = 0
    while str(lvl) in root:
        arr = root[str(lvl)]
        z, y, x = arr.shape[2:]
        arr[t, c, :z, :y, :x] = vol[:z, :y, :x]
        lvl += 1
        if str(lvl) in root:
            vol = downsample2(vol)


def read_progress(out_path: Path) -> dict | None:
    try:
        return json.loads((Path(out_path) / PROGRESS_NAME).read_text())
    except (OSError, ValueError):
        return None


def write_progress(
    out_path: Path, *, n_t: int, n_c: int, done: list[list[int]], state: str
) -> None:
    """`done` is a list of [t, c] pairs written so far. `state` is running,
    complete or failed."""
    path = Path(out_path) / PROGRESS_NAME
    tmp = path.with_name(f".{PROGRESS_NAME}.tmp")
    tmp.write_text(
        json.dumps(
            {
                "state": state,
                "n_t": n_t,
                "n_c": n_c,
                "done": sorted([int(t), int(c)] for t, c in done),
                "updated_at": time.time(),
            }
        )
    )
    os.replace(tmp, path)


def complete_timepoints(progress: dict | None) -> list[int]:
    """Timepoints with every channel written, ascending."""
    if not progress:
        return []
    n_c = int(progress.get("n_c", 0))
    per_t: dict[int, set[int]] = {}
    for t, c in progress.get("done", []):
        per_t.setdefault(int(t), set()).add(int(c))
    return sorted(t for t, cs in per_t.items() if len(cs) >= n_c)


def is_processed_store(path: Path | str | None) -> bool:
    """A processed (bioformats2raw layout) store, as the one-format live
    lane writes -- as opposed to a legacy store or anything else."""
    if path is None:
        return False
    try:
        attrs = json.loads((Path(path) / ".zattrs").read_text())
    except (OSError, ValueError):
        return False
    return isinstance(attrs, dict) and "bioformats2raw.layout" in attrs


def mip_stacks(store: Path) -> dict[int, np.ndarray]:
    """channel -> (T, y, x) Z-MIPs of a processed store, for every timepoint
    whose channels are all written (per the progress file), in time order:
    what the per-frame MIP TIFFs used to provide."""
    group = image_group(store, MIP_SERIES)
    arr = group["0"]
    done = complete_timepoints(read_progress(store))
    if not done:
        return {}
    return {c: np.stack([arr[t, c, 0] for t in done]) for c in range(arr.shape[1])}
