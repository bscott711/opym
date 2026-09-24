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
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

# DSR resamples onto an isotropic grid whose spacing is the lateral pixel size.
DSR_VOXEL_UM = 0.136
# napari opens a pyramid lazily; a single full-resolution level per timepoint
# is slow to pan. Three levels cost ~14% extra storage.
PYRAMID_LEVELS = 3
# Distinct, colour-blind-safe-ish emission colours; index = channel number.
CHANNEL_COLORS = ("00FF00", "FF3D3D", "00B3FF", "FFC400")
PROGRESS_NAME = ".opym_live.json"


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
    """Allocate an empty store (all zeros, nothing written) at `out_path`."""
    import zarr

    out_path = Path(out_path)
    root = zarr.open_group(str(out_path), mode="w")
    for lvl, shp in enumerate(level_shapes(n_t, n_c, shape_zyx, levels)):
        root.create_dataset(
            str(lvl),
            shape=shp,
            dtype=dtype,
            chunks=(1, 1, min(64, shp[2]), min(256, shp[3]), min(256, shp[4])),
            dimension_separator="/",
        )
    labels = channel_labels or [f"C{c}" for c in range(n_c)]
    root.attrs["multiscales"] = [
        {
            "version": "0.4",
            "name": out_path.name.removesuffix(".ome.zarr"),
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
    root.attrs["omero"] = {
        "name": out_path.name,
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
    write_progress(out_path, n_t=n_t, n_c=n_c, done=[], state="running")
    return root


def write_timepoint(out_path: Path, t: int, c: int, vol: np.ndarray) -> None:
    """Write one (t, c) volume into every pyramid level. Safe to call from
    several threads for different (t, c): each writes its own chunks."""
    import zarr

    root = zarr.open_group(str(out_path), mode="r+")
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
