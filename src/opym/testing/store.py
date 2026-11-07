# Ruff style: Compliant
"""A tiny layout-3 processed store for consumers' tests.

`golden_layout3_attrs()` is the attribute tree `create_processed_store` writes,
built by the writer's own helpers (numpy only). `write_golden_store(path)` writes
a real store with `create_processed_store` (zarr, numcodecs and ome-types, imported
lazily: the `opym[server]` stack), fills it, and marks it complete.
`write_golden_attrs(path)` writes only the JSON (`.zgroup`, `.zattrs`, the progress
record and each pyramid level's `.zarray` shape), which is all the zarr-free readers
in `opym.store` need.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from opym.ome_zarr_writer import PYRAMID_LEVELS, _multiscales, _omero
from opym.store import DSR_SERIES, LAYOUT_KEY, MIP_SERIES, PROGRESS_NAME

GOLDEN_NAME = "golden_dsr.ome.zarr"
GOLDEN_N_T = 3
GOLDEN_N_C = 2
GOLDEN_SHAPE_ZYX = (8, 16, 16)
GOLDEN_LABELS = ("GFP 488", "mScarlet 561")
GOLDEN_VOXEL_UM = 0.136
GOLDEN_TIME_INTERVAL_S = 7.5
# Timepoints with every channel written (the third has channel 0 only).
GOLDEN_FRAME_COUNT = GOLDEN_N_T  # `opym.store.frame_count`: t, the first axis
GOLDEN_DONE = ([0, 0], [0, 1], [1, 0], [1, 1], [2, 0])


def golden_layout3_attrs(*, legacy_red: bool = False) -> dict[str, dict]:
    """Group attributes of the golden store by path relative to its root
    ("" is the root, "0" the DSR series, "1" its Z-MIP). With `legacy_red`,
    channel 1 names FF3D3D, as stores written before 2026-09-26 do."""
    name = GOLDEN_NAME.removesuffix(".ome.zarr").removesuffix("_dsr")
    labels = list(GOLDEN_LABELS)

    def omero(n):
        om = _omero(n, GOLDEN_N_C, labels)
        if legacy_red:
            om["channels"][1]["color"] = "FF3D3D"
        return om

    def ms(n, levels):
        return _multiscales(n, levels, GOLDEN_VOXEL_UM, GOLDEN_TIME_INTERVAL_S)

    root_ms = ms(name, PYRAMID_LEVELS)
    for ds in root_ms[0]["datasets"]:
        ds["path"] = f"{DSR_SERIES}/{ds['path']}"
    return {
        "": {LAYOUT_KEY: 3, "multiscales": root_ms, "omero": omero(name)},
        DSR_SERIES: {"multiscales": ms(name, PYRAMID_LEVELS), "omero": omero(name)},
        MIP_SERIES: {
            "multiscales": ms(f"{name} Z-MIP", 1),
            "omero": omero(f"{name} Z-MIP"),
        },
        "OME": {"series": [DSR_SERIES, MIP_SERIES]},
    }


def _progress(state: str = "running") -> dict:
    return {
        "state": state,
        "n_t": GOLDEN_N_T,
        "n_c": GOLDEN_N_C,
        "done": [list(d) for d in GOLDEN_DONE],
        "updated_at": 0.0,
    }


def write_golden_attrs(path: Path | str, *, legacy_red: bool = False) -> Path:
    """The golden store's JSON only (no arrays, no zarr), enough for `opym.store`."""
    path = Path(path)
    for rel, attrs in golden_layout3_attrs(legacy_red=legacy_red).items():
        group = path / rel
        group.mkdir(parents=True, exist_ok=True)
        (group / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
        (group / ".zattrs").write_text(json.dumps(attrs))
    for series, z in ((DSR_SERIES, GOLDEN_SHAPE_ZYX[0]), (MIP_SERIES, 1)):
        for level in range(PYRAMID_LEVELS if series == DSR_SERIES else 1):
            shape = [
                GOLDEN_N_T,
                GOLDEN_N_C,
                z,
                GOLDEN_SHAPE_ZYX[1] >> level,
                GOLDEN_SHAPE_ZYX[2] >> level,
            ]
            arr = path / series / str(level)
            arr.mkdir(parents=True, exist_ok=True)
            (arr / ".zarray").write_text(json.dumps({"zarr_format": 2, "shape": shape}))
    (path / PROGRESS_NAME).write_text(json.dumps(_progress()))
    return path


def write_golden_store(
    path: Path | str | None = None, *, root: Path | str | None = None
):
    """Write the golden store with `create_processed_store` and fill the timepoints
    `GOLDEN_DONE` lists with `golden_volume(t, c)`. Pass `path` (the store) or `root`
    (a directory to put `GOLDEN_NAME` in). Returns the store path."""
    from opym.ome_zarr_writer import (
        create_processed_store,
        write_progress,
        write_timepoint,
    )

    if path is None:
        if root is None:
            raise ValueError("pass path or root")
        path = Path(root) / GOLDEN_NAME
    path = Path(path)
    create_processed_store(
        path,
        n_t=GOLDEN_N_T,
        n_c=GOLDEN_N_C,
        shape_zyx=GOLDEN_SHAPE_ZYX,
        channel_labels=list(GOLDEN_LABELS),
        voxel_um=GOLDEN_VOXEL_UM,
        time_interval_s=GOLDEN_TIME_INTERVAL_S,
    )
    for t, c in GOLDEN_DONE:
        write_timepoint(path, t, c, golden_volume(t, c))
    write_progress(
        path, n_t=GOLDEN_N_T, n_c=GOLDEN_N_C, done=list(GOLDEN_DONE), state="running"
    )
    return path


def golden_volume(t: int, c: int) -> np.ndarray:
    """The uint16 (z, y, x) volume written at (t, c): 1 + 1000 t + 100 c + z."""
    z = np.arange(GOLDEN_SHAPE_ZYX[0], dtype=np.uint16)[:, None, None]
    base = np.uint16(1 + 1000 * t + 100 * c)
    return np.broadcast_to(base + z, GOLDEN_SHAPE_ZYX).copy()
