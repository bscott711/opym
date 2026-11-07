# Ruff style: Compliant
"""The processed store (bioformats2raw layout 3) as consumers read it.

The names and shape of the store `opym.ome_zarr_writer.create_processed_store`
writes, and readers that need only the stdlib: attributes are plain JSON, so
CORE, ChimeraX, dashboards and viewers read a store's voxel size, channels and
progress without zarr. Only `image_group` (and `ome_zarr_writer.mip_stacks`,
which reads pixels) opens zarr, lazily.

Layout: series "0" is the DSR multiscale (t, c, z, y, x) with `PYRAMID_LEVELS`
levels `0/0`, `0/1`, ...; series "1" its Z-MIP `1/0` (t, c, 1, y, x);
`.opym_live.json` at the root lists the (t, c) written. A legacy store has its
multiscales at the root and only the DSR image.

Golden store for consumers' tests: `opym.testing.store`.
"""

from __future__ import annotations

import json
from pathlib import Path

from opym.decon_config import XY_PIXEL_SIZE_UM
from opym.display import default_channel_color, normalize_color

# The root attribute that marks a processed store.
LAYOUT_KEY = "bioformats2raw.layout"
BF2RAW_LAYOUT = 3
DSR_SERIES = "0"
MIP_SERIES = "1"
PROGRESS_NAME = ".opym_live.json"
# napari opens a pyramid lazily; a single full-resolution level per timepoint
# is slow to pan. Three levels cost ~14% extra storage.
PYRAMID_LEVELS = 3
CHUNKS_ZYX = (64, 256, 256)
# DSR resamples onto an isotropic grid whose spacing is the lateral pixel size:
# one value, decon_config's.
DSR_VOXEL_UM = XY_PIXEL_SIZE_UM

# The names of the output tree below a dataset's output base
# (`opym.utils.resolve_output_base`): <base>/viewer/<leaf>_dsr.ome.zarr, the
# MIP movies in <base>/mip_movies, and the channel-aligned copy that the aligner
# writes beside a viewer store (<stem>_aligned.ome.zarr).
VIEWER_DIR = "viewer"
DSR_STORE_SUFFIX = "_dsr.ome.zarr"
MIP_MOVIES_DIR = "mip_movies"
ALIGNED_STORE_SUFFIX = "_aligned.ome.zarr"


def dsr_store_name(leaf_name: str) -> str:
    """The viewer store's folder name for a dataset leaf: `<leaf>_dsr.ome.zarr`."""
    return f"{leaf_name}{DSR_STORE_SUFFIX}"


def viewer_store_path(output_base: Path | str, leaf_name: str) -> Path:
    """`<output base>/viewer/<leaf>_dsr.ome.zarr`, where the pipeline writes (and
    the viewers look for) a dataset's processed store. `output_base` is
    `opym.utils.resolve_output_base(leaf_dir)`."""
    return Path(output_base) / VIEWER_DIR / dsr_store_name(leaf_name)


def mip_movies_dir(output_base: Path | str) -> Path:
    """`<output base>/mip_movies`, the dataset's MIP movies and posters."""
    return Path(output_base) / MIP_MOVIES_DIR


def aligned_store_name(stem: str) -> str:
    """The folder name of the channel-aligned copy: `<stem>_aligned.ome.zarr`."""
    return f"{stem}{ALIGNED_STORE_SUFFIX}"


def is_aligned_store(path: Path | str) -> bool:
    """Is `path` named like a channel-aligned copy (so not a primary viewer store)?"""
    return Path(path).name.endswith(ALIGNED_STORE_SUFFIX)


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def read_attrs(path: Path | str) -> dict:
    """A zarr group's attributes by plain JSON: v2 `.zattrs`, else v3
    `zarr.json` (its "ome" namespace merged in). {} if there are none."""
    path = Path(path)
    attrs = _read_json(path / ".zattrs")
    if not isinstance(attrs, dict):
        meta = _read_json(path / "zarr.json")
        attrs = meta.get("attributes") if isinstance(meta, dict) else None
        if not isinstance(attrs, dict):
            return {}
        attrs = dict(attrs)
        if isinstance(attrs.get("ome"), dict):
            attrs.update(attrs["ome"])
    return attrs


def is_processed_store(path: Path | str | None) -> bool:
    """A processed (bioformats2raw layout) store, as the one-format live
    lane writes -- as opposed to a legacy store or anything else."""
    if path is None:
        return False
    attrs = _read_json(Path(path) / ".zattrs")
    return isinstance(attrs, dict) and LAYOUT_KEY in attrs


def read_progress(out_path: Path | str) -> dict | None:
    """The `.opym_live.json` record ({"state", "n_t", "n_c", "done", ...}), or None."""
    rec = _read_json(Path(out_path) / PROGRESS_NAME)
    return rec if rec is not None else None


def complete_timepoints(progress: dict | None) -> list[int]:
    """Timepoints with every channel written, ascending."""
    if not progress:
        return []
    n_c = int(progress.get("n_c", 0))
    per_t: dict[int, set[int]] = {}
    for t, c in progress.get("done", []):
        per_t.setdefault(int(t), set()).add(int(c))
    return sorted(t for t, cs in per_t.items() if len(cs) >= n_c)


def image_group(store: Path | str, series: str = DSR_SERIES, mode: str = "r"):
    """The zarr group holding `series`' multiscales: the numbered series
    group of a processed (bioformats2raw) store, or the root of a legacy
    one, which only has the DSR image. Opens zarr (lazily imported)."""
    import zarr

    root = zarr.open_group(str(store), mode=mode)
    if LAYOUT_KEY in root.attrs:
        return root[series]
    if series != DSR_SERIES:
        raise KeyError(f"{store} is a legacy store; it has no series {series!r}")
    return root


def level_path(store: Path | str, series: str = DSR_SERIES, level: int = 0) -> Path:
    """Where pyramid `level` of `series` lives: <store>/<series>/<level>."""
    return Path(store) / series / str(level)


def frame_count(store: Path | str, series: str = DSR_SERIES) -> int | None:
    """The number of timepoints (t, the first axis) in `series`' full-resolution
    image, read from the level-0 array's `.zarray` (v2) or `zarr.json` (v3)
    metadata without zarr; None if the store, the series or the metadata is
    missing. Works for a processed (layout 3) store and a legacy flat one."""
    store = Path(store)
    if LAYOUT_KEY in read_attrs(store):
        level0 = level_path(store, series, 0)
    elif series == DSR_SERIES:
        level0 = store / "0"  # legacy: the multiscales sit at the root
    else:
        return None
    for name in (".zarray", "zarr.json"):
        meta = _read_json(level0 / name)
        try:
            return int(meta["shape"][0])
        except (KeyError, IndexError, TypeError, ValueError):
            continue
    return None


def _image_attrs(store: Path | str, series: str = DSR_SERIES) -> dict:
    """The attributes of the group holding `series`' image (the series group of
    a processed store, the root of a legacy one)."""
    store = Path(store)
    root = read_attrs(store)
    if LAYOUT_KEY in root:
        return read_attrs(store / series)
    return root if series == DSR_SERIES else {}


def _scale(store: Path | str, series: str, level: int) -> list[float] | None:
    try:
        ms = _image_attrs(store, series)["multiscales"][0]
        for tr in ms["datasets"][level]["coordinateTransformations"]:
            if tr.get("type") == "scale":
                return [float(s) for s in tr["scale"]]
    except (KeyError, IndexError, TypeError, ValueError):
        pass
    return None


def voxel_size_um(
    store: Path | str, series: str = DSR_SERIES, level: int = 0
) -> float | None:
    """The store's voxel size in micrometres (the last scale of `level`'s
    coordinateTransformation; doubles with each pyramid level), or None."""
    scale = _scale(store, series, level)
    return scale[-1] if scale else None


def time_interval_s(store: Path | str, series: str = DSR_SERIES) -> float | None:
    """The t scale of the multiscales (seconds between timepoints), or None.

    The writer records 1.0 when it did not know the interval; real per-volume
    times are the root `opym_time` record (`opym.timeline.read_times`)."""
    scale = _scale(store, series, 0)
    return scale[0] if scale else None


def _omero_channels(store: Path | str, series: str) -> list[dict]:
    channels = (_image_attrs(store, series).get("omero") or {}).get("channels") or []
    return [c if isinstance(c, dict) else {} for c in channels]


def channel_labels(store: Path | str, series: str = DSR_SERIES) -> list[str]:
    """The omero channel labels ("" where a channel has none), [] if no omero block."""
    return [str(c.get("label") or "") for c in _omero_channels(store, series)]


def channel_colors(store: Path | str, series: str = DSR_SERIES) -> list[str]:
    """The omero channel colours as "RRGGBB" with the legacy remap applied (a
    missing or unparsable one falls back to `default_channel_color`), [] if
    no omero block."""
    return [
        normalize_color(c.get("color")) or default_channel_color(i)
        for i, c in enumerate(_omero_channels(store, series))
    ]
