# Ruff style: Compliant
"""
Recursive discovery of raw OPM acquisition leaf directories across one or
more data roots, for the unattended bulk backfill (crop -> zarr -> DSR ->
MIP, decon skipped). Nothing recursive existed in this repo before this
module -- every prior entry point (cli.py, submit_opm.py, batch.py) takes a
single, pre-specified dataset directory.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

# Output-directory names written by this pipeline (and the legacy one) --
# pruned during the walk so a resumed backfill doesn't re-descend into
# hundreds of already-written per-frame files on GPFS.
_OUTPUT_DIR_NAMES = frozenset(
    {
        "processed_ngff",
        "processed_tiff_series_split",
        "Decon",
        "decon",
        "DS",
        "DSR",
        "DSR_nodecon",
        "MIPs",
        "mip_movies",
    }
)


# Discriminates the two raw-acquisition shapes this module can discover --
# see LeafDataset.kind.
KIND_OME_TIF = "ome_tif"
KIND_ZARR_PRECROPPED = "zarr_precropped"


@dataclass(frozen=True)
class LeafDataset:
    """A single raw-acquisition dataset discovered under a data root.

    `kind` distinguishes two acquisition shapes written by two different
    versions of the microscope's acquisition software, both from the same
    physical dual-camera OPM hardware:
    - `KIND_OME_TIF` (legacy Micro-Manager writer): one raw directory is one
      dataset. `leaf_dir` is that real directory; `master_file` is the
      single raw OME-TIF this pipeline crops + channel-remaps itself.
    - `KIND_ZARR_PRECROPPED` (newer pymmcore-based MDA writer): the camera
      regions are already cropped and split one-per-channel at capture time
      into per-channel `*.ome.zarr` stores, and ONE RAW DIRECTORY CAN HOLD
      MANY INDEPENDENT SAMPLES' worth of these (e.g. `cell_001_mScarlet_561`
      through `cell_010_mScarlet_561`, each its own capture, no relation to
      each other) -- confirmed against real data, not assumed. Files are
      grouped into datasets by shared name prefix before the trailing
      `_<Channel>_<Wavelength>` suffix (see `parse_zarr_group_prefix`), so
      e.g. `bead_005_GFP_488` and `bead_005_mScarlet_561` (same prefix) group
      as one 2-channel dataset, while `cell_001_...` (no sibling sharing
      that exact prefix) is its own single-channel dataset. Because of this,
      `leaf_dir` here is a SYNTHETIC per-group path (`raw_dir / prefix`) used
      only as this dataset's own output namespace (mip_movies, registry
      bookkeeping) -- it is NOT a real directory containing the raw data.
      The real raw files are `channel_zarr_paths` (siblings under the real
      `raw_dir`, i.e. `leaf_dir.parent`); `master_file` is one of them
      (picked for cheap reference-frame reads), not the whole dataset.
      These skip this pipeline's own crop stage entirely and dispatch
      straight to deskew/decon -- see `bioimaging/backfill/pipeline.py`.
    """

    root: Path
    leaf_dir: Path
    master_file: Path
    kind: str = KIND_OME_TIF
    channel_zarr_paths: tuple[Path, ...] = field(default_factory=tuple)

    @property
    def dataset_key(self) -> str:
        """Stable, human-readable primary key -- the resolved leaf directory path."""
        return str(self.leaf_dir)

    @property
    def raw_dir(self) -> Path:
        """The real filesystem directory containing this dataset's raw
        input file(s). Equal to `leaf_dir` for `KIND_OME_TIF` (where
        `leaf_dir` already is that real directory); for
        `KIND_ZARR_PRECROPPED`, `leaf_dir` is a synthetic per-group output
        namespace, so this is `master_file.parent` instead -- the real
        shared directory `channel_zarr_paths` live in.
        """
        return self.master_file.parent


def is_leaf_dataset_dir(path: Path) -> bool:
    """True iff `path` directly (non-recursively) contains at least one
    `*.ome.tif`. Verified against real data in all three data roots: this
    alone correctly excludes every known non-dataset top-level folder
    (`PSF/`, `OTF/`, `OPM_preprocessed/`, `OPM_Testing/`, `Macropinocytosis/`,
    `Phagocytosis-Bcell/`) with no exclude-list required.
    """
    return next(path.glob("*.ome.tif"), None) is not None


def select_master_ome_tif(leaf_dir: Path) -> Path | None:
    """Picks the base multi-series file over Micro-Manager's numbered
    continuation files (`*_1.ome.tif`, `*_2.ome.tif`, ...) by shortest
    filename -- the same heuristic already proven in run_pipeline_cli.py.
    """
    ome_tifs = list(leaf_dir.glob("*.ome.tif"))
    if not ome_tifs:
        return None
    return min(ome_tifs, key=lambda p: len(p.name))


def find_channel_zarr_stores(leaf_dir: Path) -> list[Path]:
    """Every per-channel `*.ome.zarr` store directly under `leaf_dir`,
    sorted for determinism (channel-pattern matching downstream doesn't
    depend on order, but a stable list makes logs/registry entries
    reproducible run to run).
    """
    return sorted(leaf_dir.glob("*.ome.zarr"))


# Matches the trailing "_<ChannelName>_<Wavelength>" the newer acquisition
# writer appends to every per-channel zarr store's name, e.g.
# "cell_001_mScarlet_561" -> prefix "cell_001", or "bead_GFP_488" -> prefix
# "bead". Verified against every real example seen so far (mScarlet_561,
# GFP_488, ...); a name that doesn't match this shape falls back to being
# its own singleton group (see parse_zarr_group_prefix) rather than
# guessing wrong.
_ZARR_CHANNEL_SUFFIX_RE = re.compile(r"^(?P<prefix>.+)_(?P<channel>[A-Za-z]+)_(?P<wavelength>\d+)$")


def parse_zarr_group_prefix(zarr_store: Path) -> str:
    """The shared-sample grouping key for a per-channel zarr store name --
    strips the trailing `_<ChannelName>_<Wavelength>` suffix (e.g.
    "cell_001_mScarlet_561.ome.zarr" -> "cell_001", "bead_GFP_488.ome.zarr"
    -> "bead"). Falls back to the full stem (i.e. its own singleton group)
    if the name doesn't match the convention.
    """
    stem = zarr_store.name.removesuffix(".ome.zarr")
    m = _ZARR_CHANNEL_SUFFIX_RE.match(stem)
    return m.group("prefix") if m else stem


def group_channel_zarr_stores(leaf_dir: Path) -> dict[str, list[Path]]:
    """Groups every per-channel `*.ome.zarr` store in `leaf_dir` by shared
    sample prefix (see `parse_zarr_group_prefix`) -- confirmed against real
    acquisition folders: `bead_005_GFP_488` and `bead_005_mScarlet_561`
    (same prefix "bead_005") group together as one 2-channel dataset, while
    ten separate `cell_001_mScarlet_561` .. `cell_010_mScarlet_561` (no
    sibling sharing any one of those exact prefixes) are ten independent
    single-channel datasets, not one 10-channel dataset.
    """
    groups: dict[str, list[Path]] = {}
    for store in find_channel_zarr_stores(leaf_dir):
        groups.setdefault(parse_zarr_group_prefix(store), []).append(store)
    return groups


def is_zarr_leaf_dataset_dir(path: Path) -> bool:
    """True iff `path` is a pre-cropped acquisition from the newer
    pymmcore-based MDA writer: at least one per-channel `*.ome.zarr` store.

    `MDA_settings.yaml` is NOT required here -- confirmed with the user
    it's a one-time-per-experiment-set file the acquisition GUI doesn't
    currently write per dataset, so gating discovery on it silently drops
    real, fully-uploaded datasets (as it did for a real backfill upload).
    Its absence is already handled gracefully downstream: `parse_zarr_z_step`
    / `parse_zarr_expected_timepoints` (metadata.py) fall back to defaults
    when the sidecar is missing, same as the rest of this metadata-parsing
    chain. This module only ever runs against already-landed backfill data
    (not live acquisition), so there's no in-flight-transfer case to guard
    against the way there was when this required both markers.
    """
    return bool(find_channel_zarr_stores(path))


def walk_leaf_datasets(root: Path) -> Iterator[tuple[Path, str]]:
    """Unbounded-depth walk (real data ranges 7-11 path segments) that finds
    every leaf dataset directory under `root`, yielding `(leaf_dir, kind)`.
    Uses `os.walk` (not `Path.rglob`, which can't prune) so already-processed
    output subdirectories are skipped entirely rather than merely ignored.
    """
    root = Path(root)
    for dirpath, dirnames, _filenames in os.walk(root):
        current = Path(dirpath)
        dirnames[:] = [d for d in dirnames if d not in _OUTPUT_DIR_NAMES]
        if is_leaf_dataset_dir(current):
            yield current, KIND_OME_TIF
            # A leaf dataset dir's own subdirectories (if any) are not
            # further raw-data leaves -- don't descend further.
            dirnames[:] = []
        elif is_zarr_leaf_dataset_dir(current):
            yield current, KIND_ZARR_PRECROPPED
            dirnames[:] = []


def discover_leaf_datasets(roots: Sequence[Path]) -> list[LeafDataset]:
    """Production entry point: walk every given root independently and
    return every discovered `LeafDataset`.

    Deliberately does NOT attempt cross-root duplicate detection. A full
    recursive filename+size diff across every name that happens to collide
    between two roots showed the "duplicate" almost always diverges (one
    side a near-empty stub, or a different subset of sub-experiments, or
    partially-processed output the other side lacks) -- there is no safe
    "prefer root A" rule. Roots are walked independently; per-stage
    idempotency (see `registry.py`) absorbs the negligible cost of the rare
    true duplicate.
    """
    datasets: list[LeafDataset] = []
    for root in roots:
        root = Path(root)
        for raw_dir, kind in walk_leaf_datasets(root):
            if kind == KIND_OME_TIF:
                master_file = select_master_ome_tif(raw_dir)
                if master_file is None:
                    continue
                datasets.append(
                    LeafDataset(root=root, leaf_dir=raw_dir, master_file=master_file, kind=kind)
                )
            else:
                # One raw directory can hold many independent samples --
                # each shared-prefix group is its own dataset, with a
                # synthetic per-group leaf_dir (see LeafDataset docstring).
                for prefix, members in group_channel_zarr_stores(raw_dir).items():
                    members_sorted = tuple(sorted(members))
                    datasets.append(
                        LeafDataset(
                            root=root,
                            leaf_dir=raw_dir / prefix,
                            master_file=members_sorted[0],
                            kind=kind,
                            channel_zarr_paths=members_sorted,
                        )
                    )
    return datasets
