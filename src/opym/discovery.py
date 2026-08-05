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
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
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


@dataclass(frozen=True)
class LeafDataset:
    """A single raw-acquisition directory discovered under a data root."""

    root: Path
    leaf_dir: Path
    master_file: Path

    @property
    def dataset_key(self) -> str:
        """Stable, human-readable primary key -- the resolved leaf directory path."""
        return str(self.leaf_dir)


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


def walk_leaf_datasets(root: Path) -> Iterator[Path]:
    """Unbounded-depth walk (real data ranges 7-11 path segments) that finds
    every leaf dataset directory under `root`. Uses `os.walk` (not
    `Path.rglob`, which can't prune) so already-processed output
    subdirectories are skipped entirely rather than merely ignored.
    """
    root = Path(root)
    for dirpath, dirnames, _filenames in os.walk(root):
        current = Path(dirpath)
        dirnames[:] = [d for d in dirnames if d not in _OUTPUT_DIR_NAMES]
        if is_leaf_dataset_dir(current):
            yield current
            # A leaf dataset dir's own subdirectories (if any) are not
            # further raw-data leaves -- don't descend further.
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
        for leaf_dir in walk_leaf_datasets(root):
            master_file = select_master_ome_tif(leaf_dir)
            if master_file is None:
                continue
            datasets.append(
                LeafDataset(root=root, leaf_dir=leaf_dir, master_file=master_file)
            )
    return datasets
