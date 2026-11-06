# Ruff style: Compliant
"""
opym: OPM processing library (deskew/decon, stream receiver, backfill, live view)
and the shared contracts its consumers import.

The package namespace is lazy: `from opym import run_batch_cropping` still works,
but `import opym` (and so `import opym.timeline` etc.) imports none of the heavy
dependencies. Each re-exported name is resolved on first access from its
submodule; see `_LAZY` and `docs/adr/0001-light-base.md`.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # static analysis / IDEs only; never executed
    from .batch import (
        run_batch_cropping,
    )
    from .core import (
        process_dataset,
        run_processing_job,
    )
    from .dataloader import (
        find_dsr_directory,
        get_channel_count,
        load_llsm_tiff_series,
        load_tiff_series,
    )
    from .discovery import (
        KIND_OME_TIF,
        KIND_ZARR_PRECROPPED,
        LeafDataset,
        discover_leaf_datasets,
        find_channel_zarr_stores,
        is_leaf_dataset_dir,
        is_zarr_leaf_dataset_dir,
    )
    from .metadata import (
        create_processing_log,
        parse_expected_timepoints,
        parse_mda_settings,
        parse_timestamps,
        parse_z_step,
        parse_zarr_expected_timepoints,
        parse_zarr_z_step,
    )
    from .petakit import (
        monitor_job_background,
        run_petakit_processing,
        submit_crop_and_save_sidecar,
        submit_remote_crop_job,
        submit_remote_deskew_job,
        wait_for_job,
    )
    from .registry import (
        StatusRegistry,
    )
    from .roi_detect import (
        auto_detect_rois,
        compute_reference_projection,
    )
    from .roi_utils import (
        align_rois,
        load_rois_from_log,
        process_rois_from_selector,
        save_rois_to_log,
    )
    from .ui import (
        create_crop_settings_ui,
        create_deskew_ui,
    )
    from .utils import (
        DerivedPaths,
        MicroscopyDataType,
        OutputFormat,
        derive_paths,
        detect_microscopy_data_type,
        parse_roi_string,
        scan_channel_patterns,
    )
    from .viewer import (
        composite_viewer,
        create_mip,
        interactive_roi_selector,
        single_channel_viewer,
        visualize_alignment,
    )
    from .widgets import (
        DeconvolutionViewer,
        PSFAverager,
        PSFExtractor,
    )

# Public name -> submodule that defines it.
_LAZY: dict[str, str] = {
    "run_batch_cropping": "batch",
    "process_dataset": "core",
    "run_processing_job": "core",
    "find_dsr_directory": "dataloader",
    "get_channel_count": "dataloader",
    "load_llsm_tiff_series": "dataloader",
    "load_tiff_series": "dataloader",
    "KIND_OME_TIF": "discovery",
    "KIND_ZARR_PRECROPPED": "discovery",
    "LeafDataset": "discovery",
    "discover_leaf_datasets": "discovery",
    "find_channel_zarr_stores": "discovery",
    "is_leaf_dataset_dir": "discovery",
    "is_zarr_leaf_dataset_dir": "discovery",
    "create_processing_log": "metadata",
    "parse_expected_timepoints": "metadata",
    "parse_mda_settings": "metadata",
    "parse_timestamps": "metadata",
    "parse_z_step": "metadata",
    "parse_zarr_expected_timepoints": "metadata",
    "parse_zarr_z_step": "metadata",
    "monitor_job_background": "petakit",
    "run_petakit_processing": "petakit",
    "submit_crop_and_save_sidecar": "petakit",
    "submit_remote_crop_job": "petakit",
    "submit_remote_deskew_job": "petakit",
    "wait_for_job": "petakit",
    "StatusRegistry": "registry",
    "auto_detect_rois": "roi_detect",
    "compute_reference_projection": "roi_detect",
    "align_rois": "roi_utils",
    "load_rois_from_log": "roi_utils",
    "process_rois_from_selector": "roi_utils",
    "save_rois_to_log": "roi_utils",
    "create_crop_settings_ui": "ui",
    "create_deskew_ui": "ui",
    "DerivedPaths": "utils",
    "MicroscopyDataType": "utils",
    "OutputFormat": "utils",
    "derive_paths": "utils",
    "detect_microscopy_data_type": "utils",
    "parse_roi_string": "utils",
    "scan_channel_patterns": "utils",
    "composite_viewer": "viewer",
    "create_mip": "viewer",
    "interactive_roi_selector": "viewer",
    "single_channel_viewer": "viewer",
    "visualize_alignment": "viewer",
    "DeconvolutionViewer": "widgets",
    "PSFAverager": "widgets",
    "PSFExtractor": "widgets",
}

__all__ = [
    "process_dataset",
    "run_processing_job",
    "create_processing_log",
    "parse_timestamps",
    "derive_paths",
    "parse_roi_string",
    "OutputFormat",
    "DerivedPaths",
    "save_rois_to_log",
    "load_rois_from_log",
    "align_rois",
    "process_rois_from_selector",
    "single_channel_viewer",
    "composite_viewer",
    "load_tiff_series",
    "load_llsm_tiff_series",
    "find_dsr_directory",
    "create_mip",
    "interactive_roi_selector",
    "visualize_alignment",
    "submit_remote_crop_job",
    "submit_remote_deskew_job",
    "monitor_job_background",
    "run_petakit_processing",
    "detect_microscopy_data_type",
    "MicroscopyDataType",
    "wait_for_job",
    "run_batch_cropping",
    "create_crop_settings_ui",
    "create_deskew_ui",
    "get_channel_count",
    "submit_crop_and_save_sidecar",
    "parse_z_step",
    "scan_channel_patterns",
    # Public API for new widgets
    "PSFExtractor",
    "PSFAverager",
    "DeconvolutionViewer",
    # Bulk backfill support (discovery, ROI auto-detect, status registry)
    "LeafDataset",
    "discover_leaf_datasets",
    "is_leaf_dataset_dir",
    "compute_reference_projection",
    "auto_detect_rois",
    "StatusRegistry",
    # Pre-cropped zarr acquisitions (newer pymmcore-based MDA writer)
    "KIND_OME_TIF",
    "KIND_ZARR_PRECROPPED",
    "is_zarr_leaf_dataset_dir",
    "find_channel_zarr_stores",
    "parse_expected_timepoints",
    "parse_mda_settings",
    "parse_zarr_z_step",
    "parse_zarr_expected_timepoints",
]


def __getattr__(name: str) -> Any:
    submodule = _LAZY.get(name)
    if submodule is not None:
        value = getattr(importlib.import_module(f".{submodule}", __name__), name)
        globals()[name] = value  # cache: later lookups skip __getattr__
        return value
    if not name.startswith("_"):
        # `opym.timeline`, `opym.petakit`, ... as attribute access, like the
        # eager package used to allow.
        try:
            return importlib.import_module(f".{name}", __name__)
        except ModuleNotFoundError as exc:
            if exc.name != f"{__name__}.{name}":
                raise
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
