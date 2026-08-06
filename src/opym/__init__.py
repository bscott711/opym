# Ruff style: Compliant
"""
opym: OPM Cropper Package
"""

from __future__ import annotations

# Expose the main functions for library use (e.g., in notebooks)
from .batch import run_batch_cropping
from .core import process_dataset, run_processing_job
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
from .registry import StatusRegistry
from .roi_detect import auto_detect_rois, compute_reference_projection
from .roi_utils import (
    align_rois,
    load_rois_from_log,
    process_rois_from_selector,
    save_rois_to_log,
)
from .ui import create_crop_settings_ui, create_deskew_ui
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

# New PSF and Decon Widgets
from .widgets import DeconvolutionViewer, PSFAverager, PSFExtractor

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
