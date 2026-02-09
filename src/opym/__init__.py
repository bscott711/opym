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
from .metadata import create_processing_log, parse_timestamps
from .petakit import (
    monitor_job_background,
    run_petakit_processing,
    submit_crop_and_save_sidecar,
    submit_remote_crop_job,
    submit_remote_deskew_job,
    wait_for_job,
)
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
)
from .viewer import (
    composite_viewer,
    create_mip,
    interactive_roi_selector,
    single_channel_viewer,
    visualize_alignment,
)

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
]
