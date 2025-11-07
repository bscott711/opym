# Ruff style: Compliant
"""
opym: OPM Cropper Package
"""

# Expose the main functions for library use (e.g., in notebooks)
from .core import process_dataset, run_processing_job
from .dataloader import (
    load_llsm_tiff_series,
    load_tiff_series,
)
from .metadata import create_processing_log, parse_timestamps
from .petakit import (
    PetaKitContext,
    get_petakit_context,
    run_llsm_petakit_processing,
    run_petakit_from_config,
    run_petakit_processing,
)

# --- MODIFIED: Import from new roi_utils module ---
from .roi_utils import (
    align_rois,
    load_rois_from_log,
    save_rois_to_log,
)
from .utils import (
    DerivedPaths,
    MicroscopyDataType,
    OutputFormat,
    derive_paths,
    detect_microscopy_data_type,
    parse_roi_string,
)

# --- END MODIFICATION ---
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
    "single_channel_viewer",
    "composite_viewer",
    "load_tiff_series",
    "load_llsm_tiff_series",
    "create_mip",
    "interactive_roi_selector",
    "visualize_alignment",
    "PetaKitContext",
    "get_petakit_context",
    "run_petakit_processing",
    "run_llsm_petakit_processing",
    "run_petakit_from_config",
    "detect_microscopy_data_type",
    "MicroscopyDataType",
]
