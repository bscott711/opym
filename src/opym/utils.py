# Ruff style: Compliant
"""
Core utilities, definitions, and path helpers for the OPM Cropper.
"""

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal


class OutputFormat(str, Enum):
    """Defines the allowed output formats."""

    ZARR = "ZARR"
    TIFF_SERIES = "TIFF_SERIES_SPLIT_C"

    def __str__(self):
        return self.value


MicroscopyDataType = Literal["LLSM", "OPM", "UNKNOWN"]


def detect_microscopy_data_type(directory: Path) -> MicroscopyDataType:
    """
    Detects the microscopy data type by inspecting filenames.

    Args:
        directory: The Path object to the data directory.

    Returns:
        "LLSM", "OPM", or "UNKNOWN".
    """
    if not directory.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {directory}")

    # --- MODIFICATION: Check for OPM using the new C..._T... format ---
    if next(directory.glob("*_C[0-9]_T[0-9][0-9][0-9].tif"), None):
        return "OPM"

    # Check for LLSM
    if next(
        directory.glob("*_Cam[AB]_ch[0-9]_stack[0-9][0-9][0-9][0-9]*.tif"),
        None,
    ):
        return "LLSM"

    return "UNKNOWN"


@dataclass(frozen=True)
class DerivedPaths:
    """Holds all paths derived from the base input file."""

    base_file: Path
    metadata_file: Path
    output_dir: Path
    output_log: Path
    sanitized_name: str


def sanitize_filename(name: str) -> str:
    """Removes .ome.tif and replaces spaces."""
    return name.replace(".ome.tif", "").replace(" ", "_")


def derive_paths(base_file: Path, output_format: OutputFormat) -> DerivedPaths:
    """Derives all associated input and output paths from the base file."""
    base_name_no_ext = base_file.name.replace(".ome.tif", "")
    sanitized_name = sanitize_filename(base_file.name)
    metadata_file = base_file.parent / (base_name_no_ext + "_metadata.txt")

    if output_format == OutputFormat.ZARR:
        output_dir_name = "processed_ngff"
    else:
        output_dir_name = "processed_tiff_series_split"

    output_dir = base_file.parent / output_dir_name
    output_log = output_dir / (sanitized_name + "_processing_log.json")

    return DerivedPaths(
        base_file=base_file,
        metadata_file=metadata_file,
        output_dir=output_dir,
        output_log=output_log,
        sanitized_name=sanitized_name,
    )


def parse_roi_string(roi_str: str) -> tuple[slice, slice]:
    """
    Parses a CLI string like "y1:y2, x1:x2" into a NumPy slice.
    e.g., "0:512, 0:512" -> (slice(0, 512), slice(0, 512))
    """
    if not re.match(r"^\d+:\d+,\s*\d+:\d+$", roi_str):
        raise ValueError(
            f"Invalid ROI format: '{roi_str}'. Expected 'y_start:y_stop,x_start:x_stop'"
        )

    y_str, x_str = roi_str.split(",")
    y_start, y_stop = map(int, y_str.strip().split(":"))
    x_start, x_stop = map(int, x_str.strip().split(":"))

    return (slice(y_start, y_stop), slice(x_start, x_stop))
