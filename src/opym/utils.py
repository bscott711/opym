# Ruff style: Compliant
"""
Utilities for the OPM Cropper
"""

import json
import re
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
from skimage.registration import phase_cross_correlation


class OutputFormat(str, Enum):
    """Defines the allowed output formats."""

    ZARR = "ZARR"
    TIFF_SERIES = "TIFF_SERIES_SPLIT_C"

    def __str__(self):
        return self.value


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
        print(
            f"ERROR: Invalid ROI format: '{roi_str}'. "
            "Expected 'y_start:y_stop,x_start:x_stop'",
            file=sys.stderr,
        )
        sys.exit(1)

    y_str, x_str = roi_str.split(",")
    y_start, y_stop = map(int, y_str.strip().split(":"))
    x_start, x_stop = map(int, x_str.strip().split(":"))

    return (slice(y_start, y_stop), slice(x_start, x_stop))


# --- NEW ROI LOG FUNCTIONS ---


def _roi_to_tuple(roi: tuple[slice, slice]) -> tuple[int, int, int, int]:
    """Converts (slice(y1, y2), slice(x1, x2)) to (y1, y2, x1, x2)"""
    # Ensure slice attributes are not None before accessing
    y_start = roi[0].start if roi[0].start is not None else 0
    y_stop = roi[0].stop if roi[0].stop is not None else -1
    x_start = roi[1].start if roi[1].start is not None else 0
    x_stop = roi[1].stop if roi[1].stop is not None else -1
    return (y_start, y_stop, x_start, x_stop)


def _tuple_to_roi(tpl: tuple[int, int, int, int]) -> tuple[slice, slice]:
    """Converts (y1, y2, x1, x2) to (slice(y1, y2), slice(x1, x2))"""
    return (slice(tpl[0], tpl[1]), slice(tpl[2], tpl[3]))


def _tuple_to_cli_string(tpl: tuple[int, int, int, int]) -> str:
    """Converts (y1, y2, x1, x2) to 'y1:y2,x1:x2'"""
    return f"{tpl[0]}:{tpl[1]},{tpl[2]}:{tpl[3]}"


def save_rois_to_log(
    log_file: Path,
    base_file: Path,
    top_roi: tuple[slice, slice],
    bottom_roi: tuple[slice, slice],
):
    """Appends the ROIs for a given file to a central JSON log."""
    data = {}
    if log_file.exists():
        try:
            with log_file.open("r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Overwriting corrupted ROI log {log_file.name}")
            data = {}

    data[base_file.name] = {
        "top_roi": _roi_to_tuple(top_roi),
        "bottom_roi": _roi_to_tuple(bottom_roi),
    }

    try:
        # --- FIX: Create the output directory if it doesn't exist ---
        log_file.parent.mkdir(parents=True, exist_ok=True)
        # -----------------------------------------------------------

        with log_file.open("w") as f:
            json.dump(data, f, indent=4)
        print(f"✅ Saved ROIs for {base_file.name} to {log_file.name}")
    except Exception as e:
        print(f"Error saving ROI log: {e}", file=sys.stderr)


def load_rois_from_log(
    log_file: Path,
) -> dict[str, dict[str, tuple[int, int, int, int]]]:
    """Loads the ROI log. Returns an empty dict if not found."""
    if not log_file.exists():
        return {}
    try:
        with log_file.open("r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading ROI log: {e}", file=sys.stderr)
        return {}


def align_rois(
    mip_data: np.ndarray,
    top_roi: tuple[slice, slice],
    bottom_roi: tuple[slice, slice],
) -> tuple[slice, slice]:
    """
    Calculates the pixel shift between two ROIs from a 2D MIP
    using phase cross-correlation and returns the adjusted second ROI.

    Args:
        mip_data: The 2D (Y, X) Max Intensity Projection array.
        top_roi: (slice, slice) for the reference ROI (Y, X).
        bottom_roi: (slice, slice) for the target ROI (Y, X).

    Returns:
        The adjusted (slice, slice) for the bottom ROI.
    """
    print("Aligning ROIs using 2D MIP...")

    try:
        # Crop the data from the MIP for registration
        top_crop = mip_data[top_roi[0], top_roi[1]]
        bottom_crop_before = mip_data[bottom_roi[0], bottom_roi[1]]

        # --- 1. Calculate the shift ---
        shift, _, _ = phase_cross_correlation(
            top_crop, bottom_crop_before, upsample_factor=10
        )
        dy, dx = shift
        print(f"Detected shift (dy, dx): ({dy:.2f}, {dx:.2f}) pixels.")

        # --- 2. Apply the shift to create the new ROI slice ---
        old_y_start, old_y_end = bottom_roi[0].start, bottom_roi[0].stop
        old_x_start, old_x_end = bottom_roi[1].start, bottom_roi[1].stop

        # --- 3. SUBTRACT the shift (as in your original code) ---
        new_y_start = old_y_start - int(round(dy))
        new_y_end = old_y_end - int(round(dy))
        new_x_start = old_x_start - int(round(dx))
        new_x_end = old_x_end - int(round(dx))

        aligned_bottom_roi = (
            slice(new_y_start, new_y_end),
            slice(new_x_start, new_x_end),
        )
        print(f"Adjusted Bottom ROI Slice: {aligned_bottom_roi}")
        return aligned_bottom_roi

    except Exception as e:
        print(f"❌ ERROR during alignment: {e}")
        print("Returning original bottom ROI.")
        return bottom_roi
