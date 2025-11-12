# Ruff style: Compliant
"""
Utilities for handling, logging, and aligning Regions of Interest (ROIs).
"""

import json
import sys
from pathlib import Path
from typing import overload

import numpy as np
from skimage.registration import phase_cross_correlation


@overload
def _roi_to_tuple(roi: tuple[slice, slice]) -> tuple[int, int, int, int]: ...


@overload
def _roi_to_tuple(roi: None) -> None: ...


def _roi_to_tuple(
    roi: tuple[slice, slice] | None,
) -> tuple[int, int, int, int] | None:
    """Converts (slice(y1, y2), slice(x1, x2)) to (y1, y2, x1, x2) or None."""
    if roi is None:
        return None
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
    top_roi: tuple[slice, slice] | None,
    bottom_roi: tuple[slice, slice] | None,
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
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("w") as f:
            json.dump(data, f, indent=4)
        print(f"✅ Saved ROIs for {base_file.name} to {log_file.name}")
    except Exception as e:
        print(f"Error saving ROI log: {e}", file=sys.stderr)


def load_rois_from_log(
    log_file: Path,
) -> dict[str, dict[str, tuple[int, int, int, int] | None]]:
    """
    Loads the ROI log. Returns an empty dict if not found.
    Values for ROIs can be tuples or None.
    """
    if not log_file.exists():
        return {}
    try:
        with log_file.open("r") as f:
            data: dict[str, dict[str, tuple[int, int, int, int] | None]] = json.load(f)
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

        shift, _, _ = phase_cross_correlation(
            top_crop, bottom_crop_before, upsample_factor=10
        )
        dy, dx = shift
        print(f"Detected shift (dy, dx): ({dy:.2f}, {dx:.2f}) pixels.")

        old_y_start, old_y_end = bottom_roi[0].start, bottom_roi[0].stop
        old_x_start, old_x_end = bottom_roi[1].start, bottom_roi[1].stop

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


def process_rois_from_selector(
    mip_data: np.ndarray,
    unaligned_rois: list[tuple[slice, slice]],
    valid_threshold: float = 1.0,
    require_top: bool = True,
    require_bottom: bool = True,
) -> tuple[tuple[slice, slice] | None, tuple[slice, slice] | None]:
    """
    Validates ROIs from the selector, aligns if both are valid, and returns them.

    Args:
        mip_data: The 2D MIP data used for validation/alignment.
        unaligned_rois: The list of ROIs from selector.get_rois().
        valid_threshold: The mean pixel value above which an ROI is "valid".
        require_top: Whether a top ROI was expected.
        require_bottom: Whether a bottom ROI was expected.

    Returns:
        A tuple of (final_top_roi, final_bottom_roi), where either can be None.
    """
    final_top_roi: tuple[slice, slice] | None = None
    final_bottom_roi: tuple[slice, slice] | None = None

    # --- Case 1: Both ROIs required ---
    if require_top and require_bottom:
        if len(unaligned_rois) != 2:
            print("❌ ERROR: Expected 2 ROIs, but got different amount.")
            return None, None

        top_roi_unaligned = unaligned_rois[0]
        bottom_roi_unaligned = unaligned_rois[1]

        # Validate contents
        top_mean = np.mean(mip_data[top_roi_unaligned[0], top_roi_unaligned[1]])
        bot_mean = np.mean(mip_data[bottom_roi_unaligned[0], bottom_roi_unaligned[1]])

        top_valid = top_mean > valid_threshold
        bot_valid = bot_mean > valid_threshold

        print(f"  Top ROI mean: {top_mean:.2f} (Valid: {top_valid})")
        print(f"  Bottom ROI mean: {bot_mean:.2f} (Valid: {bot_valid})")

        if top_valid and bot_valid:
            print("\n--- Auto-Aligning ROIs ---")
            final_bottom_roi = align_rois(
                mip_data, top_roi_unaligned, bottom_roi_unaligned
            )
            final_top_roi = top_roi_unaligned
            print("\n✅ Both ROIs valid. Alignment complete.")
        elif top_valid:
            print("\nℹ️ Bottom ROI is empty. Using Top only.")
            final_top_roi = top_roi_unaligned
        elif bot_valid:
            print("\nℹ️ Top ROI is empty. Using Bottom only.")
            final_bottom_roi = bottom_roi_unaligned
        else:
            print("\n❌ ERROR: Both ROIs appear empty.")

    # --- Case 2: Top Only ---
    elif require_top and not require_bottom:
        if len(unaligned_rois) != 1:
            print("❌ ERROR: Expected 1 ROI (Top).")
            return None, None

        top_roi_unaligned = unaligned_rois[0]
        top_mean = np.mean(mip_data[top_roi_unaligned[0], top_roi_unaligned[1]])

        if top_mean > valid_threshold:
            print(f"✅ Top ROI valid (mean={top_mean:.2f}). Skipping alignment.")
            final_top_roi = top_roi_unaligned
        else:
            print(f"❌ Top ROI empty (mean={top_mean:.2f}).")

    # --- Case 3: Bottom Only ---
    elif require_bottom and not require_top:
        if len(unaligned_rois) != 1:
            print("❌ ERROR: Expected 1 ROI (Bottom).")
            return None, None

        bottom_roi_unaligned = unaligned_rois[0]
        bot_mean = np.mean(mip_data[bottom_roi_unaligned[0], bottom_roi_unaligned[1]])

        if bot_mean > valid_threshold:
            print(f"✅ Bottom ROI valid (mean={bot_mean:.2f}). Skipping alignment.")
            final_bottom_roi = bottom_roi_unaligned
        else:
            print(f"❌ Bottom ROI empty (mean={bot_mean:.2f}).")

    return final_top_roi, final_bottom_roi
