# Ruff style: Compliant
"""
Consolidated ROI auto-detection: finds the biological-sample subregion
within each half (top camera / bottom camera) of a raw OPM frame and
returns a same-size (top_roi, bottom_roi) crop pair.

Before this module, two near-duplicate implementations existed:
- `auto_detect_rois()` in bioimaging/run_pipeline_cli.py -- force-fit a
  FIXED 576x1152 box regardless of actual detected signal extent, risking
  silently clipping real signal on an outlier dataset.
- `_auto_detect_rois()` in bioimaging/run_napari_opym.py -- regionprops/
  connected-components with an area filter, sizing the box as
  max(detected_extent, expected) -- only grows past the fixed box, never
  shrinks below real signal.

This module canonicalizes on the second (safer) algorithm, since the bulk
backfill runs unattended across ~150-300 heterogeneous datasets where a
silently-clipped crop would go unnoticed. Both prior implementations shared
the same `master_roi.json` {"max_h", "max_w"} cache contract; this module
preserves it, so any already-cached files remain valid.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.ndimage
import skimage.measure

EXPECTED_H = 576
EXPECTED_W = 1152


def compute_reference_projection(
    z_array: np.ndarray, timepoint: int | None = None
) -> np.ndarray:
    """Builds a 2D (Y, X) reference plane for ROI auto-detection: a true
    max-intensity projection over every leading axis (Z and/or C) at one
    timepoint.

    `timepoint=None` selects the middle timepoint of a full 5D
    (T, C, Z, Y, X) or (T, Z, C, Y, X) array (matches the CLI's historical
    convention). `timepoint=0` (or any int) selects that specific index --
    also the right call for an already T-sliced 4D array (e.g. a napari
    layer's `.data[0]`), where no further T-slicing should happen.
    """
    arr = z_array
    if arr.ndim >= 5:
        t = timepoint if timepoint is not None else arr.shape[0] // 2
        arr = arr[t]
    elif arr.ndim == 4 and timepoint is not None:
        arr = arr[timepoint]

    frame = np.asarray(arr)
    if frame.ndim > 2:
        proj_axes = tuple(range(frame.ndim - 2))
        frame = np.max(frame, axis=proj_axes)
    return frame


def _load_master_bounds(master_roi_path: Path | None) -> tuple[int, int]:
    if master_roi_path and master_roi_path.exists():
        try:
            with open(master_roi_path) as f:
                d = json.load(f)
                return d.get("max_h", EXPECTED_H), d.get("max_w", EXPECTED_W)
        except Exception:  # noqa: BLE001 - corrupt/partial cache is not fatal
            pass
    return EXPECTED_H, EXPECTED_W


def _find_half_roi(image_half: np.ndarray, offset_y: int, sigma: float, area_threshold: float) -> dict | None:
    smoothed = scipy.ndimage.gaussian_filter(image_half, sigma=sigma)
    thresh = np.mean(smoothed) + 2 * np.std(smoothed)
    mask = smoothed > thresh

    labels = skimage.measure.label(mask)
    props = skimage.measure.regionprops(labels)
    if not props:
        return None

    min_row, min_col = image_half.shape[0], image_half.shape[1]
    max_row, max_col = 0, 0
    found = False
    for p in props:
        if p.area > area_threshold:  # ignore tiny hot-pixel clusters
            r0, c0, r1, c1 = p.bbox
            min_row, min_col = min(min_row, r0), min(min_col, c0)
            max_row, max_col = max(max_row, r1), max(max_col, c1)
            found = True

    if not found:
        return None
    return {
        "ymin": min_row + offset_y,
        "ymax": max_row + offset_y,
        "xmin": min_col,
        "xmax": max_col,
    }


def auto_detect_rois(
    max_proj: np.ndarray,
    master_roi_path: Path | None = None,
    *,
    gaussian_sigma: float = 5.0,
    area_threshold: float = 500.0,
    expected_h: int = EXPECTED_H,
    expected_w: int = EXPECTED_W,
    needs_top: bool = True,
    needs_bot: bool = True,
) -> tuple[tuple[slice, slice] | None, tuple[slice, slice] | None]:
    """Detects the sample ROI in the top and bottom camera halves of
    `max_proj` (a 2D (Y, X) reference plane, see
    `compute_reference_projection`), unifies them to one common box size,
    and returns (top_roi, bottom_roi) as (slice, slice) pairs (or None for a
    half with no detected signal / not requested).

    Persists {"max_h", "max_w"} to `master_roi_path` so later calls against
    the same dataset (or sibling datasets sharing one `master_roi.json`,
    the existing per-dataset-directory convention) reproduce the exact same
    crop dimensions.
    """
    prior_h, prior_w = _load_master_bounds(master_roi_path)

    max_y, max_x = max_proj.shape
    half_y = max_y // 2

    top_roi_dict = (
        _find_half_roi(max_proj[:half_y, :], 0, gaussian_sigma, area_threshold)
        if needs_top
        else None
    )
    bot_roi_dict = (
        _find_half_roi(max_proj[half_y:, :], half_y, gaussian_sigma, area_threshold)
        if needs_bot
        else None
    )

    valid_rois = [r for r in (top_roi_dict, bot_roi_dict) if r is not None]
    if not valid_rois:
        return None, None

    detected_max_h = max(r["ymax"] - r["ymin"] for r in valid_rois)
    detected_max_w = max(r["xmax"] - r["xmin"] for r in valid_rois)

    # Enforce the box is AT LEAST the expected optical FOV size -- only
    # grows past it, never shrinks below real detected signal. Reuses any
    # previously-cached bounds too, so a dataset processed in multiple
    # passes stays consistent.
    max_h = max(detected_max_h, expected_h, prior_h)
    max_w = max(detected_max_w, expected_w, prior_w)

    if master_roi_path:
        try:
            with open(master_roi_path, "w") as f:
                json.dump({"max_h": max_h, "max_w": max_w}, f)
        except Exception:  # noqa: BLE001 - cache write failure is not fatal
            pass

    rois_out: list[tuple[slice, slice] | None] = []
    for r_dict in (top_roi_dict, bot_roi_dict):
        if r_dict is None:
            rois_out.append(None)
            continue

        y_center = (r_dict["ymin"] + r_dict["ymax"]) // 2
        x_center = (r_dict["xmin"] + r_dict["xmax"]) // 2

        new_ymin = max(0, y_center - max_h // 2)
        new_ymax = new_ymin + max_h

        new_xmin = max(0, x_center - max_w // 2)
        new_xmax = min(max_x, new_xmin + max_w)
        if new_xmax > max_x:
            new_xmax = max_x
            new_xmin = max(0, new_xmax - max_w)

        rois_out.append((slice(new_ymin, new_ymax), slice(new_xmin, new_xmax)))

    return rois_out[0], rois_out[1]
