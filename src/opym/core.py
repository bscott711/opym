# Ruff style: Compliant
# Description:
# This module contains the core processing functions. It uses the
# OfflineDeskewer class to generate 2D maximum intensity projections from
# raw 3D light-sheet data.

from pathlib import Path

import numpy as np
import tifffile

# Import the new standalone deskewer class
from .deskewer import OfflineDeskewer


def process_file(
    filepath: Path,
    output_dir: Path,
    params: dict,
) -> bool:
    """
    Load, process, and save a single light-sheet file as a deskewed
    2D projection.
    """
    try:
        # Load the raw data using tifffile
        raw_stack = tifffile.imread(filepath)
        print(f"  - Loaded raw data with shape: {raw_stack.shape}")

        # --- Instantiate the deskewer with parameters from metadata ---
        deskewer = OfflineDeskewer(
            sheet_angle_deg=params["angle"],
            xy_pixel_pitch_um=params["dx"],
            z_step_um=params["voxel_size_z"],
        )

        microscope = params["microscope"]
        dx = params["dx"]
        rois = params["rois"]
        projections = []

        # --- Handle data based on microscope type ---
        if microscope == "OPM":
            print("  - Starting OPM processing pipeline...")
            # OPM data is split into channels first
            channels = crop_and_split_opm(raw_stack, rois)
            print(f"  - Split raw data into {len(channels)} channels.")
            for i, channel_stack in enumerate(channels):
                print(f"  - Deskewing channel {i}...")
                projections.append(deskewer.deskew_stack(channel_stack))

        elif microscope == "LLSM":
            print("  - Starting LLSM processing pipeline...")
            # LLSM data is a single channel
            projections.append(deskewer.deskew_stack(raw_stack))
        else:
            raise ValueError(f"Unknown microscope type: {microscope}")

        # --- Save Processed Projections ---
        for i, projection in enumerate(projections):
            output_filename = f"{filepath.stem}_channel_{i}_deskewed_projection.tif"
            output_path = output_dir / output_filename
            tifffile.imwrite(
                output_path,
                projection.astype(np.uint16),
                imagej=True,
                metadata={"axes": "YX", "spacing": dx},
            )
            print(f"  - Saved deskewed projection {i} to: {output_path}")

        return True

    except Exception as e:
        print(f"  - ERROR processing {filepath.name}: {e}")
        return False


def crop_and_split_opm(raw_stack: np.ndarray, rois: list) -> list:
    """
    Crops the OPM raw data into four channels from two cameras based on
    a list of regions of interest (ROIs).
    """
    cam1_data = raw_stack[:, 0, ...]
    cam2_data = raw_stack[:, 1, ...]

    y1_start, y1_end, x1_start, x1_end = rois[0]
    ch0 = cam1_data[:, y1_start:y1_end, x1_start:x1_end]
    ch1 = cam2_data[:, y1_start:y1_end, x1_start:x1_end]

    y2_start, y2_end, x2_start, x2_end = rois[1]
    ch2 = cam1_data[:, y2_start:y2_end, x2_start:x2_end]
    ch3 = cam2_data[:, y2_start:y2_end, x2_start:x2_end]

    return [ch0, ch1, ch2, ch3]
