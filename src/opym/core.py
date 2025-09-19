# Ruff style: Compliant
# Description:
# This module contains the corrected core LLSM processing functions, adapted
# from the robust MATLAB-to-Python translation. It handles both LLSM and OPM
# data processing workflows.

import math
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage


def process_file(
    filepath: Path,
    output_dir: Path,
    params: dict,
) -> bool:
    """
    Load, crop (if needed), and process a single light sheet file.
    """
    try:
        # Load the raw data using tifffile
        raw_stack = tifffile.imread(filepath)
        print(f"  - Loaded raw data with shape: {raw_stack.shape}")

        # --- Get parameters from the metadata ---
        dx = params["dx"]
        dz = params["voxel_size_z"]
        angle = params["angle"]
        rois = params["rois"]
        microscope = params["microscope"]

        # --- Handle data based on microscope type ---
        if microscope == "OPM":
            # OPM processing would still require its specific transform.
            # For now, we focus on correcting the LLSM path.
            print(
                "  - OPM processing path is defined but not implemented with new logic."
            )
            # OPM data would be split and processed here.
            channels = crop_and_split_opm(raw_stack, rois)

        elif microscope == "LLSM":
            print("  - Starting LLSM processing pipeline...")
            # --- LLSM Processing Pipeline ---
            # Effective dz for sample scan is modified by the sine of the angle
            dz_data_eff = dz * math.sin(math.radians(angle))

            # 1. Deskew data
            print("  - Step 1: Deskewing data...")
            deskewed_data = deskew(
                image=raw_stack.astype(np.float32),
                angle=angle,
                dz=dz,
                xy_pixelsize=dx,
            )
            print(f"  - Data deskewed to shape: {deskewed_data.shape}")

            # 2. Final rotation
            print("  - Step 2: Rotating final volume...")
            zx_aspratio = dz_data_eff / dx
            processed_data = rotate_3d(deskewed_data, -angle, zx_aspratio)
            print(f"  - Volume rotated to shape: {processed_data.shape}")

            channels = [processed_data]  # Keep it in a list for uniform handling
        else:
            raise ValueError(f"Unknown microscope type: {microscope}")

        # --- Save Processed Channels ---
        for i, channel_stack in enumerate(channels):
            # OPM data would be uint16, LLSM is float32, can be converted if needed
            if microscope == "LLSM":
                final_image = channel_stack.astype(np.float32)
            else:  # Placeholder for OPM
                final_image = channel_stack.astype(np.uint16)

            output_filename = f"{filepath.stem}_channel_{i}_processed.tif"
            output_path = output_dir / output_filename
            tifffile.imwrite(
                output_path,
                final_image,
                imagej=True,
                metadata={"axes": "ZYX", "spacing": dx},
            )
            print(f"  - Saved processed channel {i} to: {output_path}")

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


# --- Core Image Transformation Functions from llsm_decon ---


def deskew(
    image: np.ndarray,
    angle: float,
    dz: float,
    xy_pixelsize: float,
    b_reverse: bool = False,
    trans: float = 0.0,
    fill_val: float = 0.0,
) -> np.ndarray:
    """
    Deskews a 3D image stack acquired at an angle.
    """
    shear_factor = math.cos(math.radians(angle)) * dz / xy_pixelsize
    if b_reverse:
        shear_factor *= -1

    matrix = np.array([[1, 0, 0], [0, 1, 0], [shear_factor, 0, 1]])

    original_shape = image.shape
    widen_by = math.ceil(abs(original_shape[0] * shear_factor))
    output_shape = (original_shape[0], original_shape[1], original_shape[2] + widen_by)

    offset = np.array([0, 0, -trans])

    deskewed_image = ndimage.affine_transform(
        image,
        matrix,
        offset=offset, # type: ignore
        output_shape=output_shape,
        order=1,
        cval=fill_val,
    )
    return deskewed_image


def rotate_3d(
    image: np.ndarray,
    angle: float,
    zx_aspratio: float,
    z_trans: float = 0.0,
) -> np.ndarray:
    """
    Rotates a 3D array around the Y-axis after correcting for voxel anisotropy.
    """
    zoomed_image = ndimage.zoom(image, (zx_aspratio, 1, 1), order=1)

    if z_trans != 0.0:
        zoomed_image = ndimage.shift(zoomed_image, (z_trans, 0, 0), order=1)

    rotated_image = ndimage.rotate(
        zoomed_image,
        angle,
        axes=(2, 0),
        reshape=True,
        order=1,
        cval=0.0,
    )
    return rotated_image
