# Ruff style: Compliant
# Description:
# This module contains the core OPM processing functions.

from pathlib import Path

import cupy as cp
import numpy as np
import tifffile
from cupyx.scipy.ndimage import affine_transform


def process_file(
    filepath: Path,
    output_dir: Path,
    params: dict,
) -> bool:
    """
    Load, crop (if needed), and process a single light sheet file with a
    combined deskew and rotation transformation.
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
            channels = crop_and_split_opm(raw_stack, rois)
            print(f"  - Cropped and split into {len(channels)} channels.")
        elif microscope == "LLSM":
            channels = [raw_stack]
            print("  - Treating LLSM file as a single channel stack.")
        else:
            raise ValueError(f"Unknown microscope type: {microscope}")

        for i, channel_stack in enumerate(channels):
            print(f"\n  --- Processing Channel {i} ---")

            # --- Combined Deskew and Rotation ---
            print("  - Applying combined deskew and rotation on GPU...")
            processed_gpu = combined_transform(cp.asarray(channel_stack), dz, dx, angle)
            processed_shape = tuple(int(dim) for dim in processed_gpu.shape)
            print(f"  - Final shape: {processed_shape}")

            # --- Save Processed Channel ---
            final_image = cp.asnumpy(processed_gpu).astype(np.uint16)
            output_filename = f"{filepath.stem}_channel_{i}_processed.tif"
            output_path = output_dir / output_filename
            tifffile.imwrite(
                output_path,
                final_image,
                imagej=True,
                metadata={"axes": "ZYX"},
            )
            print(f"  - Saved processed channel to: {output_path}")

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


def combined_transform(
    image_stack: cp.ndarray, dz: float, dx: float, angle: float
) -> cp.ndarray:
    """
    Applies a single, combined affine transformation for deskewing and
    rotation, meticulously following the logic from the MATLAB script
    deskewRotateFrame3D.m and accounting for row-major ordering.
    """
    nz, ny, nx = image_stack.shape
    angle_rad = np.deg2rad(angle)

    # --- Calculate Geometric Parameters ---
    # Z-anisotropy: new z-spacing in pixel units after projection
    z_aniso = dz * np.sin(np.abs(angle_rad)) / dx

    # --- Calculate Final Output Shape (as per MATLAB script) ---
    # This is a direct translation, accounting for Python's (Z, Y, X) order
    out_nz = int(np.round((nx - 1) * np.sin(np.abs(angle_rad)) - 4))
    out_ny = ny
    out_nx = int(
        np.round(
            (nx - 1) * np.cos(angle_rad)
            + (nz - 1) * z_aniso / np.sin(np.abs(angle_rad))
        )
    )
    output_shape = (out_nz, out_ny, out_nx)

    # --- Build the Forward Transformation Matrix (Input -> Output) ---
    # This matrix maps points from the original image space (Z,Y,X) to the new,
    # transformed space, following the MATLAB logic.

    # 1. Shear transformation (Deskew)
    x_step = dz * np.cos(angle_rad) / dx
    x_shift = -x_step
    ds_s = np.array([[1, 0, x_step, x_shift], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    # 2. Center, Scale, Rotate, and Translate back
    center_in = np.array([(nx - 1) / 2, (ny - 1) / 2, (nz - 1) / 2])
    t1 = np.eye(4)
    t1[:3, 3] = -center_in

    s = np.eye(4)
    s[2, 2] = z_aniso

    theta = -angle_rad
    r = np.eye(4)
    r[0, 0] = np.cos(theta)
    r[0, 2] = -np.sin(theta)
    r[2, 0] = np.sin(theta)
    r[2, 2] = np.cos(theta)

    center_out = np.array([(out_nx - 1) / 2, (out_ny - 1) / 2, (out_nz - 1) / 2])
    t2 = np.eye(4)
    t2[:3, 3] = center_out

    # Combine all matrices to create the full forward transformation
    forward_transform = ds_s @ t1 @ s @ r @ t2

    # --- Invert Matrix for the Library Function ---
    inverse_transform = np.linalg.inv(forward_transform)

    return affine_transform(
        image_stack,
        cp.asarray(inverse_transform),
        output_shape=output_shape,
        order=1,
        prefilter=False,
    )
