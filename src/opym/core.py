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
    dx: float,
    dz: float,
    angle: float,
) -> bool:
    """
    Load, crop, de-interlace, and process a single OPM file with a
    combined deskew and rotation transformation.

    Returns:
        bool: True if processing was successful, False otherwise.
    """
    try:
        # Load the raw data using tifffile
        raw_stack = tifffile.imread(filepath)
        print(f"  - Loaded raw data with shape: {raw_stack.shape}")

        # --- Define ROIs and Split Channels ---
        rois = [
            (657, 1161, 1224, 2262),  # ch0 (cam1) & ch1 (cam2)
            (1296, 1800, 1224, 2262),  # ch2 (cam1) & ch3 (cam2)
        ]
        channels = crop_and_split_channels(raw_stack, rois)
        print(f"  - Cropped and split into {len(channels)} channels.")

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


def crop_and_split_channels(raw_stack: np.ndarray, rois: list) -> list:
    """
    Crops the raw data into four channels from two cameras based on
    a list of regions of interest (ROIs).
    """
    cam1_data = raw_stack[:, 0, ...]
    cam2_data = raw_stack[:, 1, ...]

    # ROI for ch0 and ch1
    y1_start, y1_end, x1_start, x1_end = rois[0]
    ch0 = cam1_data[:, y1_start:y1_end, x1_start:x1_end]
    ch1 = cam2_data[:, y1_start:y1_end, x1_start:x1_end]

    # ROI for ch2 and ch3
    y2_start, y2_end, x2_start, x2_end = rois[1]
    ch2 = cam1_data[:, y2_start:y2_end, x2_start:x2_end]
    ch3 = cam2_data[:, y2_start:y2_end, x2_start:x2_end]

    return [ch0, ch1, ch2, ch3]


def combined_transform(
    image_stack: cp.ndarray, dz: float, dx: float, angle: float
) -> cp.ndarray:
    """
    Applies a single, combined affine transformation for deskewing and
    rotation to coverslip coordinates, based on the logic from the
    deskewRotateFrame3D.m script.
    """
    nz, ny, nx = image_stack.shape
    angle_rad = np.deg2rad(angle)

    # --- Build the forward transformation matrix (input -> output) ---
    # This matrix maps points from the original image space (Z,Y,X) to the new,
    # transformed space.

    # 1. Deskew (shear X as a function of Z)
    # x_new = x_old + z_old * shear_factor
    shear_factor = dz * np.cos(angle_rad) / dx
    deskew_mat = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64
    )
    deskew_mat[2, 0] = shear_factor  # This should affect X based on Z

    # 2. Scale Z to make voxels isotropic for rotation
    z_aniso = dz * np.sin(angle_rad) / dx
    scale_mat = np.eye(4, dtype=np.float64)
    scale_mat[0, 0] = z_aniso

    # 3. Rotate around the Y-axis
    theta = -angle_rad
    rot_mat = np.eye(4, dtype=np.float64)
    rot_mat[0, 0] = np.cos(theta)
    rot_mat[0, 2] = -np.sin(theta)
    rot_mat[2, 0] = np.sin(theta)
    rot_mat[2, 2] = np.cos(theta)

    # Combine the deskew, scaling, and rotation
    transform = rot_mat @ scale_mat @ deskew_mat

    # --- Calculate the output shape and offset ---
    # Find the bounding box of the transformed volume
    corners = np.array(
        [
            [0, 0, 0, 1],
            [0, 0, nx, 1],
            [0, ny, 0, 1],
            [0, ny, nx, 1],
            [nz, 0, 0, 1],
            [nz, 0, nx, 1],
            [nz, ny, 0, 1],
            [nz, ny, nx, 1],
        ]
    ).T

    transformed_corners = transform @ corners
    min_coords = transformed_corners[:3].min(axis=1)
    max_coords = transformed_corners[:3].max(axis=1)

    # The final output shape is the size of this bounding box
    output_shape = tuple(np.ceil(max_coords - min_coords).astype(int))

    # --- Build the inverse transformation matrix for the library function ---
    # This maps points from the new, transformed space back to the original image.

    # Create a translation matrix to move the transformed volume's corner to the origin
    offset_mat = np.eye(4, dtype=np.float64)
    offset_mat[:3, 3] = -min_coords

    # The final forward matrix includes this offset
    final_forward_mat = offset_mat @ transform

    # The inverse matrix is what we need for the transformation function
    inverse_mat = np.linalg.inv(final_forward_mat)

    return affine_transform(
        image_stack,
        cp.asarray(inverse_mat),
        output_shape=output_shape,
        order=1,
        prefilter=False,
    )
