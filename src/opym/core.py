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
        # Using the exact coordinates you provided.
        # Format is (y_start, y_end, x_start, x_end).
        rois = [
            (657, 1161, 1224, 2262),  # ROI for channels 0 and 1
            (1296, 1800, 1224, 2262),  # ROI for channels 2 and 3
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
    rotation to coverslip coordinates.
    """
    nz, ny, nx = image_stack.shape
    angle_rad = np.deg2rad(angle)

    # --- Build the forward transformation matrix ---
    # This maps points from the original image to the new, transformed space.

    # 1. Deskew (shear)
    deskew_mat = np.eye(4)
    deskew_mat[2, 0] = dz * np.cos(angle_rad) / dx

    # 2. Z-scaling (to make voxels isotropic before rotation)
    scale_mat = np.eye(4)
    scale_mat[0, 0] = np.sin(angle_rad)

    # 3. Rotation around Y-axis
    rot_mat = np.eye(4)
    theta = -angle_rad
    rot_mat[0, 0] = np.cos(theta)
    rot_mat[0, 2] = -np.sin(theta)
    rot_mat[2, 0] = np.sin(theta)
    rot_mat[2, 2] = np.cos(theta)

    # Combine transformations
    forward_transform = rot_mat @ scale_mat @ deskew_mat

    # --- Calculate the output shape ---
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

    transformed_corners = forward_transform @ corners
    min_coords = transformed_corners[:3].min(axis=1)
    max_coords = transformed_corners[:3].max(axis=1)
    output_shape = tuple(np.ceil(max_coords - min_coords).astype(int))

    # --- Build the inverse transformation matrix for cupyx.scipy.ndimage ---
    # This maps points from the new, transformed space back to the original image.

    # Create a translation matrix to move the transformed volume to the origin
    translation_mat = np.eye(4)
    translation_mat[:3, 3] = -min_coords

    # The final forward transformation matrix
    final_forward_mat = translation_mat @ forward_transform

    # The inverse transformation matrix
    inverse_mat = np.linalg.inv(final_forward_mat)

    return affine_transform(
        image_stack,
        cp.asarray(inverse_mat),
        output_shape=output_shape,
        order=1,
        prefilter=False,
    )
