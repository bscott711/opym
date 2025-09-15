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
        # NOTE: These ROI coordinates are placeholders based on the previous discussion.
        # Format is (y_start, y_end, x_start, x_end).
        rois = [
            (1200, 2400, 0, 2400),  # ch0: cam1_bottom (blue)
            (0, 1200, 0, 2400),  # ch1: cam1_top (green)
            (0, 1200, 0, 2400),  # ch2: cam2_top (red)
            (1200, 2400, 0, 2400),  # ch3: cam2_bottom (far red)
        ]
        channels = crop_and_split_channels(raw_stack, rois)
        print(f"  - Cropped and split into {len(channels)} channels.")

        # Standard pixel size for this type of microscope
        dx = 0.104

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

    y1_start, y1_end, x1_start, x1_end = rois[0]
    ch0 = cam1_data[:, y1_start:y1_end, x1_start:x1_end]

    y2_start, y2_end, x2_start, x2_end = rois[1]
    ch1 = cam1_data[:, y2_start:y2_end, x2_start:x2_end]

    y3_start, y3_end, x3_start, x3_end = rois[2]
    ch2 = cam2_data[:, y3_start:y3_end, x3_start:x3_end]

    y4_start, y4_end, x4_start, x4_end = rois[3]
    ch3 = cam2_data[:, y4_start:y4_end, x4_start:x4_end]

    return [ch0, ch1, ch2, ch3]


def combined_transform(
    image_stack: cp.ndarray, dz: float, dx: float, angle: float
) -> cp.ndarray:
    """
    Applies a single, combined affine transformation for deskewing and
    rotation to coverslip coordinates, correctly handling memory allocation.
    """
    nz, ny, nx = image_stack.shape
    angle_rad = np.deg2rad(angle)

    # --- Check Skew Factor ---
    fsk = dz / (dx / np.tan(angle_rad))
    if fsk > 2:
        print(
            f"  - WARNING: Skew factor is {fsk:.2f} > 2. "
            "Interpolation of raw data may be required for optimal results."
        )

    # --- Build Transformation Matrices (Forward mapping) ---
    # NOTE: These matrices map points from the original image to the new canvas.
    # The coordinate system is (Z, Y, X).

    # 1. Deskew (shear X based on Z)
    s_ds = cp.eye(4)
    s_ds[2, 0] = dz * np.cos(angle_rad) / dx

    # 2. Translate to center for rotation
    t1 = cp.eye(4)
    t1[:3, 3] = cp.array([-nz / 2, -ny / 2, -nx / 2])

    # 3. Z-scaling for isotropic voxels after rotation
    s_z = cp.eye(4)
    s_z[0, 0] = np.sin(angle_rad)

    # 4. Rotate around Y-axis
    theta = -angle_rad
    r_y = cp.eye(4)
    r_y[0, 0] = np.cos(theta)
    r_y[0, 2] = -np.sin(theta)
    r_y[2, 0] = np.sin(theta)
    r_y[2, 2] = np.cos(theta)

    # --- Calculate Final Output Shape and Offset ---
    # Combine all forward transformations except the final translation
    transform_no_offset = r_y @ s_z @ t1 @ s_ds

    # Find the bounding box of the transformed volume
    corners = cp.array(
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
    transformed_corners = transform_no_offset @ corners
    min_coords = transformed_corners[:3].min(axis=1)
    max_coords = transformed_corners[:3].max(axis=1)

    # The final output shape is the size of this bounding box
    output_shape = tuple(cp.ceil(max_coords - min_coords).astype(int).get())

    # Offset is the translation needed to bring the most negative corner to the origin
    offset = min_coords

    # --- Create the final inverse matrix for the transform function ---
    # The affine_transform function requires a matrix that maps output
    # coordinates to input coordinates (the inverse transformation).

    # Create the final translation matrix
    t2 = cp.eye(4)
    t2[:3, 3] = -offset

    # Combine all matrices and take the inverse
    full_forward_transform = t2 @ transform_no_offset
    inverse_transform = cp.linalg.inv(full_forward_transform)

    return affine_transform(
        image_stack,
        inverse_transform,
        output_shape=output_shape,
        order=1,
        prefilter=False,
    )
