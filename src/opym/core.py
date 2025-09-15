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

    # --- Calculate Geometric Parameters ---
    # Shear factor: pixels shifted in x per z-slice
    shear_factor = dz * np.cos(angle_rad) / dx
    # Z-anisotropy: new z-spacing in pixel units after projection
    z_aniso = dz * np.sin(angle_rad) / dx

    # --- Calculate Final Output Shape (as per MATLAB script) ---
    # The new dimensions are calculated based on the geometry of the rotation.
    # Note the coordinate system differences: MATLAB is (Y, X, Z), NumPy is (Z, Y, X).
    out_nz = int(
        np.round((nx - 1) * np.sin(angle_rad) + (nz - 1) * z_aniso * np.cos(angle_rad))
    )
    out_ny = ny
    out_nx = int(
        np.round((nx - 1) * np.cos(angle_rad) + (nz - 1) * z_aniso * np.sin(angle_rad))
    )
    output_shape = (out_nz, out_ny, out_nx)

    # --- Build Forward Transformation Matrix (Input -> Output) ---
    # 1. Deskew Matrix
    deskew_mat = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [shear_factor, 0, 1, 0], [0, 0, 0, 1]]
    )

    # 2. Translate to Center for Rotation
    center_in = np.array([(nz - 1) / 2, (ny - 1) / 2, (nx - 1) / 2])
    t1 = np.eye(4)
    t1[:3, 3] = -center_in

    # 3. Scale Z for Isotropic Rotation
    scale_mat = np.eye(4)
    scale_mat[0, 0] = z_aniso

    # 4. Rotate around Y-axis
    theta = -angle_rad
    rot_mat = np.eye(4)
    rot_mat[0, 0] = np.cos(theta)
    rot_mat[0, 2] = -np.sin(theta)
    rot_mat[2, 0] = np.sin(theta)
    rot_mat[2, 2] = np.cos(theta)

    # 5. Translate to Center of Output Volume
    center_out = np.array([(out_nz - 1) / 2, (out_ny - 1) / 2, (out_nx - 1) / 2])
    t2 = np.eye(4)
    t2[:3, 3] = center_out

    # Combine matrices to create the full forward transformation
    forward_transform = t2 @ rot_mat @ scale_mat @ t1 @ deskew_mat

    # --- Invert Matrix for the Library Function ---
    # The affine_transform function requires a matrix that maps points from
    # the output space back to the original input space.
    inverse_transform = np.linalg.inv(forward_transform)

    # Apply the transformation
    return affine_transform(
        image_stack,
        cp.asarray(inverse_transform),
        output_shape=output_shape,
        order=1,
        prefilter=False,
    )
