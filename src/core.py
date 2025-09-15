# Ruff style: Compliant
# Description:
# This file contains the core image processing functions for the opym pipeline.
# Imports are updated to use the local cudawrapper module.

import glob
import os
import tempfile
import time

import cupy as cp
import numpy as np
import tifffile
from cupyx.scipy.ndimage import affine_transform

from . import cudawrapper  # Updated import


def find_tiff_files(directory: str, prefix: str) -> list[str]:
    """Find all TIFF files in a directory that start with a given prefix."""
    search_path = os.path.join(directory, f"{prefix}*.tif")
    return sorted(glob.glob(search_path))


def load_and_deinterlace(filepath: str, num_channels: int) -> dict[int, np.ndarray]:
    """
    Load an interlaced TIFF and split it into separate channels.

    Returns a dictionary mapping channel ID to the 3D numpy array for that channel.
    """
    try:
        print(f"Loading and de-interlacing {os.path.basename(filepath)}...")
        with tifffile.TiffFile(filepath) as tif:
            interlaced_stack = tif.asarray()

        if interlaced_stack.ndim != 3:
            raise ValueError(
                f"Expected a 3D stack (frames, y, x), but got shape {interlaced_stack.shape}"
            )

        channel_stacks = {}
        for i in range(num_channels):
            channel_stacks[i] = interlaced_stack[i::num_channels, :, :]
        return channel_stacks
    except Exception as e:
        print(f"Error loading or de-interlacing file {filepath}: {e}")
        return {}


def deskew_gpu(
    stack_cpu: np.ndarray, angle: float, voxel_size_z: float, voxel_size_xy: float
) -> np.ndarray:
    """Deskew a 3D stack on the GPU using an affine transformation."""
    # Calculate the shear factor
    shear_factor = 1 / np.tan(np.deg2rad(angle))

    # Z-step per slice in microns
    z_step_um = voxel_size_z

    # XY pixel size in microns
    xy_pixel_size_um = voxel_size_xy

    # Calculate the shift per slice in pixels
    shift_per_slice_pixels = (z_step_um / xy_pixel_size_um) * shear_factor

    # Create the affine transformation matrix for deskewing
    # This matrix applies a shear in the x-z plane
    transform_matrix = np.array(
        [[1, 0, 0], [0, 1, 0], [shift_per_slice_pixels, 0, 1]]
    )

    # Transfer the data to the GPU
    stack_gpu = cp.asarray(stack_cpu)

    # Apply the affine transformation on the GPU
    deskewed_gpu = affine_transform(stack_gpu, transform_matrix, order=1, prefilter=False)

    # Transfer the result back to the CPU
    return cp.asnumpy(deskewed_gpu)


def deconvolve_channel(
    deskewed_stack: np.ndarray,
    psf_path: str | None,
    iterations: int,
    voxel_size_xy: float,
) -> np.ndarray:
    """Deconvolve a 3D stack using the cudaDeconv binary."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Prepare paths for temporary files
        img_path = os.path.join(tmpdir, "deskewed_image.tif")
        decon_path = os.path.join(tmpdir, "deconvolved_image.tif")

        # Use a default PSF if none is provided
        if psf_path is None or not os.path.exists(psf_path):
            print("Warning: No valid PSF provided. Using a generated Gaussian PSF.")
            psf = generate_psf(deskewed_stack.shape, voxel_size_xy)
            psf_path_to_use = os.path.join(tmpdir, "generated_psf.tif")
            tifffile.imwrite(psf_path_to_use, psf)
        else:
            psf_path_to_use = psf_path

        # Save the deskewed stack to the temporary file
        tifffile.imwrite(img_path, deskewed_stack.astype(np.float32))

        # Run the deconvolution
        result = cudawrapper.deconvolve(
            image=img_path,
            psf=psf_path_to_use,
            output=decon_path,
            iterations=iterations,
        )

        if result.returncode != 0:
            print(f"Error during deconvolution: {result.stderr}")
            return deskewed_stack  # Return the deskewed stack on failure

        # Load the deconvolved result
        deconvolved_stack = tifffile.imread(decon_path)

    return deconvolved_stack


def generate_psf(shape: tuple, voxel_size_xy: float) -> np.ndarray:
    """Generate a simple Gaussian PSF for testing."""
    sigma_xy_pix = 1.5
    sigma_z_pix = 3.0
    z, y, x = np.mgrid[-shape[0] // 2 : shape[0] // 2, -shape[1] // 2 : shape[1] // 2, -shape[2] // 2 : shape[2] // 2]
    psf = np.exp(
        -(
            (x**2 + y**2) / (2 * sigma_xy_pix**2)
            + (z**2) / (2 * sigma_z_pix**2)
        )
    )
    return (psf / psf.sum()).astype(np.float32)


def process_channel(
    channel_id: int,
    stack: np.ndarray,
    psf_path: str | None,
    output_dir: str,
    base_name: str,
    params: dict,
):
    """Run the full deskew and deconvolution pipeline for a single channel."""
    print(f"\n--- Processing Channel {channel_id} ---")

    # Deskew
    start_deskew = time.time()
    print("Deskewing on GPU...")
    deskewed = deskew_gpu(
        stack,
        params["angle"],
        params["voxel_size_z"],
        params["voxel_size_xy"],
    )
    print(f"Deskewing complete in {time.time() - start_deskew:.2f} seconds.")

    # Deconvolve
    start_decon = time.time()
    print("Deconvolving with cudaDeconv...")
    deconvolved = deconvolve_channel(
        deskewed, psf_path, params["iterations"], params["voxel_size_xy"]
    )
    print(f"Deconvolution complete in {time.time() - start_decon:.2f} seconds.")

    # Save the final result
    output_filename = f"{base_name}_ch{channel_id}_processed.tif"
    output_path = os.path.join(output_dir, output_filename)
    print(f"Saving final result to: {output_path}")
    tifffile.imwrite(output_path, deconvolved.astype(np.uint16))

