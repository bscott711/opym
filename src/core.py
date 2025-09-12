# Ruff style: Compliant
# Description:
# This module contains the core functions for the OPM processing pipeline,
# including data loading, de-interlacing, deskewing, and deconvolution.

import os
import tempfile
import time

import cupy as cp
import numpy as np
import tifffile
from cupyx.scipy.ndimage import affine_transform
from llspy import cudabinwrapper


def load_and_deinterlace(
    filepath: str, num_channels: int
) -> dict[int, np.ndarray]:
    """
    Loads a multi-page TIFF and de-interlaces it into separate channels.
    """
    print(f"Loading and de-interlacing {filepath}...")
    try:
        with tifffile.TiffFile(filepath) as tif:
            stack = tif.asarray()
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return {}

    if stack.ndim != 3:
        raise ValueError("Input TIFF must be a 3D stack (Z, Y, X).")

    channels = {i: stack[i::num_channels, :, :] for i in range(num_channels)}
    print(f"Separated into {num_channels} channels.")
    return channels


def deskew_gpu(
    volume: cp.ndarray,
    angle: float,
    voxel_size_z: float,
    voxel_size_xy: float,
) -> cp.ndarray:
    """
    Deskews a 3D volume on the GPU using an affine transformation.
    """
    angle_rad = np.deg2rad(angle)
    shear_factor = 1 / np.tan(angle_rad)
    transform_matrix = cp.asarray(
        [
            [voxel_size_z / voxel_size_xy * np.cos(angle_rad), 0, 0],
            [0, 1, 0],
            [shear_factor, 0, 1],
        ]
    )
    output_shape = (
        int(volume.shape[0] * (voxel_size_z / voxel_size_xy) * np.cos(angle_rad)),
        volume.shape[1],
        int(volume.shape[2] + volume.shape[0] * shear_factor),
    )
    print(f"Deskewing volume of shape {volume.shape} to {output_shape} on GPU...")
    return affine_transform(
        volume,
        matrix=transform_matrix,
        output_shape=output_shape,
        order=1,
        prefilter=True,
    )


def deconvolve_llspy_gpu(image: np.ndarray, psf: np.ndarray, num_iter: int) -> np.ndarray:
    """
    Deconvolves a 3D image using LLSpy's cudaDeconv binary.
    """
    print(f"Performing {num_iter} deconvolution iterations via LLSpy binary...")
    with tempfile.TemporaryDirectory() as tempdir:
        im_path = os.path.join(tempdir, "image.tif")
        psf_path = os.path.join(tempdir, "psf.tif")
        out_path = os.path.join(tempdir, "deconvolved.tif")

        tifffile.imwrite(im_path, image.astype(np.float32))
        tifffile.imwrite(psf_path, psf.astype(np.float32))

        cudabinwrapper.run(im_path, psf_path, out_path, n_iters=num_iter)

        if os.path.exists(out_path):
            with tifffile.TiffFile(out_path) as tif:
                return tif.asarray()
        raise RuntimeError("cudaDeconv binary failed to produce an output file.")


def process_channel(
    channel_id: int,
    skewed_stack: np.ndarray,
    psf_path: str,
    output_dir: str,
    base_filename: str,
    params: dict,
):
    """Processes a single channel through the entire pipeline."""
    print(f"\n--- Processing Channel {channel_id} ---")
    channel_start_time = time.time()

    if psf_path:
        print(f"Loading PSF from {psf_path}...")
        with tifffile.TiffFile(psf_path) as tif:
            psf = tif.asarray()
    else:
        # Generate a placeholder PSF if none is provided
        print("Warning: No PSF provided. Generating a placeholder Gaussian PSF.")
        psf_shape = (15, 15, 15)
        psf_sigma = 1.5
        psf_grid = np.mgrid[
            -psf_shape[0] // 2 + 1 : psf_shape[0] // 2 + 1,
            -psf_shape[1] // 2 + 1 : psf_shape[1] // 2 + 1,
            -psf_shape[2] // 2 + 1 : psf_shape[2] // 2 + 1,
        ]
        psf = np.exp(-((psf_grid**2).sum(0)) / (2 * psf_sigma**2))
        psf /= psf.sum()

    print("Transferring data to GPU for deskew...")
    skewed_stack_gpu = cp.asarray(skewed_stack, dtype=cp.float32)

    deskewed_gpu = deskew_gpu(
        skewed_stack_gpu,
        params["angle"],
        params["voxel_size_z"],
        params["voxel_size_xy"],
    )

    print("Transferring deskewed data back to CPU...")
    deskewed_cpu = cp.asnumpy(deskewed_gpu)
    del skewed_stack_gpu, deskewed_gpu
    cp.get_default_memory_pool().free_all_blocks()

    processed_stack = deconvolve_llspy_gpu(deskewed_cpu, psf, params["iterations"])

    output_filename = f"{base_filename}_channel_{channel_id}_processed.tif"
    output_filepath = os.path.join(output_dir, output_filename)

    print(f"Saving processed channel to {output_filepath}")
    processed_stack = np.clip(processed_stack, 0, 65535).astype(np.uint16)
    tifffile.imwrite(output_filepath, processed_stack, imagej=True)

    print(f"Channel {channel_id} processed in {time.time() - channel_start_time:.2f}s.")
