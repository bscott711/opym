# Ruff style: Compliant
# Description:
# This module contains the core OPM processing functions.

from pathlib import Path

import cupy as cp
import numpy as np
import tifffile
from cupyx.scipy.ndimage import affine_transform
from pycudadecon import rl_decon


def process_file(
    filepath: Path,
    output_dir: Path,
    dz: float,
    angle: float,
    decon_iterations: int,
) -> bool:
    """
    Load, de-interlace, deskew, and deconvolve a single OPM file.

    Returns:
        bool: True if processing was successful, False otherwise.
    """
    try:
        # Load the raw data using tifffile
        raw_stack = tifffile.imread(filepath)
        print(f"  - Loaded raw data with shape: {raw_stack.shape}")

        # --- De-interlace Channels ---
        if raw_stack.ndim != 4:
            raise ValueError(
                f"Expected a 4D stack (Z, C, Y, X), but got shape {raw_stack.shape}"
            )
        num_channels = raw_stack.shape[1]
        channels = [raw_stack[:, c, :, :] for c in range(num_channels)]
        print(f"  - De-interlaced into {len(channels)} channels.")

        for i, channel_stack in enumerate(channels):
            print(f"\n  --- Processing Channel {i} ---")

            # --- Deskewing ---
            print("  - Deskewing on GPU...")
            deskewed_gpu = deskew_on_gpu(cp.asarray(channel_stack), dz, angle)
            print(f"  - Deskewed shape: {deskewed_gpu.shape}")

            # --- Deconvolution ---
            # NOTE: For real data, you must load an experimentally measured PSF.
            # Here, we generate a placeholder PSF for demonstration.
            # FIX: Convert the CuPy shape object to a standard Python tuple.
            psf = generate_psf(tuple(deskewed_gpu.shape))
            print(f"  - Generated placeholder PSF with shape: {psf.shape}")

            psf_gpu = cp.asarray(psf)
            print(f"  - Deconvolving for {decon_iterations} iterations on GPU...")
            deconvolved_gpu = rl_decon(deskewed_gpu, psf_gpu, n_iters=decon_iterations)
            print("  - Deconvolution complete.")

            # --- Save Processed Channel ---
            final_image = cp.asnumpy(deconvolved_gpu).astype(np.uint16)
            output_filename = f"{filepath.stem}_channel_{i}_processed.tif"
            output_path = output_dir / output_filename
            tifffile.imwrite(output_path, final_image, imagej=True)
            print(f"  - Saved processed channel to: {output_path}")

        return True

    except Exception as e:
        print(f"  - ERROR processing {filepath.name}: {e}")
        return False


def deskew_on_gpu(
    image_stack: cp.ndarray, dz: float, angle: float = 31.5
) -> cp.ndarray:
    """Deskews a 3D image stack on the GPU using an affine transform."""
    angle_rad = cp.deg2rad(angle)
    shear_matrix = cp.array(
        [[1, 0, 0], [0, 1, 0], [-cp.cos(angle_rad) / cp.sin(angle_rad) * dz, 0, 1]]
    )
    return affine_transform(image_stack, shear_matrix, order=1, prefilter=False)


def generate_psf(shape: tuple) -> np.ndarray:
    """Generates a simple Gaussian PSF for demonstration."""
    sigma = [2, 5, 5]
    psf = np.zeros(shape, dtype=np.float32)
    center = tuple(s // 2 for s in shape)
    psf[center] = 1
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(psf, sigma=sigma).astype(np.float32)
