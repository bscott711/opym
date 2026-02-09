# Ruff style: Compliant
"""
Contains the MIP (Maximum Intensity Projection) generation utility.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import tifffile
import zarr
from tqdm.auto import tqdm


def create_mip(
    file_path: Path | str, t_index: int = 0
) -> tuple[np.ndarray, float, float, zarr.Array, int]:
    """
    Opens a 5D OME-TIF as a lazy Zarr array and computes the
    max intensity projection for a specific timepoint.

    Args:
        file_path: Path to the OME-TIF file.
        t_index: The T index to project.

    Returns:
        A tuple containing:
        - mip_data (np.ndarray): The 2D (Y, X) MIP array.
        - vmin (float): 1st percentile for contrast.
        - vmax (float): 99.9th percentile for contrast.
        - lazy_data (zarr.Array): The opened zarr array (4D or 5D).
        - t_max (int): The maximum T index.
    """
    file_path = Path(file_path)
    print(f"Opening {file_path.name} as lazy Zarr array...")

    try:
        # --- NEW: Handle 4D (single timepoint) vs 5D data (STREAM-SAFE) ---
        store = tifffile.TiffFile(file_path).series[0].aszarr()
        lazy_data: zarr.Array = zarr.open_array(store, mode="r")

        shape = lazy_data.shape
        ndim = lazy_data.ndim

        if ndim == 4:  # ZCYX
            print(f"  Info: 4D data detected (shape {shape}). Assuming T=1.")
            T = 1
            Z, C, Y, X = shape
        elif ndim == 5:  # TZCXY
            print(f"  Info: 5D data detected (shape {shape}).")
            T, Z, C, Y, X = shape
        else:
            # Raise error for 3D or 6D+ data
            raise ValueError(
                f"Unsupported data shape: {shape}. Expected 4D (ZCYX) or 5D (TZCXY)."
            )
        # --- END NEW ---

    except Exception as e:
        print(f"❌ ERROR: Could not open {file_path.name} as zarr.")
        print(f"  Details: {e}")
        raise

    # This print statement will now show the *true* shape (4D or 5D)
    print(f"Full data shape: {lazy_data.shape}")

    if t_index >= T:
        print(f"Warning: T_INDEX ({t_index}) is out of range. Using T=0.")
        t_index = 0

    print(f"Selecting data for Max Projection: (T={t_index})")

    # --- NEW: Select correct indexing based on dimensions ---
    # In both cases, stack_to_project becomes a 4D (Z, C, Y, X) array
    if ndim == 5:
        stack_to_project = cast(zarr.Array, lazy_data[t_index, :, :, :, :])
    else:  # ndim == 4
        # T is 1, so t_index is 0. The array is already (Z, C, Y, X)
        stack_to_project = lazy_data
    # --- END NEW ---

    total_planes = Z * C
    print(f"Calculating Max Projection from {total_planes} (Z*C) planes...")

    # Indexing [0, 0, :, :] works on the 4D stack_to_project
    plane_0: np.ndarray = np.asarray(stack_to_project[0, 0, :, :])
    z_mip = np.copy(plane_0)

    with tqdm(total=total_planes, desc="  Projecting") as pbar:
        for z in range(Z):
            for c in range(C):
                if z == 0 and c == 0:
                    pbar.update(1)
                    continue
                # Indexing [z, c, :, :] also works on the 4D stack_to_project
                plane: np.ndarray = np.asarray(stack_to_project[z, c, :, :])
                np.maximum(z_mip, plane, out=z_mip)
                pbar.update(1)

    print("✅ Max Projection complete.")

    if z_mip.max() > 0:
        vmin, vmax = np.percentile(z_mip, [1, 99.9])
        if vmax <= vmin:
            vmax = z_mip.max()
    else:
        vmin, vmax = 0, 1

    print(f"  MIP display range (vmin, vmax): ({vmin:.0f}, {vmax:.0f})")
    # Return the original 4D or 5D array, T-1 is the max index
    return z_mip, float(vmin), float(vmax), lazy_data, T - 1
