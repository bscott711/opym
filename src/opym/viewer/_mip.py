# Ruff style: Compliant
"""
Contains the MIP (Maximum Intensity Projection) generation utility.
"""

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

    This function correctly handles both 5D (T, Z, C, Y, X) and
    4D (Z, C, Y, X) OME-TIF files.

    Args:
        file_path: Path to the OME-TIF file.
        t_index: The T index to project.

    Returns:
        A tuple containing:
        - mip_data (np.ndarray): The 2D (Y, X) MIP array.
        - vmin (float): 1st percentile for contrast.
        - vmax (float): 99.9th percentile for contrast.
        - lazy_data (zarr.Array): The opened zarr array (still 4D or 5D).
        - t_max (int): The maximum T index (0 for 4D files).
    """
    file_path = Path(file_path)
    print(f"Opening {file_path.name} as lazy Zarr array...")

    try:
        store = tifffile.TiffFile(file_path).series[0].aszarr()
        lazy_data = zarr.open(store, mode="r")

    except Exception as e:
        print(f"❌ ERROR: Could not open {file_path.name} as zarr.")
        print(f"  Details: {e}")
        raise

    if not isinstance(lazy_data, zarr.Array):
        raise TypeError(f"Expected zarr.Array, but found {type(lazy_data)}.")

    shape = lazy_data.shape
    print(f"Full data shape: {shape}")

    # --- START FIX: Handle 4D (squeezed T) vs 5D arrays ---
    if lazy_data.ndim == 5:
        T, Z, C, Y, X = shape
        t_max = T - 1
        if t_index >= T:
            print(f"Warning: T_INDEX ({t_index}) is out of range. Using T=0.")
            t_index = 0

        print(f"Selecting 5D data for Max Projection: (T={t_index})")
        # Select the 4D (Z, C, Y, X) stack for the chosen timepoint
        stack_to_project = cast(zarr.Array, lazy_data[t_index, :, :, :, :])

    elif lazy_data.ndim == 4:
        print("Detected 4D array, assuming T=1 (squeezed).")
        Z, C, Y, X = shape
        t_index = 0  # Only one timepoint exists
        t_max = 0  # Max T-index is 0

        print("Selecting 4D data for Max Projection.")
        # The whole array is the 4D (Z, C, Y, X) stack
        stack_to_project = cast(zarr.Array, lazy_data)

    else:
        raise ValueError(f"Incompatible data shape: {shape}. Expected 4D or 5D array.")
    # --- END FIX ---

    total_planes = Z * C
    print(f"Calculating Max Projection from {total_planes} (Z*C) planes...")

    # Use np.asarray() to explicitly convert slice to ndarray
    plane_0: np.ndarray = np.asarray(stack_to_project[0, 0, :, :])
    z_mip = np.copy(plane_0)

    with tqdm(total=total_planes, desc="  Projecting") as pbar:
        for z in range(Z):
            for c in range(C):
                if z == 0 and c == 0:
                    pbar.update(1)
                    continue

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

    # Return t_max (0 for 4D files, T-1 for 5D files)
    return z_mip, float(vmin), float(vmax), lazy_data, t_max
