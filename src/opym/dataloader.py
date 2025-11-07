# Ruff style: Compliant
"""
Data loading utilities for the opym viewer.
Parses a directory of 3D TIFFs to prepare them for viewing.
"""

import functools
import re
from pathlib import Path

import numpy as np
import tifffile


# --- NEW: LLSM-specific preview loader ---
def load_llsm_preview(
    directory: Path, preview_channel: int = 0
) -> tuple[np.ndarray, float, float, str]:
    """
    Loads a preview MIP for LLSM data from a specific channel.

    Finds the first timepoint (stack0000) for the specified channel,
    loads it, and calculates the MIP.

    Args:
        directory: The Path object to the LLSM data directory.
        preview_channel: The channel index (e.g., 0, 1, 2) to load.

    Returns:
        A tuple containing:
        - mip_original (np.ndarray): The calculated MIP.
        - vmin_original (float): 1st percentile for contrast.
        - vmax_original (float): 99.9th percentile for contrast.
        - preview_file (str): The name of the file used for the preview.
    """
    print(f"  Loading LLSM preview for ch{preview_channel}, stack0000...")

    # Find the first stack for the specified channel
    first_stack_file = next(
        directory.glob(f"*_Cam[AB]_ch{preview_channel}_stack0000*.tif"),
        None,
    )

    if not first_stack_file:
        # Fallback to *any* stack for that channel if stack0000 is missing
        first_stack_file = next(
            directory.glob(f"*_Cam[AB]_ch{preview_channel}_stack*.tif"),
            None,
        )

    if not first_stack_file:
        raise FileNotFoundError(
            f"Could not find any LLSM stack files for ch{preview_channel} "
            f"in: {directory}"
        )

    print(f"  Loading preview from: {first_stack_file.name}")
    stack_original = tifffile.imread(first_stack_file)
    mip_original = np.max(stack_original, axis=0)

    vmin_original, vmax_original = np.percentile(mip_original, [1, 99.9])
    if vmin_original >= vmax_original:
        vmax_original = np.max(mip_original)

    return mip_original, vmin_original, vmax_original, first_stack_file.name


# --- END NEW ---


def load_tiff_series(directory: Path):
    """
    Parses a directory of processed TIFFs and returns viewer parameters.

    This function finds all files, parses T/C/Z limits, and returns
    a (get_stack, T_min, T_max, C_min, C_max, Z_max, Y, X, base_name) tuple.

    Args:
        directory: The Path object pointing to the processed_tiff_series_split
                   directory.

    Returns:
        A tuple containing:
        - get_stack (Callable): A cached function to load a (T, C) stack.
        - T_min (int): The minimum time index found.
        - T_max (int): The maximum time index found.
        - C_min (int): The minimum channel index found.
        - C_max (int): The maximum channel index found.
        - Z_max (int): The maximum Z index.
        - Y (int): The Y dimension of the images.
        - X (int): The X dimension of the images.
        - base_name (str): The parsed base name of the files.
    """
    print("Loading TIFF series...")
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    # --- START OF FIX: Robust base_name finding ---
    # Find any file matching the pattern, not just T0/C0
    first_file = next(directory.glob("*_T[0-9][0-9][0-9]_C[0-9].tif"), None)
    if not first_file:
        raise FileNotFoundError(
            f"No processed TIFF files (e.g., '*_T000_C0.tif') found in {directory}"
        )

    # Use regex to parse the base name
    file_pattern_re = re.compile(r"^(.*?)_T\d{3}_C\d\.tif$")
    match = file_pattern_re.match(first_file.name)
    if not match:
        raise ValueError(f"Could not parse base name from file: {first_file.name}")

    BASE_NAME = match.group(1)
    # --- END OF FIX ---

    print(f"Found base name: {BASE_NAME}")

    # --- 2. Parse T, C, and Z limits from files ---
    t_vals = set()
    c_vals = set()
    # Use the parsed BASE_NAME for the pattern
    file_pattern = re.compile(f"{re.escape(BASE_NAME)}_T(\\d+)_C(\\d+).tif")

    for f in directory.glob(f"{BASE_NAME}_T*_C*.tif"):
        match = file_pattern.match(f.name)
        if match:
            t_vals.add(int(match.group(1)))
            c_vals.add(int(match.group(2)))

    if not t_vals or not c_vals:
        raise Exception("Could not parse T or C values from filenames.")

    # --- MODIFIED: Get min and max ---
    T_min = min(t_vals)
    T_max = max(t_vals)
    C_min = min(c_vals)
    C_max = max(c_vals)
    # --- END MODIFICATION ---

    # Use the first_file we already found
    first_stack = tifffile.imread(first_file)
    Z_max, Y, X = first_stack.shape
    Z_max -= 1  # Max index is shape - 1

    print(
        f"Data shape: T={T_min}-{T_max}, Z={Z_max + 1}, C={C_min}-{C_max}, Y={Y}, X={X}"
    )

    # --- 3. Caching Function (for speed) ---
    @functools.lru_cache(maxsize=8)
    def get_stack(t, c):
        """Loads a 3D ZYX stack for a given T and C."""
        file_path = directory / f"{BASE_NAME}_T{t:03d}_C{c:d}.tif"
        if not file_path.exists():
            print(f"Warning: File not found {file_path.name}")
            return np.zeros((Z_max + 1, Y, X), dtype=first_stack.dtype)
        return tifffile.imread(file_path)

    print("✅ Data loaded. You can now run the viewer cells below.")

    # --- MODIFIED: Return min/max and base_name ---
    return get_stack, T_min, T_max, C_min, C_max, Z_max, Y, X, BASE_NAME
