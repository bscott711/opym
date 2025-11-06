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
