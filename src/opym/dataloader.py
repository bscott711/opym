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
    a (get_stack, T_max, Z_max, C_max, Y, X) tuple that can be passed
    directly to the opym viewers.

    Args:
        directory: The Path object pointing to the processed_tiff_series_split
                   directory.

    Returns:
        A tuple containing:
        - get_stack (Callable): A cached function to load a (T, C) stack.
        - T_max (int): The maximum time index.
        - Z_max (int): The maximum Z index.
        - C_max (int): The maximum Channel index.
        - Y (int): The Y dimension of the images.
        - X (int): The X dimension of the images.
    """
    print("Loading TIFF series...")
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    first_file = next(directory.glob("*_T000_C0.tif"), None)
    if not first_file:
        raise FileNotFoundError(f"No '*_T000_C0.tif' files found in {directory}")

    BASE_NAME = first_file.name.replace("_T000_C0.tif", "")
    print(f"Found base name: {BASE_NAME}")

    # --- 2. Parse T, C, and Z limits from files ---
    t_vals = set()
    c_vals = set()
    file_pattern = re.compile(f"{re.escape(BASE_NAME)}_T(\\d+)_C(\\d+).tif")

    for f in directory.glob(f"{BASE_NAME}_T*_C*.tif"):
        match = file_pattern.match(f.name)
        if match:
            t_vals.add(int(match.group(1)))
            c_vals.add(int(match.group(2)))

    if not t_vals or not c_vals:
        raise Exception("Could not parse T or C values from filenames.")

    T_max = max(t_vals)
    C_max = max(c_vals)

    first_stack = tifffile.imread(first_file)
    Z_max, Y, X = first_stack.shape
    Z_max -= 1  # Max index is shape - 1

    print(f"Data shape: T={T_max + 1}, Z={Z_max + 1}, C={C_max + 1}, Y={Y}, X={X}")

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

    return get_stack, T_max, Z_max, C_max, Y, X
