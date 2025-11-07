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


def load_llsm_tiff_series(directory: Path):
    """
    Parses a directory of LLSM TIFFs and returns viewer parameters.

    Returns a tuple of:
    (get_stack, T_min, T_max, C_min, C_max, Z_max, Y, X, base_name)
    """
    print("Loading LLSM TIFF series...")
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    # Parse T, C, and Z limits from files
    t_vals = set()
    c_vals = set()
    file_map = {}  # Dictionary to map (t, c) -> file_path
    base_name = None
    first_file = None

    # Regex for LLSM: (base_name)_Cam(A/B)_ch(c)_stack(t)...tif
    file_pattern = re.compile(
        r"^(.*?)_Cam([AB])_ch(\d+)_stack(\d+).*?\.tif$",
        re.IGNORECASE,
    )

    for f in directory.glob("*_Cam*_ch*_stack*.tif"):
        match = file_pattern.match(f.name)
        if match:
            if base_name is None:
                base_name = match.group(1)
                first_file = f  # Store the first file we find

            t = int(match.group(4))
            c = int(match.group(3))

            t_vals.add(t)
            c_vals.add(c)
            file_map[(t, c)] = f

    if not file_map or not first_file or base_name is None:
        raise FileNotFoundError(
            "No valid LLSM TIFF files "
            "(e.g., '*_CamA_ch0_stack0000*.tif') "
            f"found in {directory}"
        )

    print(f"Found base name: {base_name}")

    # Get min/max values
    T_min = min(t_vals)
    T_max = max(t_vals)
    C_min = min(c_vals)
    C_max = max(c_vals)

    # Use the first_file we already found
    first_stack = tifffile.imread(first_file)
    Z_max, Y, X = first_stack.shape
    Z_max -= 1  # Max index is shape - 1

    print(
        f"Data shape: T={T_min}-{T_max}, Z={Z_max + 1}, C={C_min}-{C_max}, Y={Y}, X={X}"
    )

    @functools.lru_cache(maxsize=8)
    def get_stack(t, c):
        """Loads a 3D ZYX stack for a given T and C."""
        file_path = file_map.get((t, c))
        if not file_path or not file_path.exists():
            print(f"Warning: File not found for T={t}, C={c}")
            return np.zeros((Z_max + 1, Y, X), dtype=first_stack.dtype)
        return tifffile.imread(file_path)

    print("✅ LLSM Data loaded.")

    return get_stack, T_min, T_max, C_min, C_max, Z_max, Y, X, base_name


def load_tiff_series(directory: Path):
    """
    Parses a directory of processed OPM TIFFs and returns viewer parameters.

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
    print("Loading OPM TIFF series...")
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    # Find any file matching the pattern to determine the base name
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
    print(f"Found base name: {BASE_NAME}")

    # Parse T, C, and Z limits from files
    t_vals = set()
    c_vals = set()
    file_pattern = re.compile(f"{re.escape(BASE_NAME)}_T(\\d+)_C(\\d+).tif")

    for f in directory.glob(f"{BASE_NAME}_T*_C*.tif"):
        match = file_pattern.match(f.name)
        if match:
            t_vals.add(int(match.group(1)))
            c_vals.add(int(match.group(2)))

    if not t_vals or not c_vals:
        raise FileNotFoundError("Could not parse T or C values from filenames.")

    T_min = min(t_vals)
    T_max = max(t_vals)
    C_min = min(c_vals)
    C_max = max(c_vals)

    # Use the first_file we already found to get shape
    first_stack = tifffile.imread(first_file)
    Z_max, Y, X = first_stack.shape
    Z_max -= 1  # Max index is shape - 1

    print(
        f"Data shape: T={T_min}-{T_max}, Z={Z_max + 1}, C={C_min}-{C_max}, Y={Y}, X={X}"
    )

    @functools.lru_cache(maxsize=8)
    def get_stack(t, c):
        """Loads a 3D ZYX stack for a given T and C."""
        file_path = directory / f"{BASE_NAME}_T{t:03d}_C{c:d}.tif"
        if not file_path.exists():
            print(f"Warning: File not found {file_path.name}")
            return np.zeros((Z_max + 1, Y, X), dtype=first_stack.dtype)
        return tifffile.imread(file_path)

    print("✅ OPM Data loaded.")

    return get_stack, T_min, T_max, C_min, C_max, Z_max, Y, X, BASE_NAME
