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
    Updated to handle variable padding (e.g. C0 vs C00, T000 vs T0001).

    Args:
        directory: The Path object pointing to the processed tiff series.

    Returns:
        Standard viewer tuple.
    """
    print(f"Loading OPM TIFF series from: {directory.name}")
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    # 1. Flexible Glob: Find anything looking like *_C*_T*.tif
    # This matches 'img_C00_T0001.tif' AND 'Name_C0_T000.tif'
    files = sorted(list(directory.glob("*_C*_T*.tif")))

    if not files:
        raise FileNotFoundError(
            f"No compatible TIFF files (e.g., '*_Cxx_Txxxx.tif') found in {directory}"
        )

    # 2. Flexible Regex: Capture Base, C, and T regardless of digit count
    # Anchored to end to handle extensions correctly
    file_pattern_re = re.compile(r"^(.*?)_C(\d+)_T(\d+)\.tif$", re.IGNORECASE)

    t_vals = set()
    c_vals = set()
    file_map = {}  # Map (t, c) -> Path

    base_name = None
    first_file = None

    print(f"Scanning {len(files)} files...")

    for f in files:
        match = file_pattern_re.match(f.name)
        if match:
            # Capture base name from the first valid match
            if base_name is None:
                base_name = match.group(1)
                first_file = f

            # Ensure we don't mix base names (e.g. if folder has junk)
            if match.group(1) != base_name:
                continue

            c = int(match.group(2))
            t = int(match.group(3))

            c_vals.add(c)
            t_vals.add(t)
            file_map[(t, c)] = f

    if not file_map:
        raise ValueError("Files found but regex failed to parse C/T values.")

    print(f"Found base name: {base_name}")

    T_min = min(t_vals)
    T_max = max(t_vals)
    C_min = min(c_vals)
    C_max = max(c_vals)

    # Get dimensions from the first valid file
    first_stack = tifffile.imread(first_file)
    if len(first_stack.shape) == 2:
        # Handle 2D images (Z=1) gracefully
        Z_max = 0
        Y, X = first_stack.shape
    else:
        Z_max, Y, X = first_stack.shape
        Z_max -= 1

    print(
        f"Data shape: T={T_min}-{T_max}, Z={Z_max + 1}, C={C_min}-{C_max}, Y={Y}, X={X}"
    )

    @functools.lru_cache(maxsize=8)
    def get_stack(t, c):
        """Loads a 3D ZYX stack using the pre-built file map."""
        file_path = file_map.get((t, c))

        if not file_path or not file_path.exists():
            print(f"Warning: Frame missing for T={t}, C={c}")
            return np.zeros((Z_max + 1, Y, X), dtype=first_stack.dtype)

        return tifffile.imread(file_path)

    print("✅ OPM Data loaded.")

    return get_stack, T_min, T_max, C_min, C_max, Z_max, Y, X, base_name
