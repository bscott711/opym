# Ruff style: Compliant
"""
Data loading utilities for the opym viewer.
Parses a directory of 3D TIFFs to prepare them for viewing.
"""

from __future__ import annotations

import functools
import re
from pathlib import Path

import numpy as np
import tifffile


def load_llsm_tiff_series(directory: Path):
    """
    Parses a directory of LLSM TIFFs and returns viewer parameters.
    Normalizes indices to start at 0.

    Returns a tuple of:
    (get_stack, T_min, T_max, C_min, C_max, Z_max, Y, X, base_name)
    """
    print("Loading LLSM TIFF series...")
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    # Temporary storage for matches before normalization
    raw_matches = []
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
                first_file = f

            t_raw = int(match.group(4))
            c_raw = int(match.group(3))
            raw_matches.append((t_raw, c_raw, f))

    if not raw_matches or not first_file or base_name is None:
        raise FileNotFoundError(
            "No valid LLSM TIFF files "
            "(e.g., '*_CamA_ch0_stack0000*.tif') "
            f"found in {directory}"
        )

    # --- NORMALIZE INDICES TO 0 ---
    t_min_raw = min(m[0] for m in raw_matches)
    c_min_raw = min(m[1] for m in raw_matches)

    file_map = {}
    t_vals = set()
    c_vals = set()

    for t_raw, c_raw, f in raw_matches:
        t_norm = t_raw - t_min_raw
        c_norm = c_raw - c_min_raw
        file_map[(t_norm, c_norm)] = f
        t_vals.add(t_norm)
        c_vals.add(c_norm)

    print(f"Found base name: {base_name}")
    if t_min_raw != 0:
        t_max_raw = max(m[0] for m in raw_matches)
        print(f"  -> Normalizing Time: {t_min_raw}..{t_max_raw} -> 0..{max(t_vals)}")

    # Get min/max values (now 0-based)
    T_min = 0
    T_max = max(t_vals)
    C_min = 0
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
    Automatically normalizes T and C indices to start at 0.

    Args:
        directory: The Path object pointing to the processed tiff series.

    Returns:
        Standard viewer tuple.
    """
    print(f"Loading OPM TIFF series from: {directory.name}")
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    # 1. Flexible Glob
    files = sorted(list(directory.glob("*_C*_T*.tif")))

    if not files:
        raise FileNotFoundError(
            f"No compatible TIFF files (e.g., '*_Cxx_Txxxx.tif') found in {directory}"
        )

    # 2. Flexible Regex
    file_pattern_re = re.compile(r"^(.*?)_C(\d+)_T(\d+)\.tif$", re.IGNORECASE)

    raw_matches = []
    base_name = None
    first_file = None

    print(f"Scanning {len(files)} files...")

    for f in files:
        match = file_pattern_re.match(f.name)
        if match:
            if base_name is None:
                base_name = match.group(1)
                first_file = f

            # Ensure consistency
            if match.group(1) != base_name:
                continue

            c_raw = int(match.group(2))
            t_raw = int(match.group(3))
            raw_matches.append((t_raw, c_raw, f))

    if not raw_matches:
        raise ValueError("Files found but regex failed to parse C/T values.")

    # --- NORMALIZE INDICES TO 0 ---
    # Find the lowest value for T and C in the folder
    t_min_raw = min(m[0] for m in raw_matches)
    c_min_raw = min(m[1] for m in raw_matches)

    file_map = {}
    t_vals = set()
    c_vals = set()

    # Rebuild map with 0-based keys
    for t_raw, c_raw, f in raw_matches:
        t_norm = t_raw - t_min_raw
        c_norm = c_raw - c_min_raw

        file_map[(t_norm, c_norm)] = f
        t_vals.add(t_norm)
        c_vals.add(c_norm)

    print(f"Found base name: {base_name}")

    # Inform user of shift if it happened
    if t_min_raw != 0:
        t_max_raw = max(m[0] for m in raw_matches)
        print(f"  -> Normalizing Time: {t_min_raw}..{t_max_raw} -> 0..{max(t_vals)}")

    if c_min_raw != 0:
        c_max_raw = max(m[1] for m in raw_matches)
        print(f"  -> Normalizing Chan: {c_min_raw}..{c_max_raw} -> 0..{max(c_vals)}")

    T_min = 0
    T_max = max(t_vals)
    C_min = 0
    C_max = max(c_vals)

    # Get dimensions from the first valid file
    first_stack = tifffile.imread(first_file)
    if len(first_stack.shape) == 2:
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
        """Loads a 3D ZYX stack using the pre-built file map (0-based keys)."""
        file_path = file_map.get((t, c))

        if not file_path or not file_path.exists():
            print(f"Warning: Frame missing for T={t}, C={c}")
            return np.zeros((Z_max + 1, Y, X), dtype=first_stack.dtype)

        return tifffile.imread(file_path)

    print("✅ OPM Data loaded.")

    return get_stack, T_min, T_max, C_min, C_max, Z_max, Y, X, base_name


def find_dsr_directory(
    explicit_path: Path | None = None, search_root: Path = Path(".")
) -> Path:
    """
    Locates the Deskewed-Rotated (DSR) output directory.

    Args:
        explicit_path: Known path from previous processing steps.
        search_root: Where to search if explicit_path is missing.

    Returns:
        Path object pointing to the DSR directory.

    Raises:
        FileNotFoundError: If no valid DSR directory is found.
    """
    # 1. Check explicit path
    if explicit_path is not None:
        candidate = explicit_path / "DSR"
        if candidate.exists():
            return candidate

    # 2. Search locally
    found_dsrs = sorted(
        list(search_root.glob("*/DSR")), key=lambda p: p.stat().st_mtime, reverse=True
    )

    if found_dsrs:
        return found_dsrs[0]

    raise FileNotFoundError(
        "Could not auto-detect any 'DSR' folders. "
        "Please ensure the Deskew job completed successfully."
    )


def get_channel_count(file_path: Path) -> int:
    """
    Detects the number of channels in an OME-TIFF file.
    Returns 4 by default if detection fails.
    """
    try:
        import tifffile

        with tifffile.TiffFile(file_path) as tif:
            shape = tif.series[0].shape
            # (T, C, Z, Y, X) or (C, Z, Y, X)
            if len(shape) == 5:
                return shape[2]  # Adjust index based on your specific axis order
            elif len(shape) == 4:
                return shape[1]
    except Exception:  # nosec
        pass
    return 4
