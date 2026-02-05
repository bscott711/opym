#!/usr/bin/env python3
"""
Lightweight OPM Job Submitter for Login Nodes.
Zero dependencies (Standard Lib only).
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

# --- Configuration ---
# Matches your system's folder structure
QUEUE_DIR = Path.home() / "petakit_jobs" / "queue"

# Default Physics Parameters (Fallback if metadata fails)
DEFAULTS = {
    "angle": 122.0,  # 90 + 32
    "xy": 0.136,
    "z": 1.0,
    "iter": 10,
}


def parse_z_step(data_dir):
    """
    Lightweight scan of AcqSettings.txt to find Z-step.
    """
    try:
        # 1. Look for base name to find metadata file
        # Try finding the metadata file directly in parent (common structure)
        parent = data_dir.parent
        meta_files = list(parent.glob("*_metadata.txt"))

        # If we can't find it easily, try the AcqSettings.txt which is often cleaner
        acq_file = parent / "AcqSettings.txt"

        target_file = None
        if acq_file.exists():
            target_file = acq_file
        elif meta_files:
            target_file = meta_files[0]

        if not target_file:
            return None

        # 2. Parse the file
        with open(target_file, encoding="latin-1") as f:
            data = json.load(f)

        # Check standard keys
        step = data.get("stepSizeUm") or data.get("zStep_um")
        return float(step) if step else None

    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Minimal OPM Job Submitter")

    parser.add_argument("input_dir", type=Path, help="Path to input directory")

    # Physics Overrides
    parser.add_argument(
        "--angle", type=float, default=DEFAULTS["angle"], help="Sheet Angle"
    )
    parser.add_argument(
        "--xy", type=float, default=DEFAULTS["xy"], help="XY Pixel Size"
    )
    parser.add_argument(
        "--z", type=float, default=None, help="Z-Step (Auto-detected if skipped)"
    )

    # ROI (Optional): xmin xmax ymin ymax
    parser.add_argument("--roi", type=int, nargs=4, help="Crop: xmin xmax ymin ymax")

    # Deconvolution
    parser.add_argument(
        "--psf", type=str, default=None, help="Path to PSF file (Enables Decon)"
    )
    parser.add_argument(
        "--iter", type=int, default=DEFAULTS["iter"], help="Decon Iterations"
    )

    args = parser.parse_args()

    # 1. Validate Input
    if not args.input_dir.exists():
        print(f"Error: {args.input_dir} not found.")
        sys.exit(1)

    # 2. Determine Z-Step
    z_step = args.z
    if z_step is None:
        z_step = parse_z_step(args.input_dir)
        if z_step is None:
            z_step = DEFAULTS["z"]
            print(f"Warning: Could not auto-detect Z-step. Using default: {z_step}")

    # 3. Determine Base Name (Smart Naming Fix)
    # If the folder is named 'processed_tiff_series_split'
    # grab the parent name instead.
    if args.input_dir.name == "processed_tiff_series_split":
        base_name = args.input_dir.parent.name
    else:
        base_name = args.input_dir.name

    # 4. Construct Payload
    payload = {
        "jobType": "deskew",
        "dataDir": str(args.input_dir.resolve()),
        "baseName": base_name,
        "parameters": {
            "xy_pixel_size": args.xy,
            "z_step_um": z_step,
            "sheet_angle_deg": args.angle,
            "deskew": True,
            "rotate": True,
            # Pass ROI if present
            "crop_box": args.roi if args.roi else None,
            # Decon args
            "run_decon": bool(args.psf),
            "decon_iter": args.iter,
            "psf_path": args.psf,
        },
    }

    # 5. Write to Queue
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = int(time.time() * 1000)
    # Clean filename (replace non-alphanumeric with _)
    safe_name = re.sub(r"[^\w\-_\.]", "_", base_name)
    job_filename = f"{safe_name}_{timestamp}.json"
    job_file = QUEUE_DIR / job_filename

    try:
        with open(job_file, "w") as f:
            json.dump(payload, f, indent=4)
        print(f"Job submitted: {job_file}")
        decon_status = bool(args.psf)
        print(
            f"ID: {base_name} | Params: Z={z_step}, "
            f"Angle={args.angle}, Decon={decon_status}"
        )
    except Exception as e:
        print(f"Failed to write job ticket: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
