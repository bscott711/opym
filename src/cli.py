# Ruff style: Compliant
# Description:
# This module provides the command-line interface for the opym processing
# pipeline. Imports are updated to use the local modules.

import argparse
import os
import sys
import time
import tkinter as tk
from tkinter import filedialog

from . import core, libinstall, metadata  # Updated imports


def main():
    """Main function to parse arguments and run the pipeline."""
    parser = argparse.ArgumentParser(
        description=(
            "Process a folder of raw OPM data using a GPU-accelerated pipeline. "
            "Finds AcqSettings.txt in the folder to determine parameters and "
            "the prefix for raw data files to be processed."
        )
    )
    # (Argument parsing remains the same as before)
    parser.add_argument(
        "input_dir",
        type=str,
        nargs="?",
        default=None,
        help=(
            "Path to the directory containing the raw data and AcqSettings.txt. "
            "If not provided, a file dialog will open."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory to save the processed files. "
            "Defaults to a 'processed' subfolder in the input directory."
        ),
    )
    parser.add_argument(
        "--num-channels",
        type=int,
        default=None,
        help="Number of interlaced channels. Overrides metadata.",
    )
    parser.add_argument(
        "--angle",
        type=float,
        default=31.5,
        help="Angle of the light sheet in degrees.",
    )
    parser.add_argument(
        "--voxel-size-z",
        type=float,
        default=None,
        help="Voxel size in Z (microns). Overrides metadata.",
    )
    parser.add_argument(
        "--voxel-size-xy",
        type=float,
        default=0.1,
        help="Voxel size in XY (microns).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of deconvolution iterations.",
    )
    parser.add_argument(
        "--psf-path",
        type=str,
        default=None,
        help="Path to the 3D PSF TIFF file. If not provided, a default is used.",
    )
    args = parser.parse_args()

    # --- Determine Input Directory ---
    if args.input_dir is None:
        print("No input directory specified. Opening file dialog...")
        root = tk.Tk()
        root.withdraw()
        input_dir = filedialog.askdirectory(title="Select Data Directory")
        if not input_dir:
            print("No directory selected. Exiting.")
            sys.exit(0)
    else:
        input_dir = args.input_dir

    # --- Determine Output Directory ---
    if args.output_dir is None:
        output_dir = os.path.join(input_dir, "processed")
        print(f"No output directory specified. Defaulting to: {output_dir}")
    else:
        output_dir = args.output_dir

    # (The rest of the main function remains the same, but the libinstall call is now local)
    print("\n--- OPM GPU Processing Pipeline ---")
    start_total_time = time.time()

    settings_path = os.path.join(input_dir, "AcqSettings.txt")
    meta_params = metadata.parse_acq_settings(settings_path)

    if not meta_params:
        print("Error: Could not find or parse AcqSettings.txt. Cannot proceed.")
        return

    prefix = meta_params.get("save_name_prefix")
    if not prefix:
        print(
            "Error: 'saveNamePrefix' not found in AcqSettings.txt. "
            "Cannot find files to process."
        )
        return

    tiff_files = core.find_tiff_files(input_dir, prefix)
    if not tiff_files:
        print(f"Error: No TIFF files found with prefix '{prefix}'. Nothing to process.")
        return

    params = {
        "angle": args.angle,
        "voxel_size_xy": args.voxel_size_xy,
        "iterations": args.iterations,
    }
    params["num_channels"] = meta_params.get("num_channels", 2)
    params["voxel_size_z"] = meta_params.get("voxel_size_z", 0.4)

    if args.num_channels is not None:
        params["num_channels"] = args.num_channels
        print(f"User override: Number of channels set to {args.num_channels}")
    if args.voxel_size_z is not None:
        params["voxel_size_z"] = args.voxel_size_z
        print(f"User override: Voxel size Z set to {args.voxel_size_z} um")

    if not libinstall.check_lib_installed():
        print("cudaDeconv binary not found. Attempting to install...")
        libinstall.install_lib()
        if not libinstall.check_lib_installed():
            print("Failed to install cudaDeconv binary. Exiting.")
            return

    os.makedirs(output_dir, exist_ok=True)

    for filepath in tiff_files:
        print(f"\n===== Processing File: {os.path.basename(filepath)} =====")
        channel_stacks = core.load_and_deinterlace(filepath, params["num_channels"])
        if not channel_stacks:
            print(f"Skipping file {filepath} due to loading error.")
            continue

        base_name, _ = os.path.splitext(os.path.basename(filepath))

        for channel_id, stack in channel_stacks.items():
            core.process_channel(
                channel_id, stack, args.psf_path, output_dir, base_name, params
            )

    print(
        f"\n--- Pipeline Complete ---"
        f"\nTotal execution time: {time.time() - start_total_time:.2f} seconds."
    )


if __name__ == "__main__":
    main()

