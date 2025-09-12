# Ruff style: Compliant
# Description:
# This module provides the command-line interface for the opym processing pipeline.

import argparse
import os
import time
from . import core
from llspy import libinstall


def main():
    """Main function to parse arguments and run the pipeline."""
    parser = argparse.ArgumentParser(
        description="Process raw OPM data using a GPU-accelerated pipeline."
    )
    parser.add_argument(
        "input_file", type=str, help="Path to the raw, interlaced TIFF file."
    )
    parser.add_argument(
        "output_dir", type=str, help="Directory to save the processed files."
    )
    parser.add_argument(
        "--num-channels",
        type=int,
        default=2,
        help="Number of interlaced channels in the input file.",
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
        default=0.4,
        help="Voxel size in Z (microns).",
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

    print("--- OPM GPU Processing Pipeline with LLSpy ---")
    start_total_time = time.time()

    if not libinstall.check_lib_installed():
        print("cudaDeconv binary not found. Attempting to install...")
        libinstall.install_lib()
        if not libinstall.check_lib_installed():
            print("Failed to install cudaDeconv binary. Exiting.")
            return

    os.makedirs(args.output_dir, exist_ok=True)

    channel_stacks = core.load_and_deinterlace(args.input_file, args.num_channels)
    if not channel_stacks:
        print("Halting pipeline due to loading error.")
        return

    base_name, _ = os.path.splitext(os.path.basename(args.input_file))
    params = {
        "angle": args.angle,
        "voxel_size_z": args.voxel_size_z,
        "voxel_size_xy": args.voxel_size_xy,
        "iterations": args.iterations,
    }

    for channel_id, stack in channel_stacks.items():
        core.process_channel(
            channel_id, stack, args.psf_path, args.output_dir, base_name, params
        )

    print(
        f"\n--- Pipeline Complete ---"
        f"\nTotal execution time: {time.time() - start_total_time:.2f} seconds."
    )


if __name__ == "__main__":
    main()

