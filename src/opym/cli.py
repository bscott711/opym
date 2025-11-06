# Ruff style: Compliant
"""
Command-Line Interface (CLI) for OPM Cropper.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

from .core import process_dataset
from .metadata import create_processing_log
from .utils import DerivedPaths, OutputFormat, derive_paths, parse_roi_string


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="OPM OME-TIF Cropper/Splitter for pypetakit5d.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "base_file",
        type=Path,
        help="Path to the source .ome.tif file.",
    )
    parser.add_argument(
        "--top-roi",
        required=True,
        type=str,
        help="Top ROI in 'y_start:y_stop,x_start:x_stop' format (e.g., '0:512,0:512').",
    )
    parser.add_argument(
        "--bottom-roi",
        required=True,
        type=str,
        help="Bottom ROI in 'y_start:y_stop,x_start:x_stop' format "
        "(e.g., '512:1024,0:512').",
    )
    parser.add_argument(
        "-f",
        "--format",
        type=str,
        default=OutputFormat.TIFF_SERIES.value,
        choices=[f.value for f in OutputFormat],
        help="Output format. 'TIFF_SERIES_SPLIT_C' is required for pypetakit5d.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean (delete) any existing files from the output directory "
        "before processing.",
    )
    args = parser.parse_args()

    # --- Start Processing ---
    print("--- Starting OPM Cropper Job ---")
    try:
        base_file = args.base_file.resolve()
        output_format = OutputFormat(args.format)
        top_roi = parse_roi_string(args.top_roi)
        bottom_roi = parse_roi_string(args.bottom_roi)

        # 1. Derive all paths
        paths: DerivedPaths = derive_paths(base_file, output_format)
        print(f"  Source: {paths.base_file}")
        print(f"  Metadata: {paths.metadata_file}")
        print(f"  Output Dir: {paths.output_dir}")
        print(f"  Format: {output_format.value}")

        # 2. Validate inputs
        if not paths.base_file.exists():
            raise FileNotFoundError(f"Input file not found: {paths.base_file}")
        if not paths.metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {paths.metadata_file}")

        # 3. Create output directory
        paths.output_dir.mkdir(parents=True, exist_ok=True)

        # 4. Clean output directory if requested
        if args.clean:
            print(f"Cleaning output directory: {paths.output_dir.name}...")
            if output_format == OutputFormat.TIFF_SERIES:
                # Delete old TIFFs but keep the directory
                for f in paths.output_dir.glob(f"{paths.sanitized_name}_T*.tif"):
                    os.remove(f)
            elif output_format == OutputFormat.ZARR:
                zarr_path = paths.output_dir / (
                    paths.sanitized_name + "_processed.zarr"
                )
                if zarr_path.exists():
                    shutil.rmtree(zarr_path)

        # 5. Run main processing
        print("\nStarting stream processing...")
        num_timepoints = process_dataset(
            paths.base_file,
            paths.output_dir,
            paths.sanitized_name,
            top_roi,
            bottom_roi,
            output_format,
        )

        # 6. Create the log file
        print("\nCreating processing log...")
        create_processing_log(
            paths,
            num_timepoints,
            top_roi,
            bottom_roi,
            output_format,
        )

        print("\n--- Processing Job Complete ---")

    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
