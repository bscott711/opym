# Ruff style: Compliant
"""
Command-Line Interface (CLI) for OPM Cropper.
"""

import argparse
import sys
from pathlib import Path

# --- FIX: Import the high-level wrapper and necessary utils ---
from .core import run_processing_job
from .utils import OutputFormat, parse_roi_string

# --- End fix ---


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
    # --- FIX: Removed redundant --clean argument ---
    # The core.run_processing_job function now handles cleaning by default.
    args = parser.parse_args()

    # --- Start Processing ---
    print("--- Starting OPM Cropper Job ---")
    try:
        # --- FIX: Refactored to call the high-level wrapper ---
        base_file = args.base_file.resolve()
        output_format = OutputFormat(args.format)

        # Parse ROIs here to provide immediate user feedback
        top_roi = parse_roi_string(args.top_roi)
        bottom_roi = parse_roi_string(args.bottom_roi)

        # Call the single processing function
        run_processing_job(
            base_file=base_file,
            top_roi=top_roi,
            bottom_roi=bottom_roi,
            output_format=output_format,
            # Uses default cli_log_file="opm_roi_log.json"
        )
        # --- End fix ---

        print("\n--- Processing Job Complete ---")

    except (ValueError, FileNotFoundError, RuntimeError) as e:
        # --- FIX: Catch exceptions raised from the library ---
        print(f"\n❌ An error occurred: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
