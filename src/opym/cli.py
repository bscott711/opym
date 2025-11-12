# Ruff style: Compliant
"""
Command-Line Interface (CLI) for OPM Cropper.
"""

import argparse
import sys
from pathlib import Path

from .core import run_processing_job
from .roi_utils import _tuple_to_roi, load_rois_from_log
from .utils import OutputFormat, parse_roi_string


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="OPM OME-TIF Cropper/Splitter for pypetakit5d.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input-file",
        type=Path,
        help="A single 5D OME-TIF file to process.",
    )
    input_group.add_argument(
        "--input-dir",
        type=Path,
        help="A directory of 5D OME-TIF files to process (requires --roi-from-log).",
    )

    roi_group = parser.add_argument_group("ROI Selection")
    roi_group.add_argument(
        "--top-roi",
        type=str,
        default=None,
        help="Top ROI in 'y_start:y_stop,x_start:x_stop' format (e.g., '0:512,0:512').",
    )
    roi_group.add_argument(
        "--bottom-roi",
        type=str,
        default=None,
        help=(
            "Bottom ROI in 'y_start:y_stop,x_start:x_stop' format "
            "(e.g., '512:1024,0:512')."
        ),
    )
    roi_group.add_argument(
        "--roi-from-log",
        type=Path,
        default=None,
        help="Path to a JSON log file containing ROIs (e.g., 'opm_roi_log.json').",
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
        "--rotate",
        action="store_true",
        help="Rotate the cropped ROIs by 90 degrees counter-clockwise before saving.",
    )
    # --- NEW: --channels argument ---
    parser.add_argument(
        "-c",
        "--channels",
        type=str,
        default="0,1,2,3",
        help="Comma-separated list of output channels to save (e.g., '0,1,2,3').",
    )
    # --- END NEW ---

    args = parser.parse_args()

    print("--- Starting OPM Cropper Job ---")
    try:
        output_format = OutputFormat(args.format)

        # --- NEW: Parse channels list ---
        try:
            channels_to_output = [int(c.strip()) for c in args.channels.split(",")]
            if not all(0 <= c <= 3 for c in channels_to_output):
                raise ValueError
        except ValueError:
            print(
                "❌ ERROR: Invalid --channels format. "
                "Expected comma-separated numbers 0-3.",
                file=sys.stderr,
            )
            sys.exit(1)
        # --- END NEW ---

        if args.input_dir:
            if not args.roi_from_log:
                print("❌ ERROR: --input-dir requires --roi-from-log.", file=sys.stderr)
                sys.exit(1)

            print(f"Loading ROIs from {args.roi_from_log.name}...")
            roi_data = load_rois_from_log(args.roi_from_log)
            if not roi_data:
                print("❌ ERROR: ROI log is empty or not found.", file=sys.stderr)
                sys.exit(1)

            input_files = list(args.input_dir.glob("*.ome.tif"))
            if not input_files:
                print(
                    f"❌ ERROR: No .ome.tif files found in {args.input_dir}.",
                    file=sys.stderr,
                )
                sys.exit(1)

            print(f"Found {len(input_files)} files to process.")

            for base_file in input_files:
                file_rois = roi_data.get(base_file.name)
                if not file_rois:
                    print(f"  Skipping {base_file.name}: No ROI found in log.")
                    continue

                print(f"\n--- Processing {base_file.name} ---")
                top_roi_data = file_rois.get("top_roi")
                bottom_roi_data = file_rois.get("bottom_roi")

                top_roi = _tuple_to_roi(top_roi_data) if top_roi_data else None
                bottom_roi = _tuple_to_roi(bottom_roi_data) if bottom_roi_data else None

                # --- NEW: Validate ROIs against selected channels ---
                need_top = (1 in channels_to_output) or (2 in channels_to_output)
                need_bottom = (0 in channels_to_output) or (3 in channels_to_output)

                if (need_top and top_roi is None) or (
                    need_bottom and bottom_roi is None
                ):
                    print(
                        f"  Skipping {base_file.name}: Required ROIs for channels "
                        f"{channels_to_output} not found in log."
                    )
                    continue
                # --- END NEW ---

                run_processing_job(
                    base_file=base_file.resolve(),
                    top_roi=top_roi,
                    bottom_roi=bottom_roi,
                    output_format=output_format,
                    cli_log_file=args.roi_from_log,
                    rotate_90=args.rotate,
                    channels_to_output=channels_to_output,
                )

        elif args.input_file:
            if args.roi_from_log:
                print(f"Loading ROIs from {args.roi_from_log.name}...")
                roi_data = load_rois_from_log(args.roi_from_log)
                file_rois = roi_data.get(args.input_file.name)
                if not file_rois:
                    print(
                        f"❌ ERROR: No ROI found for {args.input_file.name} in log.",
                        file=sys.stderr,
                    )
                    sys.exit(1)

                top_roi_data = file_rois.get("top_roi")
                bottom_roi_data = file_rois.get("bottom_roi")
                top_roi = _tuple_to_roi(top_roi_data) if top_roi_data else None
                bottom_roi = _tuple_to_roi(bottom_roi_data) if bottom_roi_data else None
            else:
                top_roi = parse_roi_string(args.top_roi) if args.top_roi else None
                bottom_roi = (
                    parse_roi_string(args.bottom_roi) if args.bottom_roi else None
                )

            # --- NEW: Validate ROIs against selected channels ---
            need_top = (1 in channels_to_output) or (2 in channels_to_output)
            need_bottom = (0 in channels_to_output) or (3 in channels_to_output)

            if need_top and top_roi is None:
                print(
                    "❌ ERROR: Channels 1 or 2 selected, but --top-roi not provided.",
                    file=sys.stderr,
                )
                sys.exit(1)
            if need_bottom and bottom_roi is None:
                print(
                    "❌ ERROR: Channels 0 or 3 selected, --bottom-roi not provided.",
                    file=sys.stderr,
                )
                sys.exit(1)
            # --- END NEW ---

            print(f"\n--- Processing {args.input_file.name} ---")
            run_processing_job(
                base_file=args.input_file.resolve(),
                top_roi=top_roi,
                bottom_roi=bottom_roi,
                output_format=output_format,
                cli_log_file=args.roi_from_log or Path("opm_roi_log.json"),
                rotate_90=args.rotate,
                channels_to_output=channels_to_output,
            )

        print("\n--- Processing Job Complete ---")

    except (ValueError, FileNotFoundError, RuntimeError) as e:
        print(f"\n❌ An error occurred: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
