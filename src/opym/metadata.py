# Ruff style: Compliant
"""
Handles Micro-Manager metadata parsing and processing log generation.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

from .utils import DerivedPaths, OutputFormat


def parse_timestamps(metadata_file: Path, num_timepoints: int) -> list[float]:
    """
    Parses the Micro-Manager metadata file to extract the
    elapsed time for the start of each timepoint (Z-stack).
    """
    print(f"Parsing timestamps from {metadata_file.name}...")
    timestamps_sec = []
    backup_interval_sec = 6.0

    try:
        # Open with 'latin-1' encoding to handle special characters
        with metadata_file.open("r", encoding="latin-1") as f:
            metadata = json.load(f)

        # Try to get the real interval from the summary
        try:
            summary = metadata.get("Summary", {})
            spim_settings_str = summary.get("SPIMAcqSettings", "{}")
            spim_settings = json.loads(spim_settings_str)
            backup_interval_sec = spim_settings.get("timepointInterval", 6.0)
        except Exception:
            pass  # Keep default

        for t in range(num_timepoints):
            # Key is "FrameKey-T-Z-C". We want the start of each
            # Z-stack, so we use Z=0 and C=0.
            frame_key = f"FrameKey-{t}-0-0"
            frame_data = metadata.get(frame_key)

            if frame_data and "ElapsedTime-ms" in frame_data:
                ms = frame_data["ElapsedTime-ms"]
                timestamps_sec.append(ms / 1000.0)
            else:
                # Fallback if key is missing
                print(
                    f"Warning: Could not find metadata for {frame_key}. "
                    "Using calculated time.",
                    file=sys.stderr,
                )
                timestamps_sec.append(t * backup_interval_sec)

        print(f"Successfully parsed {len(timestamps_sec)} timestamps.")

    except Exception as e:
        print(
            f"CRITICAL: Could not parse metadata: {e}. Using calculated times.",
            file=sys.stderr,
        )
        timestamps_sec = [(t * backup_interval_sec) for t in range(num_timepoints)]

    return timestamps_sec


def _format_slice(s: slice) -> tuple[int | None, int | None]:
    """Helper to format slice objects for JSON serialization."""
    return (s.start, s.stop)


def create_processing_log(
    paths: DerivedPaths,
    num_timepoints: int,
    top_roi: tuple[slice, slice],
    bottom_roi: tuple[slice, slice],
    output_format: OutputFormat,
):
    """Writes a JSON log file with all processing parameters."""

    timestamps = parse_timestamps(paths.metadata_file, num_timepoints)

    log_data = {
        "processing_version": f"3.0-format-{output_format.value}",
        "processing_date": datetime.now().isoformat(),
        "source_base_file": str(paths.base_file),
        "source_metadata_file": str(paths.metadata_file),
        "rois": {
            "top_roi": (
                _format_slice(top_roi[0]),
                _format_slice(top_roi[1]),
            ),
            "bottom_roi": (
                _format_slice(bottom_roi[0]),
                _format_slice(bottom_roi[1]),
            ),
        },
        "notes": (
            "Cropped two ROIs from each of 2 camera channels, "
            f"stacking to 4 channels. Output format: {output_format.value}"
        ),
        "timestamps_sec": timestamps,
    }

    try:
        with paths.output_log.open("w") as f:
            json.dump(log_data, f, indent=4)
        print(f"✅ Successfully wrote processing log to {paths.output_log.name}")
    except Exception as e:
        print(f"Error writing log file: {e}", file=sys.stderr)
