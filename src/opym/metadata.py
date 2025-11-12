# Ruff style: Compliant
"""
Handles Micro-Manager metadata parsing and processing log generation.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import overload

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
            pass  # nosec B110

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


@overload
def _format_slice(s: slice) -> tuple[int | None, int | None]: ...


@overload
def _format_slice(s: None) -> None: ...


def _format_slice(s: slice | None) -> tuple[int | None, int | None] | None:
    """Helper to format slice objects for JSON serialization."""
    if s is None:
        return None
    return (s.start, s.stop)


def create_processing_log(
    paths: DerivedPaths,
    num_timepoints: int,
    top_roi: tuple[slice, slice] | None,
    bottom_roi: tuple[slice, slice] | None,
    output_format: OutputFormat,
    rotate_90: bool = False,
    channels_to_output: list[int] | None = None,
):
    """Writes a JSON log file with all processing parameters."""

    timestamps = parse_timestamps(paths.metadata_file, num_timepoints)

    if channels_to_output is None:
        channels_to_output = [0, 1, 2, 3]  # Default fallback

    notes = (
        f"Cropped {len(channels_to_output)} channels: {channels_to_output}. "
        f"Output format: {output_format.value}"
    )

    log_data = {
        "processing_version": "3.1-selective-channel",
        "processing_date": datetime.now().isoformat(),
        "channels_exported": channels_to_output,
        "rotate_90_degrees": rotate_90,
        "source_base_file": str(paths.base_file),
        "source_metadata_file": str(paths.metadata_file),
        "rois": {
            "top_roi": (
                (_format_slice(top_roi[0]), _format_slice(top_roi[1]))
                if top_roi
                else None
            ),
            "bottom_roi": (
                (_format_slice(bottom_roi[0]), _format_slice(bottom_roi[1]))
                if bottom_roi
                else None
            ),
        },
        "notes": notes,
        "timestamps_sec": timestamps,
    }

    try:
        with paths.output_log.open("w") as f:
            json.dump(log_data, f, indent=4)
        print(f"✅ Successfully wrote processing log to {paths.output_log.name}")
    except Exception as e:
        print(f"Error writing log file: {e}", file=sys.stderr)
