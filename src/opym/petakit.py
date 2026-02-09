# Ruff style: Compliant
"""
Utilities for interacting with the PetaKit job queue system.
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

import ipywidgets as widgets

from .roi_utils import _roi_to_tuple, _tuple_to_cli_string

# Constants
PETAKIT_JOBS_DIR = Path.home() / "petakit_jobs"
QUEUE_DIR = PETAKIT_JOBS_DIR / "queue"


def _ensure_directories():
    """Ensures the job queue directory exists."""
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)


def submit_remote_crop_job(
    base_file: Path,
    top_roi: tuple[slice, slice] | None,
    bottom_roi: tuple[slice, slice] | None,
    channels: list[int] | None = None,
    output_format: str = "tiff-series",
    rotate: bool = True,
    queue_dir: Path = QUEUE_DIR,
) -> Path:
    """
    Creates a JSON job ticket for Cropping.
    Automatically handles BigTiff naming conventions.
    """
    _ensure_directories()
    base_file = Path(base_file).resolve()

    # ROI formatting for CLI
    rois = {}
    if top_roi:
        rois["top"] = _tuple_to_cli_string(_roi_to_tuple(top_roi))
    if bottom_roi:
        rois["bottom"] = _tuple_to_cli_string(_roi_to_tuple(bottom_roi))

    # Calculate clean base name (removing .ome.tif or .tif)
    # This matches the logic in run_bigtiff_cropper.m
    base_name = base_file.name
    if base_name.lower().endswith(".ome.tif"):
        base_name = base_name[:-8]
    elif base_name.lower().endswith(".tif"):
        base_name = base_name[:-4]

    payload = {
        "jobType": "crop",
        "dataDir": str(base_file),
        "baseName": base_name,
        "parameters": {
            "rois": rois,
            "channels": channels,
            "rotate": rotate,
            "format": output_format,
        },
    }

    job_file = _write_ticket(payload, base_name, "CROP", queue_dir)

    print(f"✅ Job Ticket Created: {job_file.name}")
    print(f"   Target Output Dir: {base_name}")

    return job_file


def submit_remote_deskew_job(
    input_target: Path,
    z_step_um: float,
    xy_pixel_size: float = 0.136,
    sheet_angle_deg: float = 31.8,
    deskew: bool = True,
    rotate: bool = True,
    interp_method: str = "cubic",
    ds_dir_name: str = "DS",
    dsr_dir_name: str = "DSR",
    queue_dir: Path = QUEUE_DIR,
) -> Path:
    """
    Creates a JSON job ticket for Deskew/Rotate.

    INTELLIGENT PATH HANDLING:
    If input_target is a file (e.g., 'Data.ome.tif'), it automatically
    redirects to the folder ('Data') where the cropped files live.
    """
    _ensure_directories()
    input_target = Path(input_target).resolve()

    # --- PATH REDIRECTION LOGIC ---
    if input_target.is_file():
        # Strip extension to find the folder (e.g. 'Data.ome.tif' -> 'Data')
        folder_name = input_target.name
        if folder_name.lower().endswith(".ome.tif"):
            folder_name = folder_name[:-8]
        elif folder_name.lower().endswith(".tif"):
            folder_name = folder_name[:-4]

        # Redirect target to the folder next to the file
        potential_dir = input_target.parent / folder_name
        if potential_dir.exists():
            print(
                f"-> Redirecting input from file '{input_target.name}' "
                f"to folder '{folder_name}'"
            )
            input_target = potential_dir
        else:
            # Fallback: Check for legacy folder
            legacy_dir = input_target.parent / "processed_tiff_series_split"
            if legacy_dir.exists():
                print(
                    "   -> Redirecting to legacy folder 'processed_tiff_series_split'"
                )
                input_target = legacy_dir

    if not input_target.exists():
        raise FileNotFoundError(f"Input directory not found: {input_target}")

    base_name = input_target.name

    payload = {
        "jobType": "deskew",
        "dataDir": str(input_target),
        "baseName": f"{base_name}*",  # Wildcard for PetaKit regex
        "parameters": {
            "ds_dir_name": ds_dir_name,
            "dsr_dir_name": dsr_dir_name,
            "deskew": deskew,
            "rotate": rotate,
            "interp_method": interp_method,
            "xy_pixel_size": xy_pixel_size,
            "z_step_um": z_step_um,
            "sheet_angle_deg": sheet_angle_deg,
        },
    }

    return _write_ticket(payload, base_name, "DESKEW", queue_dir)


# --- BACKWARD COMPATIBILITY ALIASES ---
def run_petakit_processing(
    processed_dir_path: Path,
    z_step_um: float,
    xy_pixel_size: float = 0.136,
    sheet_angle_deg: float = 31.8,
    deskew: bool = True,
    rotate: bool = True,
) -> Path:
    """Alias for submit_remote_deskew_job to match old notebook calls."""
    return submit_remote_deskew_job(
        input_target=processed_dir_path,
        z_step_um=z_step_um,
        xy_pixel_size=xy_pixel_size,
        sheet_angle_deg=sheet_angle_deg,
        deskew=deskew,
        rotate=rotate,
    )


def wait_for_job(job_path: Path, poll_interval: int = 2) -> bool:
    """
    Blocks and monitors the job ticket (Blocking version).
    """
    queue_dir = job_path.parent
    base_dir = queue_dir.parent
    completed_path = base_dir / "completed" / job_path.name
    failed_path = base_dir / "failed" / job_path.name

    print(f"⏳ Monitoring Job: {job_path.name}")
    start_time = time.time()

    try:
        while True:
            if completed_path.exists():
                print(f"✅ Job Finished! ({time.time() - start_time:.1f}s)")
                return True
            if failed_path.exists():
                print("❌ Job Failed.")
                return False

            time.sleep(poll_interval)
    except KeyboardInterrupt:
        return False


def _write_ticket(payload: dict, base_name: str, prefix: str, queue_dir: Path) -> Path:
    """Helper to write the JSON file."""
    timestamp = int(time.time() * 1000)
    # Sanitize name
    safe_name = re.sub(r"[^\w\-_\.]", "_", base_name)
    job_file = queue_dir / f"{prefix}_{safe_name}_{timestamp}.json"

    with open(job_file, "w") as f:
        json.dump(payload, f, indent=4)
    return job_file


def monitor_job_background(job_path: Path, status_label: widgets.Label):
    """
    Spawns a background thread to monitor the job (Async version).
    """

    def _poll():
        queue_dir = job_path.parent
        base_dir = queue_dir.parent
        completed_path = base_dir / "completed" / job_path.name
        failed_path = base_dir / "failed" / job_path.name

        start_time = time.time()

        while True:
            elapsed = int(time.time() - start_time)
            if completed_path.exists():
                status_label.value = f"✅ Job Finished! ({elapsed}s)"
                status_label.style.background = "#e6fffa"
                break
            if failed_path.exists():
                status_label.value = f"❌ Job Failed. ({elapsed}s)"
                status_label.style.background = "#ffe6e6"
                break

            status_label.value = f"⏳ Running... (Elapsed: {elapsed}s)"
            time.sleep(2)

    t = threading.Thread(target=_poll, daemon=True)
    t.start()
