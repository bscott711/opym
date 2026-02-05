# Ruff style: Compliant
"""
Utilities for interacting with the PetaKit job queue system.
"""

import json
import re
import threading
import time
from pathlib import Path

import ipywidgets as widgets

from .roi_utils import _roi_to_tuple, _tuple_to_cli_string


def submit_remote_crop_job(
    base_file: Path,
    top_roi: tuple[slice, slice] | None,
    bottom_roi: tuple[slice, slice] | None,
    channels: list[int],
    output_format: str = "tiff-series",
    rotate: bool = True,
    queue_dir: Path = Path.home() / "petakit_jobs" / "queue",
) -> Path:
    """Creates a JSON job ticket for Cropping."""
    rois = {}
    if top_roi:
        rois["top"] = _tuple_to_cli_string(_roi_to_tuple(top_roi))
    if bottom_roi:
        rois["bottom"] = _tuple_to_cli_string(_roi_to_tuple(bottom_roi))

    payload = {
        "jobType": "crop",
        "dataDir": str(base_file.resolve()),
        "baseName": base_file.stem,
        "parameters": {
            "rois": rois,
            "channels": channels,
            "rotate": rotate,
            "format": output_format,
        },
    }
    return _write_ticket(payload, base_file.stem, "CROP", queue_dir)


def submit_remote_deskew_job(
    input_dir: Path,
    z_step_um: float,
    xy_pixel_size: float = 0.136,
    sheet_angle_deg: float = 31.8,
    deskew: bool = True,
    rotate: bool = True,
    queue_dir: Path = Path.home() / "petakit_jobs" / "queue",
) -> Path:
    """Creates a JSON job ticket for Deskew/Rotate."""
    payload = {
        "jobType": "deskew",
        "dataDir": str(input_dir.resolve()),
        "baseName": input_dir.name,
        "parameters": {
            "xy_pixel_size": xy_pixel_size,
            "z_step_um": z_step_um,
            "sheet_angle_deg": sheet_angle_deg,
            "deskew": deskew,
            "rotate": rotate,
        },
    }
    return _write_ticket(payload, input_dir.name, "DESKEW", queue_dir)


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
        input_dir=processed_dir_path,
        z_step_um=z_step_um,
        xy_pixel_size=xy_pixel_size,
        sheet_angle_deg=sheet_angle_deg,
        deskew=deskew,
        rotate=rotate,
    )


def wait_for_job(job_path: Path, poll_interval: int = 2) -> bool:
    """
    Blocks and monitors the job ticket (Blocking version).
    Useful if you DON'T want async background updates.
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
    queue_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time() * 1000)
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
