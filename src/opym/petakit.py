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
    sheet_angle_deg: float = 32.0,
    deskew: bool = True,
    rotate: bool = True,
    interp_method: str = "cubic",
    ds_dir_name: str = "DS",
    dsr_dir_name: str = "DSR",
    queue_dir: Path = QUEUE_DIR,
    psf_path: str | Path | None = None,
    n_iters: int | None = None,
    channel_patterns: list[str] | None = None,
    input_axis_order: str = "zyx",  # ✅ NEW: Default to standard TIFF order
    output_axis_order: str = "zyx",  # ✅ NEW: Default to standard TIFF order
) -> Path:
    """
    Creates a JSON job ticket for Deskew/Rotate and optional Deconvolution.

    Parameters
    ----------
    input_axis_order : str, default 'zyx'
        Axis order of input data. Options: 'zyx', 'yxz', 'xyz', etc.
        'zyx' = standard TIFF stack (Z-slices, Y-rows, X-columns)
    output_axis_order : str, default 'zyx'
        Desired axis order of output data.
    """
    _ensure_directories()
    input_target = Path(input_target).resolve()

    # --- PATH REDIRECTION LOGIC ---
    if input_target.is_file():
        folder_name = input_target.name
        if folder_name.lower().endswith(".ome.tif"):
            folder_name = folder_name[:-8]
        elif folder_name.lower().endswith(".tif"):
            folder_name = folder_name[:-4]

        potential_dir = input_target.parent / folder_name
        if potential_dir.exists():
            input_target = potential_dir
        else:
            legacy_dir = input_target.parent / "processed_tiff_series_split"
            if legacy_dir.exists():
                input_target = legacy_dir

    if not input_target.exists():
        raise FileNotFoundError(f"Input directory not found: {input_target}")

    base_name = input_target.name

    # Build parameters dictionary
    params = {
        "ds_dir_name": ds_dir_name,
        "dsr_dir_name": dsr_dir_name,
        "deskew": deskew,
        "rotate": rotate,
        "interp_method": interp_method,
        "xy_pixel_size": xy_pixel_size,
        "z_step_um": z_step_um,
        "sheet_angle_deg": sheet_angle_deg,
        "channel_patterns": channel_patterns,
        "input_axis_order": input_axis_order,  # ✅ NEW
        "output_axis_order": output_axis_order,  # ✅ NEW
    }

    # Add deconvolution parameters if a PSF is provided
    if psf_path:
        params["run_decon"] = True
        params["psf_path"] = str(psf_path)
        params["decon_iter"] = n_iters if n_iters is not None else 10

    payload = {
        "jobType": "deskew",
        "dataDir": str(input_target),
        "baseName": f"{base_name}*",
        "parameters": params,
    }

    return _write_ticket(payload, base_name, "DESKEW", queue_dir)


# --- BACKWARD COMPATIBILITY ALIASES ---
def run_petakit_processing(
    processed_dir_path: Path,
    z_step_um: float,
    xy_pixel_size: float = 0.136,
    sheet_angle_deg: float = 32.0,
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
                break
            if failed_path.exists():
                status_label.value = f"❌ Job Failed. ({elapsed}s)"
                break

            status_label.value = f"⏳ Running... (Elapsed: {elapsed}s)"
            time.sleep(2)

    t = threading.Thread(target=_poll, daemon=True)
    t.start()


def submit_crop_and_save_sidecar(
    file_path: Path,
    top_roi: tuple[slice, slice] | None,
    bottom_roi: tuple[slice, slice] | None,
    channels: list[int],
    output_format: str,
    rotate: bool,
) -> tuple[Path, Path]:
    """
    Submits a crop job and immediately saves the settings sidecar
    to the destination folder.

    Returns:
        tuple[Path, Path]: (job_ticket_path, output_directory_path)
    """
    # 1. Submit the job
    job_path = submit_remote_crop_job(
        base_file=file_path,
        top_roi=top_roi,
        bottom_roi=bottom_roi,
        channels=channels,
        output_format=output_format,
        rotate=rotate,
    )

    # 2. Determine and create output directory
    name = file_path.name
    if name.endswith(".ome.tif"):
        clean_name = name[:-8]
    elif name.endswith(".tif"):
        clean_name = name[:-4]
    else:
        clean_name = file_path.stem

    output_dir = file_path.parent / clean_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3. Create JSON Sidecar (Safely handling None ROIs)
    sidecar = output_dir / "petakit_settings.json"

    rois = {}
    if top_roi:
        rois["top"] = _tuple_to_cli_string(_roi_to_tuple(top_roi))
    if bottom_roi:
        rois["bottom"] = _tuple_to_cli_string(_roi_to_tuple(bottom_roi))

    settings = {
        "source_file": str(file_path.name),
        "rois": rois,
        "channels": channels,
        "rotate": rotate,
        "format": output_format,
    }

    with open(sidecar, "w") as f:
        json.dump(settings, f, indent=4)

    return job_path, output_dir
