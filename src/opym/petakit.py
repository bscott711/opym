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
BASE_DIR = Path("/dev/shm/petakit_jobs")
QUEUE_DIR = BASE_DIR / "queue"
DONE_DIR = BASE_DIR / "completed"
FAIL_DIR = BASE_DIR / "failed"


def _ensure_directories():
    """Ensures the job queue directory exists."""
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)


def submit_remote_crop_job(
    base_file: Path,
    top_roi: tuple[slice, slice] | None,
    bottom_roi: tuple[slice, slice] | None,
    channels: list[int] | None = None,
    timepoints: list[int] | None = None,
    output_format: str = "tiff-series",
    rotate: bool = True,
    z_step_um: float | None = None,
    xy_pixel_size: float | None = None,
    test_mode: bool = False,
    exposure_mode: str = "Single Exposure (All Lasers)",
    active_channels: list[str] | None = None,
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

    if test_mode:
        base_name = f"{base_name}_test"

    payload = {
        "jobType": "crop",
        "dataDir": str(base_file),
        "baseName": base_name,
        "parameters": {
            "rois": rois,
            "channels": channels,
            "timepoints": timepoints,
            "rotate": rotate,
            "format": output_format,
            "exposure_mode": exposure_mode,
            "active_channels": active_channels,
        },
    }

    if z_step_um is not None:
        payload["parameters"]["z_step_um"] = z_step_um
    if xy_pixel_size is not None:
        payload["parameters"]["xy_pixel_size"] = xy_pixel_size

    job_file = _write_ticket(payload, base_name, "CROP", queue_dir)

    print(f"✅ Job Ticket Created: {job_file.name}")
    print(f"   Target Output Dir: {base_name}")

    return job_file


def submit_remote_deskew_job(
    input_target: Path,
    z_step_um: float,
    xy_pixel_size: float = 0.136,
    sheet_angle_deg: float = 60.0,
    deskew: bool = True,
    rotate: bool = True,
    interp_method: str = "cubic",
    ds_dir_name: str = "DS",
    dsr_dir_name: str = "DSR",
    queue_dir: Path = QUEUE_DIR,
    psf_path: str | Path | None = None,
    n_iters: int | None = None,
    channel_patterns: list[str] | None = None,
    input_axis_order: str = "yxz",
    output_axis_order: str = "yxz",
    objective_scan: bool = False,
    z_stage_scan: bool = False,
    reverse: bool = True,
    gpu_decon: bool = False,
    crop_was_rotated: bool = False,
) -> Path:
    """
    Creates a JSON job ticket for Deskew/Rotate and optional Deconvolution.

    Parameters
    ----------
    input_axis_order : str, default 'yxz'
        Axis order of input data. Must match PetaKit5D conventions.
        'yxz' = MATLAB cropper output (rows=Y, cols=X, planes=Z).
        Automatically set to 'xyz' if crop_was_rotated=True, because
        rot90 in MATLAB transposes dim1/dim2 (Y↔X).
    output_axis_order : str, default 'yxz'
        Desired axis order of output data.
    objective_scan : bool, default False
        True if the objective moves during scanning. For standard galvo-
        scanned OPM this should be False.
    z_stage_scan : bool, default False
        True if the sample stage physically moves in Z during acquisition.
        For standard galvo-scanned OPM this should be False.
    reverse : bool, default False
        Reverse the shear direction along the Z axis.
    gpu_decon : bool, default False
        Use GPU for deconvolution (requires CUDA-capable GPU on the
        processing node).
    crop_was_rotated : bool, default False
        If True, the crop step applied rot90 to match LLSM visual
        orientation. This swaps Y↔X in the output TIFFs, so
        input_axis_order is auto-corrected to 'xyz'.
    """
    _ensure_directories()
    input_target = Path(input_target).resolve()

    # PetaKit5D's default 'yxz' shears the 2nd dimension (X).
    # Since the galvo sweeps in the depth-Z plane, the coverslip (Y) is the invariant axis.
    # Therefore, we MUST shear the X axis. So 'yxz' is mathematically perfect.
    input_axis_order = "yxz"
    output_axis_order = "yxz"

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
        "input_axis_order": input_axis_order,
        "output_axis_order": output_axis_order,
        "objective_scan": objective_scan,
        "z_stage_scan": z_stage_scan,
        "reverse": reverse,
    }

    if psf_path:
        params["run_decon"] = True
        params["psf_path"] = str(psf_path)
        params["decon_iter"] = n_iters if n_iters is not None else (2 if rl_method == "omw" else 25)
        params["rl_method"] = rl_method
        params["gpu_decon"] = gpu_decon

    payload = {
        "jobType": "deskew",
        "dataDir": str(input_target),
        "baseName": f"{base_name}*",
        "parameters": params,
    }

    return _write_ticket(payload, base_name, "DESKEW", queue_dir)


def submit_remote_decon_job(
    input_target: Path,
    psf_paths: list[str] | str | Path,
    iterations: int | None = None,
    gpu_job: bool = True,
    skewed: bool = True,
    result_dir_name: str = "Decon",
    channel_patterns: list[str] | None = None,
    rl_method: str = "simple",
    queue_dir: Path = QUEUE_DIR,
) -> Path:
    """
    Creates a JSON job ticket for standalone Deconvolution.
    """
    _ensure_directories()
    input_target = Path(input_target).resolve()

    if not input_target.exists():
        raise FileNotFoundError(f"Input directory not found: {input_target}")

    base_name = input_target.name

    params = {
        "result_dir_name": result_dir_name,
        "iterations": iterations if iterations is not None else (2 if rl_method == "omw" else 25),
        "gpu_job": gpu_job,
        "skewed": skewed,
        "rl_method": rl_method,
        "save_16bit": True,
    }

    if channel_patterns:
        params["channel_patterns"] = channel_patterns

    payload = {
        "jobType": "decon",
        "dataDir": str(input_target),
        "baseName": f"{base_name}*",
        "parameters": params,
    }

    return _write_ticket(payload, base_name, "DECON", queue_dir)


def submit_pipeline_job(
    output_file: Path,
    shm_path: Path,
    psf_paths: list[str] | str | Path,
    z_step_um: float,
    xy_pixel_size: float = 0.136,
    sheet_angle_deg: float = 60.0,
    interp_method: str = "cubic",
    iterations: int | None = None,
    rl_method: str = "simple",
    channel_patterns: list[str] | None = None,
    z_crop_end: int | None = None,
    save_zarr: bool = True,
    debug: bool = False,
    queue_dir: Path = QUEUE_DIR,
) -> Path:
    """
    Creates a JSON job ticket for the unified GPU pipeline.
    This job instructs MATLAB to load the temporary file from /dev/shm/,
    perform Decon -> DSR -> Z-Trim on the GPU, and save the final result to output_file.
    """
    _ensure_directories()
    output_file = Path(output_file).resolve()
    data_dir = output_file.parent
    base_name = output_file.name

    params = {
        "shm_path": str(shm_path),
        "xy_pixel_size": xy_pixel_size,
        "z_step_um": z_step_um,
        "sheet_angle_deg": sheet_angle_deg,
        "interp_method": interp_method,
        "iterations": iterations if iterations is not None else (2 if rl_method == "omw" else 25),
        "rl_method": rl_method,
        "save_zarr": save_zarr,
        "debug": debug,
    }
    if z_crop_end is not None:
        params["z_crop_end"] = int(z_crop_end)

    if channel_patterns:
        params["channel_patterns"] = channel_patterns

    if psf_paths is None:
        params["psf_paths"] = []
    elif isinstance(psf_paths, (str, Path)):
        params["psf_paths"] = [str(psf_paths)]
    else:
        params["psf_paths"] = [str(p) for p in psf_paths]

    payload = {
        "jobType": "pipeline",
        "dataDir": str(data_dir),
        "baseName": base_name,
        "parameters": params,
    }

    return _write_ticket(payload, base_name, "PIPELINE", queue_dir)


# --- BACKWARD COMPATIBILITY ALIASES ---
def run_petakit_processing(
    processed_dir_path: Path,
    z_step_um: float,
    xy_pixel_size: float = 0.136,
    sheet_angle_deg: float = 60.0,
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
        timepoints=None,  # Adjust if you want this wrapper to support it
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
