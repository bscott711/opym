# Ruff style: Compliant
"""
Module for submitting PyPetaKit5D processing jobs to a persistent MATLAB server.
"""

import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import tifffile  # Needed to inspect dimensions for cropping jobs

# --- CONFIGURATION ---
BASE_JOB_DIR = Path.home() / "petakit_jobs"
QUEUE_DIR = BASE_JOB_DIR / "queue"
COMPLETED_DIR = BASE_JOB_DIR / "completed"
FAILED_DIR = BASE_JOB_DIR / "failed"


@dataclass(frozen=True)
class PetaKitContext:
    """Holds paths required to identify the dataset."""

    base_data_dir: Path
    processed_dir: Path
    base_name: str


def get_petakit_context(processed_dir_path: Path) -> PetaKitContext:
    """Identifies the dataset structure from the user's selected folder."""
    processed_dir = processed_dir_path.resolve()
    if not processed_dir.exists():
        raise FileNotFoundError(f"Directory not found: {processed_dir}")

    base_data_dir = processed_dir.parent

    log_file = next(processed_dir.glob("*_processing_log.json"), None)
    if log_file:
        base_name = log_file.stem.replace("_processing_log", "")
    else:
        first_file = next(processed_dir.glob("*_C[0-9]_T[0-9][0-9][0-9].tif"), None)
        if not first_file:
            raise FileNotFoundError("Could not determine base name.")
        match = re.search(r"^(.*?)_C\d_T\d{3}\.tif$", first_file.name)
        if not match:
            raise ValueError(f"Could not parse base name from file: {first_file.name}")
        base_name = match.group(1)

    return PetaKitContext(
        base_data_dir=base_data_dir,
        processed_dir=processed_dir,
        base_name=base_name,
    )


def _submit_job(payload: dict) -> str:
    """Writes the job payload to a JSON file in the queue directory."""
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    COMPLETED_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = int(time.time() * 1000)
    safe_name = payload.get("baseName", "job")
    safe_name = re.sub(r"[^\w\-_\.]", "_", safe_name)
    job_filename = f"{safe_name}_{timestamp}.json"
    job_file = QUEUE_DIR / job_filename

    print(f"--- Submitting job to queue: {job_file.name} ---")

    try:
        with job_file.open("w") as f:
            json.dump(payload, f, indent=4)
        print("✅ Job ticket created.")
        return job_filename
    except Exception as e:
        print(f"❌ Failed to write job file: {e}")
        raise


def _slice_to_list(s: slice | None) -> list[int] | None:
    """Converts a python slice object to a [start, stop] list for JSON."""
    if s is None:
        return None
    # Handle None in slice (e.g. [:100])
    start = s.start if s.start is not None else 0
    # We assume stop is provided for ROIs, but handle edge case
    stop = s.stop if s.stop is not None else -1
    return [int(start), int(stop)]


def run_petakit_processing(
    processed_dir_path: Path,
    interp_method: str = "cubic",
    **kwargs,
) -> str:
    """Prepares context and submits a 'deskew' job for OPM data."""
    try:
        print(f"--- Preparing PetaKit5D job for: {processed_dir_path.name} ---")
        ctx = get_petakit_context(processed_dir_path)
        kwargs["interp_method"] = interp_method

        payload = {
            "jobType": "deskew",
            "dataDir": str(ctx.processed_dir),
            "baseName": ctx.base_name,
            "parameters": kwargs,
        }
        return _submit_job(payload)
    except Exception as e:
        print(f"\n❌ Error submitting job: {e}")
        raise


def run_llsm_petakit_processing(
    source_dir: Path, interp_method: str = "cubic", **kwargs
) -> str:
    """Submits a 'deskew' job for an LLSM dataset."""
    try:
        print(f"--- Preparing LLSM PetaKit5D job for: {source_dir.name} ---")
        source_dir = source_dir.resolve()
        if not source_dir.exists():
            raise FileNotFoundError(f"Directory not found: {source_dir}")

        first_file = next(
            source_dir.glob("*_Cam[AB]_ch[0-9]_stack[0-9][0-9][0-9][0-9]*.tif"), None
        )
        if not first_file:
            first_file = next(source_dir.glob("*.tif"), None)
        if not first_file:
            raise FileNotFoundError(f"No TIFF files found in {source_dir}")

        match = re.search(r"^(.*?)_Cam[AB]_", first_file.name)
        base_name = match.group(1) if match else source_dir.name

        kwargs["interp_method"] = interp_method
        payload = {
            "jobType": "deskew",
            "dataDir": str(source_dir),
            "baseName": base_name,
            "parameters": kwargs,
        }
        return _submit_job(payload)
    except Exception as e:
        print(f"\n❌ Error submitting LLSM job: {e}")
        raise


def run_decon_processing(
    data_dir: Path,
    channel_patterns: list[str],
    psf_paths: list[str],
    result_dir_name: str = "decon",
    iterations: int = 10,
    gpu_job: bool = True,
    skewed: bool = True,
    rl_method: str = "simplified",
    **kwargs,
) -> str:
    """Submits a 'decon' job."""
    try:
        print(f"--- Preparing Decon job for: {data_dir.name} ---")
        data_dir = data_dir.resolve()
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        payload = {
            "jobType": "decon",
            "dataDir": str(data_dir),
            "baseName": data_dir.name,
            "parameters": {
                "channel_patterns": channel_patterns,
                "psf_paths": psf_paths,
                "result_dir_name": result_dir_name,
                "iterations": iterations,
                "gpu_job": gpu_job,
                "skewed": skewed,
                "rl_method": rl_method,
                "save_16bit": True,
                **kwargs,
            },
        }
        return _submit_job(payload)
    except Exception as e:
        print(f"\n❌ Error submitting Decon job: {e}")
        raise


def run_cropping_processing(
    base_file: Path,
    output_dir: Path,
    sanitized_name: str,
    top_roi: tuple[slice, slice] | None,
    bottom_roi: tuple[slice, slice] | None,
    channels_to_output: list[int],
    rotate_90: bool = False,
    **kwargs,
) -> str:
    """
    Submits a 'crop' job to the MATLAB server.
    Inspects file dimensions locally first to pass to MATLAB.
    """
    try:
        print(f"--- Preparing Crop job for: {base_file.name} ---")
        base_file = base_file.resolve()
        output_dir = output_dir.resolve()

        if not base_file.exists():
            raise FileNotFoundError(f"Input file not found: {base_file}")

        # Inspect dimensions locally using tifffile
        # This saves MATLAB from guessing/scanning the whole file header
        with tifffile.TiffFile(base_file) as tif:
            series = tif.series[0]
            shape = series.shape
            # Detect dimensions (ZCYX or TZCXY)
            ndim = len(shape)
            if ndim == 4:  # ZCYX
                T, Z, C, Y, X = 1, shape[0], shape[1], shape[2], shape[3]
            elif ndim == 5:  # TZCXY
                T, Z, C, Y, X = shape
            else:
                raise ValueError(f"Unsupported dimensions: {shape}")

        print(f"  Detected Shape: T={T}, Z={Z}, C={C}, Y={Y}, X={X}")

        # Convert python slices to JSON-safe lists [start, stop]
        top_roi_list = (
            [_slice_to_list(top_roi[0]), _slice_to_list(top_roi[1])] if top_roi else []
        )
        bot_roi_list = (
            [_slice_to_list(bottom_roi[0]), _slice_to_list(bottom_roi[1])]
            if bottom_roi
            else []
        )

        payload = {
            "jobType": "crop",
            "dataDir": str(base_file),
            "baseName": sanitized_name,
            "parameters": {
                "output_dir": str(output_dir),
                "channels_to_output": channels_to_output,
                "rotate_90": rotate_90,
                "top_roi": top_roi_list,
                "bottom_roi": bot_roi_list,
                "dims": {"T": T, "Z": Z, "C": C, "Y": Y, "X": X},
                **kwargs,
            },
        }

        return _submit_job(payload)

    except Exception as e:
        print(f"\n❌ Error submitting Crop job: {e}")
        raise


def wait_for_job(job_filename: str, poll_interval: int = 5) -> None:
    """Blocks execution and polls for the job completion."""
    print(f"\n⏳ Waiting for MATLAB server to process: {job_filename} ...")
    job_path_done = COMPLETED_DIR / job_filename
    job_path_fail = FAILED_DIR / job_filename

    start_time = time.time()

    while True:
        if job_path_done.exists():
            print(
                f"\n✅ Job Completed Successfully! "
                f"(Time: {time.time() - start_time:.1f}s)"
            )
            return

        if job_path_fail.exists():
            print(f"\n❌ Job Failed! (Time: {time.time() - start_time:.1f}s)")
            log_file = FAILED_DIR / f"{job_filename}.log"
            if log_file.exists():
                print("-" * 40)
                print(log_file.read_text())
                print("-" * 40)
            else:
                print("   No error log found.")
            raise RuntimeError("MATLAB processing failed.")

        sys.stdout.write(".")
        sys.stdout.flush()
        time.sleep(poll_interval)
