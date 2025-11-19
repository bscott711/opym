# Ruff style: Compliant
"""
Module for submitting PyPetaKit5D processing jobs to a persistent MATLAB server.

This module uses a producer-consumer model. It generates a JSON job file in a
watched queue directory, which a running MATLAB server picks up and processes.
"""

import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

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
    """
    Identifies the dataset structure from the user's selected folder.
    """
    processed_dir = processed_dir_path.resolve()
    if not processed_dir.exists():
        raise FileNotFoundError(f"Directory not found: {processed_dir}")

    base_data_dir = processed_dir.parent

    # 1. Try finding the log file to get the base name
    log_file = next(processed_dir.glob("*_processing_log.json"), None)
    if log_file:
        base_name = log_file.stem.replace("_processing_log", "")
    else:
        # 2. Fallback: Parse from the first TIFF file
        first_file = next(processed_dir.glob("*_C[0-9]_T[0-9][0-9][0-9].tif"), None)
        if not first_file:
            raise FileNotFoundError(
                "Could not determine base name. No log or data files found."
            )
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
    """
    Writes the job payload to a JSON file in the queue directory.
    Returns: The filename of the submitted job.
    """
    # Ensure all directories exist
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    COMPLETED_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_DIR.mkdir(parents=True, exist_ok=True)

    # Create a unique filename based on timestamp and base name
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


def run_petakit_processing(
    processed_dir_path: Path,
    **kwargs,
) -> str:
    """
    Prepares context and submits a 'deskew' job for OPM data.
    """
    try:
        print(f"--- Preparing PetaKit5D job for: {processed_dir_path.name} ---")
        ctx = get_petakit_context(processed_dir_path)

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


def run_llsm_petakit_processing(source_dir: Path, **kwargs) -> str:
    """
    Submits a 'deskew' job for an LLSM dataset.
    """
    try:
        print(f"--- Preparing LLSM PetaKit5D job for: {source_dir.name} ---")
        source_dir = source_dir.resolve()
        if not source_dir.exists():
            raise FileNotFoundError(f"Directory not found: {source_dir}")

        # Try to find an LLSM file to determine base name
        # Pattern: name_CamA_ch0_stack0000.tif
        first_file = next(
            source_dir.glob("*_Cam[AB]_ch[0-9]_stack[0-9][0-9][0-9][0-9]*.tif"),
            None,
        )
        if not first_file:
            # Fallback for generic folders (like 'decon' output) that might have TIFs
            # Try finding *any* TIF if the specific pattern fails
            first_file = next(source_dir.glob("*.tif"), None)

        if not first_file:
            raise FileNotFoundError(f"No TIFF files found in {source_dir}")

        # Attempt to parse LLSM base name, or fallback to folder name
        match = re.search(r"^(.*?)_Cam[AB]_", first_file.name)
        if match:
            base_name = match.group(1)
        else:
            base_name = source_dir.name

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
    """
    Submits a 'decon' job to the MATLAB server.
    """
    try:
        print(f"--- Preparing Decon job for: {data_dir.name} ---")
        data_dir = data_dir.resolve()
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        # Base name for the job file (just use folder name)
        base_name = data_dir.name

        payload = {
            "jobType": "decon",
            "dataDir": str(data_dir),
            "baseName": base_name,
            "parameters": {
                "channel_patterns": channel_patterns,
                "psf_paths": psf_paths,
                "result_dir_name": result_dir_name,
                "iterations": iterations,
                "gpu_job": gpu_job,
                "skewed": skewed,
                "rl_method": rl_method,
                "save_16bit": True,
                **kwargs,  # include any extras
            },
        }

        return _submit_job(payload)

    except Exception as e:
        print(f"\n❌ Error submitting Decon job: {e}")
        raise


def wait_for_job(job_filename: str, poll_interval: int = 5) -> None:
    """
    Blocks execution and polls for the job completion.
    """
    print(f"\n⏳ Waiting for MATLAB server to process: {job_filename} ...")
    print("   (This cell will remain running until the job finishes)")

    job_path_done = COMPLETED_DIR / job_filename
    job_path_fail = FAILED_DIR / job_filename

    start_time = time.time()

    while True:
        # Check Success
        if job_path_done.exists():
            duration = time.time() - start_time
            print(f"\n✅ Job Completed Successfully! (Time: {duration:.1f}s)")
            return

        # Check Failure
        if job_path_fail.exists():
            print(f"\n❌ Job Failed! (Time: {time.time() - start_time:.1f}s)")
            # Try to read the error log
            log_file = FAILED_DIR / f"{job_filename}.log"
            if log_file.exists():
                print("-" * 40)
                print(log_file.read_text())
                print("-" * 40)
            else:
                print("   No error log found.")
            raise RuntimeError("MATLAB processing failed.")

        # Still processing?
        sys.stdout.write(".")
        sys.stdout.flush()

        time.sleep(poll_interval)
