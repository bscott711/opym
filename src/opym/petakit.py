# Ruff style: Compliant
"""
Module for submitting PyPetaKit5D processing jobs to a persistent MATLAB server.

This module uses a producer-consumer model. It generates a JSON job file in a
watched queue directory, which a running MATLAB server picks up and processes.
"""

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

# --- CONFIGURATION ---
# Use Path.home() to make this generic for any user
# The server script must use the same relative path: ~/petakit_jobs/queue
QUEUE_DIR = Path.home() / "petakit_jobs/queue"


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


def _submit_job(payload: dict) -> None:
    """Writes the job payload to a JSON file in the queue directory."""
    # Ensure queue exists
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)

    # Create a unique filename based on timestamp and base name
    timestamp = int(time.time() * 1000)
    safe_name = payload.get("baseName", "job")
    # Sanitize filename
    safe_name = re.sub(r"[^\w\-_\.]", "_", safe_name)
    job_filename = f"{safe_name}_{timestamp}.json"
    job_file = QUEUE_DIR / job_filename

    print(f"--- Submitting job to queue: {job_file} ---")

    try:
        with job_file.open("w") as f:
            json.dump(payload, f, indent=4)
        print("✅ Job submitted successfully. The MATLAB server will pick it up.")
    except Exception as e:
        print(f"❌ Failed to write job file: {e}")
        raise


def run_petakit_processing(
    processed_dir_path: Path,
    **kwargs,
) -> None:
    """
    Prepares context and submits a job for the provided directory.

    Any keyword arguments passed (e.g., xy_pixel_size, z_step_um) are
    forwarded to the MATLAB server in the JSON payload.

    Args:
        processed_dir_path: Path to 'processed_tiff_series_split'
        **kwargs: Parameters to pass to MATLAB (e.g., z_step_um=1.0)
    """
    try:
        print(f"--- Preparing PetaKit5D job for: {processed_dir_path.name} ---")
        ctx = get_petakit_context(processed_dir_path)

        # Construct the payload
        # The server reads 'dataDir', 'baseName', and 'parameters'
        payload = {
            "dataDir": str(ctx.processed_dir),
            "baseName": ctx.base_name,
            "parameters": kwargs,
        }

        _submit_job(payload)

    except Exception as e:
        print(f"\n❌ Error submitting job: {e}")
        raise


def run_llsm_petakit_processing(source_dir: Path, **kwargs) -> None:
    """
    Submits a job for an LLSM dataset (skipping opym preprocessing).
    """
    try:
        print(f"--- Preparing LLSM PetaKit5D job for: {source_dir.name} ---")
        source_dir = source_dir.resolve()
        if not source_dir.exists():
            raise FileNotFoundError(f"Directory not found: {source_dir}")

        # Determine base name for LLSM
        first_file = next(
            source_dir.glob("*_Cam[AB]_ch[0-9]_stack[0-9][0-9][0-9][0-9]*.tif"),
            None,
        )
        if not first_file:
            raise FileNotFoundError(f"No LLSM files found in {source_dir}")

        match = re.search(r"^(.*?)_Cam[AB]_", first_file.name)
        if not match:
            raise ValueError(f"Could not parse base name from: {first_file.name}")
        base_name = match.group(1)

        payload = {
            "dataDir": str(source_dir),
            "baseName": base_name,
            "parameters": kwargs,
        }

        _submit_job(payload)

    except Exception as e:
        print(f"\n❌ Error submitting LLSM job: {e}")
        raise
