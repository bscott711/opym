# Ruff style: Compliant
"""
Module for running PyPetaKit5D processing on 'opym' pre-processed data.

This module is designed to be called after 'opym.run_processing_job'
has successfully generated a 'processed_tiff_series_split' directory.
"""

import inspect
import json
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

import PyPetaKit5D as ppk

# No longer need OutputFormat from utils
# from .utils import OutputFormat


@dataclass(frozen=True)
class PetaKitContext:
    """Holds all paths required for a PyPetaKit5D processing job."""

    base_data_dir: Path
    processed_dir: Path
    log_file: Path
    base_name: str
    job_log_dir: Path
    ds_output_dir: Path
    dsr_output_dir: Path


def get_petakit_context(processed_dir_path: Path) -> PetaKitContext:
    """
    Dynamically generates all necessary paths for a PyPetaKit5D job
    from the user-selected processed directory.

    Args:
        processed_dir_path: The path to the processed TIFF directory
                            (e.g., '.../processed_tiff_series_split').

    Returns:
        A PetaKitContext dataclass instance with all paths.
    """
    processed_dir = processed_dir_path.resolve()
    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed TIFF directory not found: {processed_dir}")

    # 1. The base data dir is one level up
    base_data_dir = processed_dir.parent

    # 2. Find the opym processing log *inside* the provided path
    log_files = list(processed_dir.glob("*_processing_log.json"))
    if not log_files:
        raise FileNotFoundError(
            f"No '*_processing_log.json' file found in {processed_dir}. "
            "Please ensure this is the correct opym output folder."
        )
    log_file = log_files[0]

    # 3. Get the base name from the log file
    base_name = log_file.stem.replace("_processing_log", "")

    # 4. Define all output paths
    # Job logs are stored in the base data dir
    job_log_dir = base_data_dir / "job_logs"
    # PyPetaKit outputs are stored *inside* the processed dir
    ds_output_dir = processed_dir / "DS"
    dsr_output_dir = processed_dir / "DSR"

    return PetaKitContext(
        base_data_dir=base_data_dir,
        processed_dir=processed_dir,
        log_file=log_file,
        base_name=base_name,
        job_log_dir=job_log_dir,
        ds_output_dir=ds_output_dir,
        dsr_output_dir=dsr_output_dir,
    )


def run_petakit_processing(
    processed_dir_path: Path,
    *,
    # --- Physical Parameters ---
    xy_pixel_size: float = 0.108,
    z_step_um: float = 1.0,
    sheet_angle_deg: float = 21.72,
    # --- Reader/Writer Flags ---
    large_file: bool = False,
    dsr_combined: bool = False,
    zarr_file: bool = False,
    save_zarr: bool = False,
    save_16bit: bool = True,
    save_3d_stack: bool = True,
    save_mip: bool = True,
    # --- MCC Logic Flags ---
    objective_scan: bool = False,
    reverse_z: bool = True,
    # --- Processing ---
    deskew: bool = True,
    rotate: bool = True,
    interp_method: str = "linear",
    block_size: list[int] | None = None,
    # --- Cluster & Config ---
    parse_cluster: bool = False,
    master_compute: bool = True,
    config_file: str = "",
    mcc_mode: bool = True,
    # --- Redundant/Unused PyPetaKit5D args ---
    ff_correction: bool = False,
    lower_limit: float = 0.4,
    const_offset: float = 1.0,
    ff_image_paths: list[str] | None = None,
    background_paths: list[str] | None = None,
    bk_removal: bool = False,
) -> None:
    """
    (Base function) Runs the PyPetaKit5D deskew/rotate wrapper on
    an 'opym' processed TIFF series.

    Args:
        processed_dir_path: The path to the processed TIFF directory
                            (e.g., '.../processed_tiff_series_split').
        **kwargs: See function signature for all processing options.
    """
    if block_size is None:
        block_size = [256, 256, 256]
    if ff_image_paths is None:
        ff_image_paths = [""]
    if background_paths is None:
        background_paths = [""]

    try:
        # 1. Get all paths
        print(f"--- Setting up PetaKit5D for: {processed_dir_path.name} ---")
        ctx = get_petakit_context(processed_dir_path)

        # 2. Create required directories
        os.makedirs(ctx.job_log_dir, exist_ok=True)
        # Note: PyPetaKit5D creates the DS/DSR directories

        print(f"\nRunning job locally for TIFF series in: {ctx.processed_dir.name}")
        print(f"  Base name: {ctx.base_name}")
        print(f"  Job log directory: {ctx.job_log_dir.name}")
        print(f"  Deskew output: {ctx.ds_output_dir.name}")
        print(f"  Rotate output: {ctx.dsr_output_dir.name}")

        # 3. Run the PyPetaKit5D wrapper
        ppk.XR_deskew_rotate_data_wrapper(
            [str(ctx.processed_dir)],
            deskew=deskew,
            rotate=rotate,
            DSRCombined=dsr_combined,
            xyPixelSize=xy_pixel_size,
            dz=z_step_um,
            skewAngle=sheet_angle_deg,
            objectiveScan=objective_scan,
            reverse=reverse_z,
            channelPatterns=[ctx.base_name],
            # Define separate output directories
            DSDirName=ctx.ds_output_dir.name,
            DSRDirName=ctx.dsr_output_dir.name,
            # Pass through all other flags
            FFCorrection=ff_correction,
            lowerLimit=lower_limit,
            constOffset=const_offset,
            FFImagePaths=ff_image_paths,
            backgroundPaths=background_paths,
            largeFile=large_file,
            zarrFile=zarr_file,
            saveZarr=save_zarr,
            blockSize=block_size,
            save16bit=save_16bit,
            parseCluster=parse_cluster,
            masterCompute=master_compute,
            configFile=config_file,
            mccMode=mcc_mode,
            BKRemoval=bk_removal,
            save3DStack=save_3d_stack,
            saveMIP=save_mip,
            interpMethod=interp_method,
        )

        print("--- PyPetaKit5D Processing Complete ---")

    except FileNotFoundError as fnfe:
        print(f"\n❌ SETUP ERROR: {fnfe}", file=sys.stderr)
        print("Ensure the input path and files exist.", file=sys.stderr)
        raise  # Re-raise for the notebook to catch
    except Exception as e:
        print(f"\n❌ FATAL ERROR in PyPetaKit5D: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        raise  # Re-raise for the notebook to catch


def run_petakit_from_config(
    processed_dir_path: Path,
    config_file: Path,
) -> None:
    """
    Loads parameters from a JSON file and runs petakit processing.

    This function is a wrapper around 'run_petakit_processing'.
    Any parameters defined in the JSON file will override the defaults
    in 'run_petakit_processing'.

    Args:
        processed_dir_path: The path to the processed TIFF directory
                            (e.g., '.../processed_tiff_series_split').
        config_file: The path to the .json parameter file.
    """
    print(f"--- Loading parameters from: {config_file.name} ---")
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    try:
        with config_file.open("r") as f:
            params = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: Could not parse JSON config file: {e}")
        raise

    # Get all valid argument names from the base function
    sig = inspect.signature(run_petakit_processing)
    valid_keys = {p.name for p in sig.parameters.values()}

    # Filter the loaded params to only include valid keyword arguments
    filtered_params = {}
    for key, value in params.items():
        if key in valid_keys:
            filtered_params[key] = value
        else:
            print(f"  Warning: Ignoring unknown parameter '{key}' in JSON.")

    # Call the base function, "splatting" the loaded params.
    run_petakit_processing(
        processed_dir_path=processed_dir_path,
        **filtered_params,
    )
