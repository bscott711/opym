# Ruff style: Compliant
"""
Module for running PyPetaKit5D processing on 'opym' pre-processed data.

This module is designed to be called after 'opym.run_processing_job'
has successfully generated a 'processed_tiff_series_split' directory.
"""

import inspect
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import PyPetaKit5D as ppk


@dataclass(frozen=True)
class PetaKitContext:
    """Holds all paths required for a PyPetaKit5D processing job."""

    base_data_dir: Path
    processed_dir: Path
    log_file: Path | None
    base_name: str
    job_log_dir: Path
    ds_output_dir: Path
    dsr_output_dir: Path


def get_petakit_context(
    processed_dir_path: Path,
    ds_dir_name: str = "DS",
    dsr_dir_name: str = "DSR",
) -> PetaKitContext:
    """
    Dynamically generates all necessary paths for a PyPetaKit5D job
    from the user-selected processed directory.

    Args:
        processed_dir_path: The path to the processed TIFF directory
                            (e.g., '.../processed_tiff_series_split').
        ds_dir_name: The name for the deskewed output subdirectory.
        dsr_dir_name: The name for the deskewed+rotated output subdirectory.

    Returns:
        A PetaKitContext dataclass instance with all paths.
    """
    processed_dir = processed_dir_path.resolve()
    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed TIFF directory not found: {processed_dir}")

    base_data_dir = processed_dir.parent

    # Find the opym processing log *inside* the provided path
    log_file = next(processed_dir.glob("*_processing_log.json"), None)
    if log_file:
        base_name = log_file.stem.replace("_processing_log", "")
    else:
        # Fallback: find the first data file and parse its name
        # --- MODIFICATION: Update glob pattern to new format ---
        first_file = next(processed_dir.glob("*_C[0-9]_T[0-9][0-9][0-9].tif"), None)
        if not first_file:
            raise FileNotFoundError(
                "Could not find a '*_processing_log.json' file or any "
                "'*_C..._T..tif' file to determine base_name."
            )
        # --- MODIFICATION: Update regex to new format ---
        match = re.search(r"^(.*?)_C\d_T\d{3}\.tif$", first_file.name)
        if not match:
            raise ValueError(f"Could not parse base name from file: {first_file.name}")
        base_name = match.group(1)

    # Define all output paths
    job_log_dir = base_data_dir / "job_logs"
    # PyPetaKit outputs are stored *inside* the processed dir
    ds_output_dir = processed_dir / ds_dir_name
    dsr_output_dir = processed_dir / dsr_dir_name

    return PetaKitContext(
        base_data_dir=base_data_dir,
        processed_dir=processed_dir,
        log_file=log_file,
        base_name=base_name,
        job_log_dir=job_log_dir,
        ds_output_dir=ds_output_dir,
        dsr_output_dir=dsr_output_dir,
    )


def _run_petakit_base(
    input_dir: Path,
    output_ds_dir: Path,
    output_dsr_dir: Path,
    base_name: str,
    *,
    # Physical Parameters
    xy_pixel_size: float = 0.108,
    z_step_um: float = 1.0,
    sheet_angle_deg: float = 21.72,
    # Reader/Writer Flags
    large_file: bool = False,
    dsr_combined: bool = False,
    zarr_file: bool = False,
    save_zarr: bool = False,
    save_16bit: bool = True,
    save_3d_stack: bool = True,
    save_mip: bool = True,
    # MCC Logic Flags
    objective_scan: bool = False,
    reverse_z: bool = True,
    # Processing
    deskew: bool = True,
    rotate: bool = True,
    interp_method: str = "linear",
    block_size: list[int] | None = None,
    # Cluster & Config
    parse_cluster: bool = False,
    master_compute: bool = False,  # <-- CHANGED from True
    config_file: str = "",
    mcc_mode: bool = False,  # <-- CHANGED from True
    # Redundant/Unused PyPetaKit5D args
    ff_correction: bool = False,
    lower_limit: float = 0.4,
    const_offset: float = 1.0,
    ff_image_paths: list[str] | None = None,
    background_paths: list[str] | None = None,
    bk_removal: bool = False,
) -> None:
    """Internal base function that runs the PyPetaKit5D wrapper."""
    if block_size is None:
        block_size = [256, 256, 256]
    if ff_image_paths is None:
        ff_image_paths = [""]
    if background_paths is None:
        background_paths = [""]

    # Run the PyPetaKit5D wrapper
    ppk.XR_deskew_rotate_data_wrapper(
        [str(input_dir)],
        deskew=deskew,
        rotate=rotate,
        DSRCombined=dsr_combined,
        xyPixelSize=xy_pixel_size,
        dz=z_step_um,
        skewAngle=sheet_angle_deg,
        objectiveScan=objective_scan,
        reverse=reverse_z,
        channelPatterns=[base_name],
        # Define separate output directories
        DSDirName=output_ds_dir.name,
        DSRDirName=output_dsr_dir.name,
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


def run_llsm_petakit_processing(
    source_dir: Path,
    *,
    ds_dir_name: str = "DS",
    dsr_dir_name: str = "DSR",
    **kwargs,
) -> None:
    """
    Runs the PyPetaKit5D wrapper *directly* on an LLSM dataset.

    Args:
        source_dir: Path to the raw LLSM TIFF directory.
        ds_dir_name: Name for the deskewed output subdirectory.
        dsr_dir_name: Name for the deskewed+rotated output subdirectory.
        **kwargs: All other processing parameters for PyPetaKit5D.
    """
    # These need to be defined *before* the try block for the check
    ds_output_dir = source_dir / ds_dir_name
    dsr_output_dir = source_dir / dsr_dir_name
    try:
        print(f"--- Setting up PetaKit5D for LLSM: {source_dir.name} ---")
        source_dir = source_dir.resolve()
        if not source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {source_dir}")

        # 1. Determine base name
        first_file = next(
            source_dir.glob("*_Cam[AB]_ch[0-9]_stack[0-9][0-9][0-9][0-9]*.tif"),
            None,
        )
        if not first_file:
            raise FileNotFoundError(f"No LLSM files found in {source_dir}")

        # Base name is everything before "_CamA" or "_CamB"
        match = re.search(r"^(.*?)_Cam[AB]_", first_file.name)
        if not match:
            raise ValueError(f"Could not parse base name from: {first_file.name}")
        base_name = match.group(1)

        # 2. Define output paths
        job_log_dir = source_dir.parent / "job_logs"
        os.makedirs(job_log_dir, exist_ok=True)

        print(f"\nRunning job locally for LLSM TIFF series in: {source_dir.name}")
        print(f"  Base name (channelPattern): {base_name}")
        print(f"  Job log directory: {job_log_dir.name}")
        print(f"  Deskew output: {ds_output_dir.name}")
        print(f"  Rotate output: {dsr_output_dir.name}")

        # 3. Run the base processor
        _run_petakit_base(
            input_dir=source_dir,
            output_ds_dir=ds_output_dir,
            output_dsr_dir=dsr_output_dir,
            base_name=base_name,
            **kwargs,
        )

        # --- FIX: Verify that the output directory was actually created ---
        # This check is crucial because PyPetaKit5D may fail without
        # raising a Python exception.
        if not dsr_output_dir.is_dir():
            # Check if the DS (deskew-only) dir was made, for a more specific error
            if ds_output_dir.is_dir():
                print(
                    f"  ⚠️ Warning: Deskew dir '{ds_output_dir.name}' was created, "
                    f"but rotated dir '{dsr_output_dir.name}' was NOT."
                )
                print("   This might indicate an error during the 'rotate' step.")

            raise RuntimeError(
                f"PyPetaKit5D completed but FAILED to create the output directory: "
                f"{dsr_output_dir.name}. "
                "The underlying process likely failed silently."
            )
        # --- END FIX ---

        print("--- PyPetaKit5D Processing Complete ---")

    except FileNotFoundError as fnfe:
        print(f"\n❌ SETUP ERROR: {fnfe}")
        print("Ensure the input path and files exist.")
        raise
    except Exception as e:
        print(f"\n❌ FATAL ERROR in PyPetaKit5D: {e}")
        raise


def run_petakit_processing(
    processed_dir_path: Path,
    *,
    ds_dir_name: str = "DS",
    dsr_dir_name: str = "DSR",
    **kwargs,
) -> None:
    """
    Runs the PyPetaKit5D wrapper on an 'opym' processed OPM TIFF series.

    Args:
        processed_dir_path: The path to the processed TIFF directory
                            (e.g., '.../processed_tiff_series_split').
        ds_dir_name: The name for the deskewed output subdirectory.
        dsr_dir_name: The name for the deskewed+rotated output subdirectory.
        **kwargs: All other processing parameters for PyPetaKit5D.
    """
    ctx = None  # Define ctx outside try for broader scope
    try:
        # 1. Get all paths
        print(f"--- Setting up PetaKit5D for: {processed_dir_path.name} ---")
        ctx = get_petakit_context(
            processed_dir_path,
            ds_dir_name=ds_dir_name,
            dsr_dir_name=dsr_dir_name,
        )

        # 2. Create required directories
        os.makedirs(ctx.job_log_dir, exist_ok=True)

        print(f"\nRunning job locally for TIFF series in: {ctx.processed_dir.name}")
        print(f"  Base name: {ctx.base_name}")
        print(f"  Job log directory: {ctx.job_log_dir.name}")
        print(f"  Deskew output: {ctx.ds_output_dir.name}")
        print(f"  Rotate output: {ctx.dsr_output_dir.name}")

        # 3. Run the base processor
        _run_petakit_base(
            input_dir=ctx.processed_dir,
            output_ds_dir=ctx.ds_output_dir,
            output_dsr_dir=ctx.dsr_output_dir,
            base_name=ctx.base_name,
            **kwargs,
        )

        # --- FIX: Verify that the output directory was actually created ---
        # This check is crucial because PyPetaKit5D may fail without
        # raising a Python exception.
        if not ctx.dsr_output_dir.is_dir():
            # Check if the DS (deskew-only) dir was made, for a more specific error
            if ctx.ds_output_dir.is_dir():
                print(
                    f"  ⚠️ Warning: Deskew dir '{ctx.ds_output_dir.name}' was created, "
                    f"but rotated dir '{ctx.dsr_output_dir.name}' was NOT."
                )
                print("   This might indicate an error during the 'rotate' step.")

            raise RuntimeError(
                f"PyPetaKit5D completed but FAILED to create the output directory: "
                f"{ctx.dsr_output_dir.name}. "
                "The underlying process likely failed silently."
            )
        # --- END FIX ---

        print("--- PyPetaKit5D Processing Complete ---")

    except FileNotFoundError as fnfe:
        print(f"\n❌ SETUP ERROR: {fnfe}")
        print("Ensure the input path and files exist.")
        raise
    except Exception as e:
        print(f"\n❌ FATAL ERROR in PyPetaKit5D: {e}")
        # Add context if we have it
        if ctx:
            print(f"  Context: Checking for output in {ctx.dsr_output_dir}")
        raise


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

    # Get all valid argument names from the base function's signature
    # (The signature of the wrapper is what matters here)
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
