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
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import PyPetaKit5D as ppk
import tifffile


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


def run_petakit_processing(
    processed_dir_path: Path,
    *,
    # --- Custom Output Dirs ---
    ds_dir_name: str = "DS",
    dsr_dir_name: str = "DSR",
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
        ds_dir_name: The name for the deskewed output subdirectory.
        dsr_dir_name: The name for the deskewed+rotated output subdirectory.
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
        raise
    except Exception as e:
        print(f"\n❌ FATAL ERROR in PyPetaKit5D: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
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


def tune_single_stack(
    stack_3d: np.ndarray,
    base_name: str,
    *,
    xy_pixel_size: float,
    z_step_um: float,
    sheet_angle_deg: float,
    interp_method: str = "linear",
    reverse_z: bool = True,
    objective_scan: bool = False,
) -> np.ndarray:
    """
    Runs in-memory deskew/rotate on a single 3D (ZYX) stack
    by saving to a temp directory and calling the wrapper.

    Args:
        stack_3d: The (ZYX) numpy array to process.
        base_name: The base name for the file (e.g., 'cell_MMStack_Pos0').
        xy_pixel_size: Pixel size in XY (microns).
        z_step_um: Z-step size (microns).
        sheet_angle_deg: Sheet angle (degrees).
        interp_method: Interpolation method ('linear', 'cubic', etc.).
        reverse_z: Z-stack direction (True is standard for OPM).
        objective_scan: Whether this is an objective-scan dataset (False).

    Returns:
        The processed 3D (ZYX) numpy array.
    """
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_path = Path(temp_dir_str)
        temp_input_dir = temp_path / "input_tiffs"
        temp_output_ds_dir = "output_ds"
        temp_output_dsr_dir = "output_dsr"
        temp_job_log_dir = temp_path / "job_logs"
        os.makedirs(temp_input_dir)
        os.makedirs(temp_job_log_dir)

        file_name = f"{base_name}_T000_C0.tif"
        input_file_path = temp_input_dir / file_name
        tifffile.imwrite(
            input_file_path,
            stack_3d,
            imagej=True,
            metadata={"axes": "ZYX"},
        )

        try:
            # Call the full wrapper on our temporary directory
            ppk.XR_deskew_rotate_data_wrapper(
                [str(temp_input_dir)],
                deskew=True,
                rotate=True,
                DSRCombined=False,
                xyPixelSize=xy_pixel_size,
                dz=z_step_um,
                skewAngle=sheet_angle_deg,
                objectiveScan=objective_scan,
                reverse=reverse_z,
                channelPatterns=[base_name],
                DSDirName=temp_output_ds_dir,
                DSRDirName=temp_output_dsr_dir,
                jobLogDir=str(temp_job_log_dir),
                largeFile=False,
                zarrFile=False,
                saveZarr=False,
                save16bit=True,
                parseCluster=False,
                masterCompute=True,
                configFile="",
                mccMode=True,
                save3DStack=True,
                saveMIP=False,
                interpMethod=interp_method,
            )

            # PyPetaKit *should* create this file.
            # It does NOT add a _DSR suffix.
            result_file_name = f"{base_name}_T000_C0.tif"
            result_path = temp_path / temp_output_dsr_dir / result_file_name

            if not result_path.exists():
                # --- THIS IS THE BUG ---
                # The wrapper must have failed silently.
                # The file it was *supposed* to create is missing.
                # We will now raise the error message you saw.
                raise FileNotFoundError(
                    f"PyPetaKit did not produce output file: {result_path.name}"
                )

            # Load the processed stack
            result_stack = tifffile.imread(result_path)
            return result_stack

        except Exception as e:
            print(
                f"❌ FATAL ERROR during tune_single_stack: {e}",
                file=sys.stderr,
            )
            traceback.print_exc(file=sys.stderr)
            # --- MODIFICATION ---
            # Re-raise the exception so the notebook
            # catches it and doesn't display the old image.
            raise e
        finally:
            # The temporary directory is automatically cleaned up
            pass
