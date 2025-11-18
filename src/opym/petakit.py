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
import subprocess  # nosec B404
import sys
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


def generate_hpc_config(
    mcr_root: str = "/cm/shared/apps_local/matlab/R2024B",
    mcc_master_script: str = (
        "/mmfs2/cm/shared/apps_local/petakit5d/mcc/linux/run_mccMaster.sh"
    ),
    output_file: Path | None = None,
) -> Path:
    """
    Generates a 'pypetakit_config.json' file for HPC use.

    Args:
        mcr_root: The path to the MATLAB MCR root.
        mcc_master_script: The path to the 'run_mccMaster.sh' script.
        output_file: The path to save the file. If None, defaults to
                     ~/pypetakit_config.json.

    Returns:
        The Path to the file that was written.
    """
    if output_file is None:
        output_file = Path.home() / "pypetakit_config.json"

    print(f"--- Generating PyPetaKit5D config at: {output_file} ---")

    config_data = {
        "MCCMasterStr": mcc_master_script,
        "MCRParam": mcr_root,
        "memPerCPU": 5.0,
        "jobTimeLimit": 48,
        "maxCPUNum": 24,  # Set to your allocated core count
        "GNUparallel": True,  # RESTORED: Needed to trigger MCC bypass mode
        "masterCompute": True,  # Run locally on the allocated node
        "parseCluster": False,  # Do NOT submit sub-jobs
        "SlurmParam": "",
    }

    try:
        with output_file.open("w") as f:
            json.dump(config_data, f, indent=4)
        print(f"✅ Config saved to {output_file}")
        return output_file
    except Exception as e:
        print(f"❌ ERROR: Could not write config file: {e}")
        raise


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
        first_file = next(processed_dir.glob("*_C[0-9]_T[0-9][0-9][0-9].tif"), None)
        if not first_file:
            raise FileNotFoundError(
                "Could not find a '*_processing_log.json' file or any "
                "'*_C..._T..tif' file to determine base_name."
            )
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


# --- NEW: Helper function to convert Python types to MATLAB CLI strings ---
def _to_matlab_str(val: object) -> str:
    """Converts a Python type to its MATLAB string representation for mcc."""
    if isinstance(val, bool):
        return "true" if val else "false"  # <-- FIX: Must be true/false
    if isinstance(val, int | float):
        return str(val)
    if isinstance(val, str | Path):
        return str(val)  # Do NOT add extra quotes
    if isinstance(val, list):
        if not val:
            return "''"  # Use empty string for empty lists
        # Check if it's a list of numbers or strings
        if all(isinstance(v, int | float) for v in val):
            # Numeric array: [1,2,3]
            return "[" + ",".join(map(str, val)) + "]"
        else:
            # Cell array of strings: {'str1','str2'}
            return "{" + ",".join([f"'{v}'" for v in val]) + "}"
    if val is None:
        return "''"  # Use empty string for None
    raise TypeError(f"Unsupported MATLAB arg type: {type(val)}")


def _get_cpu_count() -> int:
    """
    Detects the number of available CPUs.

    Prioritizes Slurm environment variables, then process affinity, then physical count.
    """
    # 1. Check Slurm allocation (specifically set by --cpus-per-task)
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        try:
            count = int(slurm_cpus)
            print(
                f"--- [opym.petakit] Auto-detected {count} CPUs "
                "from SLURM_CPUS_PER_TASK."
            )
            return count
        except ValueError:
            pass

    # 2. Check process affinity (accurate for Slurm allocations without env var)
    # Available on Linux systems (most HPC clusters)
    if hasattr(os, "sched_getaffinity"):
        try:
            # type: ignore[attr-defined]
            count = len(os.sched_getaffinity(0))  # type: ignore
            print(
                f"--- [opym.petakit] Auto-detected {count} CPUs "
                "from process affinity."
            )
            return count
        except (OSError, AttributeError):
            pass

    # 3. Fallback to total physical cores
    count = os.cpu_count() or 1
    print(f"--- [opym.petakit] Auto-detected {count} CPUs from os.cpu_count().")
    return count


# In petakit.py
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
    master_compute: bool = False,
    config_file: str = "",
    mcc_mode: bool = False,
    parse_parfor: bool = True,
    cpus_per_task: int | None = None,
    batch_size: list[int] | None = None,
    # Redundant/Unused PyPetaKit5D args
    ff_correction: bool = False,
    lower_limit: float = 0.4,
    const_offset: float = 1.0,
    ff_image_paths: list[str] | None = None,
    background_paths: list[str] | None = None,
    bk_removal: bool = False,
) -> None:
    """
    Internal base function that runs the PyPetaKit5D wrapper.
    """
    if block_size is None:
        block_size = [256, 256, 256]
    if batch_size is None:
        batch_size = [512, 512, 512]
    if ff_image_paths is None:
        ff_image_paths = [""]
    if background_paths is None:
        background_paths = [""]

    # --- Auto-detect CPUs if not provided ---
    if cpus_per_task is None:
        cpus_per_task = _get_cpu_count()
    # ---------------------------------------------

    # --- NEW: MCC/HPC BYPASS LOGIC ---
    if not mcc_mode or not config_file:
        print("--- [opym.petakit] Running in standard (non-MCC) Python mode. ---")
        # Fallback to the old, buggy way (for local, non-mcc runs)
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
            DSDirName=output_ds_dir.name,
            DSRDirName=output_dsr_dir.name,
            FFCorrection=ff_correction,
            lowerLimit=lower_limit,
            constOffset=const_offset,
            FFImagePaths=ff_image_paths,
            backgroundPaths=background_paths,
            largeFile=large_file,
            zarrFile=zarr_file,
            saveZarr=save_zarr,
            blockSize=block_size,
            batchSize=batch_size,  # Added batchSize
            save16bit=save_16bit,
            parseCluster=parse_cluster,
            masterCompute=master_compute,
            configFile=config_file,
            mccMode=mcc_mode,
            parseParfor=parse_parfor,
            cpusPerTask=cpus_per_task,
            BKRemoval=bk_removal,
            save3DStack=save_3d_stack,
            saveMIP=save_mip,
            interpMethod=interp_method,
        )
        return

    # --- [opym.petakit] Running in HPC/MCC Bypass Mode. ---
    print("--- [opym.petakit] Running in HPC/MCC Bypass Mode. ---")

    # 1. Load config file
    config_path = Path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r") as f:
        config_data = json.load(f)

    mcc_script = config_data.get("MCCMasterStr")
    mcr_path = config_data.get("MCRParam")

    if not mcc_script or not mcr_path:
        raise ValueError("MCCMasterStr or MCRParam missing from config file.")

    # 2. Build the command
    cmd = [
        mcc_script,  # Path to run_mccMaster.sh
        mcr_path,  # Path to MATLAB Runtime
        "XR_deskew_rotate_data_wrapper",  # Function to call
    ]

    # 3. Build arguments
    # Add the first positional argument (input dir)
    cmd.append(_to_matlab_str([str(input_dir)]))

    if mcc_mode:
        parse_parfor = True

    # --- NEW: Overriding GNUparallel to ensure parseParfor works ---
    # If the Python function explicitly asked for internal MATLAB parallelization,
    # we must ensure the conflicting GNUparallel flag is set to false in the
    # command-line parameters to override the config file's value.
    if parse_parfor:
        matlab_gnu_parallel = False
    else:
        # Fall back to the value defined in the config file
        matlab_gnu_parallel = config_data.get("GNUparallel", False)
    # --- END NEW LOGIC ---

    # Collect all other parameters
    matlab_params = {
        "deskew": deskew,
        "rotate": rotate,
        "DSRCombined": dsr_combined,
        "xyPixelSize": xy_pixel_size,
        "dz": z_step_um,
        "skewAngle": sheet_angle_deg,
        "objectiveScan": objective_scan,
        "reverse": reverse_z,
        "channelPatterns": [base_name],
        "DSDirName": str(output_ds_dir.name),
        "DSRDirName": str(output_dsr_dir.name),
        "FFCorrection": ff_correction,
        "lowerLimit": lower_limit,
        "constOffset": const_offset,
        "FFImagePaths": ff_image_paths,
        "backgroundPaths": background_paths,
        "largeFile": large_file,
        "zarrFile": zarr_file,
        "saveZarr": save_zarr,
        "blockSize": block_size,
        "batchSize": batch_size,  # Pass batchSize
        "save16bit": save_16bit,
        # Explicitly pass cluster params to ensure override
        "parseCluster": parse_cluster,
        "masterCompute": master_compute,
        "configFile": config_file,
        "mccMode": mcc_mode,
        "parseParfor": parse_parfor,
        "cpusPerTask": cpus_per_task,
        "GNUparallel": matlab_gnu_parallel,  # <--- USED NEW LOGIC HERE
        "BKRemoval": bk_removal,
        "save3DStack": save_3d_stack,
        "saveMIP": save_mip,
        "interpMethod": interp_method,
    }

    # Add all key-value pairs to the command
    for key, val in matlab_params.items():
        cmd.append(key)
        cmd.append(_to_matlab_str(val))

    # 4. Set the environment for the subprocess
    env = os.environ.copy()
    matlab_root = mcr_path
    mcr_paths = [
        f"{matlab_root}/runtime/glnxa64",
        f"{matlab_root}/bin/glnxa64",
        f"{matlab_root}/sys/os/glnxa64",
        f"{matlab_root}/sys/opengl/lib/glnxa64",
    ]
    current_ld_path = env.get("LD_LIBRARY_PATH", "")
    all_paths = mcr_paths + [current_ld_path]
    env["LD_LIBRARY_PATH"] = ":".join(filter(None, all_paths))
    env["MW_MCR_ROOT"] = matlab_root

    print(f"--- [opym.petakit] Calling: {mcc_script} ... ---")
    print(f"--- [opym.petakit] With MCR: {mcr_path} ---")

    # 5. Run the command
    try:
        # Run without capturing output to stream in real-time
        subprocess.run(  # nosec B603
            cmd,
            check=True,
            env=env,
            text=True,
        )
        print("--- [opym.petakit] Subprocess finished. ---")
    except subprocess.CalledProcessError as e:
        print(f"❌ FATAL ERROR in mccMaster subprocess: {e}", file=sys.stderr)
        raise
    except FileNotFoundError as e:
        print(f"❌ FATAL ERROR: Could not find {mcc_script}: {e}", file=sys.stderr)
        print("   Please check the 'MCCMasterStr' path in your config JSON.")
        raise


def run_petakit_processing(
    processed_dir_path: Path,
    *,
    ds_dir_name: str = "DS",
    dsr_dir_name: str = "DSR",
    cpus_per_task: int | None = None,
    **kwargs,
) -> None:
    """
    Runs the PyPetaKit5D wrapper on an 'opym' processed OPM TIFF series.

    Args:
        processed_dir_path: The path to the processed TIFF directory.
        ds_dir_name: The name for the deskewed output subdirectory.
        dsr_dir_name: The name for the deskewed+rotated output subdirectory.
        cpus_per_task: Number of CPUs to use. If None, auto-detects from Slurm/System.
        **kwargs: All other processing parameters for PyPetaKit5D.
    """
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

        # --- Auto-detect config file ---
        kwargs = _autodetect_hpc_config(kwargs)

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
            cpus_per_task=cpus_per_task,  # Pass None or value down
            **kwargs,
        )

        print("--- PyPetaKit5D Processing Complete ---")

    except FileNotFoundError as fnfe:
        print(f"\n❌ SETUP ERROR: {fnfe}")
        print("Ensure the input path and files exist.")
        raise
    except Exception as e:
        print(f"\n❌ FATAL ERROR in PyPetaKit5D: {e}")
        raise


def _autodetect_hpc_config(kwargs: dict) -> dict:
    """
    Checks for a default HPC config file if one is not provided.
    Also enables mcc_mode if a config file is in use.
    """
    # Check if a config file is already specified by the user
    if "config_file" not in kwargs:
        # If not, check if the default HPC config exists
        default_config = Path.home() / "pypetakit_config.json"
        if default_config.exists():
            print(f"--- Found default config at {default_config}. Using it. ---")
            kwargs["config_file"] = str(default_config)

    # Automatically enable mcc_mode if a config file is being used
    # (and not explicitly disabled by the user)
    if "config_file" in kwargs and "mcc_mode" not in kwargs:
        print("--- Config file detected, setting mcc_mode=True ---")
        kwargs["mcc_mode"] = True

    return kwargs


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
        ds_output_dir = source_dir / ds_dir_name
        dsr_output_dir = source_dir / dsr_dir_name
        job_log_dir = source_dir.parent / "job_logs"
        os.makedirs(job_log_dir, exist_ok=True)

        # --- NEW: Auto-detect config file ---
        kwargs = _autodetect_hpc_config(kwargs)
        # --- END NEW ---

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

        print("--- PyPetaKit5D Processing Complete ---")

    except FileNotFoundError as fnfe:
        print(f"\n❌ SETUP ERROR: {fnfe}")
        print("Ensure the input path and files exist.")
        raise
    except Exception as e:
        print(f"\n❌ FATAL ERROR in PyPetaKit5D: {e}")
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
