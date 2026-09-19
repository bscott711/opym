# Ruff style: Compliant
"""
Utilities for interacting with the PetaKit job queue system.
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from pathlib import Path

import ipywidgets as widgets

from .roi_utils import _roi_to_tuple, _tuple_to_cli_string

# Constants
BASE_DIR = Path("/dev/shm/petakit_jobs")
QUEUE_DIR = BASE_DIR / "queue"
DONE_DIR = BASE_DIR / "completed"
FAIL_DIR = BASE_DIR / "failed"


def _ensure_directories(queue_dir: Path = QUEUE_DIR):
    """Ensures the job queue directory exists.

    Every `submit_*_job` function below accepts its own `queue_dir`
    override (tests and non-default callers pass one), but this used to
    unconditionally `mkdir` the module-level default `QUEUE_DIR` regardless
    of what was actually passed in -- silently a no-op for any real
    override, masked only because every caller that overrides `queue_dir`
    happened to `mkdir` it manually first. Takes the actual target dir now.
    """
    Path(queue_dir).mkdir(parents=True, exist_ok=True)


def resolve_deskew_working_dir(master_file: Path) -> Path:
    """Resolves the directory PetaKit5D actually treats as `dataDir` for a
    given raw master file: the current convention names it after the
    master file's stem (e.g. `foo_MMStack_Pos0.ome.tif` ->
    `foo_MMStack_Pos0/`); `processed_tiff_series_split/` is an older,
    legacy convention some already-cropped datasets still use. Prefers the
    newer convention, same order `submit_remote_deskew_job`'s path-
    redirection logic below already applies when building the ticket's
    `dataDir` -- callers that need to find output *after* the job finishes
    (e.g. `backfill/cli.py` locating `DSR_nodecon/`) must resolve it the
    same way or they'll look in the wrong place.
    """
    folder_name = master_file.name
    if folder_name.lower().endswith(".ome.tif"):
        folder_name = folder_name[:-8]
    elif folder_name.lower().endswith(".tif"):
        folder_name = folder_name[:-4]

    potential_dir = master_file.parent / folder_name
    # Existence alone isn't enough -- confirmed live that a master-stem dir
    # can exist with zero cropped frames in it (only stale DSR/DSR_nodecon
    # output left behind by an earlier deskew attempt that itself used this
    # same dir as its working directory). Preferring it anyway sends
    # PetaKit5D a baseName glob that matches nothing, so `inputFullpaths{1}`
    # comes back an empty string and `getImageSize('')` dies with "Index
    # exceeds array bounds" -- a hard-to-diagnose failure for what's really
    # just "wrong directory". Require an actual TIFF frame before trusting it.
    if potential_dir.exists() and (
        next(potential_dir.glob("*.tif"), None) is not None
        or next(potential_dir.glob("*.tiff"), None) is not None
    ):
        return potential_dir
    legacy_dir = master_file.parent / "processed_tiff_series_split"
    if legacy_dir.exists():
        return legacy_dir
    raise FileNotFoundError(
        f"Neither {potential_dir} (with real TIFF frames) nor {legacy_dir} "
        "exists -- crop stage output not found for this master file."
    )


def _apply_omw_params(
    params: dict,
    wiener_alpha: float | None = None,
    otf_cum_thresh: float | None = None,
    hann_win_bounds: list[float] | None = None,
) -> dict:
    """Inject OMW back-projector knobs into a ticket ``params`` dict, but only
    the ones explicitly provided.

    These are consumed by run_petakit_server.m and forwarded to PetaKit5D's
    ``omw_backprojector_generation`` (used only when ``rl_method='omw'``).
    Leaving a value as None omits its key entirely, so non-omw jobs -- and the
    historical stock-default behavior -- are unchanged. Keys use the ticket's
    snake_case convention; the server maps them to wienerAlpha / OTFCumThresh /
    hannWinBounds.
    """
    if wiener_alpha is not None:
        params["wiener_alpha"] = float(wiener_alpha)
    if otf_cum_thresh is not None:
        params["otf_cum_thresh"] = float(otf_cum_thresh)
    if hann_win_bounds is not None:
        params["hann_win_bounds"] = [float(v) for v in hann_win_bounds]
    return params


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
    _ensure_directories(queue_dir)
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
    rl_method: str = "simple",
    wiener_alpha: float | None = None,
    otf_cum_thresh: float | None = None,
    hann_win_bounds: list[float] | None = None,
    save_mip: bool = False,
    zarr_input: bool = False,
) -> Path:
    """
    Creates a JSON job ticket for Deskew/Rotate and optional Deconvolution.

    Parameters
    ----------
    input_axis_order : str, default 'yxz'
        Ignored -- derived from `zarr_input` below, because the on-disk
        axis order is a property of the input format, not a free choice.
        TIFF input is already (y, x, z) in MATLAB so it gets 'yxz'; a zarr
        mirror store is (z, y, x) and gets 'zxy'. See the comment at the
        assignment for why the correct string for this microscope's
        convention is 'zxy' and not the more intuitive-looking 'zyx'.
    output_axis_order : str, default 'yxz'
        Ignored -- always 'yxz'; the DSR result is (y, x, z) and is written
        out as TIFF.
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
    rl_method : str, default 'simple'
        Richardson-Lucy variant for the optional decon step. Only used
        when psf_path is given; also affects the default iteration count
        (2 for 'omw', 25 otherwise) unless n_iters overrides it.
    save_mip : bool, default False
        Have PetaKit5D write a per-timepoint Z-MIP TIFF (to
        ``<dsrDirName>/MIPs/``) alongside the DS/DSR output. Kept opt-in
        (default False, matching every caller's behavior before this
        parameter existed) so existing callers (batch.py, the PSF-tuning
        scripts, notebooks) are unaffected; only the bulk no-decon backfill
        driver passes True.
    zarr_input : bool, default False
        `input_target` is a directory of already-cropped, per-channel
        `.ome.zarr` stores (the newer pymmcore-based MDA acquisition
        format) rather than TIFF -- confirmed against PetaKit5D's own
        source (`XR_deskewRotateFrame.m` dispatches on file extension via
        `readtiff`/`readzarr`; `XR_decon_data_wrapper`/
        `XR_deskew_rotate_data_wrapper` both have a first-class `zarrFile`
        parameter controlling how they discover input files by
        `channel_patterns`). Threaded straight through to the MATLAB
        ticket's `zarr_input` field; `run_petakit_server.m` maps it to
        `zarrFile` for whichever stage actually reads the original raw
        input.
    """
    _ensure_directories(queue_dir)
    input_target = Path(input_target).resolve()

    # PetaKit5D shears its 2nd dimension and holds the 1st invariant, so it
    # must end up with the frame as (y, x, z) where x is the TILTED camera
    # axis -- the one that sweeps in depth as the galvo scans -- and y is the
    # axis lying along the coverslip.
    #
    # A TIFF stack loads in MATLAB as (rows, cols, planes) == (y, x, z)
    # already, so the TIFF path passes 'yxz' and no permute happens.
    #
    # A zarr mirror store is (z, y, x) on disk, where `y` is 490 px (the
    # tilted axis) and `x` is 1458 px (along the coverslip) -- i.e. the
    # acquisition's axis names are the opposite way round from what
    # PetaKit5D means by them. The frame it needs is therefore
    # (1458, 490, 41): the long coverslip axis invariant, the short tilted
    # axis sheared. 'zxy' produces exactly that.
    #
    # Confirmed by running one timepoint through PetaKit5D at all four
    # candidate orders and looking at the projections: 'zxy' is the only one
    # whose YZ view shows a flattened adherent cell sitting on the coverslip
    # line, and whose XY view is a real top-down footprint. It also agrees
    # with `opym.utils.orient_zyx_for_dsr`, which solves the same problem for
    # the streaming path by physically reorienting the array (rot90 +
    # moveaxis -> ny=X, nx=Y, nz=Z) and mirrors what the legacy MATLAB
    # BigTiff cropper did.
    #
    # Beware `axis_order_mapping` (PetaKit5D utils/axis_order_mapping.m): it
    # returns `order(i) = strfind('yxz', inputAxisOrder(i))`, the INVERSE of
    # the vector MATLAB's `permute` consumes, and XR_deskewRotateFrame.m
    # applies it directly as `permute(frame, order)`. 'zyx' and 'xzy' are a
    # 3-cycle pair and get swapped by that; 'zxy' -> [3,2,1] is self-inverse,
    # so it is unambiguous. On a (41, 490, 1458) mirror store:
    #     'yxz' -> [1,2,3] -> (41, 490, 1458)   ny=41    the original bug
    #     'xzy' -> [2,3,1] -> (490, 1458, 41)   ny=490   shears the wrong axis
    #     'zxy' -> [3,2,1] -> (1458, 490, 41)   ny=1458  correct
    # Pinned by tests/test_zarr_deskew_geometry.py.
    input_axis_order = "zxy" if zarr_input else "yxz"
    # The DSR result is (y, x, z) either way and is written out as TIFF.
    output_axis_order = "yxz"

    # --- PATH REDIRECTION LOGIC ---
    if input_target.is_file():
        input_target = resolve_deskew_working_dir(input_target)

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
        "save_mip": save_mip,
        "zarr_input": zarr_input,
    }

    if psf_path:
        params["run_decon"] = True
        params["psf_path"] = str(psf_path)
        params["decon_iter"] = n_iters if n_iters is not None else (2 if rl_method == "omw" else 25)
        params["rl_method"] = rl_method
        params["gpu_decon"] = gpu_decon
        _apply_omw_params(params, wiener_alpha, otf_cum_thresh, hann_win_bounds)

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
    wiener_alpha: float | None = None,
    otf_cum_thresh: float | None = None,
    hann_win_bounds: list[float] | None = None,
    queue_dir: Path = QUEUE_DIR,
) -> Path:
    """
    Creates a JSON job ticket for standalone Deconvolution.
    """
    _ensure_directories(queue_dir)
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
    _apply_omw_params(params, wiener_alpha, otf_cum_thresh, hann_win_bounds)

    if channel_patterns:
        params["channel_patterns"] = channel_patterns

    payload = {
        "jobType": "decon",
        "dataDir": str(input_target),
        "baseName": f"{base_name}*",
        "parameters": params,
    }

    return _write_ticket(payload, base_name, "DECON", queue_dir)


def _read_psf_dz(psf_path: str | Path) -> float | None:
    """Reads the PSF's own z-step (microns) from its ImageJ 'spacing' tag.

    Returns None if the tag is absent so the caller can fail loudly instead
    of silently assuming the PSF's z-step matches the data's z-step.
    """
    try:
        import tifffile
    except ImportError:
        return None

    try:
        with tifffile.TiffFile(str(psf_path)) as tf:
            meta = tf.imagej_metadata
            if meta and "spacing" in meta:
                return float(meta["spacing"])
    except Exception:
        return None
    return None


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
    dz_psf: float | None = None,
    wiener_alpha: float | None = None,
    otf_cum_thresh: float | None = None,
    hann_win_bounds: list[float] | None = None,
    queue_dir: Path = QUEUE_DIR,
) -> Path:
    """
    Creates a JSON job ticket for the unified GPU pipeline.
    This job instructs MATLAB to load the temporary file from /dev/shm/,
    perform Decon -> DSR -> Z-Trim on the GPU, and save the final result to output_file.

    dz_psf : float, optional
        The z-step (microns) the PSF was acquired at. Required for correct
        deconvolution, since the PSF's z-sampling generally differs from the
        raw data's z-step and must be resampled to match before use. If not
        given explicitly, this is read from the first PSF file's ImageJ
        'spacing' metadata tag (written by psf_tools/extract_bead_psf.py).
        Raises ValueError if it cannot be determined either way -- a silent
        wrong default here previously caused a real decon/DSR regression.
    """
    _ensure_directories(queue_dir)
    output_file = Path(output_file).resolve()
    data_dir = output_file.parent
    base_name = output_file.name

    if psf_paths is None:
        resolved_psf_paths: list[str] = []
    elif isinstance(psf_paths, (str, Path)):
        resolved_psf_paths = [str(psf_paths)]
    else:
        resolved_psf_paths = [str(p) for p in psf_paths]

    if resolved_psf_paths:
        if dz_psf is None:
            dz_psf = _read_psf_dz(resolved_psf_paths[0])
        if dz_psf is None:
            raise ValueError(
                f"dz_psf could not be determined for PSF '{resolved_psf_paths[0]}'. "
                "Pass dz_psf explicitly, or re-save the PSF with "
                "psf_tools.extract_bead_psf (which embeds the 'spacing' tag)."
            )

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
        "psf_paths": resolved_psf_paths,
    }
    if resolved_psf_paths:
        params["dz_psf"] = dz_psf
        _apply_omw_params(params, wiener_alpha, otf_cum_thresh, hann_win_bounds)
    if z_crop_end is not None:
        params["z_crop_end"] = int(z_crop_end)

    if channel_patterns:
        params["channel_patterns"] = channel_patterns

    payload = {
        "jobType": "pipeline",
        "dataDir": str(data_dir),
        "baseName": base_name,
        "parameters": params,
    }

    return _write_ticket(payload, base_name, "PIPELINE", queue_dir)


def submit_pipeline_batch_job(
    items: list[dict],
    psf_paths: list[str] | str | Path,
    z_step_um: float,
    xy_pixel_size: float = 0.136,
    sheet_angle_deg: float = 60.0,
    interp_method: str = "cubic",
    iterations: int | None = None,
    rl_method: str = "simple",
    save_zarr: bool = True,
    debug: bool = False,
    dz_psf: float | None = None,
    wiener_alpha: float | None = None,
    otf_cum_thresh: float | None = None,
    hann_win_bounds: list[float] | None = None,
    ticket_label: str = "batch",
    queue_dir: Path = QUEUE_DIR,
) -> Path:
    """
    Creates a JSON job ticket for the pipeline_batch job type (see
    run_petakit_server.m / run_gpu_pipeline_batch_async.m). Bundles N
    (shm_path, output_file) pairs that share one PSF into a single MATLAB
    call, so the GPU concurrency lock and dispatch overhead are paid once
    per batch instead of once per frame -- each item still runs through the
    exact same run_gpu_pipeline() call submit_pipeline_job() would produce
    for it individually; only dispatch granularity changes.

    items : list of {"shm_path": str, "output_file": str} dicts. All items
        must share the same PSF/geometry passed to this call -- that's the
        batchability precondition (keeps the PSF/OTF persistent caches in
        run_gpu_pipeline.m / decon_lucy_function.m warm across the batch).
    dz_psf : see submit_pipeline_job -- same resolution/validation logic.
    """
    _ensure_directories(queue_dir)
    if not items:
        raise ValueError("submit_pipeline_batch_job requires a non-empty items list.")

    if psf_paths is None:
        resolved_psf_paths: list[str] = []
    elif isinstance(psf_paths, (str, Path)):
        resolved_psf_paths = [str(psf_paths)]
    else:
        resolved_psf_paths = [str(p) for p in psf_paths]

    if resolved_psf_paths:
        if dz_psf is None:
            dz_psf = _read_psf_dz(resolved_psf_paths[0])
        if dz_psf is None:
            raise ValueError(
                f"dz_psf could not be determined for PSF '{resolved_psf_paths[0]}'. "
                "Pass dz_psf explicitly, or re-save the PSF with "
                "psf_tools.extract_bead_psf (which embeds the 'spacing' tag)."
            )

    resolved_items = [
        {"shm_path": str(item["shm_path"]), "output_file": str(item["output_file"])} for item in items
    ]

    params = {
        "items": resolved_items,
        "xy_pixel_size": xy_pixel_size,
        "z_step_um": z_step_um,
        "sheet_angle_deg": sheet_angle_deg,
        "interp_method": interp_method,
        "iterations": iterations if iterations is not None else (2 if rl_method == "omw" else 25),
        "rl_method": rl_method,
        "save_zarr": save_zarr,
        "debug": debug,
        "psf_paths": resolved_psf_paths,
    }
    if resolved_psf_paths:
        params["dz_psf"] = dz_psf
        _apply_omw_params(params, wiener_alpha, otf_cum_thresh, hann_win_bounds)

    # dataDir/baseName aren't consumed for pipeline_batch's per-item output
    # paths (those come from items[i]["output_file"]) -- kept only as a
    # human-readable ticket label, matching the other job types' schema.
    payload = {
        "jobType": "pipeline_batch",
        "dataDir": str(Path(resolved_items[0]["output_file"]).parent),
        "baseName": ticket_label,
        "parameters": params,
    }

    return _write_ticket(payload, ticket_label, "PIPELINEBATCH", queue_dir)


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
    """Helper to write the JSON file.

    `base_name` alone is not a reliable per-dataset differentiator -- for
    both `submit_remote_deskew_job` callers it collapses to a FIXED
    subdirectory name shared by every dataset (`processed_tiff_series_split`
    for OME-TIF input, `zarr_mirror` for pre-cropped zarr input), not
    something dataset-specific. A millisecond timestamp alone isn't enough
    uniqueness either: confirmed via a real run (11 zarr datasets, 2
    concurrent crop workers) that two datasets' tickets landed in the same
    millisecond and one filename collision silently overwrote the other's
    ticket JSON -- the clobbered dataset's registry entry then got marked
    done/failed based on its sibling's ticket outcome, never actually having
    been submitted to MATLAB at all. A short uuid4 suffix makes every
    ticket filename unique regardless of submission timing, independent of
    how collision-prone `base_name`/`timestamp` happen to be for a given
    caller.
    """
    timestamp = int(time.time() * 1000)
    unique_suffix = uuid.uuid4().hex[:8]
    # Sanitize name
    safe_name = re.sub(r"[^\w\-_\.]", "_", base_name)
    job_file = queue_dir / f"{prefix}_{safe_name}_{timestamp}_{unique_suffix}.json"

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
