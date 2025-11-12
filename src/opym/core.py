# Ruff style: Compliant
"""
Core OPM Cropper processing functions.
Streams from a 5D OME-TIF virtual zarr stack into the specified output format.
"""

import os
import shutil
from pathlib import Path
from typing import cast

import numpy as np
import tifffile
import zarr
from tqdm.auto import tqdm

from .metadata import create_processing_log
from .roi_utils import save_rois_to_log
from .utils import OutputFormat, derive_paths


def _get_crop_shape(
    dummy_plane: np.ndarray,
    roi: tuple[slice, slice] | None,
) -> tuple[int, int] | None:
    """Helper to get shape from a potentially None ROI."""
    if roi is None:
        return None
    return dummy_plane[roi[0], roi[1]].shape


def process_dataset(
    base_file: Path,
    output_dir: Path,
    sanitized_name: str,
    top_roi: tuple[slice, slice] | None,
    bottom_roi: tuple[slice, slice] | None,
    output_format: OutputFormat,
    rotate_90: bool = False,
    channels_to_output: list[int] | None = None,
) -> int:
    """
    Main processing function. Streams data to either a single OME-TIF file
    or a series of 3D (ZYX) TIFF files.

    Processes only the channels specified in `channels_to_output`.

    Returns:
        int: The number of timepoints processed (T).
    """
    print(f"--- Processing: {base_file.name} (Format: {output_format.value}) ---")
    if rotate_90:
        print("    90-degree rotation: ENABLED")

    # --- MODIFICATION: Default to all 4 channels if not specified ---
    if channels_to_output is None:
        channels_to_output = [0, 1, 2, 3]
    if not channels_to_output:
        raise ValueError("channels_to_output list cannot be empty.")

    print(f"  Outputting channels: {channels_to_output}")

    # Determine which ROIs are needed
    need_top = (1 in channels_to_output) or (2 in channels_to_output)
    need_bottom = (0 in channels_to_output) or (3 in channels_to_output)

    if need_top and top_roi is None:
        raise ValueError("Channels 1 or 2 selected, but top_roi is None.")
    if need_bottom and bottom_roi is None:
        raise ValueError("Channels 0 or 3 selected, but bottom_roi is None.")

    try:
        # Setup based on input TIFF
        with tifffile.TiffFile(base_file) as tif:
            series = tif.series[0]
            store = series.aszarr()
            zarr_array: zarr.Array = zarr.open_array(store, mode="r")

            T, Z, C, Y, X = series.shape
            dtype = series.dtype

            if C != 2:
                print(
                    f"Warning: Expected 2 cameras (C=2), but found C={C}",
                )

        dummy_plane = np.zeros((Y, X), dtype=dtype)
        top_shape = _get_crop_shape(dummy_plane, top_roi)
        bottom_shape = _get_crop_shape(dummy_plane, bottom_roi)

        if top_shape and bottom_shape and (top_shape != bottom_shape):
            raise ValueError(f"ROI shapes do not match: {top_shape} vs {bottom_shape}")

        # Determine output shape from the first available ROI
        Y_new, X_new = top_shape or bottom_shape or (0, 0)

        C_new = len(channels_to_output)

        if rotate_90:
            Y_out, X_out = X_new, Y_new
        else:
            Y_out, X_out = Y_new, X_new

        print(f"Using sanitized base name for output: {sanitized_name}")

        if output_format == OutputFormat.ZARR:
            output_zarr_path = output_dir / (sanitized_name + "_processed.zarr")
            if output_zarr_path.exists():
                print(
                    f"Warning: Output Zarr {output_zarr_path.name} "
                    "already exists. Deleting it..."
                )
                shutil.rmtree(output_zarr_path)

            output_shape = (T, Z, C_new, Y_out, X_out)
            chunks = (1, 1, 1, Y_out, X_out)
            zarr_out: zarr.Array = zarr.create(
                output_shape,
                store=str(output_zarr_path),
                dtype=dtype,
                chunks=chunks,
            )
            # Map output channel index (0, 1, ...) to its file name (0, 2, ...)
            channel_map = {c_out: i for i, c_out in enumerate(channels_to_output)}
            print(
                f"Created new {C_new}-channel OME-Zarr store: {output_zarr_path.name}"
            )

            with tqdm(
                total=T * Z * C, desc=" ├ Processing Planes", unit="plane"
            ) as pbar:
                for t in range(T):
                    for z in range(Z):
                        plane_cam0 = cast(np.ndarray, zarr_array[t, z, 0])
                        plane_cam1 = cast(np.ndarray, zarr_array[t, z, 1])

                        # Pre-crop ROIs only if needed.
                        # Pylance needs casting because it doesn't assume 'need_top'
                        # implies 'top_roi' is not None inside the loop.
                        if need_top:
                            t_roi = cast(tuple[slice, slice], top_roi)
                            top_crop_c0 = plane_cam0[t_roi[0], t_roi[1]]
                            top_crop_c1 = plane_cam1[t_roi[0], t_roi[1]]
                        else:
                            top_crop_c0 = None
                            top_crop_c1 = None

                        if need_bottom:
                            b_roi = cast(tuple[slice, slice], bottom_roi)
                            bot_crop_c0 = plane_cam0[b_roi[0], b_roi[1]]
                            bot_crop_c1 = plane_cam1[b_roi[0], b_roi[1]]
                        else:
                            bot_crop_c0 = None
                            bot_crop_c1 = None

                        if rotate_90:
                            # Pylance needs casts to know we aren't rotating None
                            if top_crop_c0 is not None:
                                top_crop_c0 = np.rot90(
                                    cast(np.ndarray, top_crop_c0), k=1
                                )
                            if top_crop_c1 is not None:
                                top_crop_c1 = np.rot90(
                                    cast(np.ndarray, top_crop_c1), k=1
                                )
                            if bot_crop_c0 is not None:
                                bot_crop_c0 = np.rot90(
                                    cast(np.ndarray, bot_crop_c0), k=1
                                )
                            if bot_crop_c1 is not None:
                                bot_crop_c1 = np.rot90(
                                    cast(np.ndarray, bot_crop_c1), k=1
                                )

                        # Write only selected channels
                        # Explicit casts required for __setitem__
                        if 0 in channels_to_output:
                            zarr_out[t, z, channel_map[0], :, :] = cast(
                                np.ndarray, bot_crop_c0
                            )
                        if 1 in channels_to_output:
                            zarr_out[t, z, channel_map[1], :, :] = cast(
                                np.ndarray, top_crop_c0
                            )
                        if 2 in channels_to_output:
                            zarr_out[t, z, channel_map[2], :, :] = cast(
                                np.ndarray, top_crop_c1
                            )
                        if 3 in channels_to_output:
                            zarr_out[t, z, channel_map[3], :, :] = cast(
                                np.ndarray, bot_crop_c1
                            )

                        pbar.update(C)  # Update pbar by number of input cams

            print(f"✅ Saved processed series to {output_zarr_path.name}")

        elif output_format == OutputFormat.TIFF_SERIES:
            output_stack_shape_3d = (Z, Y_out, X_out)
            tif_meta = {"axes": "ZYX"}

            for t in tqdm(range(T), desc=" ├ Streaming & Writing", unit="TP"):
                # Create empty stacks only for the channels we need
                stacks_to_write = {
                    c_out: np.zeros(output_stack_shape_3d, dtype=dtype)
                    for c_out in channels_to_output
                }

                for z in range(Z):
                    plane_cam0 = cast(np.ndarray, zarr_array[t, z, 0])
                    plane_cam1 = cast(np.ndarray, zarr_array[t, z, 1])

                    # Process Cam 0
                    if 0 in channels_to_output:
                        b_roi = cast(tuple[slice, slice], bottom_roi)
                        crop = plane_cam0[b_roi[0], b_roi[1]]
                        stacks_to_write[0][z, :, :] = (
                            np.rot90(crop, k=1) if rotate_90 else crop
                        )
                    if 1 in channels_to_output:
                        t_roi = cast(tuple[slice, slice], top_roi)
                        crop = plane_cam0[t_roi[0], t_roi[1]]
                        stacks_to_write[1][z, :, :] = (
                            np.rot90(crop, k=1) if rotate_90 else crop
                        )

                    # Process Cam 1
                    if 2 in channels_to_output:
                        t_roi = cast(tuple[slice, slice], top_roi)
                        crop = plane_cam1[t_roi[0], t_roi[1]]
                        stacks_to_write[2][z, :, :] = (
                            np.rot90(crop, k=1) if rotate_90 else crop
                        )
                    if 3 in channels_to_output:
                        b_roi = cast(tuple[slice, slice], bottom_roi)
                        crop = plane_cam1[b_roi[0], b_roi[1]]
                        stacks_to_write[3][z, :, :] = (
                            np.rot90(crop, k=1) if rotate_90 else crop
                        )

                # Write the stacks to TIFF files
                for c_out, stack_data in stacks_to_write.items():
                    out_name = f"{sanitized_name}_C{c_out}_T{t:03d}.tif"
                    tifffile.imwrite(
                        output_dir / out_name,
                        stack_data,
                        imagej=True,
                        metadata=tif_meta,
                    )

            print(f"✅ Saved {T * C_new} TIFF files to {output_dir.name}")

        else:
            raise ValueError(f"Unknown output_format: {output_format}")

        return T

    except Exception as e:
        print(f"\n❌ Error processing {base_file.name}: {e}\n")
        raise


def run_processing_job(
    base_file: Path,
    top_roi: tuple[slice, slice] | None,
    bottom_roi: tuple[slice, slice] | None,
    output_format: OutputFormat,
    channels_to_output: list[int],
    cli_log_file: Path = Path("opm_roi_log.json"),
    rotate_90: bool = False,
):
    """
    Runs a full processing job for a single file.

    This is the main high-level API function. It handles:
    1. Deriving paths.
    2. Validating inputs.
    3. Cleaning the output directory.
    4. Running the core `process_dataset` function.
    5. Creating the metadata log.
    6. Saving ROIs to the central CLI log.
    """
    print("--- Starting Processing Job ---")

    if top_roi is None and bottom_roi is None:
        raise ValueError("At least one ROI (top_roi or bottom_roi) must be provided.")
    if not channels_to_output:
        raise ValueError("channels_to_output list cannot be empty.")

    paths = derive_paths(base_file, output_format)

    if not paths.base_file.exists():
        raise FileNotFoundError(f"Input file not found: {paths.base_file}")
    if not paths.metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {paths.metadata_file}")

    paths.output_dir.mkdir(parents=True, exist_ok=True)

    if output_format == OutputFormat.ZARR:
        zarr_path = paths.output_dir / (paths.sanitized_name + "_processed.zarr")
        if zarr_path.exists():
            print(
                f"Warning: Output Zarr {zarr_path.name} already exists. Deleting it..."
            )
            shutil.rmtree(zarr_path)
    elif output_format == OutputFormat.TIFF_SERIES:
        print(f"Cleaning old files from {paths.output_dir.name}...")
        for f in paths.output_dir.glob(f"{paths.sanitized_name}_C*_T*.tif"):
            os.remove(f)

    print(f"Format selected: {output_format.value}")
    print(f"Output Directory: {paths.output_dir.name}")

    print("\nStarting stream processing...")
    num_timepoints = process_dataset(
        paths.base_file,
        paths.output_dir,
        paths.sanitized_name,
        top_roi,
        bottom_roi,
        output_format,
        rotate_90=rotate_90,
        channels_to_output=channels_to_output,
    )

    print("\nCreating processing log...")
    create_processing_log(
        paths,
        num_timepoints,
        top_roi,
        bottom_roi,
        output_format,
        rotate_90=rotate_90,
        channels_to_output=channels_to_output,
    )

    save_rois_to_log(
        cli_log_file,
        paths.base_file,
        top_roi,
        bottom_roi,
    )
