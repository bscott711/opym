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

    # --- START FIX: Validate that ROI slices are not None ---
    y_slice, x_slice = roi
    if y_slice is None or x_slice is None:
        raise ValueError(f"Invalid ROI: Contains 'None' slices: {roi}")
    # --- END FIX ---

    return dummy_plane[y_slice, x_slice].shape


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
    Handles both 4D (squeezed T) and 5D OME-TIF files.

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

    # --- START FIX: More robust ROI validation ---
    def is_roi_valid(roi: tuple[slice, slice] | None) -> bool:
        """Checks if an ROI is not None and both its slices are not None."""
        if roi is None:
            return False
        # Check that the tuple contains two slice objects, not None
        return roi[0] is not None and roi[1] is not None

    if need_top and not is_roi_valid(top_roi):
        raise ValueError(
            f"Channels 1 or 2 selected, but top_roi is invalid "
            f"or incomplete: {top_roi}"
        )
    if need_bottom and not is_roi_valid(bottom_roi):
        raise ValueError(
            f"Channels 0 or 3 selected, but bottom_roi is invalid "
            f"or incomplete: {bottom_roi}"
        )
    # --- END FIX ---

    try:
        # Setup based on input TIFF
        with tifffile.TiffFile(base_file) as tif:
            series = tif.series[0]
            store = series.aszarr()
            # Open the base array
            base_zarr_array: zarr.Array = zarr.open_array(store, mode="r")

            # --- START FIX: Handle 4D (squeezed T) vs 5D arrays ---
            shape = base_zarr_array.shape
            zarr_array: zarr.Array  # Define type for Pylance

            if base_zarr_array.ndim == 5:
                T, Z, C, Y, X = shape
                zarr_array = base_zarr_array  # It's already 5D
                print(f"Detected 5D array. Shape: {(T, Z, C, Y, X)}")
            elif base_zarr_array.ndim == 4:
                Z, C, Y, X = shape
                T = 1  # This is a 4D file, so it has 1 timepoint

                # --- PYLANCE FIX: Ignore type checker for this line ---
                # Pylance stubs for zarr are incorrect. This is valid.
                zarr_array = base_zarr_array[None, ...]  # type: ignore

                print(f"Detected 4D array. Reshaping to: {zarr_array.shape}")
            else:
                raise ValueError(
                    f"Incompatible data shape: {shape}. Expected 4D or 5D array."
                )
            # --- END FIX ---

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

        # --- START FIX: Corrected shape logic ---
        # We check if the *shape* (which is tuple[int, int] | None) is not None.
        if top_shape is not None:
            Y_new, X_new = top_shape
        elif bottom_shape is not None:
            Y_new, X_new = bottom_shape
        else:
            # Fallback if no valid ROIs were provided
            Y_new, X_new = (0, 0)
            if need_top or need_bottom:
                # This should be caught by validation, but as a fallback:
                print(
                    "Warning: No valid ROI shapes found despite channels being selected"
                )
        # --- END FIX ---

        C_new = len(channels_to_output)

        if rotate_90:
            Y_out, X_out = X_new, Y_new
        else:
            Y_out, X_out = Y_new, Y_new

        print(f"Using sanitized base name for output: {sanitized_name}")

        if output_format == OutputFormat.ZARR:
            # ... (This ZARR block is correct, no changes needed) ...
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

                        # --- START FIX: Add is_roi_valid check to ZARR block ---
                        if need_top and is_roi_valid(top_roi):
                            t_roi = cast(tuple[slice, slice], top_roi)
                            top_crop_c0 = plane_cam0[t_roi[0], t_roi[1]]
                            top_crop_c1 = plane_cam1[t_roi[0], t_roi[1]]
                        else:
                            top_crop_c0 = None
                            top_crop_c1 = None

                        if need_bottom and is_roi_valid(bottom_roi):
                            b_roi = cast(tuple[slice, slice], bottom_roi)
                            bot_crop_c0 = plane_cam0[b_roi[0], b_roi[1]]
                            bot_crop_c1 = plane_cam1[b_roi[0], b_roi[1]]
                        else:
                            bot_crop_c0 = None
                            bot_crop_c1 = None
                        # --- END FIX ---

                        if rotate_90:
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

                        pbar.update(C)

            print(f"✅ Saved processed series to {output_zarr_path.name}")

        elif output_format == OutputFormat.TIFF_SERIES:
            # --- START FIX: Logic for TIFF_SERIES ---
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

                    # --- FINAL REDUNDANT FIX: Check ROIs *inside* the loop ---
                    if need_top and is_roi_valid(top_roi):
                        t_roi = cast(tuple[slice, slice], top_roi)
                        top_crop_c0 = plane_cam0[t_roi[0], t_roi[1]]
                        top_crop_c1 = plane_cam1[t_roi[0], t_roi[1]]
                    else:
                        top_crop_c0 = None
                        top_crop_c1 = None

                    if need_bottom and is_roi_valid(bottom_roi):
                        b_roi = cast(tuple[slice, slice], bottom_roi)
                        bot_crop_c0 = plane_cam0[b_roi[0], b_roi[1]]
                        bot_crop_c1 = plane_cam1[b_roi[0], b_roi[1]]
                    else:
                        bot_crop_c0 = None
                        bot_crop_c1 = None
                    # --- END REDUNDANT FIX ---

                    if rotate_90:
                        if top_crop_c0 is not None:
                            top_crop_c0 = np.rot90(cast(np.ndarray, top_crop_c0), k=1)
                        if top_crop_c1 is not None:
                            top_crop_c1 = np.rot90(cast(np.ndarray, top_crop_c1), k=1)
                        if bot_crop_c0 is not None:
                            bot_crop_c0 = np.rot90(cast(np.ndarray, bot_crop_c0), k=1)
                        if bot_crop_c1 is not None:
                            bot_crop_c1 = np.rot90(cast(np.ndarray, bot_crop_c1), k=1)

                    # Assign cropped planes to the correct channel stack
                    if 0 in channels_to_output:
                        stacks_to_write[0][z, :, :] = cast(np.ndarray, bot_crop_c0)
                    if 1 in channels_to_output:
                        stacks_to_write[1][z, :, :] = cast(np.ndarray, top_crop_c0)
                    if 2 in channels_to_output:
                        stacks_to_write[2][z, :, :] = cast(np.ndarray, top_crop_c1)
                    if 3 in channels_to_output:
                        stacks_to_write[3][z, :, :] = cast(np.ndarray, bot_crop_c1)

                # Write the completed 3D stacks to TIFF files
                for c_out, stack_data in stacks_to_write.items():
                    out_name = f"{sanitized_name}_C{c_out}_T{t:03d}.tif"
                    tifffile.imwrite(
                        output_dir / out_name,
                        stack_data,
                        imagej=True,
                        metadata=tif_meta,
                    )
            # --- END FIX ---
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
