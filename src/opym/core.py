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


def process_dataset(
    base_file: Path,
    output_dir: Path,
    sanitized_name: str,
    top_roi: tuple[slice, slice],
    bottom_roi: tuple[slice, slice],
    output_format: OutputFormat,
    rotate_90: bool = False,  # <-- ADDED
) -> int:
    """
    Main processing function. Streams data to either a single OME-TIF file
    or a series of 3D (ZYX) TIFF files.

    Returns:
        int: The number of timepoints processed (T).
    """
    print(f"--- Processing: {base_file.name} (Format: {output_format.value}) ---")
    if rotate_90:
        print("    90-degree rotation: ENABLED")

    try:
        # Setup based on input TIFF
        with tifffile.TiffFile(base_file) as tif:
            series = tif.series[0]
            store = series.aszarr()
            zarr_array: zarr.Array = zarr.open_array(store, mode="r")

            T, Z, C, Y, X = series.shape
            dtype = series.dtype

            dummy_plane = np.zeros((Y, X), dtype=dtype)
            top_crop_shape = dummy_plane[top_roi[0], top_roi[1]].shape
            bottom_crop_shape = dummy_plane[bottom_roi[0], bottom_roi[1]].shape

            if top_crop_shape != bottom_crop_shape:
                raise ValueError(
                    f"ROI shapes do not match: {top_crop_shape} vs {bottom_crop_shape}"
                )

            Y_new, X_new = top_crop_shape
            C_new = C * 2  # Input C=2 -> Output C=4

            if C != 2:
                print(
                    f"Warning: Expected 2 cameras (C=2), but found C={C}",
                )

        # --- MODIFIED: Adjust output shape if rotating ---
        if rotate_90:
            Y_out, X_out = X_new, Y_new  # Shape becomes (X, Y)
        else:
            Y_out, X_out = Y_new, X_new  # Shape remains (Y, X)
        # --- END MODIFICATION ---

        print(f"Using sanitized base name for output: {sanitized_name}")

        # Stream data and write output
        if output_format == OutputFormat.ZARR:
            output_zarr_path = output_dir / (sanitized_name + "_processed.zarr")

            if output_zarr_path.exists():
                print(
                    f"Warning: Output Zarr {output_zarr_path.name} "
                    "already exists. Deleting it..."
                )
                shutil.rmtree(output_zarr_path)

            # --- MODIFIED: Use Y_out, X_out for shape ---
            output_shape = (T, Z, C_new, Y_out, X_out)
            chunks = (1, 1, 1, Y_out, X_out)
            # --- END MODIFICATION ---
            zarr_out: zarr.Array = zarr.create(
                output_shape,
                store=str(output_zarr_path),
                dtype=dtype,
                chunks=chunks,
            )
            print(f"Created new OME-Zarr store at: {output_zarr_path}")

            with tqdm(
                total=T * Z * C, desc=" ├ Processing Planes", unit="plane"
            ) as pbar:
                for t in range(T):
                    for z in range(Z):
                        for c in range(C):
                            plane = cast(np.ndarray, zarr_array[t, z, c])
                            top_crop = plane[top_roi[0], top_roi[1]]
                            bottom_crop = plane[bottom_roi[0], bottom_roi[1]]

                            # --- MODIFIED: Apply rotation ---
                            if rotate_90:
                                top_crop = np.rot90(top_crop, k=1)
                                bottom_crop = np.rot90(bottom_crop, k=1)
                            # --- END MODIFICATION ---

                            # Explicitly map C=0 and C=1.
                            # Any other channels (C=2, etc.) are ignored.
                            if c == 0:
                                zarr_out[t, z, 0, :, :] = bottom_crop  # C0
                                zarr_out[t, z, 1, :, :] = top_crop  # C1
                            elif c == 1:
                                zarr_out[t, z, 2, :, :] = top_crop  # C2
                                zarr_out[t, z, 3, :, :] = bottom_crop  # C3
                            pbar.update(1)

            print(f"✅ Saved processed series to {output_zarr_path.name}")

        elif output_format == OutputFormat.TIFF_SERIES:
            # --- MODIFIED: Use Y_out, X_out for shape ---
            output_stack_shape_3d = (Z, Y_out, X_out)
            # --- END MODIFICATION ---

            for t in tqdm(range(T), desc=" ├ Streaming & Writing", unit="TP"):
                # Create 4 empty 3D arrays for this timepoint
                ch0_stack = np.zeros(output_stack_shape_3d, dtype=dtype)
                ch1_stack = np.zeros(output_stack_shape_3d, dtype=dtype)
                ch2_stack = np.zeros(output_stack_shape_3d, dtype=dtype)
                ch3_stack = np.zeros(output_stack_shape_3d, dtype=dtype)

                # Inner loop: iterate over Z and C for *only* this timepoint
                for z in range(Z):
                    for c in range(C):
                        plane = cast(np.ndarray, zarr_array[t, z, c])
                        top_crop = plane[top_roi[0], top_roi[1]]
                        bottom_crop = plane[bottom_roi[0], bottom_roi[1]]

                        # --- MODIFIED: Apply rotation ---
                        if rotate_90:
                            top_crop = np.rot90(top_crop, k=1)
                            bottom_crop = np.rot90(bottom_crop, k=1)
                        # --- END MODIFICATION ---

                        # Apply swap logic. Explicitly check c=0 and c=1.
                        # Any other channels are ignored.
                        if c == 0:
                            ch0_stack[z, :, :] = bottom_crop  # C0 = Bottom, Cam 1
                            ch1_stack[z, :, :] = top_crop  # C1 = Top, Cam 1
                        elif c == 1:
                            ch2_stack[z, :, :] = top_crop  # C2 = Top, Cam 2
                            ch3_stack[z, :, :] = bottom_crop  # C3 = Bottom, Cam 2

                # --- MODIFIED: Update axes metadata if rotated ---
                if rotate_90:
                    # After rotation, the original Y is now X, and X is Y.
                    # Since ImageJ/Fiji metadata is ZYX, we don't change 'ZYX'.
                    # The dimensions in the file header are (Z, X_new, Y_new)
                    # which is correct.
                    tif_meta = {"axes": "ZYX"}
                else:
                    tif_meta = {"axes": "ZYX"}
                # --- END MODIFICATION ---

                out_name = f"{sanitized_name}_T{t:03d}"
                tifffile.imwrite(
                    output_dir / f"{out_name}_C0.tif",
                    ch0_stack,
                    imagej=True,
                    metadata=tif_meta,
                )
                tifffile.imwrite(
                    output_dir / f"{out_name}_C1.tif",
                    ch1_stack,
                    imagej=True,
                    metadata=tif_meta,
                )
                tifffile.imwrite(
                    output_dir / f"{out_name}_C2.tif",
                    ch2_stack,
                    imagej=True,
                    metadata=tif_meta,
                )
                tifffile.imwrite(
                    output_dir / f"{out_name}_C3.tif",
                    ch3_stack,
                    imagej=True,
                    metadata=tif_meta,
                )

            print(f"✅ Saved {T * C_new} TIFF files to {output_dir.name}")

        else:
            raise ValueError(f"Unknown output_format: {output_format}")

        # Return T so the caller can pass it to the logger
        return T

    except Exception as e:
        print(f"\n❌ Error processing {base_file.name}: {e}\n")
        # Re-raise the exception for the caller (e.g., CLI) to handle
        raise


def run_processing_job(
    base_file: Path,
    top_roi: tuple[slice, slice],
    bottom_roi: tuple[slice, slice],
    output_format: OutputFormat,
    cli_log_file: Path = Path("opm_roi_log.json"),
    rotate_90: bool = False,  # <-- ADDED
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

    paths = derive_paths(base_file, output_format)

    if not paths.base_file.exists():
        raise FileNotFoundError(f"Input file not found: {paths.base_file}")
    if not paths.metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {paths.metadata_file}")

    paths.output_dir.mkdir(parents=True, exist_ok=True)

    # Clear old output to ensure a clean run
    if output_format == OutputFormat.ZARR:
        zarr_path = paths.output_dir / (paths.sanitized_name + "_processed.zarr")
        if zarr_path.exists():
            print(
                f"Warning: Output Zarr {zarr_path.name} already exists. Deleting it..."
            )
            shutil.rmtree(zarr_path)
    elif output_format == OutputFormat.TIFF_SERIES:
        print(f"Cleaning old files from {paths.output_dir.name}...")
        for f in paths.output_dir.glob(f"{paths.sanitized_name}_T*.tif"):
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
        rotate_90=rotate_90,  # <-- PASS
    )

    print("\nCreating processing log...")
    create_processing_log(
        paths,
        num_timepoints,
        top_roi,
        bottom_roi,
        output_format,
        rotate_90=rotate_90,  # <-- PASS
    )

    save_rois_to_log(
        cli_log_file,
        paths.base_file,
        top_roi,
        bottom_roi,
    )
