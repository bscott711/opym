# Ruff style: Compliant
"""
Core OPM Cropper processing functions.
Streams from a 5D OME-TIF virtual zarr stack into the specified output format.
"""

import shutil
import sys
from pathlib import Path
from typing import cast

import numpy as np
import tifffile
import zarr
from tqdm.auto import tqdm

from .utils import OutputFormat


def process_dataset(
    base_file: Path,
    output_dir: Path,
    sanitized_name: str,
    top_roi: tuple[slice, slice],
    bottom_roi: tuple[slice, slice],
    output_format: OutputFormat,
) -> int:
    """
    Main processing function. Streams data to either a single OME-TIF file
    or a series of 3D (ZYX) TIFF files.

    Returns:
        int: The number of timepoints processed (T).
    """
    print(f"--- Processing: {base_file.name} (Format: {output_format.value}) ---")

    try:
        # --- 1. Setup based on input TIFF ---
        with tifffile.TiffFile(base_file) as tif:
            series = tif.series[0]
            store = series.aszarr()
            # FIX: Use zarr.open_array to guarantee an Array is returned
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
                    file=sys.stderr,
                )

        print(f"Using sanitized base name for output: {sanitized_name}")

        # --- 2. Stream data and write output ---
        if output_format == OutputFormat.ZARR:
            output_zarr_path = output_dir / (sanitized_name + "_processed.zarr")

            # Cleanup old Zarr.
            if output_zarr_path.exists():
                print(
                    f"Warning: Output Zarr {output_zarr_path.name} "
                    "already exists. Deleting it..."
                )
                shutil.rmtree(output_zarr_path)

            output_shape = (T, Z, C_new, Y_new, X_new)
            chunks = (1, 1, 1, Y_new, X_new)
            # FIX: 'shape' is the first positional arg. 'store' is a kwarg.
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
                            # FIX: Use cast to inform Pylance this is an ndarray
                            plane = cast(np.ndarray, zarr_array[t, z, c])
                            top_crop = plane[top_roi[0], top_roi[1]]
                            bottom_crop = plane[bottom_roi[0], bottom_roi[1]]

                            if c == 0:
                                zarr_out[t, z, 0, :, :] = bottom_crop  # C0
                                zarr_out[t, z, 1, :, :] = top_crop  # C1
                            else:
                                zarr_out[t, z, 2, :, :] = top_crop  # C2
                                zarr_out[t, z, 3, :, :] = bottom_crop  # C3
                            pbar.update(1)

            print(f"✅ Saved processed series to {output_zarr_path.name}")

        elif output_format == OutputFormat.TIFF_SERIES:
            output_stack_shape_3d = (Z, Y_new, X_new)

            # Loop by Timepoint (this is the outer loop)
            for t in tqdm(range(T), desc=" ├ Streaming & Writing", unit="TP"):
                # Create 4 empty 3D arrays for this timepoint
                ch0_stack = np.zeros(output_stack_shape_3d, dtype=dtype)
                ch1_stack = np.zeros(output_stack_shape_3d, dtype=dtype)
                ch2_stack = np.zeros(output_stack_shape_3d, dtype=dtype)
                ch3_stack = np.zeros(output_stack_shape_3d, dtype=dtype)

                # Inner loop: iterate over Z and C for *only* this timepoint
                for z in range(Z):
                    for c in range(C):
                        # FIX: Use cast to inform Pylance this is an ndarray
                        plane = cast(np.ndarray, zarr_array[t, z, c])
                        top_crop = plane[top_roi[0], top_roi[1]]
                        bottom_crop = plane[bottom_roi[0], bottom_roi[1]]

                        # Apply the C=0 swap logic and fill the arrays
                        if c == 0:
                            ch0_stack[z, :, :] = bottom_crop  # C0 = Bottom, Cam 1
                            ch1_stack[z, :, :] = top_crop  # C1 = Top, Cam 1
                        else:
                            ch2_stack[z, :, :] = top_crop  # C2 = Top, Cam 2
                            ch3_stack[z, :, :] = bottom_crop  # C3 = Bottom, Cam 2

                # --- FIX: Correct arguments for tifffile.imwrite ---
                tif_meta = {"axes": "ZYX"}
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
                # --- End fix ---

            print(f"✅ Saved {T * C_new} TIFF files to {output_dir.name}")

        else:
            # This should be caught by argparse, but good to have
            raise ValueError(f"Unknown output_format: {output_format}")

        # Return T so the caller can pass it to the logger
        return T

    except Exception as e:
        print(f"\n❌ Error processing {base_file.name}: {e}\n", file=sys.stderr)
        sys.exit(1)
