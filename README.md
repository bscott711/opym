# OPM Cropping Package (`opym`)

`opym` is a Python package for processing 5D (T, Z, C, Y, X) OPM (Oblique Plane Microscopy) datasets. Its primary function is to crop, deskew (via channel alignment), and save out OPM data into analysis-friendly formats like Zarr or TIFF series.

It provides two main workflows:

1. **Interactive Notebook:** A Jupyter notebook to visually inspect data, select ROIs, and test processing on a single file.
2. **Command-Line Interface (CLI):** A `typer`-based CLI for batch processing entire directories of OPM data using pre-defined ROIs.

## Installation

This package is managed with `uv` and `pyproject.toml`.

1. Clone the repository:

    ```bash
    git clone [https://github.com/your-username/opym.git](https://github.com/your-username/opym.git)
    cd opym
    ```

2. Create and activate a virtual environment:

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3. Install the package in editable mode:

    ```bash
    uv pip install -e .
    ```

This will install all required dependencies, including `numpy`, `tifffile`, `zarr`, `scikit-image`, `tqdm`, and `typer`.

## Workflow 1: Interactive ROI Selection (Jupyter)

This is the recommended starting point for any new dataset. Use the included notebook to find the correct cropping coordinates.

1. Start Jupyter Lab:

    ```bash
    jupyter lab
    ```

2. Open the `OPM_Cropping_Refactored.ipynb` notebook.

3. **Cell 0 & 1:** Select your base OME-TIF file (e.g., `..._Pos0.ome.tif`) and run the cells to generate a Max Intensity Projection (MIP).

4. **Cell 2:** Run to open the interactive ROI selector.
    * Draw a box for the **Top ROI (C=0)**.
    * Click-and-drag near the center of the **Bottom ROI (C=1)**. The box size will be matched automatically.

5. **Cell 3 & 4:** Run to trigger the auto-alignment, which fine-tunes the Bottom ROI position based on phase cross-correlation. The final aligned ROIs will be displayed.

6. **Cell 5:** Saves your selected ROIs to the central `opm_roi_log.json` file in your project directory. This file is used by the CLI for batch processing.

7. **Cell 6 (Optional):** Run the full processing job on *just this file* from within the notebook to confirm the results.

## Workflow 2: CLI Batch Processing

Once you have saved your ROIs to the `opm_roi_log.json` file, you can use the `opym` CLI to process all other files in your dataset (e.g., `..._Pos1.ome.tif`, `..._Pos2.ome.tif`, etc.).

The main command is `opym process`.

### Examples

**Process all files using the log:**

This is the most common use case. The command will find all `*Pos0.ome.tif`, `*Pos1.ome.tif`, etc. files in the input directory and automatically find their matching ROIs from the `opm_roi_log.json` file.

```bash
opym process \
    --input-dir /path/to/my/dataset \
    --output-format ZARR \
    --roi-from-log opm_roi_log.json

opym process \
    --input-file /path/to/my/dataset/my_file_Pos0.ome.tif \
    --output-format TIFF_SERIES_SPLIT_C \
    --top-roi "431:708, 557:1671" \
    --bottom-roi "1582:1859, 531:1645"
```

Cli options:

```bash
Usage: opym process [OPTIONS]
────────────────────────────────────────────────────────────────────────────────
 Process one or more OPM files from a 5D OME-TIF to a processed Zarr or
 TIFF series.

 You must provide EITHER --input-file OR --input-dir.
 You must provide ROIs via EITHER --roi-from-log OR (--top-roi and
 --bottom-roi).
────────────────────────────────────────────────────────────────────────────────
Options:
  --input-file              PATH  A single 5D OME-TIF file to process.
  --input-dir               PATH  A directory of 5D OME-TIF files to process.
  --output-format     [ZARR|TIFF_SERIES_SPLIT_C]
                                The output format.
                                [default: ZARR]
  --top-roi                 TEXT  ROI for the top channel (C=0) as
                                "y_start:y_stop,x_start:x_stop".
  --bottom-roi              TEXT  ROI for the bottom channel (C=1) as
                                "y_start:y_stop,x_start:x_stop".
  --roi-from-log            PATH  Path to a JSON log file containing ROIs.
  --help                        Show this message and exit.
