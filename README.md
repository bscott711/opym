# **opym: OPM Cropper & Channel Splitter**

`opym` (pronounced "opium") is a command-line tool for pre-processing 5D OME-TIF files from dual-camera OPM systems.

Its primary purpose is to prepare data for `pypetakit5d` by:

1. **Cropping** two separate ROIs (e.g., "top" and "bottom") from each camera.
2. **Splitting** the two-camera (C=2) stack into a four-channel (C=4) stack.
3. **Saving** the result as either a 5D OME-Zarr file or a series of 3D (ZYX) TIFF files, split by Time and Channel.

The `TIFF_SERIES_SPLIT_C` format is the required input for `pypetakit5d`.

## **Installation**

This package is designed to be installed and run using `uv`.

### 1. Create a Virtual Environment with uv

Navigate to the root `opym` project directory (where `pyproject.toml` is) and create a new virtual environment.

```bash
# Create the environment
uv venv

# Activate the environment (PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate the environment (bash/zsh)
source .venv/bin/activate
