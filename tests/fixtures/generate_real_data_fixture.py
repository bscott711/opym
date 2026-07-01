"""Generates the real-data regression fixture used by
test_gpu_pipeline_regression.py::test_real_data_regression.

Unlike the synthetic fixture, this pulls a single-timepoint, single-channel
crop from a real acquisition and writes it OUTSIDE either git repo -- per
project policy, real acquisition data (and anything derived from it) must
never be committed. Point OPYM_REGRESSION_REAL_DATA_DIR at the output
directory this script writes to when running the regression test.

Usage:

    module load matlab/R2024b
    OPYM_REGRESSION_REAL_DATA_DIR=/mmfs2/scratch/SDSMT.LOCAL/bscott/opym_regression_fixtures/real_data \\
    /path/to/bioimaging/.venv/bin/python tests/fixtures/generate_real_data_fixture.py

This only needs to be re-run if the source dataset path changes or the crop
window below is deliberately changed -- not on every code change.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# matlab.engine must be imported before numpy/zarr -- see
# test_gpu_pipeline_regression.py for why.
try:
    import matlab.engine
except ImportError:
    matlab = None  # noqa: F811

import numpy as np
import tifffile
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from opym.utils import orient_zyx_for_dsr  # noqa: E402

SOURCE_OME_TIF = (
    "/mmfs1/scratch/SDSMT.LOCAL/bscott/DataUpload/"
    "20260402_py_FLM_2XFyve_mSca_mem_NG/cell/cell_MMStack_Pos0.ome.tif"
)
TIMEPOINT = 0
CHANNEL = 0  # raw channel index (one of the 2 cameras for this single-excitation acquisition)
Y_SLICE = slice(800, 1056)  # 256px central-ish window
X_SLICE = slice(800, 1056)

PSF_PATH = Path(__file__).parent / "psf" / "averaged_psf.tif"
OPYM_SRC_DIR = Path(__file__).resolve().parents[2] / "src" / "opym"
PETAKIT_ROOT = Path(os.environ.get("PETAKIT_ROOT", "/cm/shared/apps_local/petakit5d"))

PIPELINE_KWARGS = dict(
    xyPixelSize=0.136,
    z_step_um=0.3,  # matches this dataset's AcqSettings.txt stepSizeUm
    dzPSF=0.1,
    DeconIter=25,
    RLMethod="simple",
    SkewAngle=60.0,
    interpMethod="cubic",
    debug=False,
)


def extract_real_crop() -> np.ndarray:
    store = tifffile.imread(SOURCE_OME_TIF, aszarr=True)
    z = zarr.open(store, mode="r")
    print(f"Source shape: {z.shape}")  # expect (T, Z, C, Y, X)
    cropped = np.array(z[TIMEPOINT, :, CHANNEL, Y_SLICE, X_SLICE])
    return orient_zyx_for_dsr(cropped)


def run_pipeline(raw_volume: np.ndarray, out_dir: Path) -> np.ndarray:
    eng = matlab.engine.start_matlab("-nodisplay")
    eng.gpuDevice(1, nargout=1)
    eng.addpath(str(OPYM_SRC_DIR), nargout=0)
    eng.addpath(str(OPYM_SRC_DIR / "patches"), nargout=0)
    if eng.exist("XR_deskew_rotate_data_wrapper", "file", nargout=1) == 0:
        eng.run(str(PETAKIT_ROOT / "setup.m"), nargout=0)

    shm_in = out_dir / "_tmp_in.zarr"
    if shm_in.exists():
        import shutil

        shutil.rmtree(shm_in)
    zarr.save(str(shm_in), raw_volume)

    output_fn = out_dir / "_tmp_output.tif"
    name_value_args = []
    for k, v in PIPELINE_KWARGS.items():
        name_value_args += [k, v]
    eng.run_gpu_pipeline(str(shm_in), str(output_fn), str(PSF_PATH), *name_value_args, nargout=1)
    eng.quit()

    import time

    out_zarr = out_dir / "_tmp_output.zarr"
    deadline = time.time() + 30
    while time.time() < deadline:
        if (out_zarr / ".zarray").exists():
            break
        time.sleep(0.2)
    return np.asarray(zarr.open(str(out_zarr), mode="r"))


def main():
    out_dir_str = os.environ.get("OPYM_REGRESSION_REAL_DATA_DIR")
    if not out_dir_str:
        raise SystemExit("Set OPYM_REGRESSION_REAL_DATA_DIR to an output directory outside git.")
    out_dir = Path(out_dir_str)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Extracting real crop...")
    raw_volume = extract_real_crop()
    np.save(out_dir / "raw_crop.npy", raw_volume)
    print(f"Wrote {out_dir / 'raw_crop.npy'} shape={raw_volume.shape}")

    print("Running GPU pipeline to generate golden reference...")
    golden = run_pipeline(raw_volume, out_dir)
    np.save(out_dir / "golden_output.npy", golden)
    print(f"Wrote {out_dir / 'golden_output.npy'} shape={golden.shape}")


if __name__ == "__main__":
    main()
