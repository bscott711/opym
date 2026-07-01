"""Golden-reference regression test for the GPU decon -> deskew+rotate pipeline.

This is the safety net every performance/orchestration change in this repo
must pass before merging (see .claude/plans -- "Speed up the OPM decon/
deskew/rotate pipeline without changing outputs", Phase 0). It calls
run_gpu_pipeline.m directly via matlab.engine, bypassing the JSON queue /
watchdog entirely, since those are pure orchestration and not what this
test protects: rotation angle, axis order, and channel/geometry parameters.

Requirements to run (skipped automatically if unavailable):
- A real CUDA GPU (uses gpuDevice(1)).
- `module load matlab/R2024b` must already be active in the shell that
  launches pytest -- matlab.engine cannot pick up MATLAB's license/env
  setup after the fact. From a shell where that's loaded:

    module load matlab/R2024b
    /path/to/bioimaging/.venv/bin/python -m pytest -m gpu \\
        opym_local/tests/test_gpu_pipeline_regression.py

  (matlab.engine / PyPetaKit5D live in bioimaging's venv, not opym_local's
  own -- opym is installed editable into bioimaging's venv, which is the
  one actually used to run this project day to day.)

Regenerate golden references (only do this deliberately, e.g. once to
establish a new intended baseline -- such as after the DeconIter 10->25
correctness fix):

    pytest -m gpu opym_local/tests/test_gpu_pipeline_regression.py --update-golden
"""

from __future__ import annotations

import os
import time
from pathlib import Path

# matlab.engine must be imported before numpy/zarr: its native engine library
# needs a newer libstdc++ than the one bundled in numpy/zarr's C-extension
# wheels, and whichever loads first wins symbol resolution for the process.
try:
    import matlab.engine  # noqa: F401
except ImportError:
    pass

import numpy as np
import pytest
import zarr

pytestmark = pytest.mark.gpu

FIXTURES_DIR = Path(__file__).parent / "fixtures"
GOLDEN_DIR = FIXTURES_DIR / "golden"
PSF_PATH = FIXTURES_DIR / "psf" / "averaged_psf.tif"
SYNTHETIC_RAW_PATH = FIXTURES_DIR / "synthetic_raw_volume.npy"

OPYM_SRC_DIR = Path(__file__).resolve().parents[1] / "src" / "opym"
PETAKIT_ROOT = Path(os.environ.get("PETAKIT_ROOT", "/cm/shared/apps_local/petakit5d"))

# Must match the intended production defaults -- see the performance plan's
# Phase 1.4: DeconIter is 25 for RLMethod='simple', not the buggy 10 that
# process_chunk() currently hardcodes. This test locks in the CORRECT value.
PIPELINE_KWARGS = dict(
    xyPixelSize=0.136,
    z_step_um=0.3,
    dzPSF=0.1,  # matches fixtures/psf/averaged_psf.tif's ImageJ 'spacing' tag
    DeconIter=25,
    RLMethod="simple",
    SkewAngle=60.0,
    interpMethod="cubic",  # run_gpu_pipeline.m force-downgrades to linear internally
    debug=False,
)

N_TOP_PEAKS = 4  # matches len(BEAD_CENTERS) in generate_synthetic_fixture.py


@pytest.fixture(scope="module")
def matlab_engine():
    try:
        import matlab.engine
    except ImportError:
        pytest.skip(
            "matlab.engine not importable -- run with bioimaging's venv "
            "(matlabengine is a declared dependency there, not in opym_local's own .venv)."
        )

    try:
        eng = matlab.engine.start_matlab("-nodisplay")
    except Exception as e:  # noqa: BLE001 - report exact MATLAB engine error to the user
        pytest.skip(f"Could not start MATLAB engine (is `module load matlab/R2024b` active?): {e}")

    try:
        eng.gpuDevice(1, nargout=1)
    except Exception as e:  # noqa: BLE001
        eng.quit()
        pytest.skip(f"No usable CUDA GPU for matlab.engine: {e}")

    eng.addpath(str(OPYM_SRC_DIR), nargout=0)
    eng.addpath(str(OPYM_SRC_DIR / "patches"), nargout=0)
    if eng.exist("XR_deskew_rotate_data_wrapper", "file", nargout=1) == 0:
        setup_m = PETAKIT_ROOT / "setup.m"
        if not setup_m.exists():
            eng.quit()
            pytest.skip(f"PetaKit5D setup.m not found at {setup_m}")
        eng.run(str(setup_m), nargout=0)

    yield eng
    eng.quit()


def _write_shm_zarr(path: Path, array: np.ndarray) -> None:
    if path.exists():
        import shutil

        shutil.rmtree(path)
    zarr.save(str(path), array)


def _wait_for_zarr(path: Path, timeout_s: float = 20.0) -> np.ndarray:
    """The current run_gpu_pipeline.m copies its result to `path` via a
    fire-and-forget background `cp` (see the performance plan's Phase 1.2 --
    this race is a known pre-existing bug, fixed in a later branch, not this
    one). Poll instead of assuming the copy finished by the time the
    synchronous MATLAB call returns.
    """
    deadline = time.time() + timeout_s
    zarray_marker = path / ".zarray"
    while time.time() < deadline:
        if zarray_marker.exists():
            try:
                return np.asarray(zarr.open(str(path), mode="r"))
            except Exception:  # noqa: BLE001 - copy may still be mid-flight
                pass
        time.sleep(0.2)
    raise TimeoutError(f"Timed out waiting for pipeline output at {path}")


def _top_peak_coords(volume: np.ndarray, n: int) -> list[tuple[int, ...]]:
    flat_idx = np.argsort(volume, axis=None)[::-1]
    coords: list[tuple[int, ...]] = []
    seen_near: list[np.ndarray] = []
    vol_shape = volume.shape
    for idx in flat_idx:
        coord = np.array(np.unravel_index(idx, vol_shape))
        # Skip points too close to an already-picked peak (same bead's
        # neighboring voxels), so we get N distinct beads, not N hot pixels
        # from the same blob.
        if any(np.linalg.norm(coord - s) < 5 for s in seen_near):
            continue
        seen_near.append(coord)
        coords.append(tuple(int(c) for c in coord))
        if len(coords) == n:
            break
    return sorted(coords)


def _run_pipeline(matlab_engine, tmp_path: Path, raw_volume: np.ndarray) -> np.ndarray:
    shm_in = tmp_path / "in.zarr"
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_fn = out_dir / "regression_test_output.tif"

    _write_shm_zarr(shm_in, raw_volume)

    # matlab.engine's Python binding doesn't accept MATLAB name-value pairs as
    # Python kwargs -- they must be flattened into the positional arg list,
    # exactly as run_petakit_server.m passes them to run_gpu_pipeline.
    name_value_args = []
    for k, v in PIPELINE_KWARGS.items():
        name_value_args.append(k)
        name_value_args.append(v)

    matlab_engine.run_gpu_pipeline(
        str(shm_in), str(output_fn), str(PSF_PATH), *name_value_args, nargout=1
    )

    out_zarr = out_dir / "regression_test_output.zarr"
    return _wait_for_zarr(out_zarr)


def _compare_or_update(actual: np.ndarray, golden_path: Path, update_golden: bool) -> None:
    if update_golden or not golden_path.exists():
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(golden_path, actual)
        pytest.skip(f"Wrote golden reference to {golden_path} (--update-golden or first run).")

    golden = np.load(golden_path)

    assert actual.shape == golden.shape, (
        f"Output shape changed: {actual.shape} vs golden {golden.shape}. "
        "This is exactly the class of regression this test exists to catch "
        "(axis permutation / rotation / crop-size change)."
    )

    actual_peaks = _top_peak_coords(actual, N_TOP_PEAKS)
    golden_peaks = _top_peak_coords(golden, N_TOP_PEAKS)
    assert actual_peaks == golden_peaks, (
        f"Peak voxel locations changed: {actual_peaks} vs golden {golden_peaks}. "
        "This means rotation angle / axis order / deskew geometry likely changed."
    )

    assert np.allclose(actual, golden, rtol=1e-4, atol=1), (
        "Output values drifted beyond floating-point tolerance vs golden reference."
    )

    golden_sum = float(golden.sum())
    actual_sum = float(actual.sum())
    assert abs(actual_sum - golden_sum) / golden_sum < 1e-3, (
        f"Total intensity changed beyond tolerance: {actual_sum} vs golden {golden_sum}. "
        "Could indicate a silently-zeroed channel or half-volume."
    )


def test_synthetic_regression(matlab_engine, tmp_path, update_golden):
    raw_volume = np.load(SYNTHETIC_RAW_PATH)
    actual = _run_pipeline(matlab_engine, tmp_path, raw_volume)
    _compare_or_update(actual, GOLDEN_DIR / "synthetic_golden.npy", update_golden)


def test_real_data_regression(matlab_engine, tmp_path, update_golden):
    real_data_dir = os.environ.get("OPYM_REGRESSION_REAL_DATA_DIR")
    if not real_data_dir:
        pytest.skip(
            "OPYM_REGRESSION_REAL_DATA_DIR not set -- real-data regression fixture "
            "lives outside git (see Phase 0 of the performance plan) and is only "
            "run on machines with access to it."
        )
    real_dir = Path(real_data_dir)
    raw_path = real_dir / "raw_crop.npy"
    golden_path = real_dir / "golden_output.npy"
    if not raw_path.exists():
        pytest.skip(f"Real-data raw crop not found at {raw_path}")

    raw_volume = np.load(raw_path)
    actual = _run_pipeline(matlab_engine, tmp_path, raw_volume)
    _compare_or_update(actual, golden_path, update_golden)
