"""Golden-reference regression test for the deskew-only (decon SKIPPED) DSR
path used by the bulk no-decon backfill.

Companion to test_gpu_pipeline_regression.py (decon+DSR fused, via
run_gpu_pipeline.m) -- this test instead calls XR_deskewRotateFrame directly
with no PSF, exercising exactly the code path the backfill's
`submit_remote_deskew_job(psf_path=None, ...)` dispatches to via the
'deskew' job type in run_petakit_server.m. Protects the same class of
regression (axis permutation / rotation / crop-size change) for the
skip-decon route specifically -- skipping decon should not move bead
positions, only reduce sharpness/SNR, so this compares peak *coordinates*
against the SAME golden peak locations the decon-included test locks in,
not against a decon-included golden intensity array.

Requirements to run (skipped automatically if unavailable): see
test_gpu_pipeline_regression.py's module docstring -- identical
prerequisites (real CUDA GPU, `module load matlab/R2024b`, run from
bioimaging's venv).

    module load matlab/R2024b
    /path/to/bioimaging/.venv/bin/python -m pytest -m gpu \\
        opym_local/tests/test_deskew_only_regression.py
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np
import pytest
import tifffile
import zarr

pytestmark = pytest.mark.gpu

FIXTURES_DIR = Path(__file__).parent / "fixtures"
GOLDEN_DIR = FIXTURES_DIR / "golden"
SYNTHETIC_RAW_PATH = FIXTURES_DIR / "synthetic_raw_volume.npy"

# varargin name-value pairs for XR_deskewRotateFrame -- deliberately mirrors
# exactly what run_petakit_server.m's 'deskew' job-type branch passes to
# XR_deskew_rotate_data_wrapper (see run_petakit_server.m ~line 531-555),
# EXCEPT no psf/decon parameters are supplied at all -- decon is skipped by
# never calling XR_decon_data_wrapper, not by a flag on this function (decon
# isn't even a parameter of XR_deskewRotateFrame; it's architecturally a
# separate, prior call chained by the server for the decon+DSR route).
XY_PIXEL_SIZE = 0.136
DZ = 0.3
DSR_KWARGS = dict(
    DSRDirName="DSR_nodecon",
    skewAngle=60.0,
    reverse=False,
    inputAxisOrder="yxz",
    outputAxisOrder="yxz",
    rotate=True,
    DSRCombined=True,
    interpMethod="cubic",
    save16bit=True,
    save3DStack=True,
    saveMIP=False,
)

N_TOP_PEAKS = 4  # matches len(BEAD_CENTERS) in generate_synthetic_fixture.py


def _top_peak_coords(volume: np.ndarray, n: int) -> list[tuple[int, ...]]:
    flat_idx = np.argsort(volume, axis=None)[::-1]
    coords: list[tuple[int, ...]] = []
    seen_near: list[np.ndarray] = []
    vol_shape = volume.shape
    for idx in flat_idx:
        coord = np.array(np.unravel_index(idx, vol_shape))
        if any(np.linalg.norm(coord - s) < 5 for s in seen_near):
            continue
        seen_near.append(coord)
        coords.append(tuple(int(c) for c in coord))
        if len(coords) == n:
            break
    return sorted(coords)


def _wait_for_dsr_tif(dsr_dir: Path, timeout_s: float = 20.0) -> np.ndarray:
    """Polls rather than assumes synchronous completion -- PetaKit5D's
    write-then-move pattern elsewhere in this codebase is a known source of
    a "file exists but copy still mid-flight" race (see
    test_gpu_pipeline_regression.py's `_wait_for_zarr`).
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        candidates = [p for p in dsr_dir.glob("*.tif") if "MIP" not in p.name] if dsr_dir.is_dir() else []
        if candidates:
            try:
                return tifffile.imread(candidates[0])
            except Exception:  # noqa: BLE001 - file may still be mid-write
                pass
        time.sleep(0.2)
    raise TimeoutError(
        f"Timed out waiting for deskew-only output under {dsr_dir}. "
        "If XR_deskewRotateFrame's output directory/filename convention "
        "differs from what this test assumes, adjust `_wait_for_dsr_tif`."
    )


def _run_deskew_only(matlab_engine, tmp_path: Path, raw_volume: np.ndarray) -> np.ndarray:
    frame_path = tmp_path / "in.zarr"
    if frame_path.exists():
        shutil.rmtree(frame_path)
    zarr.save(str(frame_path), raw_volume)

    name_value_args = []
    for k, v in DSR_KWARGS.items():
        name_value_args.append(k)
        name_value_args.append(v)

    # No psf_path / decon parameters anywhere in this call -- that IS the
    # decon skip. nargout=2 matches `function [ds, dsr] = XR_deskewRotateFrame(...)`;
    # DSRCombined=True means `ds` is unset/empty (the DS-only branch is
    # skipped entirely), so only `dsr` is used below.
    matlab_engine.XR_deskewRotateFrame(
        str(frame_path), XY_PIXEL_SIZE, DZ, *name_value_args, nargout=2
    )

    dsr_dir = frame_path.parent / DSR_KWARGS["DSRDirName"]
    return _wait_for_dsr_tif(dsr_dir)


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
        "This means rotation angle / axis order / deskew geometry likely changed "
        "in the no-decon path specifically."
    )


def test_deskew_only_synthetic_regression(matlab_engine, tmp_path, update_golden):
    raw_volume = np.load(SYNTHETIC_RAW_PATH)
    actual = _run_deskew_only(matlab_engine, tmp_path, raw_volume)
    _compare_or_update(actual, GOLDEN_DIR / "deskew_only_synthetic_golden.npy", update_golden)


def test_deskew_only_matches_decon_included_peaks(matlab_engine, tmp_path):
    """Skipping decon should not move bead positions, only reduce
    sharpness/SNR -- cross-checks peak coordinates against the
    decon-included golden reference from test_gpu_pipeline_regression.py,
    rather than requiring a byte-identical intensity match.
    """
    decon_golden_path = GOLDEN_DIR / "synthetic_golden.npy"
    if not decon_golden_path.exists():
        pytest.skip(
            f"{decon_golden_path} not generated yet -- run "
            "test_gpu_pipeline_regression.py --update-golden first."
        )

    raw_volume = np.load(SYNTHETIC_RAW_PATH)
    actual = _run_deskew_only(matlab_engine, tmp_path, raw_volume)
    decon_golden = np.load(decon_golden_path)

    actual_peaks = _top_peak_coords(actual, N_TOP_PEAKS)
    decon_peaks = _top_peak_coords(decon_golden, N_TOP_PEAKS)
    assert actual_peaks == decon_peaks, (
        f"No-decon peak locations {actual_peaks} differ from the decon-included "
        f"golden's {decon_peaks} -- skipping decon should only affect sharpness, "
        "not bead position."
    )
