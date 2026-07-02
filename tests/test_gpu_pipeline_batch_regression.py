"""Regression test for the pipeline_batch job type (Phase 2.1 of the
performance plan -- see .claude/plans/we-have-just-finished-reflective-flute.md).

Verifies two things about run_gpu_pipeline_batch_async.m:
1. Batched output is pixel-identical (within the same tolerance as the
   single-frame regression test) to running each item independently through
   run_gpu_pipeline -- batching must only change dispatch granularity, never
   per-frame decon/deskew/rotate math.
2. Per-item failure isolation: one deliberately-bad item in a batch does not
   prevent its batch-mates from completing and being saved correctly.

Bypasses the JSON queue/watchdog and the GPU-lock acquire loop in
run_petakit_server.m (same bypass test_gpu_pipeline_regression.py uses for
the single-frame path) but still exercises the real
jsondecode -> struct array -> run_gpu_pipeline_batch_async path via a
MATLAB-side `eval`, so the items struct array is built exactly the way a
real pipeline_batch ticket builds it.

Requirements: same as test_gpu_pipeline_regression.py -- real CUDA GPU +
`module load matlab/R2024b` active in the launching shell.
"""

from __future__ import annotations

import json
from pathlib import Path

try:
    import matlab.engine  # noqa: F401
except ImportError:
    pass

import numpy as np
import pytest

from test_gpu_pipeline_regression import (
    PIPELINE_KWARGS,
    PSF_PATH,
    SYNTHETIC_RAW_PATH,
    N_TOP_PEAKS,
    _run_pipeline,
    _top_peak_coords,
    _write_shm_zarr,
    _wait_for_zarr,
    matlab_engine,  # noqa: F401 -- imported for pytest fixture discovery
)

pytestmark = pytest.mark.gpu


def _matlab_literal(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(float(value))
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    raise TypeError(f"Unsupported MATLAB literal type: {type(value)}")


def _run_batch(
    matlab_engine, tmp_path: Path, raw_volumes: list[np.ndarray], tag: str, bad_item_index: int | None = None
) -> tuple[list[Path], Path, Path]:
    """Runs raw_volumes through one pipeline_batch ticket via
    run_gpu_pipeline_batch_async. If bad_item_index is given, that item's
    shm_path is left nonexistent to trigger a deterministic per-item failure.

    Returns (per-item output zarr paths, done_dir, fail_dir).
    """
    out_dir = tmp_path / f"{tag}_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    queue_dir = tmp_path / f"{tag}_queue"
    queue_dir.mkdir(parents=True, exist_ok=True)
    done_dir = tmp_path / f"{tag}_completed"
    done_dir.mkdir(parents=True, exist_ok=True)
    fail_dir = tmp_path / f"{tag}_failed"
    fail_dir.mkdir(parents=True, exist_ok=True)
    lock_dir = tmp_path / f"{tag}_gpu_locks"
    lock_dir.mkdir(parents=True, exist_ok=True)

    items_payload = []
    out_paths = []
    for i, vol in enumerate(raw_volumes):
        output_fn = out_dir / f"{tag}_item{i}.tif"
        out_paths.append(out_dir / f"{tag}_item{i}.zarr")
        if bad_item_index is not None and i == bad_item_index:
            shm_in = tmp_path / f"{tag}_item{i}_MISSING.zarr"  # never created
        else:
            shm_in = tmp_path / f"{tag}_item{i}_in.zarr"
            _write_shm_zarr(shm_in, vol)
        items_payload.append({"shm_path": str(shm_in), "output_file": str(output_fn)})

    active_path = queue_dir / f".active_{tag}.json"
    active_path.write_text(json.dumps({"items": items_payload}))

    name_value_args = []
    for k, v in PIPELINE_KWARGS.items():
        if k == "debug":
            continue
        name_value_args.append(k)
        name_value_args.append(v)
    nv_literal = ", ".join(_matlab_literal(v) for v in name_value_args)

    cmd = (
        f"ticket = jsondecode(fileread({_matlab_literal(str(active_path))})); "
        f"run_gpu_pipeline_batch_async("
        f"{_matlab_literal(str(active_path))}, "
        f"{_matlab_literal(str(done_dir))}, "
        f"{_matlab_literal(str(fail_dir))}, "
        f"ticket.items, "
        f"{_matlab_literal(str(PSF_PATH))}, "
        f"{_matlab_literal(str(lock_dir))}, "
        f"{_matlab_literal(tag)}"
        + (f", {nv_literal}" if nv_literal else "")
        + ");"
    )
    matlab_engine.eval(cmd, nargout=0)

    return out_paths, done_dir, fail_dir


def _shifted_volumes(n: int) -> list[np.ndarray]:
    """N distinct raw volumes derived from the synthetic fixture via small
    circular shifts, so each item in a batch has genuinely different content
    while still sharing the same PSF/geometry (the batchability precondition).
    """
    base = np.load(SYNTHETIC_RAW_PATH)
    return [np.roll(base, shift=i * 3, axis=2) for i in range(n)]


def test_batch_matches_independent_runs(matlab_engine, tmp_path):
    volumes = _shifted_volumes(3)

    independent_outputs = []
    for i, vol in enumerate(volumes):
        out = _run_pipeline(matlab_engine, tmp_path / f"indep{i}", vol)
        independent_outputs.append(out)

    batch_out_paths, done_dir, fail_dir = _run_batch(matlab_engine, tmp_path, volumes, tag="matchtest")
    assert list(fail_dir.glob("*")) == [], "no items should fail in this test"
    assert (done_dir / "matchtest.json").exists(), "ticket should be moved to done_dir"

    for i, (independent, out_path) in enumerate(zip(independent_outputs, batch_out_paths)):
        batched = _wait_for_zarr(out_path)
        assert batched.shape == independent.shape, f"item {i}: shape mismatch between batched and independent run"

        independent_peaks = _top_peak_coords(independent, N_TOP_PEAKS)
        batched_peaks = _top_peak_coords(batched, N_TOP_PEAKS)
        assert batched_peaks == independent_peaks, (
            f"item {i}: peak voxel locations differ between batched and independent run -- "
            "batching must not change rotation/axis/geometry."
        )

        assert np.allclose(batched, independent, rtol=1e-4, atol=1), (
            f"item {i}: batched output drifted beyond tolerance vs. the independent single-frame run."
        )


def test_batch_failure_isolation(matlab_engine, tmp_path):
    volumes = _shifted_volumes(4)
    bad_index = 2

    batch_out_paths, done_dir, fail_dir = _run_batch(
        matlab_engine, tmp_path, volumes, tag="failtest", bad_item_index=bad_index
    )

    # 3 of 4 succeeded -> ticket lands in done_dir, not fail_dir.
    assert (done_dir / "failtest.json").exists(), "ticket should land in done_dir when not all items failed"
    assert not (fail_dir / "failtest.json").exists()

    partial_failures = list(fail_dir.glob("*.partial_failures.json"))
    assert len(partial_failures) == 1, "a partial-failures manifest should be written to fail_dir"
    manifest = json.loads(partial_failures[0].read_text())
    statuses = {item["output_file"]: item["status"] for item in manifest}
    failed_outputs = [k for k, v in statuses.items() if v == "failed"]
    assert len(failed_outputs) == 1
    assert f"failtest_item{bad_index}.tif" in failed_outputs[0]

    for i, out_path in enumerate(batch_out_paths):
        if i == bad_index:
            assert not out_path.exists(), "the deliberately-bad item must not produce output"
            continue
        good_output = _wait_for_zarr(out_path)
        assert good_output.sum() > 0, f"item {i} (batch-mate of the failed item) should still have real output"
