"""Generates the checked-in synthetic raw volume used by the GPU pipeline
regression test (see ../test_gpu_pipeline_regression.py).

Run manually to (re)create synthetic_raw_volume.npy:

    python tests/fixtures/generate_synthetic_fixture.py

This is deliberately deterministic (no RNG) so the fixture never needs
regenerating unless someone intentionally changes the bead layout below.
The volume is already in the (ny, nx, nz) "skewed-space" layout that
run_gpu_pipeline.m expects from readzarr(shm_path) -- i.e. the same
layout opym.utils.orient_zyx_for_dsr() produces in production.
"""

from pathlib import Path

import numpy as np

# Deliberately small (fast decon/DSR on GPU) but with distinct extents on
# every axis and an asymmetric bead layout, so any axis swap/flip/rotation
# regression shows up as a peak-location mismatch, not a shape coincidence.
SHAPE = (192, 128, 64)  # (ny, nx, nz)
BACKGROUND = 100
PEAK = 6000

# (ny, nx, nz) coordinates. Chosen so no two beads share a coordinate on any
# axis, and none sit on a center/symmetry line -- this is what makes the
# fixture sensitive to axis permutation bugs instead of passing by accident.
BEAD_CENTERS = [
    (30, 20, 8),
    (150, 100, 50),
    (60, 90, 15),
    (170, 15, 40),
]


def make_volume() -> np.ndarray:
    vol = np.full(SHAPE, BACKGROUND, dtype=np.uint16)
    yy, xx, zz = np.meshgrid(
        np.arange(SHAPE[0]), np.arange(SHAPE[1]), np.arange(SHAPE[2]), indexing="ij"
    )
    accum = np.zeros(SHAPE, dtype=np.float64)
    for cy, cx, cz in BEAD_CENTERS:
        r2 = (yy - cy) ** 2 + (xx - cx) ** 2 + (zz - cz) ** 2
        accum += PEAK * np.exp(-r2 / (2 * 1.2**2))
    vol = np.clip(vol.astype(np.float64) + accum, 0, 65535).astype(np.uint16)
    return vol


if __name__ == "__main__":
    out_dir = Path(__file__).parent
    vol = make_volume()
    out_path = out_dir / "synthetic_raw_volume.npy"
    np.save(out_path, vol)
    print(f"Wrote {out_path} shape={vol.shape} dtype={vol.dtype}")
