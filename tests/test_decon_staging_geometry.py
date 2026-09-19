"""Geometry and safety contract for the deconvolution path.

Deconvolution is a 3D convolution, and PetaKit5D's decon path
(XR_decon_data_wrapper -> XR_RLdeconFrame3D -> RLdecon) has NO axis-order
parameter -- it convolves the array exactly as stored, before
XR_deskewRotateFrame's `inputAxisOrder` permute ever happens. So the
orientation of the staged input, and of the PSF, is the whole ballgame: get
it wrong and decon reports success on garbage, exactly the way the deskew
axis-order bug did (see test_zarr_deskew_geometry.py next door).

These also pin the `rl_method` guard. PetaKit5D's RLdecon.m dispatches
RLMethod through switches with no `otherwise` branch over
{original, simplified, omw, cudagen}, and initializes `deconvolved = []`, so
an unrecognized name writes an EMPTY volume rather than erroring. Our
historical default, 'simple', was exactly such a name.

Pure Python: no MATLAB, no GPU, no real data (unlike the gpu-marked
test_deskew_only_regression.py next door), so these run in a normal
`pytest tests/` pass.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from opym.petakit import _normalize_rl_method, submit_remote_deskew_job
from opym.utils import orient_zyx_for_decon_tiff, orient_zyx_for_dsr

# The production PSF. Skipped rather than failed when absent, so the suite
# still runs off the cluster.
PSF_PATH = Path("/mmfs2/scratch/SDSMT.LOCAL/bscott/DataUpload/PSF/20260910_averaged_psf.tif")


# --------------------------------------------------------------------------
# Staging orientation
# --------------------------------------------------------------------------


def test_staged_tiff_page_order_is_nz_ny_nx():
    """A staged frame must be written so `readtiff` hands MATLAB (ny, nx, nz).

    `tifffile.imwrite` of a numpy (a, b, c) array writes `a` pages of b x c,
    and MATLAB's `readtiff` returns (rows, cols, pages) == (b, c, a). Three
    distinct extents so a wrong permutation cannot coincidentally pass.
    """
    z, y, x = 7, 5, 11  # (scan Z, tilted camera axis, coverslip axis)
    vol = np.arange(z * y * x).reshape(z, y, x)

    staged = orient_zyx_for_decon_tiff(vol)
    assert staged.shape == (z, x, y)

    matlab_view = (staged.shape[1], staged.shape[2], staged.shape[0])
    assert matlab_view == (x, y, z), "MATLAB must see (ny=coverslip, nx=tilted, nz=scan)"


def test_staged_matches_orient_zyx_for_dsr():
    """The two helpers must never drift apart.

    `orient_zyx_for_decon_tiff` is deliberately `orient_zyx_for_dsr` minus
    its trailing moveaxis -- the step TIFF paging performs for you.
    """
    vol = np.arange(7 * 5 * 11).reshape(7, 5, 11)
    assert np.array_equal(
        np.moveaxis(orient_zyx_for_decon_tiff(vol), 0, -1),
        orient_zyx_for_dsr(vol),
    )


def test_rot90_and_zxy_differ_by_an_ny_flip():
    """The staging convention is a MIRROR of the deskew-only zarr convention.

    The no-decon zarr path reaches PetaKit5D via `inputAxisOrder='zxy'`,
    which is a pure transpose. Staging instead applies rot90, matching the
    legacy OME-TIFF cropper -- and, critically, matching the rot90 the
    measured PSF itself carries (psf_tools/extract_bead_psf.py). The two
    differ by a flip of the coverslip axis, so DSR_decon output is a lateral
    mirror of DSR_nodecon output. That is a deliberate, documented choice;
    this test exists so it can never become an accident.
    """
    vol = np.arange(7 * 5 * 11).reshape(7, 5, 11)
    zxy = vol.transpose(2, 1, 0)
    dsr = orient_zyx_for_dsr(vol)

    assert dsr.shape == zxy.shape
    assert np.array_equal(dsr, np.flip(zxy, axis=0))
    assert not np.array_equal(dsr, zxy)


# --------------------------------------------------------------------------
# PSF orientation
# --------------------------------------------------------------------------


@pytest.mark.skipif(not PSF_PATH.is_file(), reason=f"{PSF_PATH} not available")
def test_psf_matlab_orientation():
    """The PSF's 3rd MATLAB dimension must be the scan axis.

    Both `psf_gen_new` (background from `psf(:,:,[1:5,end-4:end])`, and it
    FFT-resamples dim 3 from dz_psf to dz_data) and
    `omw_backprojector_generation` with `skewed=true` (OTF mask built as
    `cat(3, mask_r, mask_c, mask_l)` -- the skewed OTF's three lobes
    separate along dim 3) hard-code that assumption.
    """
    psf = tifffile.imread(PSF_PATH)
    assert psf.ndim == 3
    n_pages, n_rows, n_cols = psf.shape
    matlab_shape = (n_rows, n_cols, n_pages)
    assert matlab_shape[2] == n_pages
    assert matlab_shape[2] > matlab_shape[0] and matlab_shape[2] > matlab_shape[1], (
        "dim 3 should be the long scan axis (81 planes at dz=0.1 um), not a lateral one"
    )


@pytest.mark.skipif(not PSF_PATH.is_file(), reason=f"{PSF_PATH} not available")
def test_psf_is_in_skewed_space():
    """The PSF must be a SKEWED-space PSF, since decon runs before deskew.

    Its signature is that the blob is sheared along the tilted axis -- which
    after extract_bead_psf's rot90 is numpy axis 2 (MATLAB dim 2, the axis
    PetaKit5D shears) -- so that axis carries visibly more extent than the
    coverslip axis. A deskewed/upright PSF here would silently under-correct.
    """
    psf = tifffile.imread(PSF_PATH).astype(np.float64)
    psf = np.clip(psf - np.percentile(psf, 1), 0, None)

    def extent(axis: int) -> int:
        prof = psf.sum(axis=tuple(i for i in range(3) if i != axis))
        prof = prof / prof.max()
        above = np.flatnonzero(prof >= 0.5)
        return int(above[-1] - above[0] + 1)

    coverslip, tilted = extent(1), extent(2)
    assert tilted > coverslip, (
        f"tilted axis extent {tilted} should exceed coverslip axis extent "
        f"{coverslip} for a skewed-space PSF"
    )


# --------------------------------------------------------------------------
# rl_method guard -- the empty-volume bug
# --------------------------------------------------------------------------


def test_simple_is_normalized_not_passed_through():
    assert _normalize_rl_method("simple") == "simplified"
    assert _normalize_rl_method("SIMPLE") == "simplified"


@pytest.mark.parametrize("method", ["original", "simplified", "omw", "cudagen"])
def test_methods_petakit_implements_survive(method):
    assert _normalize_rl_method(method) == method


@pytest.mark.parametrize("method", ["simpl", "richardson", "rl", "wiener", ""])
def test_unknown_method_raises_instead_of_writing_an_empty_volume(method):
    with pytest.raises(ValueError, match="EMPTY"):
        _normalize_rl_method(method)


# --------------------------------------------------------------------------
# Decon ticket contract
# --------------------------------------------------------------------------


def _decon_ticket(tmp_path: Path, **kwargs) -> dict:
    import json

    data_dir = tmp_path / "decon_stage"
    data_dir.mkdir(exist_ok=True)
    (data_dir / "ds_C0_T000.tif").touch()
    psf = tmp_path / "psf.tif"
    psf.touch()
    params = {
        "input_target": data_dir,
        "z_step_um": 0.1,
        "psf_path": psf,
        "channel_patterns": ["_C0_T"],
        "dsr_dir_name": "DSR_decon",
        "queue_dir": tmp_path / "queue",
        "zarr_input": False,
    }
    params.update(kwargs)
    return json.loads(submit_remote_deskew_job(**params).read_text())


def test_decon_ticket_declares_yxz_and_omw(tmp_path):
    """Staged input is already (ny, nx, nz), so no permute must be requested.

    `zarr_input=False` is what makes `input_axis_order` derive to 'yxz'; it
    also makes the MATLAB server's `val_deskewZarrInput` false, so the
    post-decon deskew reads decon's TIFF output as TIFF.
    """
    params = _decon_ticket(tmp_path)["parameters"]
    assert params["run_decon"] is True
    assert params["input_axis_order"] == "yxz"
    assert params["zarr_input"] is False
    assert params["dsr_dir_name"] == "DSR_decon"
    assert params["rl_method"] == "omw"
    assert params["decon_iter"] == 2, "omw's default is 2 iterations, not 25"


def test_no_psf_means_no_decon(tmp_path):
    """The entire decon switch is the presence of a PSF path."""
    params = _decon_ticket(tmp_path, psf_path=None)["parameters"]
    assert "run_decon" not in params
    assert "psf_path" not in params


def test_decon_ticket_rejects_a_bad_rl_method(tmp_path):
    with pytest.raises(ValueError, match="EMPTY"):
        _decon_ticket(tmp_path, rl_method="simpl")


def test_per_channel_psfs_must_match_channel_count(tmp_path):
    psf = tmp_path / "psf.tif"
    psf.touch()
    with pytest.raises(ValueError, match="one-to-one"):
        _decon_ticket(
            tmp_path,
            psf_path=None,
            psf_paths=[psf],
            channel_patterns=["_C0_T", "_C1_T"],
        )


def test_ticket_records_the_submitting_revision(tmp_path):
    """Provenance against the stale-MATLAB-process trap: opym-serve loads
    run_petakit_server.m once, so a restart-less code change leaves a stale
    server interpreting new tickets."""
    assert _decon_ticket(tmp_path)["submitterRev"]
