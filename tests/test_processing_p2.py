# Ruff style: Compliant
"""Processing P2s (docs/review/2026-11-review.md PROC-F3, F4, F9, F10, F11, F12).

CPU only: tickets are written to a tmp queue and read back; MATLAB and the GPU
are never involved.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from opym import petakit

# --------------------------------------------------------------------------
# PROC-F3: a ticket is never visible half-written
# --------------------------------------------------------------------------


def test_ticket_is_published_atomically(tmp_path, monkeypatch):
    """The server claims the oldest non-dot `*.json`; at the moment the final
    name appears it must already hold the whole ticket."""
    queue = tmp_path / "queue"
    queue.mkdir()
    seen = {}
    real_replace = os.replace

    def spy(src, dst):
        src, dst = Path(src), Path(dst)
        seen["src_name"] = src.name
        seen["dst_existed"] = dst.exists()
        seen["visible_json"] = [
            p.name for p in queue.glob("*.json") if not p.name.startswith(".")
        ]
        seen["body"] = json.loads(src.read_text())
        real_replace(src, dst)

    monkeypatch.setattr(petakit.os, "replace", spy)
    job = petakit._write_ticket({"jobType": "x", "n": 1}, "name", "LIVE", queue)

    assert seen["src_name"].startswith(".") and seen["src_name"].endswith(".tmp")
    assert not seen["dst_existed"]
    assert seen["visible_json"] == [], "the ticket was claimable before it was whole"
    assert seen["body"]["n"] == 1
    assert json.loads(job.read_text())["jobType"] == "x"
    assert list(queue.iterdir()) == [job], "no temp file left behind"


def test_failed_ticket_write_leaves_no_file(tmp_path):
    queue = tmp_path / "queue"
    queue.mkdir()
    with pytest.raises(TypeError):
        petakit._write_ticket({"bad": object()}, "name", "LIVE", queue)
    assert list(queue.iterdir()) == []


# --------------------------------------------------------------------------
# PROC-F4: the PSF z step is read from the PSF and sent as dz_psf
# --------------------------------------------------------------------------


def _write_psf(path: Path, spacing: float | None) -> Path:
    import numpy as np
    import tifffile

    meta = {"spacing": spacing, "unit": "um"} if spacing is not None else None
    tifffile.imwrite(
        path, np.ones((5, 4, 4), np.float32), imagej=meta is not None, metadata=meta
    )
    return path


def _deskew_params(tmp_path: Path, psf, **kwargs) -> dict:
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    (data / "ds_C0_T000.tif").touch()
    return json.loads(
        petakit.submit_remote_deskew_job(
            input_target=data,
            z_step_um=0.5,
            psf_path=psf,
            channel_patterns=["_C0_T"],
            queue_dir=tmp_path / "queue",
            **kwargs,
        ).read_text()
    )["parameters"]


@pytest.mark.parametrize("spacing", [0.1, 0.25])
def test_deskew_ticket_carries_the_psf_spacing(tmp_path, spacing):
    psf = _write_psf(tmp_path / "psf.tif", spacing)
    assert _deskew_params(tmp_path, psf)["dz_psf"] == spacing


def test_explicit_dz_psf_wins_over_the_tag(tmp_path):
    psf = _write_psf(tmp_path / "psf.tif", 0.25)
    assert _deskew_params(tmp_path, psf, dz_psf=0.2)["dz_psf"] == 0.2


def test_untagged_psf_falls_back_to_the_legacy_step_with_a_warning(tmp_path, caplog):
    """No tag: the ticket says 0.1, exactly what MATLAB assumed before."""
    psf = _write_psf(tmp_path / "psf.tif", None)
    with caplog.at_level("WARNING", logger="opym.petakit"):
        assert _deskew_params(tmp_path, psf)["dz_psf"] == petakit.PSF_DZ_DEFAULT_UM
    assert "spacing" in caplog.text


def test_deskew_only_ticket_has_no_dz_psf(tmp_path):
    assert "dz_psf" not in _deskew_params(tmp_path, None)


def test_psfs_that_disagree_on_dz_are_refused(tmp_path):
    a = _write_psf(tmp_path / "a.tif", 0.1)
    b = _write_psf(tmp_path / "b.tif", 0.2)
    with pytest.raises(ValueError, match="disagree"):
        petakit.resolve_psf_dz([a, b])
    assert petakit.resolve_psf_dz([a, _write_psf(tmp_path / "c.tif", 0.1)]) == 0.1


@pytest.mark.parametrize("bad", [0, -0.1, float("nan")])
def test_nonsense_dz_psf_is_refused(bad):
    with pytest.raises(ValueError, match="positive"):
        petakit.resolve_psf_dz([], bad)


def test_live_tickets_carry_dz_psf(tmp_path):
    psf = _write_psf(tmp_path / "psf.tif", 0.25)
    zarr_params = petakit.live_zarr_parameters(
        tmp_path / "raw.zarr",
        0,
        0,
        mask_store=tmp_path / "raw.zarr",
        levels=[tmp_path / "l0"],
        mip=tmp_path / "mip",
        psf_path=psf,
        decon_dir=tmp_path / "decon",
        z_step_um=0.5,
    )
    assert zarr_params["dz_psf"] == 0.25
    ticket = petakit.submit_live_frames_job(
        [tmp_path / "f.tif"],
        tmp_path / "decon",
        tmp_path / "f.tif",
        [psf],
        ["_C0_T"],
        0.5,
        ticket_name="t",
        queue_dir=tmp_path / "q",
    )
    assert json.loads(ticket.read_text())["parameters"]["dz_psf"] == 0.25
