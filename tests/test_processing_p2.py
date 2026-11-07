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


# --------------------------------------------------------------------------
# PROC-F12: the full decon fingerprint covers every knob and the PSF
# --------------------------------------------------------------------------


def _current_ticket_params(tmp_path: Path, psf: Path, **override) -> dict:
    from opym.decon_config import deskew_decon_kwargs

    kwargs = deskew_decon_kwargs(psf)
    kwargs.update(override)
    return _deskew_params(tmp_path, psf, **kwargs)


def test_classic_fingerprint_is_unchanged_by_default(tmp_path, monkeypatch):
    """Every recorded registry row and .opym_live.json carries this string;
    the default must not move or the archive looks stale."""
    from opym import decon_config as dc

    monkeypatch.delenv(dc.FULL_FINGERPRINT_ENV, raising=False)
    assert dc.decon_params_fingerprint() == "a0.2_o0.9_h0.4-1.0_d2"
    psf = _write_psf(tmp_path / "psf.tif", 0.1)
    params = _current_ticket_params(tmp_path, psf)
    assert dc.ticket_decon_fingerprint(params) == "a0.2_o0.9_h0.4-1.0_d2"


def test_full_fingerprint_of_a_current_ticket_matches_the_current_settings(
    tmp_path,
):
    from opym import decon_config as dc

    psf = _write_psf(tmp_path / "psf.tif", 0.1)
    params = _current_ticket_params(tmp_path, psf)
    now = dc.decon_params_fingerprint(psf, full=True)
    assert dc.ticket_decon_fingerprint(params, full=True) == now
    assert now.startswith("a0.2_o0.9_h0.4-1.0_d2_e3_romw_i2_bx_mlinear_p")


@pytest.mark.parametrize(
    "override",
    [
        {"edge_erosion": 0},
        {"n_iters": 5},
        {"background": 90.0},
        {"interp_method": "cubic"},
        {"rl_method": "simplified"},
    ],
)
def test_each_missing_knob_now_changes_the_full_fingerprint(tmp_path, override):
    from opym import decon_config as dc

    psf = _write_psf(tmp_path / "psf.tif", 0.1)
    now = dc.decon_params_fingerprint(psf, full=True)
    params = _current_ticket_params(tmp_path, psf, **override)
    assert dc.ticket_decon_fingerprint(params, full=True) != now
    # ... and the classic fingerprint is blind to it (the finding).
    assert dc.ticket_decon_fingerprint(params, full=False) == (
        dc.decon_params_fingerprint(full=False)
    )


def test_editing_the_psf_in_place_changes_the_full_fingerprint(tmp_path):
    from opym import decon_config as dc

    psf = _write_psf(tmp_path / "psf.tif", 0.1)
    before = dc.decon_params_fingerprint(psf, full=True)
    _write_psf(psf, 0.1)
    os.utime(psf, (1_000_000_000, 1_000_000_000))
    assert dc.decon_params_fingerprint(psf, full=True) != before
    assert dc.psf_stamp(tmp_path / "missing.tif") == "?"


def test_env_switch_turns_the_full_fingerprint_on(tmp_path, monkeypatch):
    from opym import decon_config as dc

    psf = _write_psf(tmp_path / "psf.tif", 0.1)
    monkeypatch.setenv("OPYM_DECON_PSF", str(psf))
    monkeypatch.setenv(dc.FULL_FINGERPRINT_ENV, "1")
    assert dc.decon_params_fingerprint() == dc.decon_params_fingerprint(psf, full=True)
    assert dc.decon_params_fingerprint().endswith(dc.psf_stamp(psf))
    monkeypatch.setenv(dc.FULL_FINGERPRINT_ENV, "0")
    assert dc.decon_params_fingerprint() == "a0.2_o0.9_h0.4-1.0_d2"


def test_old_ticket_without_the_extra_fields_never_matches_the_full_one(tmp_path):
    from opym import decon_config as dc

    legacy = {
        "run_decon": True,
        "psf_path": "/x/psf.tif",
        "wiener_alpha": 0.2,
        "otf_cum_thresh": 0.9,
        "hann_win_bounds": [0.4, 1.0],
        "damp_factor": 2.0,
    }
    assert dc.ticket_decon_fingerprint(legacy, full=False) == "a0.2_o0.9_h0.4-1.0_d2"
    assert "?" in dc.ticket_decon_fingerprint(legacy, full=True)


# --------------------------------------------------------------------------
# PROC-F9: a complete processed store is not wiped by a new session
# --------------------------------------------------------------------------


def _store_kw():
    return {"n_t": 1, "n_c": 1, "shape_zyx": (4, 8, 8)}


def _complete_store(path: Path):
    import numpy as np

    from opym import ome_zarr_writer as w

    arrays = w.create_processed_store(path, **_store_kw())
    w.write_timepoint(path, 0, 0, np.full((4, 8, 8), 7, np.uint16))
    w.write_progress(path, n_t=1, n_c=1, done=[[0, 0]], state="complete")
    return arrays


def test_create_processed_store_refuses_a_complete_store(tmp_path):
    from opym import ome_zarr_writer as w

    store = tmp_path / "Cell_dsr.ome.zarr"
    _complete_store(store)
    with pytest.raises(w.ProcessedStoreComplete):
        w.create_processed_store(store, **_store_kw())
    # nothing was wiped
    assert w.read_progress(store)["state"] == "complete"
    assert (w.image_group(store)["0"][0, 0, 0] == 7).all()


def test_create_processed_store_move_aside_keeps_the_old_one(tmp_path):
    from opym import ome_zarr_writer as w

    store = tmp_path / "Cell_dsr.ome.zarr"
    _complete_store(store)
    w.create_processed_store(store, on_complete="move_aside", **_store_kw())
    assert w.read_progress(store)["state"] == "running"
    (aside,) = tmp_path.glob("Cell_dsr.ome.zarr.superseded-*")
    assert w.read_progress(aside)["state"] == "complete"
    assert (w.image_group(aside)["0"][0, 0, 0] == 7).all()


def test_create_processed_store_overwrite_still_available(tmp_path):
    from opym import ome_zarr_writer as w

    store = tmp_path / "Cell_dsr.ome.zarr"
    _complete_store(store)
    w.create_processed_store(store, on_complete="overwrite", **_store_kw())
    assert w.read_progress(store)["state"] == "running"
    assert not list(tmp_path.glob("*.superseded-*"))


@pytest.mark.parametrize("state", ["running", "failed"])
def test_unfinished_store_is_recreated_as_before(tmp_path, state):
    from opym import ome_zarr_writer as w

    store = tmp_path / "Cell_dsr.ome.zarr"
    w.create_processed_store(store, **_store_kw())
    w.write_progress(store, n_t=1, n_c=1, done=[], state=state)
    w.create_processed_store(store, **_store_kw())
    assert w.read_progress(store)["state"] == "running"


def test_legacy_create_store_is_guarded_too(tmp_path):
    import numpy as np

    from opym import ome_zarr_writer as w

    store = tmp_path / "legacy.ome.zarr"
    kw = {**_store_kw(), "dtype": np.uint16}
    w.create_store(store, **kw)
    w.write_progress(store, n_t=1, n_c=1, done=[[0, 0]], state="complete")
    with pytest.raises(w.ProcessedStoreComplete):
        w.create_store(store, **kw)
    w.create_store(store, on_complete="overwrite", **kw)


def test_bad_on_complete_value_is_rejected(tmp_path):
    from opym import ome_zarr_writer as w

    with pytest.raises(ValueError, match="on_complete"):
        w.create_processed_store(
            tmp_path / "x.ome.zarr", on_complete="x", **_store_kw()
        )
