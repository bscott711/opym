"""opym.stream.live_status: the live lane's hand-off names and readers, and
their old import paths."""

from __future__ import annotations

import json
import time

from opym.stream import live, live_status, live_zarr


def test_names_are_the_strings_consumers_read():
    assert live_status.LIVE_LATEST_NAME == "live_latest.json"
    assert live_status.LIVE_STATUS_NAME == ".live_status.json"
    assert live_status.QC_LATEST_NAME == "qc_latest.json"
    assert live_status.QC_LOG_NAME == "live_qc.jsonl"
    assert live_status.RAWPROJ_DIR == "rawproj"
    assert live_status.LIVE_STATUS_STALE_S == 300.0
    assert live_status.HEARTBEAT_S == 30.0
    assert live_status.sidecar_name("Cell_001", 1, 7) == "Cell_001_C1_T007.npz"


def test_buffer_names_round_trip():
    assert live_status.buffer_name(3, 1) == "T0003_C1.npy"
    assert live_status.parse_buffer_name("T0003_C1.npy") == (3, 1)
    for bad in ("T0003_C1.npz", "X0003_C1.npy", "T00x3_C1.npy", "display_C0.json"):
        assert live_status.parse_buffer_name(bad) is None


def test_status_freshness():
    assert live_status.live_status_is_fresh({"updated_at": time.time()})
    assert not live_status.live_status_is_fresh({"updated_at": time.time() - 3600})
    assert live_status.live_status_is_fresh({"updated_at": 1000.0}, now=1299.0)
    assert not live_status.live_status_is_fresh({"updated_at": 1000.0}, now=1301.0)
    assert not live_status.live_status_is_fresh({})


def test_read_live_status(tmp_path):
    assert live_status.read_live_status(tmp_path) is None
    (tmp_path / ".live_status.json").write_text(json.dumps({"state": "running"}))
    assert live_status.read_live_status(tmp_path) == {"state": "running"}
    (tmp_path / ".live_status.json").write_text("{not json")
    assert live_status.read_live_status(tmp_path) is None


def test_read_live_latest_takes_the_file_or_the_jobs_dir(tmp_path):
    assert live_status.read_live_latest(tmp_path) is None
    rec = {"session_id": "s", "store": "/x"}
    (tmp_path / "live_latest.json").write_text(json.dumps(rec))
    assert live_status.read_live_latest(tmp_path) == rec
    assert live_status.read_live_latest(tmp_path / "live_latest.json") == rec
    (tmp_path / "live_latest.json").write_text("[1]")
    assert live_status.read_live_latest(tmp_path) is None


def test_key_sets():
    assert set(live_status.LIVE_LATEST_KEYS) <= set(live_status.LIVE_LATEST_ZARR_KEYS)
    assert {"view_store", "buffers_dir"} == set(
        live_status.LIVE_LATEST_ZARR_KEYS
    ) - set(live_status.LIVE_LATEST_KEYS)


def test_old_import_paths_are_the_same_objects():
    for name in (
        "LIVE_STATUS_NAME",
        "LIVE_LATEST_NAME",
        "LIVE_STATUS_STALE_S",
        "HEARTBEAT_S",
        "QC_LATEST_NAME",
        "read_live_status",
        "live_status_is_fresh",
    ):
        assert getattr(live, name) is getattr(live_status, name)
    assert live_zarr.buffer_name is live_status.buffer_name
    assert live_zarr.parse_buffer_name is live_status.parse_buffer_name
    assert live_zarr.LIVE_LATEST_NAME is live_status.LIVE_LATEST_NAME
    from opym import live_view

    assert live_view.LIVE_LATEST_NAME is live_status.LIVE_LATEST_NAME
    assert live_view.QC_LOG_NAME is live_status.QC_LOG_NAME
    assert live_view.buffer_name is live_status.buffer_name


def _session(**extra):
    from types import SimpleNamespace

    return SimpleNamespace(
        session_id="s1",
        base_name="Cell_001",
        zarr_path="/z",
        archive_store="/a",
        view_store="/v",
        buffers_dir="/b",
        dsr_dir="/d",
        started_at=1.0,
        qc_dir="/q",
        num_timepoints=3,
        n_channels=2,
        channel_labels=["a", "b"],
        z_step_um=0.5,
        **extra,
    )


def test_the_writers_write_exactly_the_documented_keys(tmp_path):
    from types import SimpleNamespace

    lane = SimpleNamespace(jobs=tmp_path / "t", qc=True)
    lane.jobs.mkdir()
    live.LiveLane._write_latest(lane, _session())
    assert set(live_status.read_live_latest(lane.jobs)) == set(
        live_status.LIVE_LATEST_KEYS
    )
    zlane = SimpleNamespace(jobs=tmp_path / "z", qc=True)
    zlane.jobs.mkdir()
    live_zarr.ZarrLiveLane._write_latest(zlane, _session())
    assert set(live_status.read_live_latest(zlane.jobs)) == set(
        live_status.LIVE_LATEST_ZARR_KEYS
    )
