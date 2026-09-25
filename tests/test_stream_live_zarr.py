"""Tests for the one-format live lane (opym.stream.live_zarr). A fake GPU
server stands in for MATLAB: for each 'live_zarr' ticket it writes what
run_live_zarr.m would -- the view buffer, then every pyramid level and the
MIP into the processed store -- and moves the ticket to completed/."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
import zarr

from opym import ome_zarr_writer as w
from opym.stream import live_zarr, trace
from opym.stream.live import read_live_status
from opym.stream.live_zarr import BUFFER_KEEP_T, ZarrLiveLane
from opym.utils import dsr_shape_zyx

RAW_ZYX = (6, 16, 12)  # small, but through the real DSR shape formula


@pytest.fixture
def psf(tmp_path):
    p = tmp_path / "psf.tif"
    p.write_bytes(b"psf")
    return p


@pytest.fixture(autouse=True)
def _view_root(tmp_path, monkeypatch):
    monkeypatch.setenv(live_zarr.VIEW_ROOT_ENV_VAR, str(tmp_path / "shm_view"))


def _lane(tmp_path, psf):
    jobs = tmp_path / "jobs"
    for d in ("queue_live", "completed", "failed"):
        (jobs / d).mkdir(parents=True, exist_ok=True)
    return ZarrLiveLane(psf, jobs=jobs, qc=False), jobs


def _raw(tmp_path, n_t, n_c):
    arrays = []
    for c in range(n_c):
        a = zarr.open(
            str(tmp_path / "stage" / f"Cell_030_C{c}.ome.zarr" / "p0"),
            mode="w",
            shape=(n_t, *RAW_ZYX),
            chunks=(1, 1, *RAW_ZYX[1:]),
            dtype="uint16",
            dimension_separator="/",
        )
        arrays.append(Path(a.store.path))
    return arrays


def _session(lane, tmp_path, n_t=4, n_c=2, sid="sess"):
    return lane.start_session(
        sid,
        base_name="Cell_030",
        num_timepoints=n_t,
        n_channels=n_c,
        raw_arrays=_raw(tmp_path, n_t, n_c),
        raw_shape_zyx=RAW_ZYX,
        stage_leaf=tmp_path / "stage" / "Cell_030",
        dest_leaf=tmp_path / "gpfs" / "exp" / "Cell_030",
        z_step_um=0.5,
        channel_labels=["GFP 488", "mScarlet 561"][:n_c],
        time_interval_s=10.0,
    )


def _tickets(jobs):
    return [
        (p, json.loads(p.read_text()))
        for p in sorted((jobs / "queue_live").glob("*.json"))
    ]


def _volume(t, c, shape):
    return np.full(shape, 100 * (t + 1) + c, dtype=np.uint16)


def _complete(jobs, path, ticket, shape):
    """What run_live_zarr.m leaves behind for one (t, c)."""
    p = ticket["parameters"]
    t, c = p["t"], p["c"]
    vol = _volume(t, c, shape)
    np.save(p["view_npy"], vol)
    for lvl in p["levels"]:
        arr = zarr.open(lvl, mode="r+")
        arr[t, c] = vol[tuple(slice(0, n) for n in arr.shape[2:])]
        vol = w.downsample2(vol)
    zarr.open(p["mip"], mode="r+")[t, c, 0] = _volume(t, c, shape).max(0)
    os.replace(path, jobs / "completed" / path.name)


def _pump_until(lane, pred, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        lane.pump()
        if pred():
            return True
        time.sleep(0.02)
    return pred()


def _stage_all(lane, s, n_t, n_c):
    for t in range(n_t):
        for c in range(n_c):
            lane.frame_staged(s.session_id, t, c)


def test_start_creates_both_stores_at_the_dsr_shape(tmp_path, psf):
    lane, jobs = _lane(tmp_path, psf)
    s = _session(lane, tmp_path)
    shape = dsr_shape_zyx(RAW_ZYX, 0.5)
    for store in (s.view_store, s.archive_store):
        assert w.image_group(store)["0"].shape == (4, 2, *shape)
        assert (store / "OME" / "METADATA.ome.xml").exists()
    assert s.view_store.is_relative_to(tmp_path / "shm_view")
    assert (
        s.archive_store == tmp_path / "gpfs/exp/Cell_030/viewer/Cell_030_dsr.ome.zarr"
    )
    latest = json.loads((jobs / "live_latest.json").read_text())
    assert latest["view_store"] == str(s.view_store)
    assert latest["store"] == str(s.archive_store)
    assert read_live_status(s.dsr_dir)["state"] == "running"


def test_nothing_is_dispatched_before_channel_0_timepoint_0(tmp_path, psf):
    lane, jobs = _lane(tmp_path, psf)
    s = _session(lane, tmp_path)
    lane.frame_staged("sess", 0, 1)
    lane.frame_staged("sess", 1, 0)
    lane.pump()
    assert _tickets(jobs) == []
    lane.frame_staged("sess", 0, 0)
    lane.pump()
    got = sorted(
        (tk["parameters"]["t"], tk["parameters"]["c"]) for _, tk in _tickets(jobs)
    )
    assert got == [(0, 0), (0, 1), (1, 0)]
    _, tk = _tickets(jobs)[0]
    p = tk["parameters"]
    assert tk["jobType"] == "live_zarr"
    assert p["raw_store"] == str(s.raw_arrays[p["c"]])
    assert p["mask_store"] == str(s.raw_arrays[0])
    assert p["levels"] == [str(x) for x in s.view_arrays.levels]
    assert p["view_npy"] == str(s.buffer(p["t"], p["c"]))


def test_one_ticket_per_channel_and_the_in_flight_cap(tmp_path, psf):
    lane, jobs = _lane(tmp_path, psf)
    _session(lane, tmp_path, n_t=4)
    _stage_all(lane, lane.sessions["sess"], 4, 2)
    lane.pump()
    assert len(_tickets(jobs)) == lane.max_outstanding == 4
    got = [(tk["parameters"]["t"], tk["parameters"]["c"]) for _, tk in _tickets(jobs)]
    assert sorted(got) == [(0, 0), (0, 1), (1, 0), (1, 1)]  # oldest first


def test_finished_volumes_are_viewable_then_archived_unchanged(tmp_path, psf):
    lane, jobs = _lane(tmp_path, psf)
    s = _session(lane, tmp_path, n_t=2)
    shape = dsr_shape_zyx(RAW_ZYX, 0.5)
    _stage_all(lane, s, 2, 2)
    lane.pump()
    for path, tk in _tickets(jobs):
        _complete(jobs, path, tk, shape)
    assert _pump_until(lane, lambda: s.done == {0, 1})

    assert w.complete_timepoints(w.read_progress(s.view_store)) == [0, 1]
    assert w.complete_timepoints(w.read_progress(s.archive_store)) == [0, 1]
    view = w.image_group(s.view_store)
    archive = w.image_group(s.archive_store)
    for lvl in ("0", "1", "2"):
        np.testing.assert_array_equal(archive[lvl][:], view[lvl][:])
    np.testing.assert_array_equal(
        w.image_group(s.archive_store, "1")["0"][:],
        w.image_group(s.view_store, "1")["0"][:],
    )
    np.testing.assert_array_equal(archive["0"][1, 1], _volume(1, 1, shape))
    assert not list(s.archive_store.rglob(".*.copying"))
    assert read_live_status(s.dsr_dir)["timepoints_done"] == [0, 1]
    events = [e["ev"] for e in trace.read(jobs=jobs)]
    assert events.count("view_ready") == 2 and events.count("view_buffer") == 4


def test_only_the_newest_timepoints_keep_uncompressed_buffers(tmp_path, psf):
    lane, jobs = _lane(tmp_path, psf)
    n_t = BUFFER_KEEP_T + 2
    s = _session(lane, tmp_path, n_t=n_t, n_c=1)
    lane.max_outstanding = 99
    shape = dsr_shape_zyx(RAW_ZYX, 0.5)
    _stage_all(lane, s, n_t, 1)
    lane.pump()
    for path, tk in _tickets(jobs):
        _complete(jobs, path, tk, shape)
    assert _pump_until(lane, lambda: len(s.done) == n_t)
    kept = sorted(int(p.name[1:5]) for p in s.buffers_dir.glob("*.npy"))
    assert kept == list(range(n_t - BUFFER_KEEP_T, n_t))


def test_a_failed_ticket_is_retried_once_then_left_to_the_backfill(tmp_path, psf):
    lane, jobs = _lane(tmp_path, psf)
    s = _session(lane, tmp_path, n_t=1, n_c=1)
    lane.frame_staged("sess", 0, 0)
    lane.pump()
    for attempt in (1, 2):
        [(path, _)] = _tickets(jobs)
        os.replace(path, jobs / "failed" / path.name)
        lane.pump()
    assert _tickets(jobs) == [] and s.failed == {0}
    lane.end_session("sess")
    assert _pump_until(lane, lambda: "sess" not in lane.sessions)
    assert read_live_status(s.dsr_dir)["state"] == "failed"


def test_finalize_keeps_the_view_until_a_later_session(tmp_path, psf):
    lane, jobs = _lane(tmp_path, psf)
    s = _session(lane, tmp_path, n_t=1, n_c=1)
    shape = dsr_shape_zyx(RAW_ZYX, 0.5)
    lane.frame_staged("sess", 0, 0)
    lane.pump()
    [(path, tk)] = _tickets(jobs)
    _complete(jobs, path, tk, shape)
    lane.end_session("sess")
    assert _pump_until(lane, lambda: "sess" not in lane.sessions)
    assert read_live_status(s.dsr_dir)["state"] == "complete"
    assert w.read_progress(s.archive_store)["state"] == "complete"
    assert not s.work_dir.exists() and s.view_dir.exists()

    # Two later sessions: the newest finished one's view is kept, older go.
    s2 = _session(lane, tmp_path, n_t=1, n_c=1, sid="sess2")
    assert s.view_dir.exists()
    lane.sessions.pop("sess2")
    _session(lane, tmp_path, n_t=1, n_c=1, sid="sess3")
    assert s2.view_dir.exists() and not s.view_dir.exists()


def test_receiver_uses_the_zarr_lane_and_stages_no_tiff(tmp_path, psf, monkeypatch):
    import zmq

    from opym.stream.protocol import MSG_FRAME, MSG_SESSION_START, pack_message
    from opym.stream.receiver import StreamReceiver

    monkeypatch.setenv("OPYM_LIVE_LANE", "1")
    monkeypatch.setenv("OPYM_LIVE_FORMAT", "zarr")
    monkeypatch.setenv("OPYM_DECON_PSF", str(psf))
    recv = StreamReceiver(bind_addr="tcp://127.0.0.1:0", ack_every_n_frames=1)
    try:
        sock = zmq.Context.instance().socket(zmq.DEALER)
        sock.setsockopt(zmq.IDENTITY, b"sess-rz")
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(recv._socket.getsockopt(zmq.LAST_ENDPOINT).decode())
        header = {
            "base_name": "Cell_031",
            "raw_root": str(tmp_path / "raw"),
            "dtype": "uint16",
            "shape_zyx": list(RAW_ZYX),
            "num_timepoints": 2,
            "channels": [0, 1],
            "channel_names": ["GFP_488", "mScarlet_561"],
            "z_step_um": 0.5,
            "t_interval_s": 10.0,
        }
        sock.send_multipart(pack_message(MSG_SESSION_START, "sess-rz", header))
        vol = np.ones(RAW_ZYX, np.uint16)
        for fi, c in enumerate((0, 1)):
            frame = {
                "t": 0,
                "c": c,
                "frame_index": fi,
                "timestamp": 0.0,
                "camera_id": 0,
                "shape_zyx": list(RAW_ZYX),
                "dtype": "uint16",
            }
            sock.send_multipart(
                pack_message(MSG_FRAME, "sess-rz", frame, vol.tobytes())
            )
        for _ in range(20):
            recv._run_once()
        assert isinstance(recv._live, ZarrLiveLane)
        s = recv._live.sessions["sess-rz"]
        assert s.staged == {0: {0, 1}}
        assert [p.name for p in s.raw_arrays] == ["p0", "p0"]
        assert s.raw_arrays[0].parent.name == "Cell_031_GFP_488.ome.zarr"
        assert not list((tmp_path / "raw").rglob("*.tif"))  # no decon TIFFs
        tickets = list(recv._live.jobs.joinpath("queue_live").glob("*.json"))
        assert len(tickets) == 2
        sock.close()
    finally:
        recv.close()
