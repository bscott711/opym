"""Tests for the live lane (opym.stream.live): dispatch, batching, copy-out,
retry, finalize and the .live_status.json hand-off. A fake server stands in
for MATLAB: it moves a ticket to completed/ and writes the DSR + MIP files
run_live_frames.m would."""

from __future__ import annotations

import json
import os
import time

import pytest

from opym import lanes
from opym.stream import live as live_mod
from opym.stream.live import LIVE_STATUS_NAME, LiveLane, read_live_status


@pytest.fixture
def psf(tmp_path):
    p = tmp_path / "psf.tif"
    p.write_bytes(b"psf")
    return p


def _lane(tmp_path, psf, **kw):
    jobs = tmp_path / "jobs"
    for d in ("queue_live", "completed", "failed"):
        (jobs / d).mkdir(parents=True, exist_ok=True)
    return LiveLane(psf, jobs=jobs, **kw), jobs


def _session(lane, tmp_path, n_t=4, n_c=2, base="Cell_002"):
    stage_leaf = tmp_path / "stage" / base
    frames_dir = stage_leaf / "decon_stage"
    frames_dir.mkdir(parents=True)
    s = lane.start_session(
        "sess",
        base_name=base,
        num_timepoints=n_t,
        n_channels=n_c,
        frames_dir=frames_dir,
        stage_leaf=stage_leaf,
        dest_leaf=tmp_path / "gpfs" / "session_dir" / base,
        z_step_um=0.5,
    )
    return s


def _stage(lane, s, t, c):
    s.frame(t, c).write_bytes(b"staged")
    lane.frame_staged(s.session_id, t, c)


def _tickets(jobs):
    out = []
    for p in sorted((jobs / "queue_live").glob("LIVE_*.json")):
        out.append((p, json.loads(p.read_text())))
    return out


def _complete(jobs, s, ticket_path, psf_dir="DSR_decon"):
    """What the MATLAB server + run_live_frames.m leave behind."""
    ticket = json.loads(ticket_path.read_text())
    dsr = s.decon_dir / psf_dir
    (dsr / "MIPs").mkdir(parents=True, exist_ok=True)
    for f in ticket["parameters"]["frames"]:
        stem = os.path.basename(f)[:-4]
        (dsr / f"{stem}.tif").write_bytes(b"dsr " + stem.encode())
        (dsr / "MIPs" / f"{stem}_MIP_z.tif").write_bytes(b"mip")
    os.replace(ticket_path, jobs / "completed" / ticket_path.name)


def _pump_until(lane, pred, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        lane.pump()
        if pred():
            return True
        time.sleep(0.02)
    return pred()


def test_gpfs_leaf_is_created_so_output_resolves_next_to_the_raw_data(tmp_path, psf):
    lane, _ = _lane(tmp_path, psf)
    s = _session(lane, tmp_path)
    dest = tmp_path / "gpfs" / "session_dir" / "Cell_002"
    assert s.dsr_dir == dest / "decon_stage" / "Decon" / "DSR_decon"
    assert read_live_status(s.dsr_dir)["state"] == "running"


def test_only_complete_timepoints_are_dispatched(tmp_path, psf):
    lane, jobs = _lane(tmp_path, psf)
    s = _session(lane, tmp_path)
    _stage(lane, s, 0, 0)
    lane.pump()
    assert _tickets(jobs) == []
    _stage(lane, s, 0, 1)
    lane.pump()
    [(_, ticket)] = _tickets(jobs)
    p = ticket["parameters"]
    assert ticket["jobType"] == "live"
    assert [os.path.basename(f) for f in p["frames"]] == [
        "Cell_002_C0_T000.tif",
        "Cell_002_C1_T000.tif",
    ]
    assert p["channel_patterns"] == ["_C0_T", "_C1_T"]
    assert p["erode_mask_source"].endswith("Cell_002_C0_T000.tif")
    assert p["decon_dir"] == str(s.decon_dir)
    # Same parameters the backfill uses (opym.decon_config).
    assert p["interp_method"] == "linear" and p["wiener_alpha"] == 0.20
    assert p["edge_erosion"] == 3 and p["gpu_decon"] is True
    assert p["z_step_um"] == 0.5


def test_backlog_is_batched_into_the_free_slots(tmp_path, psf):
    lane, jobs = _lane(tmp_path, psf, max_outstanding=2)
    s = _session(lane, tmp_path, n_t=6)
    for t in range(6):
        for c in range(2):
            _stage(lane, s, t, c)
    lane.pump()
    tickets = _tickets(jobs)
    assert len(tickets) == 2
    per_ticket = [len(t["parameters"]["frames"]) // 2 for _, t in tickets]
    assert per_ticket == [3, 3]


def test_completed_ticket_is_copied_to_gpfs_and_freed_from_ram(tmp_path, psf):
    lane, jobs = _lane(tmp_path, psf)
    s = _session(lane, tmp_path)
    for t in (0, 1):
        for c in range(2):
            _stage(lane, s, t, c)
    lane.pump()
    for path, _ in _tickets(jobs):
        _complete(jobs, s, path)
    assert _pump_until(lane, lambda: s.done == {0, 1})

    for t in (0, 1):
        for c in range(2):
            name = f"Cell_002_C{c}_T{t:03d}"
            assert (s.dsr_dir / f"{name}.tif").read_bytes() == b"dsr " + name.encode()
            assert (s.dsr_dir / "MIPs" / f"{name}_MIP_z.tif").exists()
            assert not (s.decon_dir / "DSR_decon" / f"{name}.tif").exists()
    # Staged input is freed, except the erosion-mask source (first C0 frame).
    assert s.frame(0, 0).exists()
    assert not s.frame(0, 1).exists() and not s.frame(1, 0).exists()
    assert not list(s.dsr_dir.glob(".*.copying"))
    assert read_live_status(s.dsr_dir)["timepoints_done"] == [0, 1]


def test_failed_ticket_is_retried_once_then_left_to_the_backfill(tmp_path, psf):
    lane, jobs = _lane(tmp_path, psf)
    s = _session(lane, tmp_path, n_t=1)
    for c in range(2):
        _stage(lane, s, 0, c)
    lane.end_session("sess")
    lane.pump()
    for attempt in range(2):
        [(path, _)] = _tickets(jobs)
        os.replace(path, jobs / "failed" / path.name)
        lane.pump()
    assert "sess" not in lane.sessions  # finalized
    status = read_live_status(s.dsr_dir)
    assert status["state"] == "failed" and status["timepoints_failed"] == [0]


def test_session_finalizes_complete_with_provenance(tmp_path, psf):
    lane, jobs = _lane(tmp_path, psf)
    s = _session(lane, tmp_path, n_t=2)
    for t in range(2):
        for c in range(2):
            _stage(lane, s, t, c)
    lane.pump()
    for path, _ in _tickets(jobs):
        _complete(jobs, s, path)
    lane.end_session("sess")
    assert _pump_until(lane, lambda: "sess" not in lane.sessions)

    status = json.loads((s.dsr_dir / LIVE_STATUS_NAME).read_text())
    assert status["state"] == "complete"
    assert status["timepoints_done"] == [0, 1]
    assert status["decon_psf"] == str(psf)
    assert status["decon_params"] == "a0.2_o0.9_h0.4-1.0_d2"
    assert status["interp_method"] == "linear"
    assert not s.work_dir.exists()
    assert not s.frames_dir.exists()
    assert not s.work_dir.parent.exists()  # the RAM stage leaf is gone too
    assert not lane.busy()


def test_partial_last_timepoint_does_not_block_completion(tmp_path, psf):
    """An aborted acquisition can end with only some channels of its last
    timepoint; that timepoint is not processed (the backfill's own
    dataset_timepoints drops it too) and the session still completes."""
    lane, jobs = _lane(tmp_path, psf)
    s = _session(lane, tmp_path, n_t=3)
    for c in range(2):
        _stage(lane, s, 0, c)
    _stage(lane, s, 1, 0)
    lane.pump()
    [(path, _)] = _tickets(jobs)
    _complete(jobs, s, path)
    lane.end_session("sess")
    assert _pump_until(lane, lambda: "sess" not in lane.sessions)
    assert read_live_status(s.dsr_dir)["state"] == "complete"


def test_status_freshness(tmp_path):
    assert live_mod.live_status_is_fresh({"updated_at": time.time()})
    assert not live_mod.live_status_is_fresh({"updated_at": time.time() - 3600})


def test_receiver_hands_time_lapse_sessions_to_the_live_lane(
    tmp_path, monkeypatch, psf
):
    """End to end through the real receiver: a 2T x 2C session with the live
    lane on queues live tickets, holds the lease, and does not drain the
    staged TIFFs the lane consumes."""
    import zmq

    from opym.stream.protocol import MSG_FRAME, MSG_SESSION_START, pack_message
    from opym.stream.receiver import StreamReceiver

    monkeypatch.setenv("OPYM_LIVE_LANE", "1")
    monkeypatch.setenv("OPYM_DECON_PSF", str(psf))
    monkeypatch.setenv("OPYM_STREAM_STAGE_ROOT", str(tmp_path / "stage"))
    recv = StreamReceiver(bind_addr="tcp://127.0.0.1:0", ack_every_n_frames=1)
    try:
        endpoint = recv._socket.getsockopt(zmq.LAST_ENDPOINT).decode()
        sock = zmq.Context.instance().socket(zmq.DEALER)
        sock.setsockopt(zmq.IDENTITY, b"sess-live")
        sock.connect(endpoint)
        header = {
            "base_name": "Cell_009",
            "raw_root": str(tmp_path / "raw"),
            "dtype": "uint16",
            "shape_zyx": [3, 5, 7],
            "num_timepoints": 2,
            "channels": [0, 1],
            "channel_names": ["GFP_488", "mScarlet_561"],
            "z_step_um": 0.5,
        }
        sock.send_multipart(pack_message(MSG_SESSION_START, "sess-live", header))
        recv._run_once()
        sock.recv_multipart()
        assert recv.sessions["sess-live"].live
        import numpy as np

        for i, (t, c) in enumerate([(0, 0), (0, 1)]):
            vol = np.full((3, 5, 7), 7, dtype=np.uint16)
            fh = {
                "t": t,
                "c": c,
                "frame_index": i,
                "timestamp": 0.0,
                "camera_id": 0,
                "shape_zyx": [3, 5, 7],
                "dtype": "uint16",
            }
            sock.send_multipart(pack_message(MSG_FRAME, "sess-live", fh, vol.tobytes()))
            recv._run_once()
            sock.recv_multipart()
        recv._run_once()
        tickets = list(lanes.live_queue_dir().glob("LIVE_Cell_009_*.json"))
        assert len(tickets) == 1
        assert lanes.live_lease_active()
        sock.close(linger=0)
    finally:
        recv.close()
