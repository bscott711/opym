"""Tests for live QC plumbing: the raw-projection sidecars the live lane
writes (opym.stream.qcproj) and the receiver forwarding QC verdicts to
clients that asked for them (MSG_QC)."""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from opym.stream import live as live_mod
from opym.stream import qcproj
from opym.stream.live import LIVE_LATEST_NAME, QC_LATEST_NAME, LiveLane


def _raw(shape=(5, 6, 9), seed=0):
    return np.random.default_rng(seed).integers(0, 4000, shape, dtype=np.uint16)


def test_projections_are_the_documented_reductions():
    raw = _raw()
    raw[3] += 1000  # the brightest scan plane
    p = qcproj.raw_projections(raw)
    np.testing.assert_array_equal(p["mip_scan"], raw.max(axis=0))
    np.testing.assert_array_equal(p["mip_tilted"], raw.max(axis=1))
    np.testing.assert_array_equal(p["mip_cover"], raw.max(axis=2))
    np.testing.assert_allclose(p["profile_scan"], raw.sum(axis=(1, 2)))
    assert int(p["focus_idx"]) == 3
    np.testing.assert_array_equal(p["focus_plane"], raw[3])
    # Odd tilted/cover sizes: the last row/column is dropped before binning.
    s = raw.sum(axis=0, dtype=np.float64)[:6, :8]
    np.testing.assert_allclose(
        p["sum_scan_b2"], s.reshape(3, 2, 4, 2).sum(axis=(1, 3)), rtol=1e-6
    )
    assert tuple(p["shape"]) == raw.shape


def test_sidecar_round_trips_and_lands_atomically(tmp_path):
    raw = _raw()
    dst = tmp_path / "qc" / "rawproj" / qcproj.sidecar_name("Cell_1", 1, 7)
    qcproj.write_sidecar(raw, dst, t=7, c=1, z_step_um=0.5)
    assert dst.name == "Cell_1_C1_T007.npz"
    assert [p.name for p in dst.parent.iterdir()] == [dst.name]
    d = qcproj.read_sidecar(dst)
    assert int(d["format_version"]) == qcproj.FORMAT_VERSION
    assert (int(d["t"]), int(d["c"]), float(d["z_step_um"])) == (7, 1, 0.5)
    np.testing.assert_array_equal(d["mip_scan"], raw.max(axis=0))


@pytest.fixture
def psf(tmp_path):
    p = tmp_path / "psf.tif"
    p.write_bytes(b"psf")
    return p


def _lane_session(tmp_path, psf, qc):
    jobs = tmp_path / "jobs"
    for d in ("queue_live", "completed", "failed"):
        (jobs / d).mkdir(parents=True, exist_ok=True)
    lane = LiveLane(psf, jobs=jobs, qc=qc)
    stage_leaf = tmp_path / "stage" / "Cell_002"
    (stage_leaf / "decon_stage").mkdir(parents=True)
    s = lane.start_session(
        "sess",
        base_name="Cell_002",
        num_timepoints=4,
        n_channels=2,
        frames_dir=stage_leaf / "decon_stage",
        stage_leaf=stage_leaf,
        dest_leaf=tmp_path / "gpfs" / "Cell_002",
        z_step_um=0.5,
    )
    return lane, s, jobs


def test_lane_writes_a_sidecar_per_staged_frame_next_to_the_viewer_store(tmp_path, psf):
    lane, s, jobs = _lane_session(tmp_path, psf, qc=True)
    try:
        assert s.qc_dir == tmp_path / "gpfs" / "Cell_002" / "qc"
        assert s.qc_dir.parent == s.zarr_path.parent.parent
        for c in (0, 1):
            s.frame(0, c).write_bytes(b"staged")
            lane.frame_staged("sess", 0, c, raw=_raw(seed=c))
        # A duplicate frame is not projected twice.
        lane.frame_staged("sess", 0, 1, raw=_raw(seed=9))
        lane._qc_pool.shutdown(wait=True)
        out = sorted(p.name for p in (s.qc_dir / "rawproj").iterdir())
        assert out == ["Cell_002_C0_T000.npz", "Cell_002_C1_T000.npz"]
        d = qcproj.read_sidecar(s.qc_dir / "rawproj" / "Cell_002_C1_T000.npz")
        np.testing.assert_array_equal(d["mip_scan"], _raw(seed=1).max(axis=0))
        latest = json.loads((jobs / LIVE_LATEST_NAME).read_text())
        assert latest["qc_dir"] == str(s.qc_dir)
        assert latest["num_timepoints"] == 4
        assert latest["n_channels"] == 2
        assert lane.qc_latest_path("sess") == s.qc_dir / QC_LATEST_NAME
    finally:
        lane.close()


def test_lane_writes_no_sidecars_with_qc_off(tmp_path, psf):
    lane, s, jobs = _lane_session(tmp_path, psf, qc=False)
    try:
        s.frame(0, 0).write_bytes(b"staged")
        lane.frame_staged("sess", 0, 0, raw=_raw())
        lane._qc_pool.shutdown(wait=True)
        assert not s.qc_dir.exists()
        assert json.loads((jobs / LIVE_LATEST_NAME).read_text())["qc_dir"] is None
    finally:
        lane.close()


def test_backlogged_projection_pool_skips_frames_instead_of_queueing(
    tmp_path, psf, monkeypatch
):
    lane, s, _ = _lane_session(tmp_path, psf, qc=True)
    monkeypatch.setattr(live_mod, "MAX_QC_PENDING", 0)
    try:
        lane.frame_staged("sess", 0, 0, raw=_raw())
        lane._qc_pool.shutdown(wait=True)
        assert not (s.qc_dir / "rawproj").exists()
    finally:
        lane.close()


# --- receiver: MSG_QC forwarding ------------------------------------------


def _connect(recv, sid):
    import zmq

    endpoint = recv._socket.getsockopt(zmq.LAST_ENDPOINT).decode()
    sock = zmq.Context.instance().socket(zmq.DEALER)
    sock.setsockopt(zmq.IDENTITY, sid.encode())
    sock.connect(endpoint)
    return sock


def _recv_all(sock, timeout_ms=300):
    from opym.stream.protocol import unpack_message

    out = []
    while sock.poll(timeout_ms):
        out.append(unpack_message(sock.recv_multipart()))
    return out


def test_receiver_forwards_new_qc_verdicts_only_to_clients_that_accept_them(
    tmp_path, monkeypatch, psf
):
    from opym.stream import receiver as receiver_mod
    from opym.stream.protocol import MSG_QC, MSG_SESSION_START, pack_message

    monkeypatch.setenv("OPYM_LIVE_LANE", "1")
    monkeypatch.setenv("OPYM_LIVE_QC", "1")
    monkeypatch.setenv("OPYM_DECON_PSF", str(psf))
    monkeypatch.setenv("OPYM_STREAM_STAGE_ROOT", str(tmp_path / "stage"))
    monkeypatch.setattr(receiver_mod, "QC_CHECK_SEC", 0.0)
    recv = receiver_mod.StreamReceiver(
        bind_addr="tcp://127.0.0.1:0", ack_every_n_frames=1, ack_every_sec=9999
    )
    socks = {}
    try:
        for sid, base, accepts in (
            ("s-qc", "Cell_010", ["qc"]),
            ("s-old", "Cell_011", None),
        ):
            header = {
                "base_name": base,
                "raw_root": str(tmp_path / "raw"),
                "dtype": "uint16",
                "shape_zyx": [3, 5, 7],
                "num_timepoints": 3,
                "channels": [0],
                "channel_names": ["GFP_488"],
                "z_step_um": 0.5,
            }
            if accepts:
                header["accepts"] = accepts
            socks[sid] = _connect(recv, sid)
            socks[sid].send_multipart(pack_message(MSG_SESSION_START, sid, header))
            recv._run_once()
            _recv_all(socks[sid])  # the SESSION_START ACK
        assert recv.sessions["s-qc"].accepts_qc
        assert not recv.sessions["s-old"].accepts_qc

        # A streamed frame gets its raw-projection sidecar.
        from opym.stream.protocol import MSG_FRAME

        vol = _raw((3, 5, 7))
        fh = {
            "t": 0,
            "c": 0,
            "frame_index": 0,
            "timestamp": 0.0,
            "camera_id": 0,
            "shape_zyx": [3, 5, 7],
            "dtype": "uint16",
        }
        socks["s-qc"].send_multipart(pack_message(MSG_FRAME, "s-qc", fh, vol.tobytes()))
        recv._run_once()
        _recv_all(socks["s-qc"])
        side = recv._live.sessions["s-qc"].qc_dir / "rawproj" / "Cell_010_C0_T000.npz"
        deadline = time.time() + 5
        while not side.exists() and time.time() < deadline:
            time.sleep(0.02)
        np.testing.assert_array_equal(qcproj.read_sidecar(side)["mip_scan"], vol.max(0))

        def write_verdict(sid, seq):
            path = recv._live.qc_latest_path(sid)
            path.parent.mkdir(parents=True, exist_ok=True)
            rec = {"seq": seq, "session_id": sid, "t": seq, "verdict": "warn"}
            tmp = path.with_name(".tmp")
            tmp.write_text(json.dumps(rec))
            import os

            os.replace(tmp, path)
            # A distinct mtime per write, as on a real clock.
            os.utime(path, (time.time() + seq, time.time() + seq))

        for sid in socks:
            write_verdict(sid, 0)
        recv._run_once()
        got = _recv_all(socks["s-qc"])
        assert [(m[0], m[2]["seq"], m[2]["verdict"]) for m in got] == [
            (MSG_QC, 0, "warn")
        ]
        assert _recv_all(socks["s-old"]) == []

        # Unchanged file: nothing resent. New seq: sent once.
        recv._run_once()
        assert _recv_all(socks["s-qc"]) == []
        write_verdict("s-qc", 1)
        recv._run_once()
        recv._run_once()
        assert [m[2]["seq"] for m in _recv_all(socks["s-qc"])] == [1]
    finally:
        for sock in socks.values():
            sock.close(linger=0)
        recv.close()
