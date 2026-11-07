# Ruff style: Compliant
"""Receiver hardening (docs/ROADMAP.md item 2; review ING-F2, F3, F4, F5, LIVE-02):
what the stream receiver does with a partial volume, a bad message, an unwritable
destination or an unsafe name, driven over a real loopback ZMQ DEALER."""

from __future__ import annotations

import os
import time

import numpy as np
import pytest
import zarr
import zmq

from opym.stream import outcome
from opym.stream.protocol import (
    MSG_ACK,
    MSG_FRAME,
    MSG_HEARTBEAT,
    MSG_SESSION_END,
    MSG_SESSION_START,
    pack_message,
    unpack_message,
)
from opym.stream.receiver import StreamReceiver

SHAPE_ZYX = (6, 4, 5)
SID = "sess-hard"


def _header(raw_root, num_timepoints=3, **extra):
    return {
        "base_name": "Cell_040",
        "raw_root": str(raw_root),
        "dtype": "uint16",
        "shape_zyx": list(SHAPE_ZYX),
        "num_timepoints": num_timepoints,
        "channels": [0],
        "channel_names": ["GFP_488"],
        "z_step_um": 0.5,
        **extra,
    }


def _volume(t):
    return (np.arange(np.prod(SHAPE_ZYX), dtype=np.uint16) + 1000 * t).reshape(
        SHAPE_ZYX
    )


def _slab(t, z0, planes, frame_index, **extra):
    header = {
        "t": t,
        "c": 0,
        "frame_index": frame_index,
        "timestamp": 0.0,
        "camera_id": 0,
        "dtype": "uint16",
        "z0": z0,
        "nz": SHAPE_ZYX[0],
        "shape_zyx": [planes, *SHAPE_ZYX[1:]],
        **extra,
    }
    return header, _volume(t)[z0 : z0 + planes].tobytes()


def _frame(t, frame_index, **extra):
    header = {
        "t": t,
        "c": 0,
        "frame_index": frame_index,
        "timestamp": 0.0,
        "camera_id": 0,
        "shape_zyx": list(SHAPE_ZYX),
        "dtype": "uint16",
        **extra,
    }
    return header, _volume(t).tobytes()


def _store(root, name="Cell_040"):
    return root / f"{name}_GFP_488.ome.zarr"


def _timepoint_dirs(root, name="Cell_040"):
    """The numbered directories under p0: what channel_store_timepoints counts."""
    p0 = _store(root, name) / "p0"
    return sorted(int(p.name) for p in p0.iterdir() if p.name.isdigit())


@pytest.fixture
def make_recv():
    made = []

    def _make(idle_timeout_sec=9999):
        r = StreamReceiver(
            bind_addr="tcp://127.0.0.1:0",
            ack_every_n_frames=1,
            ack_every_sec=9999,  # deterministic: only the inline per-frame ack fires
            idle_timeout_sec=idle_timeout_sec,
        )
        made.append(r)
        return r

    yield _make
    for r in made:
        r.close()


@pytest.fixture
def recv(make_recv):
    return make_recv()


@pytest.fixture
def dealer(recv):
    return _dealer_factory(recv)


def _dealer_factory(recv):
    def _connect(identity=SID):
        sock = zmq.Context.instance().socket(zmq.DEALER)
        sock.setsockopt(zmq.IDENTITY, identity.encode())
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(recv._socket.getsockopt(zmq.LAST_ENDPOINT).decode())
        return sock

    return _connect


def _drive(recv, seconds=0.2):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        recv._run_once()


def _pump(recv, sock, timeout=2.0):
    """Drive the receiver until `sock` gets a message; its header or None."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        recv._run_once()
        if sock.poll(10):
            msg_type, _sid, header, _ = unpack_message(sock.recv_multipart())
            assert msg_type == MSG_ACK
            return header
    return None


def _start(recv, sock, header, sid=SID):
    sock.send_multipart(pack_message(MSG_SESSION_START, sid, header))
    return _pump(recv, sock)


def _send(recv, sock, msg_type, header, payload=None, sid=SID):
    sock.send_multipart(pack_message(msg_type, sid, header, payload))
    _drive(recv, 0.15)


# --- ING-F2: partial volumes are discarded at finalize -------------------------


@pytest.mark.parametrize("reason", ["complete", "client_abort", "idle_timeout"])
def test_a_volume_in_flight_at_the_end_is_discarded(tmp_path, recv, dealer, reason):
    """t=0 complete, t=1 has 2 of 6 planes when the session ends. Before: p0/1
    stayed on disk, so the backfill counted two timepoints and processed t=1 with
    four planes of zeros."""
    raw = tmp_path / "raw"
    sock = dealer()
    assert _start(recv, sock, _header(raw)) is not None
    for fi, (t, z0, n) in enumerate([(0, 0, 3), (0, 3, 3), (1, 0, 2)]):
        _send(recv, sock, MSG_FRAME, *_slab(t, z0, n, fi))
    assert _timepoint_dirs(raw) == [0, 1]  # the premise: the partial plane is there

    _send(recv, sock, MSG_SESSION_END, {"reason": reason})

    assert SID not in recv.sessions
    assert _timepoint_dirs(raw) == [0]
    arr = np.asarray(zarr.open(str(_store(raw) / "p0"), mode="r")[0])
    np.testing.assert_array_equal(arr, _volume(0))


def test_a_volume_in_flight_is_discarded_when_the_session_closes_as_idle(
    tmp_path, make_recv
):
    recv = make_recv(idle_timeout_sec=0.3)
    raw = tmp_path / "raw"
    sock = _dealer_factory(recv)()
    assert _start(recv, sock, _header(raw)) is not None
    _send(recv, sock, MSG_FRAME, *_slab(0, 0, 2, 0))
    assert _timepoint_dirs(raw) == [0]
    _drive(recv, 0.8)
    assert SID not in recv.sessions
    assert _timepoint_dirs(raw) == []
    sock.close()


def test_a_completed_volume_survives_the_end(tmp_path, recv, dealer):
    raw = tmp_path / "raw"
    sock = dealer()
    assert _start(recv, sock, _header(raw, num_timepoints=1)) is not None
    _send(recv, sock, MSG_FRAME, *_slab(0, 0, 3, 0))
    _send(recv, sock, MSG_FRAME, *_slab(0, 3, 3, 1))
    _send(recv, sock, MSG_SESSION_END, {"reason": "complete"})
    assert _timepoint_dirs(raw) == [0]


# --- ING-F3: one bad message cannot take the receiver down ---------------------


def _still_serving(recv, dealer, tmp_path, sid="sess-after"):
    """The receiver answers a fresh, valid session: proof it survived."""
    sock = dealer(sid)
    ack = _start(recv, sock, _header(tmp_path / "good", base_name="Good"), sid=sid)
    assert ack is not None and sid in recv.sessions
    sock.close()


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root ignores directory permissions",
)
def test_session_start_with_an_unreadable_raw_root_is_rejected_not_fatal(
    tmp_path, recv, dealer
):
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0)
    try:
        sock = dealer()
        ack = _start(recv, sock, _header(locked / "raw"))
        assert ack is not None
        assert ack["unknown_session"] and ack["rejected"]
        assert SID not in recv.sessions
        _still_serving(recv, dealer, tmp_path)
    finally:
        locked.chmod(0o755)


def test_a_frame_without_t_is_dropped_and_counted(tmp_path, recv, dealer):
    sock = dealer()
    assert _start(recv, sock, _header(tmp_path / "raw")) is not None
    header, payload = _frame(0, 0)
    del header["t"]
    _send(recv, sock, MSG_FRAME, header, payload)
    assert recv.errors == {"bad_frame": 1}
    _still_serving(recv, dealer, tmp_path)


@pytest.mark.parametrize(
    "bad",
    [{"t": -1}, {"t": 3}, {"t": 1.0}, {"t": True}, {"c": 7}, {"frame_index": -1}],
)
def test_a_frame_outside_the_declared_grid_is_dropped(tmp_path, recv, dealer, bad):
    """t=-1 used to index the LAST timepoint of the array."""
    raw = tmp_path / "raw"
    sock = dealer()
    assert _start(recv, sock, _header(raw)) is not None
    header, payload = _frame(0, 0)
    header.update(bad)
    _send(recv, sock, MSG_FRAME, header, payload)
    assert recv.errors == {"bad_frame": 1}
    assert not _store(raw).exists()
    assert recv.sessions[SID].received_pairs == set()


@pytest.mark.parametrize(
    "bad",
    [
        {"shape_zyx": [3, 4, 9]},  # payload is 3 x 4 x 5
        {"shape_zyx": [3, 4]},
        {"shape_zyx": [0, 4, 5]},
        {"z0": 5},  # 5 + 3 planes > nz=6
        {"z0": -1},
        {"nz": 10**9},
        {"nz": 2},
        {"dtype": "not-a-dtype"},
    ],
)
def test_a_slab_that_does_not_fit_is_dropped_before_anything_is_allocated(
    tmp_path, recv, dealer, bad
):
    raw = tmp_path / "raw"
    sock = dealer()
    assert _start(recv, sock, _header(raw)) is not None
    header, payload = _slab(0, 0, 3, 0)
    header.update(bad)
    _send(recv, sock, MSG_FRAME, header, payload)
    assert recv.errors == {"bad_slab": 1}
    assert recv.sessions[SID].slabs == {}
    assert not _store(raw).exists()


def test_a_slab_that_disagrees_with_its_volume_is_dropped(tmp_path, recv, dealer):
    sock = dealer()
    assert _start(recv, sock, _header(tmp_path / "raw")) is not None
    _send(recv, sock, MSG_FRAME, *_slab(0, 0, 3, 0))
    header, payload = _slab(0, 3, 3, 1, nz=8)
    _send(recv, sock, MSG_FRAME, header, payload)
    assert recv.errors == {"bad_slab": 1}
    assert recv.sessions[SID].slabs[(0, 0)].z0s == {0}


@pytest.mark.parametrize(
    "bad",
    [
        {"num_timepoints": 0},
        {"num_timepoints": "3"},
        {"shape_zyx": [6, 4]},
        {"shape_zyx": [6, 4, 10**9]},
        {"dtype": "nonsense"},
        {"channels": [0, 0], "channel_names": ["A", "B"]},
    ],
)
def test_a_session_start_with_unusable_numbers_is_refused_without_a_crash(
    tmp_path, recv, dealer, bad
):
    sock = dealer()
    sock.send_multipart(
        pack_message(MSG_SESSION_START, SID, _header(tmp_path / "raw", **bad))
    )
    _drive(recv)
    assert SID not in recv.sessions
    _still_serving(recv, dealer, tmp_path)


def test_an_unexpected_handler_error_is_logged_counted_and_survived(
    tmp_path, recv, dealer, monkeypatch
):
    """The catch-all: whatever a handler raises, the poll loop carries on and
    the other open session keeps its state."""
    other = dealer("sess-other")
    assert _start(recv, other, _header(tmp_path / "o", base_name="O"), "sess-other")

    def _boom(*_a, **_k):
        raise RuntimeError("handler bug")

    monkeypatch.setattr(recv, "_handle_heartbeat", _boom)
    sock = dealer()
    assert _start(recv, sock, _header(tmp_path / "raw")) is not None
    sock.send_multipart(pack_message(MSG_HEARTBEAT, SID, {}))
    _drive(recv)

    assert recv.errors == {"HEARTBEAT_failed": 1}
    assert {SID, "sess-other"} <= set(recv.sessions)
    _still_serving(recv, dealer, tmp_path)


def test_a_session_start_that_fails_unexpectedly_is_answered_rejected(
    tmp_path, recv, dealer, monkeypatch
):
    def _boom(*_a, **_k):
        raise RuntimeError("lane exploded")

    monkeypatch.setattr(recv, "_send_ack", _boom)
    sock = dealer()
    ack = _start(recv, sock, _header(tmp_path / "raw"))
    assert ack["unknown_session"] and "lane exploded" in ack["rejected"]
    assert recv.errors == {"SESSION_START_failed": 1}


def test_the_health_record_carries_the_error_counts(tmp_path, recv, dealer):
    sock = dealer()
    assert _start(recv, sock, _header(tmp_path / "raw")) is not None
    header, payload = _frame(0, 0)
    header["t"] = -1
    _send(recv, sock, MSG_FRAME, header, payload)
    recv._write_health()
    assert outcome.read_health()["errors"] == {"bad_frame": 1}


# --- LIVE-02: a failing live lane cannot kill the receiver ---------------------


class _BrokenLane:
    """A live lane whose every call raises, as one with a full tmpfs does."""

    sessions: dict = {}

    def pump(self):
        raise OSError(28, "No space left on device")

    def frame_staged(self, *_a, **_k):
        raise OSError(28, "No space left on device")

    def end_session(self, *_a, **_k):
        raise OSError(28, "No space left on device")

    def qc_latest_path(self, _sid):
        raise OSError(28, "No space left on device")

    def close(self):
        pass


def test_a_pump_that_raises_does_not_end_the_poll_loop(tmp_path, recv, dealer, caplog):
    sock = dealer()
    assert _start(recv, sock, _header(tmp_path / "raw")) is not None
    recv._live = _BrokenLane()
    caplog.clear()
    for _ in range(30):
        recv._run_once()
    assert recv.errors["live_pump_failed"] == 30
    # counted every time, logged with its traceback once per minute
    assert len([r for r in caplog.records if r.exc_info]) == 1
    assert SID in recv.sessions


def test_the_other_steps_still_run_when_the_pump_raises(tmp_path, make_recv, dealer):
    recv = make_recv(idle_timeout_sec=0.2)
    sock = _dealer_factory(recv)()
    assert _start(recv, sock, _header(tmp_path / "raw")) is not None
    recv._live = _BrokenLane()
    _drive(recv, 0.6)
    assert SID not in recv.sessions  # the idle sweep ran after the failing pump
    assert recv.errors["live_pump_failed"] > 0


def test_a_lane_that_raises_on_a_frame_costs_the_view_not_the_frame(
    tmp_path, recv, dealer
):
    raw = tmp_path / "raw"
    sock = dealer()
    assert _start(recv, sock, _header(raw, num_timepoints=1)) is not None
    session = recv.sessions[SID]
    recv._live = _BrokenLane()
    session.live = session.live_zarr = session.decon_enabled = True
    _send(recv, sock, MSG_FRAME, *_frame(0, 0))
    assert session.received_pairs == {(0, 0)}  # written, counted, ACKed
    assert recv.errors["live_frame_staged_failed"] == 1
    assert session.ack_floor == 0

    _send(recv, sock, MSG_SESSION_END, {"reason": "complete"})
    assert SID not in recv.sessions  # end_session raised too; the end completed
    assert recv.errors["live_end_session_failed"] == 1
