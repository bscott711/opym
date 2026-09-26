"""Tests for streaming one session over several links, resuming a session
the receiver no longer knows, and runs the client paused."""

from __future__ import annotations

import time

import numpy as np
import pytest
import zarr
import zmq

from opym.stream.protocol import (
    MSG_ACK,
    MSG_FRAME,
    MSG_RESUME,
    MSG_SESSION_END,
    MSG_SESSION_START,
    pack_message,
    unpack_message,
)
from opym.stream.receiver import StreamReceiver, _identity_belongs

SHAPE_ZYX = (6, 4, 5)


def _header(raw_root, num_timepoints=3, **extra):
    return {
        "base_name": "Cell_030",
        "raw_root": str(raw_root),
        "dtype": "uint16",
        "shape_zyx": list(SHAPE_ZYX),
        "num_timepoints": num_timepoints,
        "channels": [0],
        "channel_names": ["GFP_488"],
        "z_step_um": 0.5,
        **extra,
    }


def _receiver():
    return StreamReceiver(
        bind_addr="tcp://127.0.0.1:0",
        ack_every_n_frames=1,
        ack_every_sec=9999,
        idle_timeout_sec=9999,
    )


def _dealer(recv, identity):
    sock = zmq.Context.instance().socket(zmq.DEALER)
    sock.setsockopt(zmq.IDENTITY, identity.encode())
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(recv._socket.getsockopt(zmq.LAST_ENDPOINT).decode())
    return sock


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


def _drive(recv, seconds=0.3):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        recv._run_once()


def _volume(t):
    return (np.arange(np.prod(SHAPE_ZYX), dtype=np.uint16) + 1000 * t).reshape(
        SHAPE_ZYX
    )


def _frame(t, frame_index):
    header = {
        "t": t,
        "c": 0,
        "frame_index": frame_index,
        "timestamp": 0.0,
        "camera_id": 0,
        "shape_zyx": list(SHAPE_ZYX),
        "dtype": "uint16",
    }
    return header, _volume(t).tobytes()


def _slab(t, z0, planes, frame_index):
    vol = _volume(t)
    header = {
        **_frame(t, frame_index)[0],
        "z0": z0,
        "nz": SHAPE_ZYX[0],
        "shape_zyx": [planes, *SHAPE_ZYX[1:]],
    }
    return header, vol[z0 : z0 + planes].tobytes()


def _store(root):
    return root / "Cell_030_GFP_488.ome.zarr"


def _read(root, t):
    return np.asarray(zarr.open(str(_store(root) / "p0"), mode="r")[t])


def _wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


@pytest.fixture
def recv():
    r = _receiver()
    yield r
    r.close()


def test_link_identities():
    assert _identity_belongs(b"abc", "abc")
    assert _identity_belongs(b"abc#1", "abc")
    assert _identity_belongs(b"abc#12", "abc")
    assert not _identity_belongs(b"abd#1", "abc")
    assert not _identity_belongs(b"abc#", "abc")
    assert not _identity_belongs(b"abc#x", "abc")
    assert not _identity_belongs(b"abc#1#2", "abc")


def test_slabs_spread_over_links_in_any_order_complete_the_volume(tmp_path, recv):
    link0 = _dealer(recv, "sess-links")
    link0.send_multipart(
        pack_message(MSG_SESSION_START, "sess-links", _header(tmp_path / "raw"))
    )
    ack = _pump(recv, link0)
    assert "links" in ack["features"] and "resume" in ack["features"]

    link1 = _dealer(recv, "sess-links#1")
    link2 = _dealer(recv, "sess-links#2")
    # Three slabs of t=0 on three links, the last one first.
    for sock, (z0, n, fi) in (
        (link2, (4, 2, 2)),
        (link0, (0, 2, 0)),
        (link1, (2, 2, 1)),
    ):
        header, payload = _slab(0, z0, n, fi)
        sock.send_multipart(pack_message(MSG_FRAME, "sess-links", header, payload))
        ack = _pump(recv, sock)  # each ACK goes back on the link heard from
        assert ack is not None
    assert ack["through_frame_index"] == 2
    np.testing.assert_array_equal(_read(tmp_path / "raw", 0), _volume(0))

    # A duplicate resent on another link after completion changes nothing.
    header, payload = _slab(0, 2, 2, 1)
    link2.send_multipart(pack_message(MSG_FRAME, "sess-links", header, payload))
    assert _pump(recv, link2)["through_frame_index"] == 2
    for sock in (link0, link1, link2):
        sock.close()


def test_a_link_identity_of_another_session_is_dropped(tmp_path, recv):
    link0 = _dealer(recv, "sess-a")
    link0.send_multipart(
        pack_message(MSG_SESSION_START, "sess-a", _header(tmp_path / "raw"))
    )
    assert _pump(recv, link0) is not None
    stranger = _dealer(recv, "sess-b#1")
    header, payload = _frame(0, 0)
    stranger.send_multipart(pack_message(MSG_FRAME, "sess-a", header, payload))
    assert _pump(recv, stranger, timeout=0.3) is None
    assert not recv.sessions["sess-a"].received_pairs
    link0.close()
    stranger.close()


def test_a_frame_for_an_unknown_session_asks_to_resume_but_not_every_frame(
    tmp_path, recv
):
    sock = _dealer(recv, "sess-lost#1")
    header, payload = _frame(0, 0)
    sock.send_multipart(pack_message(MSG_FRAME, "sess-lost", header, payload))
    ack = _pump(recv, sock)
    assert ack["unknown_session"] is True and ack["through_frame_index"] == -1
    sock.send_multipart(pack_message(MSG_FRAME, "sess-lost", header, payload))
    assert _pump(recv, sock, timeout=0.3) is None  # rate-limited

    sock.send_multipart(pack_message(MSG_RESUME, "sess-lost", {}))
    assert _pump(recv, sock)["unknown_session"] is True  # RESUME always answers
    sock.close()


def test_a_session_start_resent_for_an_open_session_is_just_reacked(tmp_path, recv):
    sock = _dealer(recv, "sess-open")
    sock.send_multipart(
        pack_message(MSG_SESSION_START, "sess-open", _header(tmp_path / "raw"))
    )
    _pump(recv, sock)
    header, payload = _frame(0, 0)
    sock.send_multipart(pack_message(MSG_FRAME, "sess-open", header, payload))
    assert _pump(recv, sock)["through_frame_index"] == 0

    sock.send_multipart(
        pack_message(MSG_SESSION_START, "sess-open", _header(tmp_path / "raw"))
    )
    assert _pump(recv, sock)["through_frame_index"] == 0
    assert (0, 0) in recv.sessions["sess-open"].received_pairs
    sock.close()


def test_resume_after_a_receiver_restart_continues_in_the_same_stores(tmp_path):
    raw = tmp_path / "raw"
    first = _receiver()
    sock = _dealer(first, "sess-restart")
    sock.send_multipart(pack_message(MSG_SESSION_START, "sess-restart", _header(raw)))
    _pump(first, sock)
    for t in (0, 1):
        header, payload = _frame(t, t)
        sock.send_multipart(pack_message(MSG_FRAME, "sess-restart", header, payload))
        assert _pump(first, sock)["through_frame_index"] == t
    sock.close()
    first.close()  # the receiver restarts

    second = _receiver()
    try:
        sock = _dealer(second, "sess-restart")
        sock.send_multipart(
            pack_message(
                MSG_SESSION_START, "sess-restart", _header(raw, resume_through=1)
            )
        )
        assert _pump(second, sock)["through_frame_index"] == 1
        session = second.sessions["sess-restart"]
        assert session.resumed and not session.live
        assert session.base_name == "Cell_030"  # not Cell_030_001

        header, payload = _frame(2, 2)
        sock.send_multipart(pack_message(MSG_FRAME, "sess-restart", header, payload))
        assert _pump(second, sock)["through_frame_index"] == 2
        for t in (0, 1, 2):
            np.testing.assert_array_equal(_read(raw, t), _volume(t))
        sock.close()
    finally:
        second.close()


def _stream_and_drain(recv, raw, stage, frames, reason="complete", resume=None):
    extra = {} if resume is None else {"resume_through": resume}
    sock = _dealer(recv, "sess-stage")
    sock.send_multipart(
        pack_message(MSG_SESSION_START, "sess-stage", _header(raw, **extra))
    )
    _pump(recv, sock)
    for t, fi in frames:
        header, payload = _frame(t, fi)
        sock.send_multipart(pack_message(MSG_FRAME, "sess-stage", header, payload))
        assert _pump(recv, sock)["through_frame_index"] == fi
    write_root = recv.sessions["sess-stage"].write_root
    sock.send_multipart(pack_message(MSG_SESSION_END, "sess-stage", {"reason": reason}))
    _drive(recv)
    sock.close()
    return write_root


def test_resume_after_the_drain_writes_into_the_kept_staging_copy(
    tmp_path, recv, monkeypatch
):
    """The client came back after the receiver idle-timed the session out
    and drained it: the session continues in its retained staging copy, so
    its next drain replaces the GPFS copy with a superset."""
    raw, stage = tmp_path / "raw", tmp_path / "stage"
    monkeypatch.setenv("OPYM_STREAM_STAGE_ROOT", str(stage))
    _stream_and_drain(recv, raw, stage, [(0, 0)], reason="idle_timeout")
    assert _wait_until(lambda: _store(raw).exists())
    assert "sess-stage" in recv._drain_pool._drained

    write_root = _stream_and_drain(recv, raw, stage, [(1, 1)], resume=0)
    assert write_root == stage
    assert _wait_until(
        lambda: "sess-stage" in recv._drain_pool._drained and _store(raw).exists()
    )
    for t in (0, 1):
        np.testing.assert_array_equal(_read(raw, t), _volume(t))


def test_resume_after_the_staging_copy_is_gone_writes_straight_to_raw_root(
    tmp_path, recv, monkeypatch
):
    raw, stage = tmp_path / "raw", tmp_path / "stage"
    monkeypatch.setenv("OPYM_STREAM_STAGE_ROOT", str(stage))
    _stream_and_drain(recv, raw, stage, [(0, 0)], reason="idle_timeout")
    assert _wait_until(lambda: "sess-stage" in recv._drain_pool._drained)
    recv._drain_pool.release([_store(stage)])  # retention expired
    assert not _store(stage).exists()

    write_root = _stream_and_drain(recv, raw, stage, [(1, 1)], resume=0)
    assert write_root == raw
    time.sleep(0.3)  # nothing may drain a fresh, partial copy over it
    for t in (0, 1):
        np.testing.assert_array_equal(_read(raw, t), _volume(t))


def test_a_paused_run_is_never_copied_to_raw_root(tmp_path, recv, monkeypatch):
    raw, stage = tmp_path / "raw", tmp_path / "stage"
    monkeypatch.setenv("OPYM_STREAM_STAGE_ROOT", str(stage))
    _stream_and_drain(recv, raw, stage, [(0, 0)], reason="paused")
    time.sleep(0.3)
    assert not _store(raw).exists()
    # Kept on staging for the usual retention, then evicted.
    assert _store(stage).exists()
    assert _store(stage) in recv._drain_pool._drained["sess-stage"].stage_dirs


def test_the_final_ack_confirms_session_end_and_a_repeat_is_confirmed_too(
    tmp_path, recv
):
    sock = _dealer(recv, "sess-end")
    sock.send_multipart(
        pack_message(MSG_SESSION_START, "sess-end", _header(tmp_path / "raw"))
    )
    _pump(recv, sock)
    sock.send_multipart(
        pack_message(MSG_SESSION_END, "sess-end", {"reason": "complete"})
    )
    assert _pump(recv, sock)["ended"] is True
    # The client missed that ACK and sends SESSION_END again.
    sock.send_multipart(
        pack_message(MSG_SESSION_END, "sess-end", {"reason": "complete"})
    )
    assert _pump(recv, sock)["unknown_session"] is True
    sock.close()
