"""Tests for streaming one session over several links, resuming a session
the receiver no longer knows, and runs the client paused."""

from __future__ import annotations

import time

import numpy as np
import pytest
import zarr
import zmq

from opym.stream import trace
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
    [ev] = [e for e in trace.read() if e["ev"] == "frame"]
    assert ev["links"] == [0, 1, 2] and ev["dup_slabs"] == 0

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


def test_resume_counts_volumes_on_disk_and_finishes_a_half_received_one(tmp_path):
    """R3: the receiver restarts with t=0 complete and t=1 half-received. The
    ACKed slabs of t=1 are never resent, so the new receiver must finish t=1
    from the planes already written, and count t=0 as received."""
    raw = tmp_path / "raw"
    first = _receiver()
    sock = _dealer(first, "sess-r3")
    sock.send_multipart(pack_message(MSG_SESSION_START, "sess-r3", _header(raw)))
    _pump(first, sock)
    slabs = [(0, 0, 3, 0), (0, 3, 3, 1), (1, 0, 2, 2), (1, 2, 2, 3)]
    for t, z0, n, fi in slabs:
        header, payload = _slab(t, z0, n, fi)
        sock.send_multipart(pack_message(MSG_FRAME, "sess-r3", header, payload))
        _pump(first, sock)
    assert first.sessions["sess-r3"].received_pairs == {(0, 0)}
    sock.close()
    first.close()  # the receiver restarts

    second = _receiver()
    try:
        sock = _dealer(second, "sess-r3")
        sock.send_multipart(
            pack_message(MSG_SESSION_START, "sess-r3", _header(raw, resume_through=3))
        )
        _pump(second, sock)
        session = second.sessions["sess-r3"]
        assert session.received_pairs == {(0, 0)}  # found complete on disk
        # z 2-3 again (unACKed before the restart, say): no double count.
        header, payload = _slab(1, 2, 2, 4)
        sock.send_multipart(pack_message(MSG_FRAME, "sess-r3", header, payload))
        _pump(second, sock)
        assert (1, 0) not in session.received_pairs
        header, payload = _slab(1, 4, 2, 5)
        sock.send_multipart(pack_message(MSG_FRAME, "sess-r3", header, payload))
        _pump(second, sock)
        assert (1, 0) in session.received_pairs
        np.testing.assert_array_equal(_read(raw, 1), _volume(1))
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


class _FakeDisk:
    """shutil.disk_usage with a free figure the test sets (GB)."""

    def __init__(self, free_gb):
        self.free_gb = free_gb

    def __call__(self, _path):
        import collections

        usage = collections.namedtuple("usage", "total used free")
        return usage(252 * 10**9, 0, int(self.free_gb * 1e9))


def test_below_the_ram_disk_floor_frames_wait_and_are_taken_once_space_is_back(
    tmp_path, recv, monkeypatch
):
    """R5: below the floor the receiver holds frames back unACKed (the
    client keeps and resends them) instead of filling tmpfs."""
    import shutil

    raw, stage = tmp_path / "raw", tmp_path / "stage"
    monkeypatch.setenv("OPYM_STREAM_STAGE_ROOT", str(stage))
    monkeypatch.setenv("OPYM_STREAM_STAGE_FLOOR_GB", "20")
    disk = _FakeDisk(free_gb=150)
    monkeypatch.setattr(shutil, "disk_usage", disk)
    sock = _dealer(recv, "sess-floor")
    sock.send_multipart(pack_message(MSG_SESSION_START, "sess-floor", _header(raw)))
    assert _pump(recv, sock) is not None

    disk.free_gb = 12
    recv._check_stage_space()
    header, payload = _frame(0, 0)
    sock.send_multipart(pack_message(MSG_FRAME, "sess-floor", header, payload))
    assert _pump(recv, sock, timeout=0.5) is None  # no ACK: held back
    assert (0, 0) not in recv.sessions["sess-floor"].received_pairs

    disk.free_gb = 150  # eviction (or anything else) freed space
    sock.send_multipart(pack_message(MSG_FRAME, "sess-floor", header, payload))
    ack = _pump(recv, sock)
    assert ack.get("resend") is True  # first: send what was held back
    if ack["through_frame_index"] != 0:
        ack = _pump(recv, sock)
    assert ack["through_frame_index"] == 0
    assert (0, 0) in recv.sessions["sess-floor"].received_pairs
    sock.close()


def test_when_space_is_back_a_held_back_session_is_asked_to_resend(
    tmp_path, recv, monkeypatch
):
    """R5 on the test stack (2026-09-27): held-back frames came back only
    when the client called its link stale, minutes later. Now the receiver
    asks as soon as the RAM disk is above its floor again, once."""
    import shutil

    raw, stage = tmp_path / "raw", tmp_path / "stage"
    monkeypatch.setenv("OPYM_STREAM_STAGE_ROOT", str(stage))
    monkeypatch.setenv("OPYM_STREAM_STAGE_FLOOR_GB", "20")
    disk = _FakeDisk(free_gb=150)
    monkeypatch.setattr(shutil, "disk_usage", disk)
    sock = _dealer(recv, "sess-resend")
    sock.send_multipart(pack_message(MSG_SESSION_START, "sess-resend", _header(raw)))
    assert _pump(recv, sock) is not None

    disk.free_gb = 12
    recv._check_stage_space()
    header, payload = _frame(0, 0)
    sock.send_multipart(pack_message(MSG_FRAME, "sess-resend", header, payload))
    assert _pump(recv, sock, timeout=0.5) is None  # held back, no ACK
    assert recv.sessions["sess-resend"].held_back

    disk.free_gb = 150
    recv._check_stage_space()
    ack = _pump(recv, sock)
    assert ack["resend"] is True and ack["through_frame_index"] == -1
    assert not recv.sessions["sess-resend"].held_back
    recv._check_stage_space()
    assert _pump(recv, sock, timeout=0.3) is None  # asked once, not every check
    sock.close()


def test_the_argus_side_sender_resends_on_request(monkeypatch):
    """opym's StreamSender (local replays, the watcher) honors "resend"."""
    import opym.stream.client as client_mod
    from opym.stream.protocol import MSG_ACK

    sender = client_mod.StreamSender.__new__(client_mod.StreamSender)
    sender.session_id = "s"
    sender.features = set()
    sender.through_frame_index = -1
    sender._retry_buffer = {
        3: ({"frame_index": 3}, b"c"),
        1: ({"frame_index": 1}, b"a"),
    }
    sent = []
    sender._send_frame_wire = lambda header, _payload: sent.append(
        header["frame_index"]
    )
    sender._sock = type("Sock", (), {"recv_multipart": lambda self: []})()
    acks = iter(
        [
            (MSG_ACK, "s", {"through_frame_index": 0}, None),
            (MSG_ACK, "s", {"through_frame_index": 0, "resend": True}, None),
        ]
    )
    monkeypatch.setattr(client_mod, "unpack_message", lambda _parts: next(acks))
    sender._handle_one_ack()
    assert sent == []
    sender._handle_one_ack()
    assert sent == [1, 3]


def test_a_session_start_frees_ram_disk_space_before_refusing(
    tmp_path, recv, monkeypatch
):
    import shutil

    raw, stage = tmp_path / "raw", tmp_path / "stage"
    monkeypatch.setenv("OPYM_STREAM_STAGE_ROOT", str(stage))
    disk = _FakeDisk(free_gb=0.0)
    monkeypatch.setattr(shutil, "disk_usage", disk)
    asked = []

    def free_up(nbytes):
        asked.append(nbytes)
        disk.free_gb = 150  # the oldest drained sessions went
        return nbytes

    monkeypatch.setattr(recv._drain_pool, "free_up", free_up)
    sock = _dealer(recv, "sess-room")
    sock.send_multipart(pack_message(MSG_SESSION_START, "sess-room", _header(raw)))
    assert _pump(recv, sock) is not None
    assert asked and "sess-room" in recv.sessions
    sock.close()
