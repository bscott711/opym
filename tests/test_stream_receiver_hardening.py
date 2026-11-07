# Ruff style: Compliant
"""Receiver hardening (docs/ROADMAP.md item 2; review ING-F2, F3, F4, F5, LIVE-02):
what the stream receiver does with a partial volume, a bad message, an unwritable
destination or an unsafe name, driven over a real loopback ZMQ DEALER."""

from __future__ import annotations

import time

import numpy as np
import pytest
import zarr
import zmq

from opym.stream.protocol import (
    MSG_ACK,
    MSG_FRAME,
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
