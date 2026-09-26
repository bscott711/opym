"""Tests for the receiver's transport features: the direct (10 GbE) endpoint
with its IP and raw_root allowlists, sessions following their client across
sockets, and volumes streamed as z-slabs while they're being acquired."""

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
    MSG_SESSION_START,
    pack_message,
    unpack_message,
)
from opym.stream.receiver import SERVER_FEATURES, StreamReceiver

SHAPE_ZYX = (5, 4, 6)


def _header(raw_root, base_name="Cell_020"):
    return {
        "base_name": base_name,
        "raw_root": str(raw_root),
        "dtype": "uint16",
        "shape_zyx": list(SHAPE_ZYX),
        "num_timepoints": 2,
        "channels": [0],
        "channel_names": ["GFP_488"],
        "z_step_um": 0.5,
    }


def _receiver(**kw):
    return StreamReceiver(
        bind_addr="tcp://127.0.0.1:0",
        ack_every_n_frames=1,
        ack_every_sec=9999,
        idle_timeout_sec=9999,
        **kw,
    )


def _endpoint(sock):
    return sock.getsockopt(zmq.LAST_ENDPOINT).decode()


def _dealer(endpoint, session_id):
    sock = zmq.Context.instance().socket(zmq.DEALER)
    sock.setsockopt(zmq.IDENTITY, session_id.encode())
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(endpoint)
    return sock


def _pump_for_ack(receiver, sock, timeout=2.0):
    """Drive the receiver until `sock` gets an ACK; None if none arrives."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        receiver._run_once()
        if sock.poll(10):
            msg_type, _sid, header, _ = unpack_message(sock.recv_multipart())
            assert msg_type == MSG_ACK
            return header
    return None


def _volume(t=0):
    return (np.arange(np.prod(SHAPE_ZYX), dtype=np.uint16) + 1000 * t).reshape(
        SHAPE_ZYX
    )


def _slab(t, z0, planes, frame_index, vol):
    header = {
        "t": t,
        "c": 0,
        "frame_index": frame_index,
        "timestamp": 0.0,
        "camera_id": 0,
        "z0": z0,
        "nz": vol.shape[0],
        "shape_zyx": [planes, *vol.shape[1:]],
        "dtype": "uint16",
    }
    return header, vol[z0 : z0 + planes].tobytes()


# --- direct endpoint --------------------------------------------------------


def test_direct_endpoint_is_not_bound_without_both_allowlists(tmp_path, caplog):
    for kw in (
        {"allow_ips": ["127.0.0.1"]},
        {"raw_roots": [tmp_path]},
    ):
        recv = _receiver(direct_bind="tcp://127.0.0.1:0", **kw)
        try:
            assert recv._direct_socket is None
        finally:
            recv.close()
    assert "Not binding the direct endpoint" in caplog.text


def test_direct_endpoint_accepts_an_allowed_host_and_advertises_features(tmp_path):
    recv = _receiver(
        direct_bind="tcp://127.0.0.1:0",
        allow_ips=["127.0.0.1"],
        raw_roots=[tmp_path / "raw"],
    )
    try:
        sock = _dealer(_endpoint(recv._direct_socket), "sess-direct")
        sock.send_multipart(
            pack_message(MSG_SESSION_START, "sess-direct", _header(tmp_path / "raw"))
        )
        ack = _pump_for_ack(recv, sock)
        assert ack is not None and ack["through_frame_index"] == -1
        assert ack["features"] == SERVER_FEATURES
        assert recv.sessions["sess-direct"].sock is recv._direct_socket
        sock.close()
    finally:
        recv.close()


def test_direct_endpoint_refuses_raw_roots_outside_its_allowlist(tmp_path):
    recv = _receiver(
        direct_bind="tcp://127.0.0.1:0",
        allow_ips=["127.0.0.1"],
        raw_roots=[tmp_path / "raw"],
    )
    try:
        sock = _dealer(_endpoint(recv._direct_socket), "sess-escape")
        header = _header(tmp_path / "raw" / ".." / "elsewhere")
        sock.send_multipart(pack_message(MSG_SESSION_START, "sess-escape", header))
        assert _pump_for_ack(recv, sock, timeout=0.5) is None
        assert "sess-escape" not in recv.sessions
        sock.close()
    finally:
        recv.close()


def test_direct_endpoint_refuses_hosts_not_on_the_ip_allowlist(tmp_path):
    recv = _receiver(
        direct_bind="tcp://127.0.0.1:0",
        allow_ips=["10.255.255.1"],
        raw_roots=[tmp_path / "raw"],
    )
    try:
        sock = _dealer(_endpoint(recv._direct_socket), "sess-intruder")
        sock.send_multipart(
            pack_message(MSG_SESSION_START, "sess-intruder", _header(tmp_path / "raw"))
        )
        assert _pump_for_ack(recv, sock, timeout=1.0) is None
        assert recv.sessions == {}
        sock.close()
    finally:
        recv.close()


def test_the_tunnel_endpoint_keeps_its_old_behaviour(tmp_path):
    """No raw_root restriction behind the SSH tunnel (SSH already
    authenticates), even when the direct endpoint is configured."""
    recv = _receiver(
        direct_bind="tcp://127.0.0.1:0",
        allow_ips=["127.0.0.1"],
        raw_roots=[tmp_path / "raw"],
    )
    try:
        sock = _dealer(_endpoint(recv._socket), "sess-tunnel")
        sock.send_multipart(
            pack_message(MSG_SESSION_START, "sess-tunnel", _header(tmp_path / "other"))
        )
        assert _pump_for_ack(recv, sock) is not None
        sock.close()
    finally:
        recv.close()


def test_a_session_follows_its_client_to_the_other_socket(tmp_path):
    """Direct link drops mid-session, the client resumes through the tunnel:
    ACKs must go back the way the client now is."""
    recv = _receiver(
        direct_bind="tcp://127.0.0.1:0",
        allow_ips=["127.0.0.1"],
        raw_roots=[tmp_path / "raw"],
    )
    try:
        direct = _dealer(_endpoint(recv._direct_socket), "sess-move")
        direct.send_multipart(
            pack_message(MSG_SESSION_START, "sess-move", _header(tmp_path / "raw"))
        )
        assert _pump_for_ack(recv, direct) is not None
        direct.close()

        tunnel = _dealer(_endpoint(recv._socket), "sess-move")
        vol = _volume()
        frame = {
            "t": 0,
            "c": 0,
            "frame_index": 0,
            "timestamp": 0.0,
            "camera_id": 0,
            "shape_zyx": list(SHAPE_ZYX),
            "dtype": "uint16",
        }
        tunnel.send_multipart(
            pack_message(MSG_FRAME, "sess-move", frame, vol.tobytes())
        )
        ack = _pump_for_ack(recv, tunnel)
        assert ack is not None and ack["through_frame_index"] == 0
        assert recv.sessions["sess-move"].sock is recv._socket
        tunnel.close()
    finally:
        recv.close()


# --- slabs ---------------------------------------------------------------------


@pytest.fixture
def receiver():
    recv = _receiver()
    yield recv
    recv.close()


def test_slabs_land_in_the_raw_store_and_are_acked_when_the_volume_completes(
    tmp_path, receiver
):
    sock = _dealer(_endpoint(receiver._socket), "sess-slab")
    sock.send_multipart(
        pack_message(MSG_SESSION_START, "sess-slab", _header(tmp_path / "raw"))
    )
    assert _pump_for_ack(receiver, sock)["through_frame_index"] == -1
    vol = _volume()

    # Planes 0-1 and 2-3: the volume is incomplete, so nothing is ACKed yet,
    # but ACKs still flow (so the client never thinks the link is stale).
    for fi, (z0, n) in enumerate([(0, 2), (2, 2)]):
        header, payload = _slab(0, z0, n, fi, vol)
        sock.send_multipart(pack_message(MSG_FRAME, "sess-slab", header, payload))
        assert _pump_for_ack(receiver, sock)["through_frame_index"] == -1
    session = receiver.sessions["sess-slab"]
    assert (0, 0) not in session.received_pairs
    store = tmp_path / "raw" / "Cell_020_GFP_488.ome.zarr" / "p0"
    np.testing.assert_array_equal(zarr.open(str(store), mode="r")[0, :4], vol[:4])

    # A resend of slab 0 changes nothing.
    header, payload = _slab(0, 0, 2, 0, vol)
    sock.send_multipart(pack_message(MSG_FRAME, "sess-slab", header, payload))
    assert _pump_for_ack(receiver, sock)["through_frame_index"] == -1

    # The last plane completes the volume: every slab is ACKed at once.
    header, payload = _slab(0, 4, 1, 2, vol)
    sock.send_multipart(pack_message(MSG_FRAME, "sess-slab", header, payload))
    assert _pump_for_ack(receiver, sock)["through_frame_index"] == 2
    assert (0, 0) in session.received_pairs and not session.slabs
    np.testing.assert_array_equal(zarr.open(str(store), mode="r")[0], vol)

    # After completion, late resends are ignored.
    header, payload = _slab(0, 2, 2, 1, vol)
    sock.send_multipart(pack_message(MSG_FRAME, "sess-slab", header, payload))
    assert _pump_for_ack(receiver, sock)["through_frame_index"] == 2

    [ev] = [e for e in trace.read() if e["ev"] == "frame"]
    assert ev["slabs"] == 3 and ev["bytes"] == vol.nbytes
    assert ev["first_recv_s"] <= ev["recv_s"]
    sock.close()


def test_whole_volume_frames_still_work_next_to_slabs(tmp_path, receiver):
    sock = _dealer(_endpoint(receiver._socket), "sess-mixed")
    sock.send_multipart(
        pack_message(MSG_SESSION_START, "sess-mixed", _header(tmp_path / "raw"))
    )
    _pump_for_ack(receiver, sock)
    v0, v1 = _volume(0), _volume(1)
    frame = {
        "t": 0,
        "c": 0,
        "frame_index": 0,
        "timestamp": 0.0,
        "camera_id": 0,
        "shape_zyx": list(SHAPE_ZYX),
        "dtype": "uint16",
    }
    sock.send_multipart(pack_message(MSG_FRAME, "sess-mixed", frame, v0.tobytes()))
    assert _pump_for_ack(receiver, sock)["through_frame_index"] == 0
    for fi, (z0, n) in enumerate([(0, 3), (3, 2)], start=1):
        header, payload = _slab(1, z0, n, fi, v1)
        sock.send_multipart(pack_message(MSG_FRAME, "sess-mixed", header, payload))
        ack = _pump_for_ack(receiver, sock)
    assert ack["through_frame_index"] == 2
    store = zarr.open(
        str(tmp_path / "raw" / "Cell_020_GFP_488.ome.zarr" / "p0"), mode="r"
    )
    np.testing.assert_array_equal(store[0], v0)
    np.testing.assert_array_equal(store[1], v1)
    sock.close()


def test_blosc_frames_and_slabs_decode_to_the_same_store(tmp_path, receiver):
    """A client that saw "blosc" may compress any FRAME's payload: a whole
    volume and a volume's slabs land exactly as their raw bytes would."""
    from opym.stream.client import compress_payload

    assert "blosc" in SERVER_FEATURES
    sock = _dealer(_endpoint(receiver._socket), "sess-blosc")
    sock.send_multipart(
        pack_message(MSG_SESSION_START, "sess-blosc", _header(tmp_path / "raw"))
    )
    assert "blosc" in _pump_for_ack(receiver, sock)["features"]
    v0, v1 = _volume(0), _volume(1)
    frame = {
        "t": 0,
        "c": 0,
        "frame_index": 0,
        "timestamp": 0.0,
        "camera_id": 0,
        "shape_zyx": list(SHAPE_ZYX),
        "dtype": "uint16",
        "codec": "blosc",
    }
    sock.send_multipart(
        pack_message(MSG_FRAME, "sess-blosc", frame, compress_payload(v0))
    )
    assert _pump_for_ack(receiver, sock)["through_frame_index"] == 0
    for fi, (z0, n) in enumerate([(0, 3), (3, 2)], start=1):
        header, _ = _slab(1, z0, n, fi, v1)
        payload = compress_payload(v1[z0 : z0 + n])
        sock.send_multipart(
            pack_message(MSG_FRAME, "sess-blosc", {**header, "codec": "blosc"}, payload)
        )
        ack = _pump_for_ack(receiver, sock)
    assert ack["through_frame_index"] == 2
    store = zarr.open(
        str(tmp_path / "raw" / "Cell_020_GFP_488.ome.zarr" / "p0"), mode="r"
    )
    np.testing.assert_array_equal(store[0], v0)
    np.testing.assert_array_equal(store[1], v1)
    evs = {e["t"]: e for e in trace.read() if e["ev"] == "frame"}
    assert evs[0]["bytes"] == evs[1]["bytes"] == v0.nbytes
    assert 0 < evs[1]["wire_bytes"] and evs[0]["wire_bytes"] == len(
        compress_payload(v0)
    )
    sock.close()


def test_an_undecodable_or_unknown_codec_frame_is_dropped(tmp_path, receiver):
    sock = _dealer(_endpoint(receiver._socket), "sess-bad")
    sock.send_multipart(
        pack_message(MSG_SESSION_START, "sess-bad", _header(tmp_path / "raw"))
    )
    _pump_for_ack(receiver, sock)
    frame = {
        "t": 0,
        "c": 0,
        "frame_index": 0,
        "timestamp": 0.0,
        "camera_id": 0,
        "shape_zyx": list(SHAPE_ZYX),
        "dtype": "uint16",
    }
    for codec, payload in (("blosc", b"not blosc"), ("zstd9", _volume().tobytes())):
        sock.send_multipart(
            pack_message(MSG_FRAME, "sess-bad", {**frame, "codec": codec}, payload)
        )
        for _ in range(20):
            receiver._run_once()
    assert not receiver.sessions["sess-bad"].received_pairs
    sock.close()


def test_the_sender_compresses_only_once_the_receiver_offers_it(tmp_path, receiver):
    from opym.stream.client import StreamSender

    sent = []
    with StreamSender(_endpoint(receiver._socket), compress=True) as s:
        s._send_frame_wire = lambda header, payload: sent.append((header, payload))
        s.send_frame({"frame_index": 0, "t": 0, "c": 0}, _volume())
        assert "codec" not in sent[-1][0]  # nothing advertised yet
        s.features.add("blosc")
        s.send_frame({"frame_index": 1, "t": 1, "c": 0}, _volume(1))
        assert sent[-1][0]["codec"] == "blosc"
        assert len(sent[-1][1]) < _volume().nbytes
