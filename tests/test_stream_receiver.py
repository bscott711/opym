# Ruff style: Compliant
"""
Tests for the real-time frame-streaming receiver (opym.stream). Drives the
receiver's `StreamReceiver._run_once()` step function directly against a
real ZMQ DEALER client over loopback TCP -- no MATLAB/GPU/PetaKit5D needed,
since `submit_pipeline_job` only writes a JSON ticket (see opym.petakit).
"""

from __future__ import annotations

import json

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
from opym.stream.receiver import StreamReceiver
from opym.utils import orient_zyx_for_dsr

SHAPE_ZYX = (3, 5, 7)


def _session_header(
    output_dir,
    base_name="sample",
    num_timepoints=2,
    channels=(0, 1),
    psf_paths=None,
):
    return {
        "base_name": base_name,
        "output_dir": str(output_dir),
        "dtype": "uint16",
        "shape_zyx": list(SHAPE_ZYX),
        "num_timepoints": num_timepoints,
        "channels": list(channels),
        "channel_names": [f"C{c}" for c in channels],
        "z_step_um": 0.3,
        "xy_pixel_size": 0.116,
        "sheet_angle_deg": 60.0,
        "t_interval_s": 1.0,
        "psf_paths": psf_paths or [],
        "dz_psf": None,
    }


def _frame(t, c, frame_index):
    vol = np.full(SHAPE_ZYX, fill_value=t * 100 + c * 10, dtype=np.uint16)
    header = {
        "t": t,
        "c": c,
        "frame_index": frame_index,
        "timestamp": 0.0,
        "camera_id": 0,
        "shape_zyx": list(SHAPE_ZYX),
        "dtype": "uint16",
    }
    return header, vol


@pytest.fixture
def receiver(tmp_path):
    recv = StreamReceiver(
        bind_addr="tcp://127.0.0.1:0",
        shm_dir=tmp_path / "shm",
        queue_dir=tmp_path / "queue",
        ack_every_n_frames=1,
        ack_every_sec=9999,  # deterministic: only the inline per-frame ack fires
        idle_timeout_sec=9999,
    )
    yield recv
    recv.close()


@pytest.fixture
def client(receiver):
    endpoint = receiver._socket.getsockopt(zmq.LAST_ENDPOINT).decode()
    ctx = zmq.Context.instance()

    def _connect(session_id: str) -> zmq.Socket:
        sock = ctx.socket(zmq.DEALER)
        sock.setsockopt(zmq.IDENTITY, session_id.encode("utf-8"))
        sock.connect(endpoint)
        return sock

    yield _connect


def _drive(receiver, n=1):
    for _ in range(n):
        receiver._run_once()


def _recv_ack(sock):
    msg_type, session_id, header, _ = unpack_message(sock.recv_multipart())
    assert msg_type == MSG_ACK
    return session_id, header


# --- protocol.py round-trip -------------------------------------------


def test_pack_unpack_round_trip_frame_message():
    header, vol = _frame(t=1, c=0, frame_index=5)
    parts = pack_message(MSG_FRAME, "sess-1", header, vol.tobytes())
    msg_type, session_id, decoded_header, payload = unpack_message(parts)
    assert msg_type == MSG_FRAME
    assert session_id == "sess-1"
    assert decoded_header == header
    assert (
        np.frombuffer(payload, dtype=np.uint16).reshape(SHAPE_ZYX).tolist()
        == vol.tolist()
    )


def test_pack_unpack_round_trip_header_only_message():
    parts = pack_message(MSG_SESSION_END, "sess-1", {"reason": "complete"})
    msg_type, session_id, header, payload = unpack_message(parts)
    assert msg_type == MSG_SESSION_END
    assert session_id == "sess-1"
    assert header == {"reason": "complete"}
    assert payload is None


def test_pack_rejects_unknown_message_type():
    with pytest.raises(ValueError):
        pack_message(b"BOGUS", "sess-1", {})


# --- receiver behavior ---------------------------------------------------


def test_frame_staged_with_correct_orientation_and_ticketed(tmp_path, receiver, client):
    session_id = "sess-orient"
    output_dir = tmp_path / "out"
    sock = client(session_id)

    sock.send_multipart(
        pack_message(MSG_SESSION_START, session_id, _session_header(output_dir))
    )
    _drive(receiver)
    _recv_ack(sock)  # SESSION_START ack

    header, vol = _frame(t=0, c=0, frame_index=0)
    sock.send_multipart(pack_message(MSG_FRAME, session_id, header, vol.tobytes()))
    _drive(receiver)
    _, ack_header = _recv_ack(sock)
    assert ack_header["through_frame_index"] == 0

    shm_path = receiver.shm_dir / "sample_T0000_C0.zarr"
    staged = np.asarray(zarr.open(str(shm_path), mode="r"))
    np.testing.assert_array_equal(staged, orient_zyx_for_dsr(vol))

    tickets = list((tmp_path / "queue").glob("*.json"))
    assert len(tickets) == 1
    payload = json.loads(tickets[0].read_text())
    assert payload["dataDir"] == str(output_dir)
    assert payload["parameters"]["shm_path"] == str(shm_path)


def test_frames_use_declared_indices_not_arrival_order(tmp_path, receiver, client):
    session_id = "sess-ooo"
    output_dir = tmp_path / "out"
    sock = client(session_id)
    sock.send_multipart(
        pack_message(MSG_SESSION_START, session_id, _session_header(output_dir))
    )
    _drive(receiver)
    _recv_ack(sock)

    # Send T=1 before T=0.
    for frame_index, (t, c) in enumerate([(1, 0), (0, 0)]):
        header, vol = _frame(t, c, frame_index)
        sock.send_multipart(pack_message(MSG_FRAME, session_id, header, vol.tobytes()))
        _drive(receiver)
        _recv_ack(sock)

    t0 = np.asarray(zarr.open(str(receiver.shm_dir / "sample_T0000_C0.zarr"), mode="r"))
    t1 = np.asarray(zarr.open(str(receiver.shm_dir / "sample_T0001_C0.zarr"), mode="r"))
    _, vol0 = _frame(0, 0, 0)
    _, vol1 = _frame(1, 0, 0)
    np.testing.assert_array_equal(t0, orient_zyx_for_dsr(vol0))
    np.testing.assert_array_equal(t1, orient_zyx_for_dsr(vol1))


def test_duplicate_frame_is_not_reticketed(tmp_path, receiver, client):
    session_id = "sess-dup"
    output_dir = tmp_path / "out"
    sock = client(session_id)
    sock.send_multipart(
        pack_message(MSG_SESSION_START, session_id, _session_header(output_dir))
    )
    _drive(receiver)
    _recv_ack(sock)

    header, vol = _frame(t=0, c=0, frame_index=0)
    for _ in range(2):  # exact resend, same frame_index
        sock.send_multipart(pack_message(MSG_FRAME, session_id, header, vol.tobytes()))
        _drive(receiver)
        _recv_ack(sock)

    assert len(list((tmp_path / "queue").glob("*.json"))) == 1


def test_session_end_writes_consolidate_sidecar(tmp_path, receiver, client):
    session_id = "sess-end"
    output_dir = tmp_path / "out"
    sock = client(session_id)
    header = _session_header(output_dir, num_timepoints=2, channels=(0, 1))
    sock.send_multipart(pack_message(MSG_SESSION_START, session_id, header))
    _drive(receiver)
    _recv_ack(sock)

    sock.send_multipart(
        pack_message(MSG_SESSION_END, session_id, {"reason": "complete"})
    )
    _drive(receiver)

    sidecar = json.loads((output_dir / ".opym_consolidate.json").read_text())
    assert sorted(sidecar["expected_zarrs"]) == sorted(
        [
            "sample_T0000_C0.zarr",
            "sample_T0000_C1.zarr",
            "sample_T0001_C0.zarr",
            "sample_T0001_C1.zarr",
        ]
    )
    assert sidecar["base_name"] == "sample"
    assert sidecar["channel_names"] == ["C0", "C1"]
    assert session_id not in receiver.sessions


def test_resume_after_reconnect_skips_already_processed_frames(
    tmp_path, receiver, client
):
    session_id = "sess-resume"
    output_dir = tmp_path / "out"
    sock_a = client(session_id)
    sock_a.send_multipart(
        pack_message(MSG_SESSION_START, session_id, _session_header(output_dir))
    )
    _drive(receiver)
    _recv_ack(sock_a)

    header0, vol0 = _frame(t=0, c=0, frame_index=0)
    sock_a.send_multipart(pack_message(MSG_FRAME, session_id, header0, vol0.tobytes()))
    _drive(receiver)
    _, ack = _recv_ack(sock_a)
    assert ack["through_frame_index"] == 0

    sock_a.close()  # simulated drop

    sock_b = client(session_id)  # reconnect: same identity, new connection
    sock_b.send_multipart(pack_message(MSG_RESUME, session_id, {}))
    _drive(receiver)
    _, resume_ack = _recv_ack(sock_b)
    assert resume_ack["through_frame_index"] == 0

    # Resend frame_index=0 (already processed) -- must not double-ticket.
    sock_b.send_multipart(pack_message(MSG_FRAME, session_id, header0, vol0.tobytes()))
    _drive(receiver)
    _recv_ack(sock_b)
    assert len(list((tmp_path / "queue").glob("*.json"))) == 1

    # A genuinely new frame after resume proceeds normally.
    header1, vol1 = _frame(t=1, c=0, frame_index=1)
    sock_b.send_multipart(pack_message(MSG_FRAME, session_id, header1, vol1.tobytes()))
    _drive(receiver)
    _, ack2 = _recv_ack(sock_b)
    assert ack2["through_frame_index"] == 1
    assert len(list((tmp_path / "queue").glob("*.json"))) == 2


def test_resume_for_unknown_session_acks_minus_one(receiver, client):
    sock = client("sess-never-started")
    sock.send_multipart(pack_message(MSG_RESUME, "sess-never-started", {}))
    _drive(receiver)
    _, header = _recv_ack(sock)
    assert header["through_frame_index"] == -1
