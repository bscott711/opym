# Ruff style: Compliant
"""
Tests for the real-time frame-streaming receiver (opym.stream). Drives the
receiver's `StreamReceiver._run_once()` step function directly against a
real ZMQ DEALER client over loopback TCP -- no MATLAB/GPU/PetaKit5D needed:
the receiver writes straight into a raw OME-Zarr mirror store (and,
optionally, decon-stage TIFFs) and submits no processing ticket at all --
see the module docstring in `opym.stream.receiver`.
"""

from __future__ import annotations

import numpy as np
import pytest
import tifffile
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
from opym.utils import orient_zyx_for_decon_tiff

SHAPE_ZYX = (3, 5, 7)


def _session_header(
    raw_root,
    base_name="sample",
    num_timepoints=2,
    channels=(0, 1),
    channel_names=None,
):
    return {
        "base_name": base_name,
        "raw_root": str(raw_root),
        "dtype": "uint16",
        "shape_zyx": list(SHAPE_ZYX),
        "num_timepoints": num_timepoints,
        "channels": list(channels),
        "channel_names": channel_names or [f"C{c}" for c in channels],
        "z_step_um": 0.3,
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


def _raw_store_path(raw_root, base_name, channel_name):
    return raw_root / f"{base_name}_{channel_name}.ome.zarr"


def _read_raw_timepoint(raw_root, base_name, channel_name, t):
    store = _raw_store_path(raw_root, base_name, channel_name)
    arr = zarr.open(str(store / "p0"), mode="r")
    return np.asarray(arr[t])


@pytest.fixture
def receiver():
    recv = StreamReceiver(
        bind_addr="tcp://127.0.0.1:0",
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


def _start_session(sock, session_id, receiver, header):
    sock.send_multipart(pack_message(MSG_SESSION_START, session_id, header))
    _drive(receiver)
    return _recv_ack(sock)


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


# --- receiver behavior: raw mirror ----------------------------------------


def test_frame_written_to_raw_mirror_unrotated(tmp_path, receiver, client):
    """Raw mirror stores the volume exactly as received -- no
    `orient_zyx_for_dsr` rotation. That rotation is applied downstream by
    the batch backfill path's own deskew-mirror step, not here (see
    `opym.stream.receiver` module docstring)."""
    session_id = "sess-orient"
    raw_root = tmp_path / "raw"
    sock = client(session_id)

    _start_session(sock, session_id, receiver, _session_header(raw_root))

    header, vol = _frame(t=0, c=0, frame_index=0)
    sock.send_multipart(pack_message(MSG_FRAME, session_id, header, vol.tobytes()))
    _drive(receiver)
    _, ack_header = _recv_ack(sock)
    assert ack_header["through_frame_index"] == 0

    got = _read_raw_timepoint(raw_root, "sample", "C0", t=0)
    np.testing.assert_array_equal(got, vol)


def test_frames_use_declared_indices_not_arrival_order(tmp_path, receiver, client):
    session_id = "sess-ooo"
    raw_root = tmp_path / "raw"
    sock = client(session_id)
    _start_session(sock, session_id, receiver, _session_header(raw_root))

    # Send T=1 before T=0.
    for frame_index, (t, c) in enumerate([(1, 0), (0, 0)]):
        header, vol = _frame(t, c, frame_index)
        sock.send_multipart(pack_message(MSG_FRAME, session_id, header, vol.tobytes()))
        _drive(receiver)
        _recv_ack(sock)

    _, vol0 = _frame(0, 0, 0)
    _, vol1 = _frame(1, 0, 0)
    np.testing.assert_array_equal(
        _read_raw_timepoint(raw_root, "sample", "C0", 0), vol0
    )
    np.testing.assert_array_equal(
        _read_raw_timepoint(raw_root, "sample", "C0", 1), vol1
    )


def test_channels_may_have_different_frame_shapes(tmp_path, receiver, client):
    """Per-FRAME `shape_zyx` overrides the SESSION_START default (see
    protocol.py) -- exercises that this actually works end-to-end, for a
    multi-camera/spectral-crop client whose per-region ROIs aren't
    necessarily all the same size.
    """
    session_id = "sess-mixed-shapes"
    raw_root = tmp_path / "raw"
    sock = client(session_id)
    _start_session(
        sock, session_id, receiver, _session_header(raw_root, num_timepoints=1)
    )

    small_shape = (3, 4, 5)
    large_shape = (3, 9, 11)
    vol_small = np.full(small_shape, fill_value=1, dtype=np.uint16)
    vol_large = np.full(large_shape, fill_value=2, dtype=np.uint16)
    header_small = {
        "t": 0, "c": 0, "frame_index": 0, "timestamp": 0.0, "camera_id": 0,
        "shape_zyx": list(small_shape), "dtype": "uint16",
    }
    header_large = {
        "t": 0, "c": 1, "frame_index": 1, "timestamp": 0.0, "camera_id": 1,
        "shape_zyx": list(large_shape), "dtype": "uint16",
    }

    for header, vol in [(header_small, vol_small), (header_large, vol_large)]:
        sock.send_multipart(pack_message(MSG_FRAME, session_id, header, vol.tobytes()))
        _drive(receiver)
        _recv_ack(sock)

    got_small = _read_raw_timepoint(raw_root, "sample", "C0", 0)
    got_large = _read_raw_timepoint(raw_root, "sample", "C1", 0)
    np.testing.assert_array_equal(got_small, vol_small)
    np.testing.assert_array_equal(got_large, vol_large)
    assert got_small.shape != got_large.shape


def test_duplicate_frame_is_not_restaged(tmp_path, receiver, client):
    session_id = "sess-dup"
    raw_root = tmp_path / "raw"
    sock = client(session_id)
    _start_session(sock, session_id, receiver, _session_header(raw_root))

    header, vol = _frame(t=0, c=0, frame_index=0)
    for _ in range(2):  # exact resend, same frame_index
        sock.send_multipart(pack_message(MSG_FRAME, session_id, header, vol.tobytes()))
        _drive(receiver)
        _recv_ack(sock)

    assert receiver.sessions[session_id].received_pairs == {(0, 0)}


def test_session_end_cleans_up_session_state(tmp_path, receiver, client):
    session_id = "sess-end"
    raw_root = tmp_path / "raw"
    sock = client(session_id)
    header = _session_header(raw_root, num_timepoints=2, channels=(0, 1))
    _start_session(sock, session_id, receiver, header)

    sock.send_multipart(
        pack_message(MSG_SESSION_END, session_id, {"reason": "complete"})
    )
    _drive(receiver)

    assert session_id not in receiver.sessions


def test_resume_after_reconnect_skips_already_processed_frames(
    tmp_path, receiver, client
):
    session_id = "sess-resume"
    raw_root = tmp_path / "raw"
    sock_a = client(session_id)
    _start_session(sock_a, session_id, receiver, _session_header(raw_root))

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

    # Resend frame_index=0 (already processed) -- must not double-process.
    sock_b.send_multipart(pack_message(MSG_FRAME, session_id, header0, vol0.tobytes()))
    _drive(receiver)
    _recv_ack(sock_b)
    assert receiver.sessions[session_id].received_pairs == {(0, 0)}

    # A genuinely new frame after resume proceeds normally.
    header1, vol1 = _frame(t=1, c=0, frame_index=1)
    sock_b.send_multipart(pack_message(MSG_FRAME, session_id, header1, vol1.tobytes()))
    _drive(receiver)
    _, ack2 = _recv_ack(sock_b)
    assert ack2["through_frame_index"] == 1
    np.testing.assert_array_equal(
        _read_raw_timepoint(raw_root, "sample", "C0", 1), vol1
    )


def test_resume_for_unknown_session_acks_minus_one(receiver, client):
    sock = client("sess-never-started")
    sock.send_multipart(pack_message(MSG_RESUME, "sess-never-started", {}))
    _drive(receiver)
    _, header = _recv_ack(sock)
    assert header["through_frame_index"] == -1


def test_malformed_session_start_is_rejected_without_crashing_receiver(
    tmp_path, receiver, client
):
    """A `channel_names` that doesn't match `channels` must not take down
    every other session this receiver process is holding."""
    bad_session_id = "sess-bad"
    good_session_id = "sess-good"
    raw_root = tmp_path / "raw"

    bad_sock = client(bad_session_id)
    bad_header = _session_header(raw_root, channels=(0, 1), channel_names=["only-one"])
    bad_sock.send_multipart(pack_message(MSG_SESSION_START, bad_session_id, bad_header))
    _drive(receiver)
    assert bad_session_id not in receiver.sessions

    good_sock = client(good_session_id)
    _start_session(
        good_sock, good_session_id, receiver,
        _session_header(raw_root, base_name="ok"),
    )
    assert good_session_id in receiver.sessions


def test_z_coordinate_array_records_z_step(tmp_path, receiver, client):
    session_id = "sess-zstep"
    raw_root = tmp_path / "raw"
    sock = client(session_id)
    _start_session(
        sock, session_id, receiver, _session_header(raw_root, num_timepoints=1)
    )

    header, vol = _frame(t=0, c=0, frame_index=0)
    sock.send_multipart(pack_message(MSG_FRAME, session_id, header, vol.tobytes()))
    _drive(receiver)
    _recv_ack(sock)

    store = _raw_store_path(raw_root, "sample", "C0")
    z = np.asarray(zarr.open(str(store / "z"), mode="r"))
    assert len(z) == SHAPE_ZYX[0]
    np.testing.assert_allclose(np.diff(z), 0.3)


# --- receiver behavior: decon staging --------------------------------------


def test_decon_stage_not_written_when_decon_disabled(
    tmp_path, receiver, client, monkeypatch
):
    monkeypatch.delenv("OPYM_DECON_PSF", raising=False)
    session_id = "sess-nodecon"
    raw_root = tmp_path / "raw"
    sock = client(session_id)
    _start_session(
        sock, session_id, receiver, _session_header(raw_root, num_timepoints=1)
    )

    header, vol = _frame(t=0, c=0, frame_index=0)
    sock.send_multipart(pack_message(MSG_FRAME, session_id, header, vol.tobytes()))
    _drive(receiver)
    _recv_ack(sock)

    assert not (raw_root / "sample" / "decon_stage").exists()


def test_decon_stage_written_with_correct_orientation_when_enabled(
    tmp_path, receiver, client, monkeypatch
):
    monkeypatch.setenv("OPYM_DECON_PSF", "/fake/psf.tif")
    session_id = "sess-decon"
    raw_root = tmp_path / "raw"
    sock = client(session_id)
    _start_session(
        sock, session_id, receiver,
        _session_header(raw_root, num_timepoints=2, channels=(0,)),
    )

    header, vol = _frame(t=0, c=0, frame_index=0)
    sock.send_multipart(pack_message(MSG_FRAME, session_id, header, vol.tobytes()))
    _drive(receiver)
    _recv_ack(sock)

    staged = raw_root / "sample" / "decon_stage" / "sample_C0_T000.tif"
    assert staged.is_file()
    np.testing.assert_array_equal(
        tifffile.imread(staged), orient_zyx_for_decon_tiff(vol)
    )


def test_decon_stage_cidx_matches_sorted_store_order_not_channel_index(
    tmp_path, receiver, client, monkeypatch
):
    """`bioimaging.backfill.pipeline.build_decon_staging_dir` assigns each
    channel's `_C{cidx}_T` suffix by the channel STORE PATHS' sort order,
    not by the wire protocol's `c` index -- the receiver's pre-staged
    filenames must match that exactly or the batch pass's skip-if-exists
    check silently misses every pre-staged file. `channel_names` is chosen
    so the store-path sort order is the REVERSE of `channels`' order
    (c=0 -> "Zchannel..." sorts last -> cidx 1; c=1 -> "Achannel..." sorts
    first -> cidx 0), and asserts on each staged file's CONTENT (not just
    filename existence, which can't distinguish "used cidx" from "used raw
    c" when there are only two channels) so a bug that used `c` directly
    instead of `channel_cidx` would be caught.
    """
    monkeypatch.setenv("OPYM_DECON_PSF", "/fake/psf.tif")
    session_id = "sess-cidx"
    raw_root = tmp_path / "raw"
    sock = client(session_id)
    header = _session_header(
        raw_root,
        # >1 to exercise the "_C{cidx}_T{t:03d}.tif" naming branch --
        # num_timepoints=1 takes build_decon_staging_dir's other,
        # cidx-independent single-timepoint naming branch instead.
        num_timepoints=2,
        channels=(0, 1),
        channel_names=["Zchannel_999", "Achannel_111"],
    )
    _start_session(sock, session_id, receiver, header)

    for c, frame_index in [(0, 0), (1, 1)]:
        h, vol = _frame(t=0, c=c, frame_index=frame_index)
        sock.send_multipart(pack_message(MSG_FRAME, session_id, h, vol.tobytes()))
        _drive(receiver)
        _recv_ack(sock)

    decon_dir = raw_root / "sample" / "decon_stage"
    _, vol_c0 = _frame(0, 0, 0)
    _, vol_c1 = _frame(0, 1, 0)
    # c=0's data must land in the cidx=1-named file, and c=1's in cidx=0's.
    np.testing.assert_array_equal(
        tifffile.imread(decon_dir / "sample_C1_T000.tif"),
        orient_zyx_for_decon_tiff(vol_c0),
    )
    np.testing.assert_array_equal(
        tifffile.imread(decon_dir / "sample_C0_T000.tif"),
        orient_zyx_for_decon_tiff(vol_c1),
    )
