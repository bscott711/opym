# Ruff style: Compliant
"""
Tests for the real-time frame-streaming client (opym.stream.client):
`StreamSender`'s wire-protocol mechanics (against a real `StreamReceiver`
running in a background thread, over loopback ZMQ), local-store completion
detection (`_z_planes_written` / `_poll_channel_once`), and
`discover_local_session`'s grouping.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import zarr
import zmq

from opym.stream.client import (
    StreamSender,
    _poll_channel_once,
    _z_planes_written,
    discover_local_session,
    watch_and_stream,
)
from opym.stream.receiver import StreamReceiver

SHAPE_ZYX = (2, 3, 4)  # (Z, Y, X)


def _write_local_store(
    path: Path,
    num_timepoints: int,
    shape_zyx: tuple[int, int, int] = SHAPE_ZYX,
    dtype: str = "uint16",
    z_step_um: float = 0.25,
) -> zarr.Array:
    """Hand-builds a minimal local `*.ome.zarr` store in the shape the real
    pymmcore MDA writer produces (see the real-store exploration this
    fixture is modeled on), for the watcher to read from.
    """
    path.mkdir(parents=True)
    (path / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    (path / ".zattrs").write_text(
        json.dumps(
            {
                "multiscales": [
                    {
                        "axes": [
                            {"name": "t", "type": "time"},
                            {"name": "z", "type": "space"},
                            {"name": "y", "type": "space"},
                            {"name": "x", "type": "space"},
                        ],
                        "datasets": [
                            {
                                "coordinateTransformations": [
                                    {"scale": [1, 1, 1, 1], "type": "scale"}
                                ],
                                "path": "p0",
                            }
                        ],
                        "name": "p0",
                        "version": "0.4",
                    }
                ]
            }
        )
    )
    nz, ny, nx = shape_zyx
    arr = zarr.open(
        str(path / "p0"),
        mode="a",
        shape=(num_timepoints, nz, ny, nx),
        chunks=(1, 1, ny, nx),
        dtype=dtype,
        compressor=None,
        dimension_separator="/",
    )
    z = zarr.open(str(path / "z"), mode="a", shape=(nz,), dtype="float64")
    z[:] = np.arange(nz, dtype=np.float64) * z_step_um
    (path / "z" / ".zattrs").write_text(json.dumps({"units": "um"}))
    return arr


def _backdate(path: Path, age_s: float) -> None:
    """Sets every file under `path` to look `age_s` seconds old, so
    `_poll_channel_once`'s mtime-age check treats it as settled without an
    actual `time.sleep`.
    """
    when = time.time() - age_s
    for f in path.rglob("*"):
        if f.is_file():
            os.utime(f, (when, when))


# --- background-thread receiver, for StreamSender/watch_and_stream's own
# blocking session_start/session_end/drain_acks calls ----------------------


@pytest.fixture
def running_receiver():
    recv = StreamReceiver(
        bind_addr="tcp://127.0.0.1:0", ack_every_n_frames=1, ack_every_sec=0.05
    )
    endpoint = recv._socket.getsockopt(zmq.LAST_ENDPOINT).decode()
    stop = threading.Event()

    def _pump():
        while not stop.is_set():
            recv._run_once()

    thread = threading.Thread(target=_pump, daemon=True)
    thread.start()
    yield recv, endpoint
    stop.set()
    thread.join(timeout=2)
    recv.close()


def _read_raw_timepoint(raw_root, base_name, channel_name, t):
    store = raw_root / f"{base_name}_{channel_name}.ome.zarr"
    arr = zarr.open(str(store / "p0"), mode="r")
    return np.asarray(arr[t])


# --- StreamSender -----------------------------------------------------


def test_sender_session_start_and_frame_land_in_raw_mirror(tmp_path, running_receiver):
    _, endpoint = running_receiver
    raw_root = tmp_path / "raw"
    header = {
        "base_name": "cell",
        "raw_root": str(raw_root),
        "dtype": "uint16",
        "shape_zyx": list(SHAPE_ZYX),
        "num_timepoints": 2,
        "channels": [0],
        "channel_names": ["C0"],
        "z_step_um": 0.3,
    }
    vol = np.arange(np.prod(SHAPE_ZYX), dtype=np.uint16).reshape(SHAPE_ZYX)

    with StreamSender(endpoint, session_id="sess-a") as sender:
        sender.session_start(header)
        sender.send_frame(
            {"t": 0, "c": 0, "frame_index": 0, "timestamp": 0.0, "camera_id": 0,
             "shape_zyx": list(SHAPE_ZYX), "dtype": "uint16"},
            vol,
        )
        sender.session_end("complete")
        sender.drain_acks(timeout_ms=1000)
        assert sender.pending_count == 0

    np.testing.assert_array_equal(_read_raw_timepoint(raw_root, "cell", "C0", 0), vol)


def test_sender_evicts_retry_buffer_as_acks_arrive(tmp_path, running_receiver):
    _, endpoint = running_receiver
    raw_root = tmp_path / "raw"
    header = {
        "base_name": "cell2", "raw_root": str(raw_root), "dtype": "uint16",
        "shape_zyx": list(SHAPE_ZYX), "num_timepoints": 3, "channels": [0],
        "channel_names": ["C0"], "z_step_um": 0.3,
    }
    vol = np.zeros(SHAPE_ZYX, dtype=np.uint16)

    with StreamSender(endpoint, session_id="sess-b") as sender:
        sender.session_start(header)
        for t in range(3):
            sender.send_frame(
                {"t": t, "c": 0, "frame_index": t, "timestamp": 0.0, "camera_id": 0,
                 "shape_zyx": list(SHAPE_ZYX), "dtype": "uint16"},
                vol,
            )
        assert sender.wait_for_all_acked(timeout_s=2.0)
        assert sender.through_frame_index == 2


def test_sender_resume_resends_unacked_frames(tmp_path, running_receiver):
    """Simulates a dropped connection: the first StreamSender's socket is
    closed before its last frame is acked; a second StreamSender with the
    SAME session_id must, after `resume()`, get that frame through.
    """
    recv, endpoint = running_receiver
    raw_root = tmp_path / "raw"
    session_id = "sess-resume-client"
    header = {
        "base_name": "cell3", "raw_root": str(raw_root), "dtype": "uint16",
        "shape_zyx": list(SHAPE_ZYX), "num_timepoints": 1, "channels": [0],
        "channel_names": ["C0"], "z_step_um": 0.3,
    }
    vol = np.full(SHAPE_ZYX, 7, dtype=np.uint16)

    sender_a = StreamSender(endpoint, session_id=session_id)
    sender_a.session_start(header)
    # Manually inject the frame into the retry buffer WITHOUT letting the
    # receiver's ack drain it, simulating a connection that dropped before
    # the ack made it back.
    frame_header = {"t": 0, "c": 0, "frame_index": 0, "timestamp": 0.0, "camera_id": 0,
                     "shape_zyx": list(SHAPE_ZYX), "dtype": "uint16"}
    sender_a._retry_buffer[0] = (frame_header, np.ascontiguousarray(vol).tobytes())
    sender_a.close()

    sender_b = StreamSender(endpoint, session_id=session_id)
    # What a real client persists locally across a reconnect.
    sender_b._retry_buffer = dict(sender_a._retry_buffer)
    sender_b.resume(timeout_ms=2000)

    acked = sender_b.wait_for_all_acked(timeout_s=2.0)
    sender_b.close()

    assert acked
    np.testing.assert_array_equal(_read_raw_timepoint(raw_root, "cell3", "C0", 0), vol)


# --- completion detection (_z_planes_written / _poll_channel_once) --------


def test_z_planes_written_counts_only_complete_chunk_dirs(tmp_path):
    store = tmp_path / "store"
    arr = _write_local_store(store, num_timepoints=2, shape_zyx=(4, 3, 3))
    pixel_dir = store / "p0"

    assert _z_planes_written(pixel_dir, 0) == (0, 0.0)

    arr[0] = np.ones((4, 3, 3), dtype=np.uint16)
    n_planes, mtime = _z_planes_written(pixel_dir, 0)
    assert n_planes == 4
    assert mtime > 0.0


def test_poll_channel_once_waits_for_chunk_completeness(tmp_path):
    from opym.stream.client import ChannelWatchState

    store = tmp_path / "store"
    arr = _write_local_store(store, num_timepoints=1, shape_zyx=(3, 2, 2))
    pixel_dir = store / "p0"
    st = ChannelWatchState(c=0, store=store, pixel_dir=pixel_dir, nz=3)

    assert _poll_channel_once(st, min_stable_age_s=0.0) is False  # nothing written yet

    arr[0] = np.ones((3, 2, 2), dtype=np.uint16)
    assert _poll_channel_once(st, min_stable_age_s=9999.0) is False  # too fresh
    _backdate(pixel_dir / "0", age_s=9999.0)
    # Complete and old enough.
    assert _poll_channel_once(st, min_stable_age_s=1.0) is True


# --- discover_local_session -------------------------------------------


def test_discover_local_session_groups_by_prefix(tmp_path):
    leaf = tmp_path / "acq"
    leaf.mkdir()
    _write_local_store(leaf / "Cell_001_GFP_488.ome.zarr", num_timepoints=1)
    _write_local_store(leaf / "Cell_001_mScarlet_561.ome.zarr", num_timepoints=1)

    session = discover_local_session(leaf)
    assert session["base_name"] == "Cell_001"
    assert set(session["channel_names"].values()) == {"GFP_488", "mScarlet_561"}
    assert set(session["local_store_paths"].keys()) == {0, 1}


def test_discover_local_session_raises_when_empty(tmp_path):
    leaf = tmp_path / "empty"
    leaf.mkdir()
    with pytest.raises(FileNotFoundError):
        discover_local_session(leaf)


# --- watch_and_stream end-to-end (pre-completed store, no live-write race) -


def test_watch_and_stream_sends_pre_completed_store(tmp_path, running_receiver):
    _, endpoint = running_receiver
    local_dir = tmp_path / "acq" / "Cell_009"
    local_dir.mkdir(parents=True)
    raw_root = tmp_path / "raw"

    store_c0 = local_dir / "Cell_009_GFP_488.ome.zarr"
    store_c1 = local_dir / "Cell_009_mScarlet_561.ome.zarr"
    arr_c0 = _write_local_store(store_c0, num_timepoints=2, z_step_um=0.4)
    arr_c1 = _write_local_store(store_c1, num_timepoints=2, z_step_um=0.4)
    for t in range(2):
        arr_c0[t] = np.full(SHAPE_ZYX, t + 1, dtype=np.uint16)
        arr_c1[t] = np.full(SHAPE_ZYX, t + 100, dtype=np.uint16)
    # Pre-completed: backdate everything so the stability-age check never
    # has to actually wait.
    _backdate(local_dir, age_s=9999.0)

    session = discover_local_session(local_dir)
    watch_and_stream(
        session["local_store_paths"],
        session["channel_names"],
        base_name=session["base_name"],
        raw_root=str(raw_root),
        connect_addr=endpoint,
        poll_interval_s=0.05,
        stability_polls=1,
        idle_exit_after_s=5,
    )

    for t in range(2):
        np.testing.assert_array_equal(
            _read_raw_timepoint(raw_root, "Cell_009", "GFP_488", t),
            np.full(SHAPE_ZYX, t + 1, dtype=np.uint16),
        )
        np.testing.assert_array_equal(
            _read_raw_timepoint(raw_root, "Cell_009", "mScarlet_561", t),
            np.full(SHAPE_ZYX, t + 100, dtype=np.uint16),
        )
