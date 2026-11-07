"""The receiver keeps each volume's acquisition time in the raw mirror."""

from __future__ import annotations

import logging

import numpy as np
import zarr

from opym import timeline
from opym.stream.protocol import MSG_FRAME, MSG_SESSION_START, pack_message
from opym.testing import stream as golden
from test_stream_transport import (
    SHAPE_ZYX,
    _dealer,
    _endpoint,
    _header,
    _pump_for_ack,
    _receiver,
    _slab,
    _volume,
)

EPOCH = 1_792_000_000.0  # a 2026 unix time on the rig's clock


def _store(tmp_path):
    return tmp_path / "raw" / "Cell_020_GFP_488.ome.zarr"


def _start(tmp_path, receiver, session_id):
    sock = _dealer(_endpoint(receiver._socket), session_id)
    sock.send_multipart(
        pack_message(MSG_SESSION_START, session_id, _header(tmp_path / "raw"))
    )
    assert _pump_for_ack(receiver, sock) is not None
    return sock


def test_whole_volume_frames_record_their_acquisition_time(tmp_path):
    receiver = _receiver()
    try:
        sock = _start(tmp_path, receiver, "sess-times")
        for fi, (t, ts) in enumerate([(0, EPOCH + 0.74), (1, EPOCH + 8.24)]):
            vol = _volume(t)
            header = {
                "t": t, "c": 0, "frame_index": fi, "timestamp": ts, "camera_id": 0,
                "shape_zyx": list(SHAPE_ZYX), "dtype": "uint16",
            }  # fmt: skip
            sock.send_multipart(
                pack_message(MSG_FRAME, "sess-times", header, vol.tobytes())
            )
            assert _pump_for_ack(receiver, sock) is not None
        arr = zarr.open(str(_store(tmp_path) / "t_epoch_s"), mode="r")
        assert arr.shape == (2,)
        np.testing.assert_allclose(arr[:], [EPOCH + 0.74, EPOCH + 8.24])
        assert dict(arr.attrs)["units"] == "s"
        rel = timeline.from_raw_stores([_store(tmp_path)])
        assert rel == [0.0, 7.5]
        assert timeline.interval_s(rel) == 7.5
    finally:
        receiver.close()


def test_slabs_record_the_time_once_per_volume(tmp_path):
    receiver = _receiver()
    try:
        sock = _start(tmp_path, receiver, "sess-slab-times")
        vol = _volume(0)
        for fi, (z0, n) in enumerate([(0, 2), (2, 2), (4, 1)]):
            header, payload = _slab(0, z0, n, fi, vol)
            header["timestamp"] = EPOCH + 3.0
            sock.send_multipart(
                pack_message(MSG_FRAME, "sess-slab-times", header, payload)
            )
            assert _pump_for_ack(receiver, sock) is not None
        arr = zarr.open(str(_store(tmp_path) / "t_epoch_s"), mode="r")
        assert arr[0] == EPOCH + 3.0
        assert np.isnan(arr[1])  # volume 1 has not arrived: unknown, not zero
        assert timeline.read_raw_times(_store(tmp_path)) == [EPOCH + 3.0, None]
    finally:
        receiver.close()


def test_a_client_that_sends_no_timestamp_leaves_the_times_unknown(tmp_path):
    receiver = _receiver()
    try:
        sock = _start(tmp_path, receiver, "sess-no-ts")
        vol = _volume(0)
        header = {
            "t": 0, "c": 0, "frame_index": 0, "timestamp": 0.0, "camera_id": 0,
            "shape_zyx": list(SHAPE_ZYX), "dtype": "uint16",
        }  # fmt: skip
        sock.send_multipart(
            pack_message(MSG_FRAME, "sess-no-ts", header, vol.tobytes())
        )
        assert _pump_for_ack(receiver, sock) is not None
        assert timeline.read_raw_times(_store(tmp_path)) is None  # 0.0 is not 1970
    finally:
        receiver.close()


def _send_frames(tmp_path, receiver, session_id, headers):
    sock = _start(tmp_path, receiver, session_id)
    for fi, header in enumerate(headers):
        t = header["t"]
        vol = _volume(t)
        header = {**header, "frame_index": fi, "shape_zyx": list(SHAPE_ZYX)}
        sock.send_multipart(pack_message(MSG_FRAME, session_id, header, vol.tobytes()))
        assert _pump_for_ack(receiver, sock) is not None


def test_an_old_rig_frame_yields_a_real_epoch_in_the_raw_store(tmp_path, caplog):
    """The rig sent ms since the run started in `timestamp` (ING-F1): 3408, 13259, ...
    Its `acq_first_s` is the real time; the raw store gets that, and the processed
    store gets real elapsed time."""
    receiver = _receiver()
    try:
        ms = [3408.0, 13259.0]
        headers = [
            {**golden.rig_frame_header(ms[t], EPOCH + 10.0 * t), "t": t}
            for t in range(2)
        ]
        with caplog.at_level(logging.INFO, logger="opym.stream.receiver"):
            _send_frames(tmp_path, receiver, "sess-old-rig", headers)
        arr = zarr.open(str(_store(tmp_path) / "t_epoch_s"), mode="r")
        np.testing.assert_allclose(arr[:], [EPOCH, EPOCH + 10.0])
        assert timeline.from_raw_stores([_store(tmp_path)]) == [0.0, 10.0]
        notes = [r for r in caplog.records if "using acq_first_s" in r.getMessage()]
        assert len(notes) == 1 and notes[0].levelno == logging.INFO  # once per session
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    finally:
        receiver.close()


def test_a_current_rig_frame_logs_nothing_about_time(tmp_path, caplog):
    receiver = _receiver()
    try:
        headers = [
            {**golden.rig_frame_header(EPOCH + 10.0 * t), "t": t} for t in (0, 1)
        ]
        headers[0]["acq_first_s"] = headers[0]["timestamp"] = EPOCH
        with caplog.at_level(logging.INFO, logger="opym.stream.receiver"):
            _send_frames(tmp_path, receiver, "sess-new-rig", headers)
        arr = zarr.open(str(_store(tmp_path) / "t_epoch_s"), mode="r")
        np.testing.assert_allclose(arr[:], [EPOCH, EPOCH + 10.0])
        assert not [r for r in caplog.records if "acq_first_s" in r.getMessage()]
    finally:
        receiver.close()


def test_frames_with_no_usable_time_warn_once_and_store_nothing(tmp_path, caplog):
    receiver = _receiver()
    try:
        headers = [
            {**golden.rig_frame_header(3408.0, acq_first_s=None), "t": t}
            for t in range(2)
        ]
        with caplog.at_level(logging.INFO, logger="opym.stream.receiver"):
            _send_frames(tmp_path, receiver, "sess-no-time", headers)
        assert timeline.read_raw_times(_store(tmp_path)) is None
        warns = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len([w for w in warns if "stay unknown" in w.getMessage()]) == 1
    finally:
        receiver.close()


def test_an_old_rig_slab_yields_a_real_epoch_too(tmp_path):
    receiver = _receiver()
    try:
        sock = _start(tmp_path, receiver, "sess-old-slab")
        vol = _volume(0)
        for fi, (z0, n) in enumerate([(0, 3), (3, 2)]):
            header, payload = _slab(0, z0, n, fi, vol)
            header.update(golden.rig_frame_header(3408.0, EPOCH + 4.0))
            header.update({"t": 0, "z0": z0, "nz": SHAPE_ZYX[0]})
            header["shape_zyx"] = [n, *SHAPE_ZYX[1:]]
            sock.send_multipart(
                pack_message(MSG_FRAME, "sess-old-slab", header, payload)
            )
            assert _pump_for_ack(receiver, sock) is not None
        arr = zarr.open(str(_store(tmp_path) / "t_epoch_s"), mode="r")
        assert arr[0] == EPOCH + 4.0
    finally:
        receiver.close()


def test_write_epoch_refuses_a_time_outside_the_epoch_window(tmp_path):
    from opym.stream import rawmirror

    store = _store(tmp_path)
    rawmirror.create_channel_store(
        store,
        num_timepoints=2,
        shape_zyx=SHAPE_ZYX,
        dtype="uint16",
        z_step_um=0.5,
    )
    for bad in (3408.0, 0.0, EPOCH * 1000.0, float("nan"), None):
        rawmirror.write_epoch(store, 0, bad)
    rawmirror.write_epoch(store, 1, EPOCH)
    arr = zarr.open(str(store / "t_epoch_s"), mode="r")
    assert np.isnan(arr[0])  # nothing usable was given for volume 0
    assert arr[1] == EPOCH
