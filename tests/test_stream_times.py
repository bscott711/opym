"""The receiver keeps each volume's acquisition time in the raw mirror."""

from __future__ import annotations

import numpy as np
import zarr
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

from opym import timeline
from opym.stream.protocol import MSG_FRAME, MSG_SESSION_START, pack_message

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
