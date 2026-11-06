"""Golden vectors for the stream and backfill wire contracts.

Light: this module imports with numpy only (no msgpack, zmq or zarr), so any repo's
tests can import it. Both ends of the wire are tested against these same bytes:
opym's own tests (tests/test_stream_golden.py) pin them to ``opym.stream.protocol``
and ``opym.stream.bfill``; the acquisition PC's tests (pymmcore-gui) import them to
check what its sender produces. If a vector has to change, the wire changed: fix both
ends.

Headers are plain dicts; ``FRAME_HEADER_HEX`` and ``BF_HEADER_HEX`` are the msgpack
encodings of the dicts in insertion order (what ``pack_message`` and ``bfill.pack``
put in part 2 of the message).
"""

from __future__ import annotations

from typing import Any

import numpy as np

SESSION_ID = "sess-1"
BF_SESSION_ID = "rig-1"

# --- FRAME header time: what the receiver makes of each -----------------------------

EPOCH_S = 1_792_000_000.5  # 2026-10-14 17:46 UTC, the rig's clock


def rig_frame_header(timestamp: float, acq_first_s: float | None = EPOCH_S) -> dict:
    """A FRAME header with exactly the keys the acquisition PC sends.

    ``timestamp`` is the field under test; ``acq_first_s`` is epoch seconds of the
    volume's first plane (None leaves it out, like a sender that predates it).
    """
    header: dict[str, Any] = {
        "t": 3,
        "c": 0,
        "frame_index": 12,
        "timestamp": timestamp,
        "camera_id": 0,
        "shape_zyx": [2, 3, 4],
        "dtype": "uint16",
    }
    if acq_first_s is not None:
        header["acq_first_s"] = acq_first_s
        header["acq_last_s"] = acq_first_s + 1.5
        header["queued_s"] = acq_first_s + 1.75
    return header


# name -> (header, expected frame_epoch_s, expected source of the time)
FRAME_TIME_CASES: dict[str, tuple[dict, float | None, str]] = {
    # a current sender: epoch seconds
    "epoch": (rig_frame_header(EPOCH_S), EPOCH_S, "timestamp"),
    # an old sender: milliseconds since the run started (the real 2026 values)
    "ms_since_start": (rig_frame_header(3408.0), EPOCH_S, "acq_first_s"),
    # an old sender whose frame meta had no runner time
    "zero": (rig_frame_header(0.0), EPOCH_S, "acq_first_s"),
    # epoch milliseconds: above the seconds window, so not trusted either
    "epoch_ms": (rig_frame_header(EPOCH_S * 1000.0), EPOCH_S, "acq_first_s"),
    # no usable time anywhere: the volume's time stays unknown
    "ms_without_acq_first": (rig_frame_header(3408.0, acq_first_s=None), None, "none"),
    "zero_acq_first_also_bad": (rig_frame_header(0.0, acq_first_s=12.0), None, "none"),
}

# --- one FRAME on the stream wire ---------------------------------------------------

VOLUME = np.arange(24, dtype="<u2").reshape(2, 3, 4)
"""The pixels of every golden FRAME / BF_VOLUME: 2 z-planes of 3 x 4 uint16."""
VOLUME_SHA256 = "e88624bf274aff4f35798f4bc27027683e9c1d78f132211a3cc4ae5b3decd4e3"
MANIFEST_SHA256 = "9fd43fc43b6bd874e33885a3b8c3cfff3ae5168ca5f7f51b1f1bbb589ea59848"

FRAME_HEADER = rig_frame_header(EPOCH_S)
FRAME_HEADER_HEX = bytes.fromhex(
    "8aa17403a16300ab6672616d655f696e6465780ca974696d657374616d70cb41"
    "dab3f000200000a963616d6572615f696400a973686170655f7a797893020304"
    "a56474797065a675696e743136ab6163715f66697273745f73cb41dab3f00020"
    "0000aa6163715f6c6173745f73cb41dab3f000800000a87175657565645f73cb"
    "41dab3f000900000"
)

# --- session ends and the receiver's health file (opym.stream.protocol / outcome) ---

# SESSION_END ``reason`` words, in the order of ``SessionEndReason``; "stalled" is
# the receiver's own (the client kept reporting in but sent no data).
SESSION_END_REASONS = (
    "complete",
    "idle_timeout",
    "client_abort",
    "paused",
    "client_error",
    "stalled",
)
HEALTH_NAME = "stream_health.json"
HEALTH_EVERY_S = 5.0
HEALTH_STALE_S = 60.0

# --- the backfill wire (opym.stream.bfill) ------------------------------------------

MARKER_NAME = ".opym_complete.json"
LOCK_NAME = ".opym_backfill.lock"
MARKER_FORMAT = 1
DEFAULT_SETTLE_S = 900.0
LOCK_STALE_S = 600.0
BF_FEATURES = ["blosc", "links", "plane_probe", "run_log", "marker_from"]

_LEAF = "/mmfs1/x/20261023"
_STORE = "Cell_001_GFP_488.ome.zarr"

# message type -> (header, has the VOLUME bytes as payload). The keys are the ones the
# sender puts on the wire (opym.stream.backfill_client); the server's BF_ACK reply is
# shown with the fields its inventory answer carries.
_SPEC = {
    "name": _STORE,
    "shape": [3, 2, 3, 4],
    "chunks": [1, 1, 3, 4],
    "dtype": "<u2",
    "written_t": [0, 1],
}
BF_MESSAGES: dict[str, tuple[dict[str, Any], bool]] = {
    "BF_INVENTORY": ({"leaf": _LEAF, "stores": [_SPEC]}, False),
    "BF_META": (
        {
            "leaf": _LEAF,
            "store": _STORE,
            "files": [{"rel": ".zgroup", "data": b'{"zarr_format": 2}'}],
        },
        False,
    ),
    "BF_VOLUME": (
        {
            "leaf": _LEAF,
            "store": _STORE,
            "t": 0,
            "z0": 0,
            "nz": 2,
            "shape_yx": [3, 4],
            "sha256": VOLUME_SHA256,
            "client": BF_SESSION_ID,
        },
        True,
    ),
    "BF_COMMIT": (
        {
            "leaf": _LEAF,
            "store": _STORE,
            "digests": {"0": VOLUME_SHA256},
            "declared_t": 3,
            "root_zattrs": None,
            "source": "EEP230-OPM:S:\\20261023\\Cell_001_GFP_488.ome.zarr",
        },
        False,
    ),
    "BF_RUN": ({"op": "start", "run_id": "20261106T000000"}, False),
    "BF_ACK": (
        {"kind": "inventory", "ok": True, "stores": [], "features": BF_FEATURES},
        False,
    ),
}

# message type -> msgpack bytes of its header
BF_HEADER_HEX: dict[str, bytes] = {
    "BF_INVENTORY": bytes.fromhex(
        "82a46c656166b12f6d6d6673312f782f3230323631303233a673746f72657391"
        "85a46e616d65b943656c6c5f3030315f4746505f3438382e6f6d652e7a617272"
        "a573686170659403020304a66368756e6b739401010304a56474797065a33c75"
        "32a97772697474656e5f74920001"
    ),
    "BF_META": bytes.fromhex(
        "83a46c656166b12f6d6d6673312f782f3230323631303233a573746f7265b943"
        "656c6c5f3030315f4746505f3438382e6f6d652e7a617272a566696c65739182"
        "a372656ca72e7a67726f7570a464617461c4127b227a6172725f666f726d6174"
        "223a20327d"
    ),
    "BF_VOLUME": bytes.fromhex(
        "88a46c656166b12f6d6d6673312f782f3230323631303233a573746f7265b943"
        "656c6c5f3030315f4746505f3438382e6f6d652e7a617272a17400a27a3000a2"
        "6e7a02a873686170655f7978920304a6736861323536d9406538383632346266"
        "3237346166663466333537393866346263323730323736383365396331643738"
        "663133323231316133636334616535623364656364346533a6636c69656e74a5"
        "7269672d31"
    ),
    "BF_COMMIT": bytes.fromhex(
        "86a46c656166b12f6d6d6673312f782f3230323631303233a573746f7265b943"
        "656c6c5f3030315f4746505f3438382e6f6d652e7a617272a764696765737473"
        "81a130d940653838363234626632373461666634663335373938663462633237"
        "3032373638336539633164373866313332323131613363633461653562336465"
        "6364346533aa6465636c617265645f7403ab726f6f745f7a6174747273c0a673"
        "6f75726365d9304545503233302d4f504d3a533a5c32303236313032335c4365"
        "6c6c5f3030315f4746505f3438382e6f6d652e7a617272"
    ),
    "BF_RUN": bytes.fromhex(
        "82a26f70a57374617274a672756e5f6964af323032363131303654303030303030"
    ),
    "BF_ACK": bytes.fromhex(
        "84a46b696e64a9696e76656e746f7279a26f6bc3a673746f72657390a8666561"
        "747572657395a5626c6f7363a56c696e6b73ab706c616e655f70726f6265a772"
        "756e5f6c6f67ab6d61726b65725f66726f6d"
    ),
}

VALID_STORE_NAMES = [
    "Cell_001_GFP_488.ome.zarr",
    "WT_Stim_005_mNG_488nm.ome.zarr",
    "a b+c-d.1.ome.zarr",
]
INVALID_STORE_NAMES = [
    "../x.ome.zarr",
    "a/b.ome.zarr",
    "x.zarr",
    "x.ome.zarr/",
    "..ome.zarr",
    "",
]
