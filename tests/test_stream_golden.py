"""The stream and backfill wire contracts against their golden vectors.

``opym.testing.stream`` holds the bytes; the acquisition PC's tests import the same
module, so a change on one end that the other does not follow fails on both.
"""

from __future__ import annotations

import typing

import numpy as np
import pytest

from opym.stream import bfill, outcome, protocol
from opym.testing import stream as golden


def test_the_golden_module_imports_without_the_heavy_stack():
    import subprocess
    import sys

    code = (
        "import sys; import opym.testing.stream, opym.stream.protocol; "
        "bad = [m for m in ('msgpack', 'zmq', 'zarr', 'scipy') if m in sys.modules]; "
        "sys.exit(1 if bad else 0)"
    )
    # no -I: the test run's PYTHONPATH decides which opym this imports
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


@pytest.mark.parametrize("name", list(golden.FRAME_TIME_CASES))
def test_frame_time_cases(name):
    header, want_epoch, want_source = golden.FRAME_TIME_CASES[name]
    assert protocol.frame_epoch_s(header) == want_epoch
    assert protocol.frame_epoch(header) == (want_epoch, want_source)


@pytest.mark.parametrize(
    "bad",
    [float("nan"), float("inf"), -1.0, 999_999_999.0, 1e11, None, "1792000000", True],
)
def test_a_timestamp_outside_the_epoch_window_is_not_a_time(bad):
    assert protocol.epoch_s_or_none(bad) is None
    header = {"timestamp": bad, "acq_first_s": golden.EPOCH_S}
    assert protocol.frame_epoch(header) == (golden.EPOCH_S, "acq_first_s")


def test_the_epoch_window_edges():
    assert protocol.epoch_s_or_none(1e9) == 1e9
    assert protocol.epoch_s_or_none(1e11 - 1) == 1e11 - 1
    assert protocol.frame_epoch({}) == (None, "none")


def test_the_golden_frame_is_what_pack_message_makes():
    parts = protocol.pack_message(
        protocol.MSG_FRAME,
        golden.SESSION_ID,
        golden.FRAME_HEADER,
        golden.VOLUME.tobytes(),
    )
    assert parts == [
        b"FRAME",
        b"sess-1",
        golden.FRAME_HEADER_HEX,
        golden.VOLUME.tobytes(),
    ]
    msg_type, sid, header, payload = protocol.unpack_message(parts)
    assert (msg_type, sid, header) == (
        protocol.MSG_FRAME,
        golden.SESSION_ID,
        golden.FRAME_HEADER,
    )
    assert payload == golden.VOLUME.tobytes()
    # the rig key set
    assert list(header) == [
        "t", "c", "frame_index", "timestamp", "camera_id",
        "shape_zyx", "dtype", "acq_first_s", "acq_last_s", "queued_s",
    ]  # fmt: skip


def test_the_wire_constants_are_the_ones_the_rig_matches():
    assert protocol.DEFAULT_BIND_ADDR == "tcp://127.0.0.1:5555"
    assert protocol.BACKFILL_BIND_ADDR == "tcp://127.0.0.1:5556"
    assert protocol.LINKBENCH_BIND_ADDR == "tcp://127.0.0.1:5599"
    assert protocol.IDLE_TIMEOUT_SEC == 600.0
    assert protocol.IDLE_CLOSED_PREFIX == "closed on Argus"
    assert protocol.IDLE_CLOSED_REASON.startswith(protocol.IDLE_CLOSED_PREFIX)
    assert protocol.IDLE_CLOSED_REASON == (
        "closed on Argus after 10 min with no new data; not resuming it"
    )
    assert protocol.SERVER_FEATURES == [
        "slabs", "blosc", "links", "resume", "heartbeat", "client_log",
    ]  # fmt: skip


def test_the_servers_use_the_protocol_constants():
    from opym.stream import backfill_client, backfill_receiver, linkbench, receiver

    assert receiver.DEFAULT_BIND_ADDR is protocol.DEFAULT_BIND_ADDR
    assert receiver.IDLE_TIMEOUT_SEC is protocol.IDLE_TIMEOUT_SEC
    assert receiver.SERVER_FEATURES is protocol.SERVER_FEATURES
    assert receiver._IDLE_CLOSED_REASON is protocol.IDLE_CLOSED_REASON
    assert backfill_receiver.DEFAULT_BIND == protocol.BACKFILL_BIND_ADDR
    assert backfill_client.DEFAULT_CONNECT == protocol.BACKFILL_BIND_ADDR
    assert backfill_client.REMOTE_PORT == protocol.BACKFILL_PORT
    assert linkbench.DEFAULT_BIND == protocol.LINKBENCH_BIND_ADDR


def test_the_backfill_names_and_digests():
    assert bfill.MARKER_NAME == golden.MARKER_NAME
    assert bfill.LOCK_NAME == golden.LOCK_NAME
    assert bfill.MARKER_FORMAT == golden.MARKER_FORMAT
    assert bfill.FEATURES == golden.BF_FEATURES
    assert bfill.volume_digest(golden.VOLUME) == golden.VOLUME_SHA256
    # the digest is of the decoded values in C order, whatever the memory layout
    assert bfill.volume_digest(np.asfortranarray(golden.VOLUME)) == golden.VOLUME_SHA256
    assert bfill.manifest_digest({0: golden.VOLUME_SHA256}) == golden.MANIFEST_SHA256


@pytest.mark.parametrize("name", list(golden.BF_MESSAGES))
def test_each_backfill_message_round_trips_the_golden_bytes(name):
    header, has_payload = golden.BF_MESSAGES[name]
    payload = golden.VOLUME.tobytes() if has_payload else None
    msg_type = getattr(bfill, name)
    parts = bfill.pack(msg_type, golden.BF_SESSION_ID, header, payload)
    want = [msg_type, b"rig-1", golden.BF_HEADER_HEX[name]]
    assert parts == want + ([payload] if has_payload else [])
    assert bfill.unpack(parts) == (msg_type, golden.BF_SESSION_ID, header, payload)


def test_store_names():
    for name in golden.VALID_STORE_NAMES:
        assert bfill.valid_store_name(name), name
    for name in golden.INVALID_STORE_NAMES:
        assert not bfill.valid_store_name(name), name


@pytest.mark.parametrize(
    "scalar", [np.float32(1.792e9), np.float64(1.792e9), np.int64(1_792_000_000)]
)
def test_numpy_scalars_are_times_too(scalar):
    assert protocol.epoch_s_or_none(scalar) == float(scalar)
    assert protocol.frame_epoch({"timestamp": scalar}) == (float(scalar), "timestamp")
    assert protocol.epoch_s_or_none(np.float32(3408.0)) is None
    assert protocol.epoch_s_or_none(np.bool_(True)) is None


def test_session_end_reasons_are_one_set_of_words():
    reasons = (
        protocol.REASON_COMPLETE,
        protocol.REASON_IDLE_TIMEOUT,
        protocol.REASON_CLIENT_ABORT,
        protocol.REASON_PAUSED,
        protocol.REASON_CLIENT_ERROR,
        protocol.REASON_STALLED,
    )
    assert reasons == golden.SESSION_END_REASONS
    assert typing.get_args(protocol.SessionEndReason) == golden.SESSION_END_REASONS
    # the end record says something different for each, never the "complete" fallback
    summaries = {
        r: outcome.summarize(r, {(0, 0)}, 3, 1, detail="why")["summary"] for r in reasons
    }
    assert len(set(summaries.values())) == len(reasons)


def test_stream_health_names_and_path(tmp_path, monkeypatch):
    assert outcome.HEALTH_NAME == golden.HEALTH_NAME
    assert outcome.HEALTH_EVERY_S == golden.HEALTH_EVERY_S
    assert outcome.HEALTH_STALE_S == golden.HEALTH_STALE_S
    assert outcome.health_path(tmp_path) == tmp_path / "stream_health.json"
    monkeypatch.setenv("PETAKIT_JOBS_DIR", str(tmp_path / "jobs"))
    assert outcome.health_path() == tmp_path / "jobs" / "stream_health.json"


def test_read_health(tmp_path):
    path = outcome.health_path(tmp_path)
    assert outcome.read_health(path) is None  # missing
    path.write_text("{not json")
    assert outcome.read_health(path) is None  # unreadable
    path.write_text("[1]")
    assert outcome.read_health(path) is None  # not a record
    path.write_text('{"updated_at": 1.5, "sessions": []}')
    assert outcome.read_health(path) == {"updated_at": 1.5, "sessions": []}


def test_the_receivers_use_the_shared_health_and_backfill_constants():
    from opym.stream import backfill_receiver, receiver

    assert receiver._HEALTH_EVERY_S == outcome.HEALTH_EVERY_S
    assert bfill.DEFAULT_SETTLE_S == golden.DEFAULT_SETTLE_S
    assert bfill.LOCK_STALE_S == golden.LOCK_STALE_S
    assert backfill_receiver.DEFAULT_SETTLE_S == bfill.DEFAULT_SETTLE_S
    assert backfill_receiver.LOCK_STALE_S == bfill.LOCK_STALE_S
