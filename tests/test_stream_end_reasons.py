"""How a streamed session ends, and that every reader can say why.

Rig 2026-10-24: from 12:23 on, ten runs in a row reached Argus with no
frames. Each was closed as idle 10 min after SESSION_START, naparym-live
said "0/100 timepoints · failed", the dashboard said "pending · 0/100", and
the client went on resyncing an idle-closed session every 30 s. These tests
pin the heartbeat, the "closed" reply, the end record and the client log.
"""

from __future__ import annotations

import json
import time

import pytest
from opym.stream import outcome
from opym.stream import receiver as receiver_mod
from opym.stream.protocol import (
    MSG_CLIENT_LOG,
    MSG_FRAME,
    MSG_HEARTBEAT,
    MSG_RESUME,
    MSG_SESSION_END,
    MSG_SESSION_START,
    pack_message,
)
from opym.stream.receiver import StreamReceiver
from test_stream_links import _dealer, _drive, _frame, _header, _pump, _wait_until


def _idle_receiver(idle_s):
    return StreamReceiver(
        bind_addr="tcp://127.0.0.1:0",
        ack_every_n_frames=1,
        ack_every_sec=9999,
        idle_timeout_sec=idle_s,
    )


@pytest.fixture
def quick(monkeypatch):
    """Heartbeats count as fresh for 0.3 s, a stall closes after 2 s."""
    monkeypatch.setattr(receiver_mod, "HEARTBEAT_FRESH_SEC", 0.3)
    monkeypatch.setattr(receiver_mod, "STALL_MAX_SEC", 2.0)
    monkeypatch.setattr(receiver_mod, "_HEALTH_EVERY_S", 0.05)


def _open(recv, sid, raw, **extra):
    sock = _dealer(recv, sid)
    sock.send_multipart(pack_message(MSG_SESSION_START, sid, _header(raw, **extra)))
    ack = _pump(recv, sock)
    assert ack is not None and "heartbeat" in ack["features"]
    return sock


def _flush(sock):
    while sock.poll(10):
        sock.recv_multipart()


def _status(raw):
    leaf = raw / "Cell_030"
    assert _wait_until(lambda: outcome.read_status(leaf) is not None)
    return outcome.read_status(leaf)


def test_summaries_name_the_outcome_and_the_remedy():
    none = outcome.summarize("idle_timeout", set(), 100, 2, "no new data for 10 min")
    assert none["outcome"] == outcome.NO_DATA
    assert none["summary"].startswith("no data reached Argus")
    assert "0/100 timepoints" in none["summary"]
    assert "opym-backfill-send" in none["remedy"]
    part = outcome.summarize("paused", {(0, 0), (0, 1), (1, 0)}, 10, 2)
    assert part["outcome"] == outcome.PARTIAL and "1/10" in part["summary"]
    full = outcome.summarize("complete", {(0, 0), (1, 0)}, 2, 1)
    assert full["outcome"] == outcome.COMPLETE and full["remedy"] == ""
    short = outcome.summarize("complete", {(0, 0)}, 2, 1)
    assert "only 1/2" in short["summary"] and short["remedy"]


def test_a_silent_client_is_closed_with_the_reason_written_beside_the_run(tmp_path):
    """The 2026-10-24 shape: SESSION_START, then nothing."""
    recv = _idle_receiver(0.3)
    raw = tmp_path / "raw"
    sock = _open(recv, "sess-silent", raw)
    try:
        _drive(recv, 0.6)
        assert "sess-silent" not in recv.sessions
        status = _status(raw)
        assert status["reason"] == "idle_timeout"
        assert status["outcome"] == outcome.NO_DATA
        assert "no new data from the microscope PC" in status["summary"]
        assert "opym-backfill-send" in status["remedy"]
        profile = (recv_jobs() / "profiling" / "receiver.jsonl").read_text()
        last = json.loads(profile.splitlines()[-1])
        assert last["outcome"] == outcome.NO_DATA and last["detail"]
    finally:
        sock.close()
        recv.close()


def recv_jobs():
    from opym import lanes

    return lanes.jobs_dir()


def test_a_heartbeating_client_is_kept_open_past_the_idle_timeout(tmp_path, quick):
    """A slow time-lapse, or a client still trying: not idle while it reports in."""
    recv = _idle_receiver(0.3)
    raw = tmp_path / "raw"
    sock = _open(recv, "sess-alive", raw)
    try:
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            hb = {"state": "acquiring", "links_up": 4, "links_total": 4}
            sock.send_multipart(pack_message(MSG_HEARTBEAT, "sess-alive", hb))
            _drive(recv, 0.1)
        assert "sess-alive" in recv.sessions
        health = json.loads((recv_jobs() / "stream_health.json").read_text())
        (open_session,) = health["sessions"]
        assert open_session["client"]["links_up"] == 4
        assert "4/4 links up" in open_session["client_summary"]
    finally:
        sock.close()
        recv.close()


def test_a_client_that_went_silent_is_closed_with_what_it_last_reported(
    tmp_path, quick
):
    recv = _idle_receiver(0.3)
    raw = tmp_path / "raw"
    sock = _open(recv, "sess-gone", raw)
    try:
        hb = {"links_up": 0, "links_total": 4, "unacked_bytes": 3_100_000_000}
        sock.send_multipart(pack_message(MSG_HEARTBEAT, "sess-gone", hb))
        _drive(recv, 1.0)
        assert "sess-gone" not in recv.sessions
        status = _status(raw)
        assert "went silent" in status["detail"]
        assert "0/4 links up" in status["detail"] and "3100 MB" in status["detail"]
    finally:
        sock.close()
        recv.close()


def test_a_client_heartbeating_without_data_is_closed_as_stalled_at_the_cap(
    tmp_path, quick
):
    recv = _idle_receiver(0.3)
    raw = tmp_path / "raw"
    sock = _open(recv, "sess-wedged", raw)
    try:
        deadline = time.monotonic() + 3.0
        while "sess-wedged" in recv.sessions and time.monotonic() < deadline:
            hb = {"state": "acquiring", "submitted": 0}
            sock.send_multipart(pack_message(MSG_HEARTBEAT, "sess-wedged", hb))
            _drive(recv, 0.1)
        assert "sess-wedged" not in recv.sessions
        status = _status(raw)
        assert status["reason"] == "stalled"
        assert "kept reporting in" in status["detail"]
    finally:
        sock.close()
        recv.close()


@pytest.mark.parametrize("msg", [MSG_FRAME, MSG_RESUME, MSG_HEARTBEAT])
def test_every_message_for_a_session_closed_as_idle_says_it_is_closed(tmp_path, msg):
    """Before: only SESSION_START was turned down; a FRAME or RESUME got a
    bare unknown_session, so the client resynced every 30 s for good."""
    recv = _idle_receiver(0.3)
    raw = tmp_path / "raw"
    sock = _open(recv, "sess-closed", raw)
    try:
        _drive(recv, 0.6)
        assert "sess-closed" not in recv.sessions
        _flush(sock)
        if msg == MSG_FRAME:
            header, payload = _frame(0, 0)
            sock.send_multipart(pack_message(MSG_FRAME, "sess-closed", header, payload))
        else:
            sock.send_multipart(pack_message(msg, "sess-closed", {}))
        ack = _pump(recv, sock)
        assert ack is not None and ack.get("unknown_session")
        assert ack.get("closed") is True
        assert "no new data" in ack.get("rejected", "")
    finally:
        sock.close()
        recv.close()


def test_a_late_session_end_is_recorded_beside_the_run(tmp_path):
    recv = _idle_receiver(0.3)
    raw = tmp_path / "raw"
    sock = _open(recv, "sess-late", raw)
    try:
        _drive(recv, 0.6)
        assert _status(raw)["reason"] == "idle_timeout"
        _flush(sock)  # the idle close's own final ACK
        end = {"reason": "paused", "detail": "Argus closed this run"}
        sock.send_multipart(pack_message(MSG_SESSION_END, "sess-late", end))
        ack = _pump(recv, sock)
        assert ack is not None and ack.get("unknown_session")
        leaf = raw / "Cell_030"
        assert _wait_until(
            lambda: (outcome.read_status(leaf) or {}).get("client_end_reason")
            == "paused"
        )
    finally:
        sock.close()
        recv.close()


def test_a_client_error_end_carries_its_detail(tmp_path):
    recv = _idle_receiver(9999)
    raw = tmp_path / "raw"
    sock = _open(recv, "sess-err", raw)
    try:
        end = {"reason": "client_error", "detail": "ValueError: boom"}
        sock.send_multipart(pack_message(MSG_SESSION_END, "sess-err", end))
        assert _pump(recv, sock)["ended"] is True
        status = _status(raw)
        assert status["reason"] == "client_error"
        assert "stream sender failed: ValueError: boom" in status["summary"]
    finally:
        sock.close()
        recv.close()


def test_client_log_lines_land_in_the_days_rig_log(tmp_path, monkeypatch):
    monkeypatch.setenv("OPYM_STREAM_CLIENT_LOG_DIR", str(tmp_path / "rig"))
    recv = _idle_receiver(9999)
    sock = _dealer(recv, "sess-log")
    try:
        lines = ["12:24:01 WARNING Argus link 2 dropped", "12:24:05 INFO up"]
        sock.send_multipart(
            pack_message(MSG_CLIENT_LOG, "sess-log", {"lines": lines, "host": "rig"})
        )
        _drive(recv, 0.2)
        (log,) = (tmp_path / "rig").glob("*.log")
        text = log.read_text()
        assert "[rig sess-log] 12:24:01 WARNING Argus link 2 dropped" in text
        assert text.count("\n") == 2
    finally:
        sock.close()
        recv.close()


def test_the_live_lane_keeps_the_end_in_its_status_files(tmp_path):
    from opym.live_view import status_text
    from opym.ome_zarr_writer import read_progress, write_progress

    store = tmp_path / "view.ome.zarr"
    store.mkdir()
    end = outcome.summarize("idle_timeout", set(), 100, 2, "no new data for 10 min")
    write_progress(store, n_t=100, n_c=2, done=[], state="failed", end=end)
    text = status_text("WTcocktail", read_progress(store))
    assert "no data reached Argus" in text and "opym-backfill-send" in text
