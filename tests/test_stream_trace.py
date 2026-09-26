"""Tests for the live-path latency trace (opym.stream.trace) and its report
(opym.stream.trace_report): the receiver, live lane and viewer each append
their events, and the report joins them with the GPU servers' profiling into
per-hop latencies per timepoint."""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pytest
import tifffile
import zmq

from opym import lanes
from opym.stream import trace, trace_report
from opym.stream.live import LiveLane
from opym.stream.protocol import (
    MSG_ACK,
    MSG_FRAME,
    MSG_SESSION_START,
    pack_message,
    unpack_message,
)
from opym.stream.receiver import StreamReceiver

# --- trace.py ---------------------------------------------------------------


def test_record_appends_json_lines_with_event_and_time(tmp_path):
    trace.record("frame", jobs=tmp_path, t=3)
    trace.record("shown", name=trace.VIEW_TRACE_NAME, jobs=tmp_path, t=3)
    [frame] = trace.read(jobs=tmp_path)
    assert frame["ev"] == "frame" and frame["t"] == 3 and frame["at"] > 0
    assert trace.read(trace.VIEW_TRACE_NAME, jobs=tmp_path)[0]["ev"] == "shown"


def test_record_never_raises_when_the_trace_cannot_be_written(tmp_path):
    blocker = tmp_path / "jobs"
    blocker.write_text("a file where the jobs dir should be")
    trace.record("frame", jobs=blocker, t=0)  # must not raise
    assert trace.read(jobs=blocker) == []


def test_default_trace_location_follows_the_jobs_dir():
    assert trace.trace_path() == lanes.jobs_dir() / "profiling" / trace.TRACE_NAME


# --- receiver ---------------------------------------------------------------

SHAPE_ZYX = (3, 5, 7)


@pytest.fixture
def receiver():
    recv = StreamReceiver(
        bind_addr="tcp://127.0.0.1:0",
        ack_every_n_frames=1,
        ack_every_sec=9999,
        idle_timeout_sec=9999,
    )
    yield recv
    recv.close()


def _connect(receiver, session_id):
    sock = zmq.Context.instance().socket(zmq.DEALER)
    sock.setsockopt(zmq.IDENTITY, session_id.encode())
    sock.connect(receiver._socket.getsockopt(zmq.LAST_ENDPOINT).decode())
    return sock


def _ack(receiver, sock):
    receiver._run_once()
    msg_type, _sid, header, _ = unpack_message(sock.recv_multipart())
    assert msg_type == MSG_ACK
    return header


def test_receiver_acks_carry_server_time_and_frames_are_traced(tmp_path, receiver):
    sock = _connect(receiver, "sess-trace")
    header = {
        "base_name": "Cell_009",
        "raw_root": str(tmp_path / "raw"),
        "dtype": "uint16",
        "shape_zyx": list(SHAPE_ZYX),
        "num_timepoints": 2,
        "channels": [0],
        "channel_names": ["GFP_488"],
        "z_step_um": 0.5,
    }
    sock.send_multipart(pack_message(MSG_SESSION_START, "sess-trace", header))
    ack = _ack(receiver, sock)
    assert ack["server_time_s"] > 1e9

    vol = np.ones(SHAPE_ZYX, dtype=np.uint16)
    frame = {
        "t": 0,
        "c": 0,
        "frame_index": 0,
        "timestamp": 0.0,
        "camera_id": 0,
        "shape_zyx": list(SHAPE_ZYX),
        "dtype": "uint16",
        "acq_last_s": 100.0,
        "sent_s": 101.0,
        "clock_offset_s": 5.0,
        "not_a_trace_field": "ignored",
    }
    for _ in range(2):  # a resend of the same (t, c) is not traced twice
        sock.send_multipart(pack_message(MSG_FRAME, "sess-trace", frame, vol.tobytes()))
        _ack(receiver, sock)
    sock.close(linger=0)

    [ev] = [e for e in trace.read() if e["ev"] == "frame"]
    assert ev["session_id"] == "sess-trace" and ev["base_name"] == "Cell_009"
    assert (ev["t"], ev["c"], ev["cidx"], ev["bytes"]) == (0, 0, 0, vol.nbytes)
    assert ev["recv_s"] <= ev["staged_s"]
    assert (ev["acq_last_s"], ev["sent_s"], ev["clock_offset_s"]) == (100.0, 101.0, 5.0)
    assert "not_a_trace_field" not in ev


# --- live lane ----------------------------------------------------------------


def test_live_lane_traces_ticket_lifecycle_and_view_ready(tmp_path):
    psf = tmp_path / "psf.tif"
    psf.write_bytes(b"psf")
    jobs = tmp_path / "jobs"
    for d in ("queue_live", "completed", "failed"):
        (jobs / d).mkdir(parents=True)
    lane = LiveLane(psf, jobs=jobs)
    stage_leaf = tmp_path / "stage" / "Cell_010"
    (stage_leaf / "decon_stage").mkdir(parents=True)
    s = lane.start_session(
        "sess-live",
        base_name="Cell_010",
        num_timepoints=2,
        n_channels=1,
        frames_dir=stage_leaf / "decon_stage",
        stage_leaf=stage_leaf,
        dest_leaf=tmp_path / "gpfs" / "Cell_010",
        z_step_um=0.5,
    )
    s.frame(0, 0).write_bytes(b"staged")
    lane.frame_staged("sess-live", 0, 0)
    lane.pump()
    [ticket_path] = sorted((jobs / "queue_live").glob("LIVE_*.json"))
    dsr = s.decon_dir / "DSR_decon"
    (dsr / "MIPs").mkdir(parents=True)
    vol = np.arange(6 * 8 * 10, dtype=np.uint16).reshape(6, 8, 10)
    tifffile.imwrite(dsr / "Cell_010_C0_T000.tif", vol)
    tifffile.imwrite(dsr / "MIPs" / "Cell_010_C0_T000_MIP_z.tif", vol.max(0))
    os.replace(ticket_path, jobs / "completed" / ticket_path.name)
    deadline = time.time() + 5
    while 0 not in s.done and time.time() < deadline:
        lane.pump()
        time.sleep(0.02)
    lane.close()
    assert 0 in s.done

    events = {e["ev"]: e for e in trace.read(jobs=jobs)}
    assert events["ticket"]["ticket"] == ticket_path.name
    assert events["ticket"]["timepoints"] == [0]
    assert events["ticket_done"]["ticket"] == ticket_path.name
    assert events["view_ready"]["t"] == 0
    assert events["ticket"]["at"] <= events["ticket_done"]["at"]
    assert events["ticket_done"]["at"] <= events["view_ready"]["at"]


# --- trace_report -----------------------------------------------------------


def _write(jobs, name, events):
    path = jobs / "profiling" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(e) + "\n" for e in events))


def _synthetic(jobs, with_client=True, legacy_profiles=False):
    """Two timepoints of a 2-channel session, 10 s apart, with known hops.
    The channels are acquired one after the other (the 488 stack, then the
    561 stack), so each point of C1 comes 2 s after C0's; a timepoint's
    time at each point is C1's. Per channel c, relative to base + 2c:

        last plane 0, sent 0.25, received 0.5, staged 0.6, ticket 0.7,
        claimed 0.8, view buffer 1.5, store 1.7, completed 1.8, reaped 1.9

    then the viewer sees both at +3.6, shows at +4.0, paints at +4.5, and
    the timepoint is archived at +4.2 (all relative to base)."""
    events, view, prof = [], [], []
    for t in (0, 1):
        base = 1000.0 + 10 * t
        for c in (0, 1):
            b = base + 2 * c
            ev = {
                "ev": "frame",
                "at": b + 0.5,
                "session_id": "sess-abc",
                "base_name": "Cell_011",
                "t": t,
                "c": c,
                "cidx": c,
                "bytes": 100_000_000,
                "recv_s": b + 0.5,
                "staged_s": b + 0.6,
            }
            if with_client:
                # client clock is 50 s behind Argus
                ev |= {
                    "acq_last_s": b - 50,
                    "sent_s": b - 50 + 0.25,
                    "clock_offset_s": 50.0,
                }
            events.append(ev)
            name = f"LIVE_Cell_011_T{t:04d}_C{c}.json"
            events.append(
                {
                    "ev": "ticket",
                    "at": b + 0.7,
                    "session_id": "sess-abc",
                    "ticket": name,
                    "timepoints": [t],
                    "c": c,
                    "attempt": 1,
                }
            )
            rec = {
                "ticket": name,
                "server_id": str(c + 1),
                "started_at": b + 0.8,
                "read_s": 0.1,
                "decon_s": 0.4,
                "dsr_s": 0.1,
                "view_s": 0.1,
                "write_s": 0.2,
                "total_s": 1.0,
            }
            if not legacy_profiles:
                rec |= {"buffer_at": b + 1.5, "store_at": b + 1.7, "done_at": b + 1.8}
            prof.append(rec)
            events.append(
                {
                    "ev": "ticket_done",
                    "at": b + 1.9,
                    "session_id": "sess-abc",
                    "ticket": name,
                    "timepoints": [t],
                    "c": c,
                }
            )
        events.append(
            {"ev": "archived", "at": base + 4.2, "session_id": "sess-abc", "t": t}
        )
        view.append(
            {
                "ev": "shown",
                "at": base + 4.0,
                "session_id": "sess-abc",
                "timepoints": [t],
                "seen_s": base + 3.6,
            }
        )
        view.append(
            {
                "ev": "painted",
                "at": base + 4.5,
                "session_id": "sess-abc",
                "timepoints": [t],
            }
        )
    _write(jobs, trace.TRACE_NAME, events)
    _write(jobs, trace.VIEW_TRACE_NAME, view)
    _write(jobs, "S1.jsonl", [r for r in prof if r["server_id"] == "1"])
    _write(jobs, "S2.jsonl", [r for r in prof if r["server_id"] == "2"])


def _summary(jobs):
    events = trace.read(jobs=jobs)
    sid = trace_report.pick_session(events, None, None)
    profiles = trace_report._server_profiles(jobs)
    rows = trace_report.timeline(
        sid, events, trace.read(trace.VIEW_TRACE_NAME, jobs=jobs), profiles
    )
    return trace_report.summarize(rows, profiles)


def test_report_breaks_each_timepoint_into_hops(tmp_path):
    _synthetic(tmp_path)
    summary = _summary(tmp_path)
    hops = {k: v["p50"] for k, v in summary["hops"].items()}
    # Each point is the timepoint's LAST channel (C1, the 561 stack).
    assert hops == pytest.approx(
        {
            "client": 0.25,
            "wire": 0.25,
            "stage": 0.1,
            "dispatch": 0.1,
            "claim": 0.1,
            "gpu": 0.7,  # claimed -> view buffer
            "detect": 0.1,  # buffer (C1, base+3.5) -> seen (base+3.6)
            "build": 0.4,
            "paint": 0.5,
        }
    )
    side = {k: v["p50"] for k, v in summary["side"].items()}
    assert side == pytest.approx(
        {"encode": 0.2, "finish": 0.1, "reap": 0.1, "archive": 0.3}
    )
    assert (summary["headline_from"], summary["headline_to"]) == (
        "acq_last",
        "painted",
    )
    assert summary["headline"]["p50"] == pytest.approx(2.5)  # base+2 -> base+4.5
    assert summary["arrival_interval"]["p50"] == pytest.approx(10.0)
    assert summary["wire_mb_per_s"]["p50"] == pytest.approx(400.0)
    gpu = summary["gpu_servers"]
    assert set(gpu) == {"1", "2"} and gpu["1"]["n"] == 2
    assert gpu["2"]["decon_s"] == pytest.approx(0.4)


def test_old_profiles_place_the_view_buffer_from_stage_durations(tmp_path):
    _synthetic(tmp_path, legacy_profiles=True)
    summary = _summary(tmp_path)
    # started 0.8 + read/decon/dsr/view 0.7 = 1.5: the same buffer time
    assert summary["hops"]["gpu"]["p50"] == pytest.approx(0.7)
    assert summary["side"]["encode"]["p50"] == pytest.approx(0.2)
    # completed = started + total_s = 1.8
    assert summary["side"]["finish"]["p50"] == pytest.approx(0.1)


def test_flatness_reports_a_growing_lag():
    rows = {
        t: {"acq_last": 100.0 * t, "painted": 100.0 * t + 2.0 + 0.1 * t}
        for t in range(9)
    }
    flat = trace_report.summarize(rows)["flatness"]
    assert flat["slope_s_per_t"] == pytest.approx(0.1)
    assert flat["first_third_p95"] == pytest.approx(2.2)
    assert flat["last_third_p95"] == pytest.approx(2.8)


def test_out_writes_the_summary_and_a_row_per_timepoint(tmp_path, capsys):
    _synthetic(tmp_path)
    out = tmp_path / "run"
    trace_report.main(["--jobs", str(tmp_path), "--out", str(out)])
    text = capsys.readouterr().out
    assert "last plane -> painted" in text and "flatness" not in text  # 2 points
    summary = json.loads((out / "summary.json").read_text())
    assert summary["session_id"] == "sess-abc"
    lines = (out / "timeline.csv").read_text().splitlines()
    assert lines[0].startswith("t,acq_last,sent,recv") and lines[0].endswith(",lag_s")
    assert len(lines) == 3 and lines[1].split(",")[-1] == "2.5000"


def test_report_falls_back_to_received_without_client_timestamps(tmp_path, capsys):
    _synthetic(tmp_path, with_client=False)
    trace_report.main(["--jobs", str(tmp_path), "--base", "Cell_011"])
    out = capsys.readouterr().out
    assert "Session sess-abc (Cell_011): 2 timepoint(s)" in out
    assert "received -> painted" in out
    trace_report.main(["--jobs", str(tmp_path), "--json"])
    summary = json.loads(capsys.readouterr().out)
    assert summary["hops"]["wire"] == {"n": 0}
    assert summary["headline"]["p50"] == pytest.approx(4.5 - 2.5)


def test_report_rejects_an_unknown_session(tmp_path):
    _synthetic(tmp_path)
    with pytest.raises(SystemExit):
        trace_report.main(["--jobs", str(tmp_path), "--session", "nope"])


def test_frames_sent_before_the_first_ack_use_the_sessions_clock_offset(tmp_path):
    _synthetic(tmp_path)
    events = trace.read(jobs=tmp_path)
    for e in events:  # both t=0 volumes left before the first ACK came back
        if e["ev"] == "frame" and e["t"] == 0:
            del e["clock_offset_s"]
    rows = trace_report.timeline("sess-abc", events, [], {})
    assert rows[0]["acq_last"] == pytest.approx(1002.0)
    assert rows[0]["sent"] == pytest.approx(1002.25)
