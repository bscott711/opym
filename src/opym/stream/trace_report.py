# Ruff style: Compliant
"""`opym-live-trace`: where a live session's timepoints spent their time.

Joins the live-path trace (`opym.stream.trace`: receiver, live lane and
naparym-live events) with the GPU servers' per-ticket profiling
(`profiling/S<id>.jsonl`, by ticket name) into one timeline per timepoint,
then prints p50/p95/max for each hop:

    client   last plane acquired -> FRAME sent      (needs client timestamps)
    wire     FRAME sent -> received on Argus        (needs client timestamps)
    stage    received -> raw store + decon input written
    dispatch staged -> live ticket queued
    claim    ticket queued -> a GPU server claimed it
    gpu      claimed -> decon + DSR finished
    reap     GPU finished -> live lane noticed
    copy     noticed -> viewer store updated (napari can show it)
    viewer   viewer store updated -> naparym-live built the new layers

A timepoint's time at each point is its LAST channel's: the view can't show
t until every channel of t has arrived. Client times are converted to Argus
time with the offset the client measured (see protocol.py). The headline is
"last plane -> shown" when client timestamps exist, else "received -> shown".
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from opym import lanes
from opym.stream import trace

HOPS = (
    ("client", "acq_last", "sent"),
    ("wire", "sent", "recv"),
    ("stage", "recv", "staged"),
    ("dispatch", "staged", "ticket"),
    ("claim", "ticket", "claim"),
    ("gpu", "claim", "gpu_done"),
    ("reap", "gpu_done", "reaped"),
    ("copy", "reaped", "view"),
    ("viewer", "view", "shown"),
)


def _client_time(ev: dict, key: str, session_offset: float | None) -> float | None:
    """A client-clock time converted to Argus time. Frames the client sent
    before its first ACK came back carry no offset of their own; they use
    the session's (one client clock, one offset)."""
    value = ev.get(key)
    offset = ev.get("clock_offset_s", session_offset)
    if value is None or offset is None:
        return None
    return float(value) + float(offset)


def _latest(values) -> float | None:
    values = [v for v in values if v is not None]
    return max(values) if values else None


def _server_profiles(jobs: Path) -> dict[str, dict]:
    """GPU-server profile records by ticket name."""
    out = {}
    for path in sorted((jobs / "profiling").glob("S*.jsonl")):
        for line in path.read_text().splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if isinstance(rec, dict) and rec.get("ticket"):
                out[rec["ticket"]] = rec
    return out


def pick_session(events: list[dict], session: str | None, base: str | None) -> str:
    """The session to report: by id prefix, by base_name, else the newest."""
    frames = [e for e in events if e.get("ev") == "frame"]
    if session:
        ids = {e["session_id"] for e in frames if e["session_id"].startswith(session)}
    elif base:
        ids = {e["session_id"] for e in frames if e.get("base_name") == base}
    else:
        ids = {e["session_id"] for e in frames}
    if not ids:
        raise SystemExit("No matching live session in the trace.")
    last_seen = {
        sid: max(e["at"] for e in frames if e["session_id"] == sid) for sid in ids
    }
    return max(last_seen, key=last_seen.get)


def timeline(
    session_id: str, events: list[dict], view_events: list[dict], profiles: dict
) -> dict[int, dict]:
    """Per timepoint: the time it reached each point on the path."""
    rows: dict[int, dict] = {}

    def row(t: int) -> dict:
        return rows.setdefault(int(t), {})

    frames: dict[int, list[dict]] = {}
    for e in events:
        if e.get("session_id") != session_id:
            continue
        ev = e.get("ev")
        if ev == "frame":
            frames.setdefault(int(e["t"]), []).append(e)
        elif ev == "ticket":
            prof = profiles.get(e["ticket"], {})
            start = prof.get("started_at")
            total = prof.get("total_s")
            for t in e["timepoints"]:
                r = row(t)
                if "ticket" in r:  # keep the first attempt
                    continue
                r["ticket"] = e["at"]
                r["ticket_name"] = e["ticket"]
                r["claim"] = start
                r["gpu_done"] = start + total if start and total else None
        elif ev == "ticket_done":
            for t in e["timepoints"]:
                row(t).setdefault("reaped", e["at"])
        elif ev == "view_ready":
            row(e["t"]).setdefault("view", e["at"])
    offsets = [
        float(e["clock_offset_s"])
        for evs in frames.values()
        for e in evs
        if e.get("clock_offset_s") is not None
    ]
    session_offset = statistics.median(offsets) if offsets else None
    for t, evs in frames.items():
        r = row(t)
        r["recv"] = _latest(e.get("recv_s") for e in evs)
        r["staged"] = _latest(e.get("staged_s") for e in evs)
        r["acq_last"] = _latest(
            _client_time(e, "acq_last_s", session_offset) for e in evs
        )
        r["sent"] = _latest(_client_time(e, "sent_s", session_offset) for e in evs)
        r["wire_mb_per_s"] = [
            int(e["bytes"]) / 1e6 / (e["recv_s"] - sent)
            for e in evs
            if e.get("bytes")
            and (sent := _client_time(e, "sent_s", session_offset)) is not None
            and e["recv_s"] > sent
        ]
    for e in view_events:
        if e.get("ev") == "shown" and e.get("session_id") == session_id:
            for t in e.get("timepoints") or ():
                r = row(t)
                if r.get("shown") is None or e["at"] < r["shown"]:
                    r["shown"] = e["at"]
    return rows


def _stats(values: list[float]) -> dict:
    values = sorted(values)
    if not values:
        return {"n": 0}
    p95 = values[min(len(values) - 1, int(round(0.95 * (len(values) - 1))))]
    return {
        "n": len(values),
        "p50": statistics.median(values),
        "p95": p95,
        "max": values[-1],
    }


def summarize(rows: dict[int, dict]) -> dict:
    hops = {}
    for name, a, b in HOPS:
        hops[name] = _stats(
            [
                r[b] - r[a]
                for r in rows.values()
                if r.get(a) is not None and r.get(b) is not None
            ]
        )
    head_from = "acq_last" if any(r.get("acq_last") for r in rows.values()) else "recv"
    headline = _stats(
        [
            r["shown"] - r[head_from]
            for r in rows.values()
            if r.get("shown") is not None and r.get(head_from) is not None
        ]
    )
    recv = sorted(r["recv"] for r in rows.values() if r.get("recv") is not None)
    # Per frame: its bytes over its own sent -> received time. A lower bound
    # on the link's rate whenever a frame queued behind another one.
    wire = [w for r in rows.values() for w in r.get("wire_mb_per_s", ())]
    return {
        "timepoints": len(rows),
        "hops": hops,
        "headline_from": head_from,
        "headline": headline,
        "arrival_interval": _stats([b - a for a, b in zip(recv, recv[1:])]),
        "wire_mb_per_s": _stats(wire),
    }


def _fmt(stats: dict) -> str:
    if not stats.get("n"):
        return "      —       —       —     0"
    return "{p50:7.2f} {p95:7.2f} {max:7.2f} {n:5d}".format(**stats)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="opym-live-trace", description=__doc__.splitlines()[0]
    )
    ap.add_argument("--session", help="session id (or a prefix of it)")
    ap.add_argument("--base", help="session base_name, e.g. Cell_001")
    ap.add_argument("--jobs", type=Path, help="jobs dir (default: PETAKIT_JOBS_DIR)")
    ap.add_argument("--json", action="store_true", help="print the summary as JSON")
    args = ap.parse_args(argv)

    jobs = args.jobs or lanes.jobs_dir()
    events = trace.read(trace.TRACE_NAME, jobs)
    session_id = pick_session(events, args.session, args.base)
    rows = timeline(
        session_id,
        events,
        trace.read(trace.VIEW_TRACE_NAME, jobs),
        _server_profiles(jobs),
    )
    summary = summarize(rows)
    if args.json:
        print(json.dumps({"session_id": session_id, **summary}, indent=1))
        return
    base = next(
        (e.get("base_name") for e in events if e.get("session_id") == session_id), "?"
    )
    print(f"Session {session_id} ({base}): {summary['timepoints']} timepoint(s)")
    print(f"{'hop':<10} {'p50 s':>7} {'p95 s':>7} {'max s':>7} {'n':>5}")
    for name, _a, _b in HOPS:
        print(f"{name:<10} {_fmt(summary['hops'][name])}")
    label = (
        "last plane -> shown"
        if summary["headline_from"] == "acq_last"
        else "received -> shown"
    )
    print(f"{label:<20}{_fmt(summary['headline'])}")
    print(f"{'arrival interval':<20}{_fmt(summary['arrival_interval'])}")
    if summary["wire_mb_per_s"].get("n"):
        print(f"{'wire MB/s':<20}{_fmt(summary['wire_mb_per_s'])}")


if __name__ == "__main__":
    main()
