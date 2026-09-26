# Ruff style: Compliant
"""`opym-live-trace`: where a live session's timepoints spent their time.

Joins the live-path trace (`opym.stream.trace`: receiver, live lane and
naparym-live events) with the GPU servers' per-ticket profiling
(`profiling/S<id>.jsonl`, by ticket name) into one timeline per timepoint,
then prints p50/p95/max for each hop.

The view path, in order -- every hop here is time the picture is behind:

    client   last plane acquired -> FRAME sent      (needs client timestamps)
    wire     FRAME sent -> received on Argus        (needs client timestamps)
    stage    received -> raw store written, volume complete
    dispatch complete -> live ticket queued
    claim    ticket queued -> a GPU server claimed it
    gpu      claimed -> view buffer written (read, decon, DSR, buffer)
    detect   view buffer written -> naparym-live saw every channel's
    build    seen -> naparym-live's layers updated
    paint    layers updated -> the frame swapped on screen

Beside it, off the view path: encode (buffer -> processed store written),
finish (store -> ticket completed), reap (completed -> the lane noticed),
archive (noticed -> every channel's chunks copied to GPFS).

A timepoint's time at each point is its LAST channel's: the view can't show
t until every channel of t is there. Client times are converted to Argus
time with the offset the client measured (see protocol.py). The headline is
"last plane -> painted" when both ends exist, falling back to "received"
for the start and "shown" (layers updated) for the end. Flatness: the
least-squares slope of the headline lag against t, and its p95 over the
first and the last third of the timepoints -- keeping pace means both stay
put.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from opym import lanes
from opym.stream import trace

VIEW_PATH = (
    ("client", "acq_last", "sent"),
    ("wire", "sent", "recv"),
    ("stage", "recv", "staged"),
    ("dispatch", "staged", "ticket"),
    ("claim", "ticket", "claim"),
    ("gpu", "claim", "buffer"),
    ("detect", "buffer", "seen"),
    ("build", "seen", "shown"),
    ("paint", "shown", "painted"),
)
SIDE_PATH = (
    ("encode", "buffer", "store"),
    ("finish", "store", "gpu_done"),
    ("reap", "gpu_done", "reaped"),
    ("archive", "reaped", "archived"),
)
POINTS = (
    "acq_last",
    "sent",
    "recv",
    "staged",
    "ticket",
    "claim",
    "buffer",
    "seen",
    "shown",
    "painted",
    "store",
    "gpu_done",
    "reaped",
    "archived",
)
GPU_STAGES = ("read_s", "decon_s", "dsr_s", "view_s", "write_s", "total_s")


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


def _num(rec: dict, key: str) -> float | None:
    """A profile number; MATLAB writes NaN as null."""
    v = rec.get(key)
    return float(v) if isinstance(v, (int, float)) and v == v else None


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


def ticket_points(prof: dict) -> dict:
    """When a ticket was claimed, wrote its view buffer, wrote its store and
    completed, from its profile: the absolute times the server records, or
    for profiles that predate them, its start plus its stage durations."""
    start = _num(prof, "started_at")
    if start is None:
        return {}
    stages = [_num(prof, k) for k in ("read_s", "decon_s", "dsr_s", "view_s")]
    buffer = _num(prof, "buffer_at")
    if buffer is None and all(s is not None for s in stages):
        buffer = start + sum(stages)  # type: ignore[arg-type]
    store = _num(prof, "store_at")
    if store is None and buffer is not None and _num(prof, "write_s") is not None:
        store = buffer + _num(prof, "write_s")  # type: ignore[operator]
    done = _num(prof, "done_at")
    if done is None and _num(prof, "total_s") is not None:
        done = start + _num(prof, "total_s")  # type: ignore[operator]
    return {"claim": start, "buffer": buffer, "store": store, "gpu_done": done}


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
    """Per timepoint: the time it reached each point on the path (the last
    channel's), plus the tickets it took."""
    frames: dict[int, list[dict]] = {}
    # (t, c) -> first attempt's points; c is None for a multi-channel ticket
    per_tc: dict[tuple[int, int | None], dict] = {}
    reaped: dict[tuple[int, int | None], float] = {}
    archived: dict[int, float] = {}
    for e in events:
        if e.get("session_id") != session_id:
            continue
        ev = e.get("ev")
        if ev == "frame":
            frames.setdefault(int(e["t"]), []).append(e)
        elif ev == "ticket":
            for t in e["timepoints"]:
                key = (int(t), e.get("c"))
                if key in per_tc:  # keep the first attempt
                    continue
                per_tc[key] = {
                    "ticket": e["at"],
                    "name": e["ticket"],
                    **ticket_points(profiles.get(e["ticket"], {})),
                }
        elif ev == "ticket_done":
            for t in e["timepoints"]:
                reaped.setdefault((int(t), e.get("c")), e["at"])
        elif ev == "archived":
            archived.setdefault(int(e["t"]), e["at"])

    rows: dict[int, dict] = {}
    for (t, c), pts in per_tc.items():
        r = rows.setdefault(t, {"tickets": []})
        r["tickets"].append(pts["name"])
        pts = {**pts, "reaped": reaped.get((t, c))}
        for k in ("ticket", "claim", "buffer", "store", "gpu_done", "reaped"):
            r[k] = _latest([r.get(k), pts.get(k)])
    for t, at in archived.items():
        rows.setdefault(t, {"tickets": []})["archived"] = at

    offsets = [
        float(e["clock_offset_s"])
        for evs in frames.values()
        for e in evs
        if e.get("clock_offset_s") is not None
    ]
    session_offset = statistics.median(offsets) if offsets else None
    for t, evs in frames.items():
        r = rows.setdefault(t, {"tickets": []})
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

    # naparym-live: the first time each timepoint was shown, and the first
    # frame painted after that.
    for e in sorted(view_events, key=lambda e: e.get("at", 0)):
        if e.get("session_id") != session_id:
            continue
        for t in e.get("timepoints") or ():
            r = rows.setdefault(int(t), {"tickets": []})
            if e.get("ev") == "shown" and r.get("shown") is None:
                r["shown"] = e["at"]
                r["seen"] = e.get("seen_s")
                r["phases"] = e.get("phases")
            elif (
                e.get("ev") == "painted"
                and r.get("shown") is not None
                and r.get("painted") is None
            ):
                r["painted"] = e["at"]
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


def _hops(rows: dict[int, dict], path) -> dict:
    return {
        name: _stats(
            [
                r[b] - r[a]
                for r in rows.values()
                if r.get(a) is not None and r.get(b) is not None
            ]
        )
        for name, a, b in path
    }


def headline_points(rows: dict[int, dict]) -> tuple[str, str]:
    start = "acq_last" if any(r.get("acq_last") for r in rows.values()) else "recv"
    end = "painted" if any(r.get("painted") for r in rows.values()) else "shown"
    return start, end


def lags(rows: dict[int, dict]) -> list[tuple[int, float]]:
    """(t, headline lag) for every timepoint that has both ends, by t."""
    start, end = headline_points(rows)
    return sorted(
        (t, r[end] - r[start])
        for t, r in rows.items()
        if r.get(start) is not None and r.get(end) is not None
    )


def flatness(points: list[tuple[int, float]]) -> dict:
    """Does the lag grow over the run? The least-squares slope of lag
    against t (seconds per timepoint), and the p95 over the first and last
    thirds of the timepoints."""
    n = len(points)
    if n < 3:
        return {"n": n}
    ts = [float(t) for t, _ in points]
    ys = [y for _, y in points]
    mt, my = statistics.fmean(ts), statistics.fmean(ys)
    var = sum((t - mt) ** 2 for t in ts)
    slope = sum((t - mt) * (y - my) for t, y in zip(ts, ys, strict=True)) / var
    third = max(1, n // 3)
    return {
        "n": n,
        "slope_s_per_t": slope,
        "first_third_p95": _stats(ys[:third])["p95"],
        "last_third_p95": _stats(ys[-third:])["p95"],
    }


def gpu_breakdown(rows: dict[int, dict], profiles: dict) -> dict[str, dict]:
    """Per GPU server, p50 of each stage over this session's tickets."""
    by_server: dict[str, list[dict]] = {}
    for r in rows.values():
        for name in r.get("tickets", ()):
            prof = profiles.get(name)
            if prof:
                by_server.setdefault(str(prof.get("server_id", "?")), []).append(prof)
    out = {}
    for sid, profs in sorted(by_server.items()):
        out[sid] = {"n": len(profs)}
        for k in GPU_STAGES:
            vals = [v for p in profs if (v := _num(p, k)) is not None]
            out[sid][k] = statistics.median(vals) if vals else None
    return out


def summarize(rows: dict[int, dict], profiles: dict | None = None) -> dict:
    start, end = headline_points(rows)
    lag_points = lags(rows)
    recv = sorted(r["recv"] for r in rows.values() if r.get("recv") is not None)
    # Per frame: its bytes over its own sent -> received time. A lower bound
    # on the link's rate whenever a frame queued behind another one.
    wire = [w for r in rows.values() for w in r.get("wire_mb_per_s", ())]
    return {
        "timepoints": len(rows),
        "hops": _hops(rows, VIEW_PATH),
        "side": _hops(rows, SIDE_PATH),
        "headline_from": start,
        "headline_to": end,
        "headline": _stats([y for _, y in lag_points]),
        "flatness": flatness(lag_points),
        "arrival_interval": _stats([b - a for a, b in zip(recv, recv[1:])]),
        "wire_mb_per_s": _stats(wire),
        "gpu_servers": gpu_breakdown(rows, profiles or {}),
    }


def write_csv(rows: dict[int, dict], path: Path) -> None:
    """One row per timepoint: each point in seconds after the headline's
    start (the last plane, or reception), and the headline lag."""
    start, end = headline_points(rows)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", *POINTS, "lag_s"])
        for t in sorted(rows):
            r = rows[t]
            t0 = r.get(start)
            rel = [
                "" if t0 is None or r.get(k) is None else f"{r[k] - t0:.4f}"
                for k in POINTS
            ]
            lag = "" if t0 is None or r.get(end) is None else f"{r[end] - t0:.4f}"
            w.writerow([t, *rel, lag])


def _fmt(stats: dict) -> str:
    if not stats.get("n"):
        return "      —       —       —     0"
    return "{p50:7.2f} {p95:7.2f} {max:7.2f} {n:5d}".format(**stats)


def _label(point: str) -> str:
    return {
        "acq_last": "last plane",
        "recv": "received",
        "painted": "painted",
        "shown": "shown",
    }[point]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="opym-live-trace", description=__doc__.splitlines()[0]
    )
    ap.add_argument("--session", help="session id (or a prefix of it)")
    ap.add_argument("--base", help="session base_name, e.g. Cell_001")
    ap.add_argument("--jobs", type=Path, help="jobs dir (default: PETAKIT_JOBS_DIR)")
    ap.add_argument("--json", action="store_true", help="print the summary as JSON")
    ap.add_argument(
        "--out",
        type=Path,
        help="also write summary.json and timeline.csv (per timepoint) here",
    )
    args = ap.parse_args(argv)

    jobs = args.jobs or lanes.jobs_dir()
    events = trace.read(trace.TRACE_NAME, jobs)
    session_id = pick_session(events, args.session, args.base)
    profiles = _server_profiles(jobs)
    rows = timeline(
        session_id, events, trace.read(trace.VIEW_TRACE_NAME, jobs), profiles
    )
    summary = {"session_id": session_id, **summarize(rows, profiles)}
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
        write_csv(rows, args.out / "timeline.csv")
    if args.json:
        print(json.dumps(summary, indent=1))
        return
    base = next(
        (e.get("base_name") for e in events if e.get("session_id") == session_id), "?"
    )
    print(f"Session {session_id} ({base}): {summary['timepoints']} timepoint(s)")
    print(f"{'view path':<20} {'p50 s':>7} {'p95 s':>7} {'max s':>7} {'n':>5}")
    for name, _a, _b in VIEW_PATH:
        print(f"  {name:<18}{_fmt(summary['hops'][name])}")
    label = f"{_label(summary['headline_from'])} -> {_label(summary['headline_to'])}"
    print(f"{label:<20}{_fmt(summary['headline'])}")
    flat = summary["flatness"]
    if flat.get("n", 0) >= 3:
        print(
            f"{'flatness':<20} slope {flat['slope_s_per_t'] * 1000:+.1f} ms/timepoint,"
            f" p95 first third {flat['first_third_p95']:.2f} s,"
            f" last third {flat['last_third_p95']:.2f} s"
        )
    print("off the view path")
    for name, _a, _b in SIDE_PATH:
        print(f"  {name:<18}{_fmt(summary['side'][name])}")
    print(f"{'arrival interval':<20}{_fmt(summary['arrival_interval'])}")
    if summary["wire_mb_per_s"].get("n"):
        print(f"{'wire MB/s':<20}{_fmt(summary['wire_mb_per_s'])}")
    if summary["gpu_servers"]:
        print("GPU server p50 s     " + " ".join(f"{k[:-2]:>7}" for k in GPU_STAGES))
        for sid, g in summary["gpu_servers"].items():
            cells = " ".join(
                "      —" if g[k] is None else f"{g[k]:7.3f}" for k in GPU_STAGES
            )
            print(f"  S{sid:<6} n={g['n']:<5}   {cells}")
    if args.out:
        print(f"Wrote {args.out / 'summary.json'} and {args.out / 'timeline.csv'}")


if __name__ == "__main__":
    main()
