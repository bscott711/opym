# Ruff style: Compliant
"""
How a streamed session ended, in words every UI can show.

The receiver decides each session's end once (`summarize`) and writes it
beside the run on GPFS (`<raw_root>/<base_name>/stream_status.json`), into
the live lane's status and progress files (so naparym-live shows it), and
into `profiling/receiver.jsonl`. The dashboard, naparym-live and the logs
then say the same thing, and a run that came in empty says why and what to
do about it instead of only "0/100 · failed" (2026-10-24: ten runs in a row
streamed nothing and nothing said so).
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

from opym import lanes
from opym.stream.protocol import (
    REASON_CLIENT_ABORT,
    REASON_CLIENT_ERROR,
    REASON_IDLE_TIMEOUT,
    REASON_PAUSED,
    REASON_STALLED,
)

STATUS_NAME = "stream_status.json"

# The receiver's live health record (`<jobs>/stream_health.json`): rewritten
# every HEALTH_EVERY_S while it runs; a reader treats one older than
# HEALTH_STALE_S as the receiver being down.
HEALTH_NAME = "stream_health.json"
HEALTH_EVERY_S = 5.0
HEALTH_STALE_S = 60.0

COMPLETE = "complete"
PARTIAL = "partial"
NO_DATA = "no_data"

REMEDY_RESEND = (
    "The run is in the microscope PC's local save: send it to Argus with "
    "opym-backfill-send (or Globus)."
)
REMEDY_IF_LONGER = (
    "If the run went further than this, send its local save with "
    "opym-backfill-send (or Globus)."
)


def span(seconds: float) -> str:
    """A duration for people: "45 s" under two minutes, else "12 min"."""
    return f"{seconds:.0f} s" if seconds < 120 else f"{seconds / 60:.0f} min"


def describe_client(status: dict[str, Any] | None) -> str:
    """One line from a client HEARTBEAT: links, backlog, last error."""
    if not status:
        return ""
    parts = []
    if "links_total" in status:
        parts.append(f"{status.get('links_up', 0)}/{status['links_total']} links up")
    if "unacked_bytes" in status:
        parts.append(f"{int(status['unacked_bytes']) / 1e6:.0f} MB waiting to send")
    if "submitted" in status:
        parts.append(f"{status['submitted']} volume(s) acquired")
    if status.get("last_error"):
        parts.append(f"last error: {status['last_error']}")
    return ", ".join(parts)


def summarize(
    reason: str,
    received_pairs: set[tuple[int, int]],
    num_timepoints: int,
    n_channels: int,
    detail: str = "",
) -> dict[str, Any]:
    """The end of one session: outcome, a one-line summary and the remedy.

    `received_pairs` are the (t, c) pairs that arrived; a timepoint counts
    once every channel of it did.
    """
    per_t: dict[int, int] = {}
    for t, _c in received_pairs:
        per_t[t] = per_t.get(t, 0) + 1
    got = sum(1 for n in per_t.values() if n >= max(1, n_channels))
    want = max(0, int(num_timepoints))
    if want and got >= want:
        outcome = COMPLETE
    elif received_pairs:
        outcome = PARTIAL
    else:
        outcome = NO_DATA
    of = f"{got}/{want} timepoints arrived"

    remedy = REMEDY_RESEND
    if outcome == COMPLETE:
        summary, remedy = f"complete: {of}", ""
    elif reason == REASON_PAUSED:
        summary = (
            f"paused by the microscope PC: it could not reach Argus for "
            f"longer than its buffer holds ({of})"
        )
    elif reason == REASON_IDLE_TIMEOUT:
        summary = f"stream stalled: {detail or 'no new data for 10 min'} ({of})"
    elif reason == REASON_STALLED:
        summary = f"stream stalled: {detail} ({of})"
    elif reason == REASON_CLIENT_ABORT:
        summary = f"stopped on the microscope (run canceled or app closed): {of}"
        remedy = REMEDY_IF_LONGER
    elif reason == REASON_CLIENT_ERROR:
        summary = f"the microscope PC's stream sender failed: {detail} ({of})"
    else:  # "complete" from a client whose run came in short
        summary = f"the microscope reported the run finished, but only {of}"
    if outcome == NO_DATA:
        summary = "no data reached Argus -- " + summary
    return {
        "reason": reason,
        "outcome": outcome,
        "timepoints_received": got,
        "num_timepoints": want,
        "summary": summary,
        "detail": detail,
        "remedy": remedy,
    }


def write_status(leaf_dir: Path, record: dict[str, Any]) -> Path:
    """Write `record` to `<leaf_dir>/stream_status.json` atomically."""
    leaf_dir = Path(leaf_dir)
    leaf_dir.mkdir(parents=True, exist_ok=True)
    path = leaf_dir / STATUS_NAME
    # One tmp name per call: the drain pool and the status thread both write
    # this file from the receiver process, so the pid alone is not unique.
    tmp = path.with_name(f".{STATUS_NAME}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(json.dumps(record, indent=1))
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def read_status(leaf_dir: Path) -> dict[str, Any] | None:
    try:
        return json.loads((Path(leaf_dir) / STATUS_NAME).read_text())
    except (OSError, ValueError):
        return None


def health_path(jobs: Path | str | None = None) -> Path:
    """`<jobs>/stream_health.json`; `jobs` defaults to `lanes.jobs_dir()`
    (so `PETAKIT_JOBS_DIR` is honoured)."""
    return (Path(jobs) if jobs is not None else lanes.jobs_dir()) / HEALTH_NAME


def read_health(path: Path | str | None = None) -> dict[str, Any] | None:
    """The receiver's health record, or None if it is missing or unreadable.

    Staleness is the caller's call: compare the record's `updated_at` (epoch
    seconds) against `HEALTH_STALE_S`.
    """
    try:
        rec = json.loads(Path(path or health_path()).read_text())
    except (OSError, ValueError):
        return None
    return rec if isinstance(rec, dict) else None
