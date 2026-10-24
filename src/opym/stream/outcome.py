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
from pathlib import Path
from typing import Any

STATUS_NAME = "stream_status.json"

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
    elif reason == "paused":
        summary = (
            f"paused by the microscope PC: it could not reach Argus for "
            f"longer than its buffer holds ({of})"
        )
    elif reason == "idle_timeout":
        summary = f"stream stalled: {detail or 'no new data for 10 min'} ({of})"
    elif reason == "stalled":
        summary = f"stream stalled: {detail} ({of})"
    elif reason == "client_abort":
        summary = f"stopped on the microscope (run canceled or app closed): {of}"
        remedy = REMEDY_IF_LONGER
    elif reason == "client_error":
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
    tmp = path.with_name(f".{STATUS_NAME}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(record, indent=1))
    os.replace(tmp, path)
    return path


def read_status(leaf_dir: Path) -> dict[str, Any] | None:
    try:
        return json.loads((Path(leaf_dir) / STATUS_NAME).read_text())
    except (OSError, ValueError):
        return None
