# Ruff style: Compliant
"""Per-hop timing trace for the live path, acquisition to screen.

Each process on the path appends one JSON line per event to a file under
`<jobs dir>/profiling/` (the same directory as the receiver's and the GPU
servers' own profiling): the receiver and the live lane write
`live_trace.jsonl`, naparym-live writes `live_view.jsonl`.
`opym.stream.trace_report` joins them, plus the GPU servers' `S<id>.jsonl`,
into per-timepoint hop latencies.

Every line carries `ev` (the event name) and `at` (Argus wall-clock epoch
seconds when it was written). Times the acquisition client measured on its
own clock are recorded as sent, with the client's `clock_offset_s`; the
report converts them. Writing is best effort: a failed trace write is logged
at debug level and never raises into the caller.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

from opym import lanes

logger = logging.getLogger(__name__)

TRACE_NAME = "live_trace.jsonl"
VIEW_TRACE_NAME = "live_view.jsonl"

_lock = threading.Lock()


def trace_path(name: str = TRACE_NAME, jobs: Path | None = None) -> Path:
    return (jobs or lanes.jobs_dir()) / "profiling" / name


def record(
    event: str, *, name: str = TRACE_NAME, jobs: Path | None = None, **fields
) -> None:
    """Append one `{"ev": event, "at": now, **fields}` line."""
    line = json.dumps({"ev": event, "at": time.time(), **fields}) + "\n"
    path = trace_path(name, jobs)
    try:
        with _lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a") as f:
                f.write(line)
    except OSError:
        logger.debug("Could not write trace event %s to %s", event, path, exc_info=True)


def read(name: str = TRACE_NAME, jobs: Path | None = None) -> list[dict]:
    """Every parseable line of one trace file, oldest first."""
    try:
        text = trace_path(name, jobs).read_text()
    except OSError:
        return []
    out = []
    for line in text.splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out
