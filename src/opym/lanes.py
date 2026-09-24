# Ruff style: Compliant
"""Priority lanes for the PetaKit GPU queue: live (streaming) work always goes
ahead of backfill.

- Live tickets go to `<jobs>/queue_live/`. run_petakit_server.m claims from
  there before it looks at the backfill queue, `<jobs>/queue/`.
- While a live acquisition is open, the stream receiver holds
  `<jobs>/LIVE_LEASE.json`, rewriting it every LEASE_REFRESH_S. While the
  lease is fresh, servers claim no backfill ticket and the backfill starts no
  pass. Freshness is the file's mtime, so a receiver that crashes releases
  the GPUs by itself within LEASE_MAX_AGE_S.
- `backfill_inflight()` lets the backfill cap how many of its tickets are
  queued or running at once. Tickets are then built shortly before they run,
  with current parameters, instead of days ahead (on 2026-09-24, 36 queued
  tickets still carried decon parameters superseded three days earlier).

The jobs dir defaults to /dev/shm/petakit_jobs. PETAKIT_JOBS_DIR overrides it
in every component (MATLAB server, supervisor, receiver, backfill), so a test
setup never touches production's queues.
"""

from __future__ import annotations

import fcntl
import json
import os
import time
from collections.abc import Collection, Iterator
from contextlib import contextmanager
from pathlib import Path

DEFAULT_JOBS_DIR = "/dev/shm/petakit_jobs"
LIVE_QUEUE_NAME = "queue_live"
BACKFILL_QUEUE_NAME = "queue"
LEASE_NAME = "LIVE_LEASE.json"
# run_petakit_server.m hard-codes the same 60 s staleness limit.
LEASE_MAX_AGE_S = 60.0
LEASE_REFRESH_S = 10.0


def jobs_dir() -> Path:
    return Path(os.environ.get("PETAKIT_JOBS_DIR") or DEFAULT_JOBS_DIR)


def live_queue_dir(jobs: Path | None = None) -> Path:
    return (jobs or jobs_dir()) / LIVE_QUEUE_NAME


def backfill_queue_dir(jobs: Path | None = None) -> Path:
    return (jobs or jobs_dir()) / BACKFILL_QUEUE_NAME


def lease_path(jobs: Path | None = None) -> Path:
    return (jobs or jobs_dir()) / LEASE_NAME


def write_lease(sessions: Collection[str], jobs: Path | None = None) -> None:
    """Create or refresh the live lease. The content is informational (who
    holds it, for which sessions); only the file's mtime is load-bearing."""
    path = lease_path(jobs)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{LEASE_NAME}.tmp")
    tmp.write_text(
        json.dumps(
            {
                "sessions": sorted(sessions),
                "pid": os.getpid(),
                "updated_at": time.time(),
            }
        )
    )
    os.replace(tmp, path)


def release_lease(jobs: Path | None = None) -> None:
    lease_path(jobs).unlink(missing_ok=True)


def live_lease_active(
    jobs: Path | None = None,
    max_age_s: float = LEASE_MAX_AGE_S,
    now: float | None = None,
) -> bool:
    """True while a live acquisition holds a fresh lease."""
    try:
        mtime = lease_path(jobs).stat().st_mtime
    except OSError:
        return False
    return (time.time() if now is None else now) - mtime <= max_age_s


def backfill_inflight(jobs: Path | None = None) -> int:
    """Backfill tickets queued or running: everything in the backfill queue,
    including `.active_*` claims and `.requeue_*` in-progress requeues
    (pathlib's glob does not hide dotfiles)."""
    return sum(1 for _ in backfill_queue_dir(jobs).glob("*.json"))


def backfill_admission_open(cap: int | None, jobs: Path | None = None) -> bool:
    """Whether the backfill may queue another ticket now: no live lease, and
    fewer than `cap` backfill tickets queued or running (`None` or <= 0 means
    no cap). A cheap unlocked pre-check; `backfill_admission` is the real one."""
    if live_lease_active(jobs):
        return False
    return not cap or cap <= 0 or backfill_inflight(jobs) < cap


@contextmanager
def backfill_admission(cap: int | None, jobs: Path | None = None) -> Iterator[bool]:
    """Yields whether one more backfill ticket may be queued, holding an
    exclusive lock across processes until the block exits. Queue the ticket
    inside the block so parallel backfill workers can't all see room at once
    and overshoot the cap together."""
    base = jobs or jobs_dir()
    base.mkdir(parents=True, exist_ok=True)
    with open(base / ".backfill_admission.lock", "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield backfill_admission_open(cap, jobs)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


class LeaseKeeper:
    """Holds the lease while `update()` is told about open sessions, and
    releases it when told there are none. Call `update` on every poll; it
    only touches the file every `refresh_s`."""

    def __init__(self, jobs: Path | None = None, refresh_s: float = LEASE_REFRESH_S):
        self._jobs = jobs
        self._refresh_s = refresh_s
        self._held = False
        self._last_write = float("-inf")

    @property
    def held(self) -> bool:
        return self._held

    def update(self, sessions: Collection[str], now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        if sessions:
            if not self._held or now - self._last_write >= self._refresh_s:
                write_lease(sessions, self._jobs)
                self._held = True
                self._last_write = now
        elif self._held:
            release_lease(self._jobs)
            self._held = False
