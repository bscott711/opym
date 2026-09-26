# opym/local_gpu_worker.py
"""
Supervisor for the PetaKit job queue: one PetaKit5D MATLAB server
(run_petakit_server.m) per GPU, each kept alive independently.

- A server is launched whenever claimable tickets exist and it isn't
  running. Servers still shut themselves down after PETAKIT_IDLE_TIMEOUT.
- A server that dies (crash, OOM, kill) is relaunched on its own. The
  previous watchdog waited on BOTH servers before doing anything, so one dead
  server silently halved throughput until the other also exited. On
  2026-09-23 server 1 died at 13:29, server 2 hung at 15:47, and nothing ran
  for 43 hours.
- A ticket held by a server that dies goes back in the queue
  (`requeue_claim`), at most MAX_REQUEUES times, then to failed/.
- A claim whose dataDir has had no file written for `hang_after_s` is
  flagged; with `kill_hung`, its server is killed, which requeues the ticket
  and frees the GPU.
- Live work goes first (see opym.lanes). Servers are started as soon as a
  live lease appears, and if a live ticket has waited `preempt_after_s`
  while a server works on a backfill ticket, that server is killed and its
  backfill ticket requeued (not counted against its requeue cap), so the
  GPU switches to live work. One server per cooldown, so a single queued
  live ticket never takes down both.

Each server records the ticket it holds in claims/S<id>.json (written by
run_petakit_server.m right after it claims one, removed when that ticket
resolves). Current state for dashboards: supervisor_status.json.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess  # nosec B404
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from opym import lanes
from opym.consolidate import run_pending_consolidations

# Dynamically locate the opym installation directory
OPYM_DIR = Path(__file__).parent.resolve()

# PETAKIT_JOBS_DIR matches run_petakit_server.m's override, for test servers.
BASE_DIR = lanes.jobs_dir()
QUEUE_DIR = BASE_DIR / "queue"

# If a Matlab server exits nonzero (e.g. a license checkout failure kills
# Matlab before it ever reaches run_petakit_server.m), relaunching it on the
# normal short poll interval would hot-loop re-attempting a license checkout
# until whatever's broken gets fixed. Back off exponentially instead, capped,
# so a stuck license degrades gracefully rather than hammering the license
# server and spamming logs.
FAILURE_BACKOFF_BASE_SEC = 15
FAILURE_BACKOFF_CAP_SEC = 300

# (PETAKIT_SERVER_ID, CUDA_VISIBLE_DEVICES): one server per physical GPU.
SERVERS: tuple[tuple[str, str], ...] = (("1", "0"), ("2", "1"))

# A ticket that has taken down its server this many times goes to failed/
# instead of back in the queue, so one poisonous dataset can't keep a GPU
# busy failing forever.
MAX_REQUEUES = 2
DEFAULT_HANG_AFTER_S = 60 * 60
HANG_CHECK_EVERY_S = 300.0
ORPHAN_SWEEP_EVERY_S = 60.0
DEFAULT_PREEMPT_AFTER_S = 30.0
KILL_GRACE_S = 20.0


def _next_backoff_sec(
    consecutive_failures: int,
    base: float = FAILURE_BACKOFF_BASE_SEC,
    cap: float = FAILURE_BACKOFF_CAP_SEC,
) -> float:
    """Seconds to wait before the next launch attempt, given how many
    consecutive attempts have failed (1 = the first failure)."""
    return min(base * (2 ** (consecutive_failures - 1)), cap)


def has_claimable_work(queue_dir: Path) -> bool:
    """True if `queue_dir` holds a ticket a server could claim.

    Path.glob("*.json") also matches dotfiles (".active_*" claims,
    ".requeue_*" in-progress requeues) -- unlike the shell, pathlib doesn't
    hide them. run_petakit_server.m skips anything starting with '.', so
    mirror that: an orphaned claim must not look like new work.
    """
    return any(not p.name.startswith(".") for p in queue_dir.glob("*.json"))


def requeue_claim(
    active_path: Path,
    reason: str,
    max_requeues: int = MAX_REQUEUES,
    count: bool = True,
) -> str:
    """Put a claimed ticket (`<lane>/.active_<name>`) back as `<lane>/<name>`.

    The ticket's `requeueCount` goes up by one each time; once it reaches
    `max_requeues`, the ticket goes to failed/ with a `.log` giving `reason`.
    `count=False` (preemption: the ticket did nothing wrong) records the
    requeue without counting it. An unreadable ticket goes straight to
    failed/. Returns "requeued" or "failed".

    The new ticket is written under a dot-name first and the claim removed
    before the rename, so a server can never claim `<name>` while
    `.active_<name>` still exists (its movefile would collide with it).
    """
    queue_dir = active_path.parent
    failed_dir = queue_dir.parent / "failed"
    name = active_path.name.removeprefix(".active_")
    try:
        ticket = json.loads(active_path.read_text())
    except (OSError, ValueError):
        ticket = None
    n_requeues = (
        int(ticket.get("requeueCount", 0)) if isinstance(ticket, dict) else max_requeues
    )

    if not isinstance(ticket, dict) or (count and n_requeues >= max_requeues):
        failed_dir.mkdir(parents=True, exist_ok=True)
        os.replace(active_path, failed_dir / name)
        (failed_dir / f"{name}.log").write_text(
            "Moved to failed/ by the opym-serve supervisor after "
            f"{n_requeues} requeue(s).\nLast reason: {reason}\n"
        )
        return "failed"

    if count:
        ticket["requeueCount"] = n_requeues + 1
    ticket.setdefault("requeueHistory", []).append(
        {"at": time.time(), "reason": reason, "counted": count}
    )
    staging = queue_dir / f".requeue_{name}"
    staging.write_text(json.dumps(ticket, indent=2))
    active_path.unlink(missing_ok=True)
    os.replace(staging, queue_dir / name)
    return "requeued"


def newest_mtime(root: Path) -> float | None:
    """Newest file mtime anywhere under `root`, or None if it has no files."""
    newest = None
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            try:
                m = os.stat(os.path.join(dirpath, fn)).st_mtime
            except OSError:
                continue
            if newest is None or m > newest:
                newest = m
    return newest


def _server_pids(server_id: str) -> list[int]:
    """Pids of this user's processes launched for PETAKIT_SERVER_ID=`server_id`
    (the server's bash wrapper, MATLAB itself, and its parpool workers, which
    all inherit the variable). Processes we can't read are skipped."""
    marker = f"PETAKIT_SERVER_ID={server_id}".encode()
    pids = []
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        try:
            with open(f"/proc/{entry.name}/environ", "rb") as f:
                if marker in f.read().split(b"\0"):
                    pids.append(int(entry.name))
        except OSError:
            continue
    return pids


def _any_server_alive() -> bool:
    return any(_server_pids(sid) for sid, _ in SERVERS)


class Claim(NamedTuple):
    """The ticket a server holds, per its claims/S<id>.json record."""

    ticket: str
    claimed_at: float
    lane: str  # the queue directory it was claimed from ("queue_live" / "queue")

    def active_path(self, base_dir: Path) -> Path:
        return base_dir / self.lane / f".active_{self.ticket}"


def oldest_claimable_age(queue_dir: Path, now: float) -> float | None:
    """Seconds the oldest claimable ticket in `queue_dir` has waited."""
    mtimes = []
    for p in queue_dir.glob("*.json"):
        if p.name.startswith("."):
            continue
        try:
            mtimes.append(p.stat().st_mtime)
        except OSError:
            continue
    return now - min(mtimes) if mtimes else None


@dataclass
class ServerSlot:
    server_id: str
    cuda_device: str
    proc: subprocess.Popen | None = None
    consecutive_failures: int = 0
    next_launch_at: float = 0.0
    last_hang_check: float = 0.0
    idle_s: float | None = None
    hang_flagged_ticket: str | None = None
    preempting: bool = False


class ServerSupervisor:
    """Drives the servers one `tick()` at a time; `process_queue` loops it.

    `launch(slot) -> Popen`, `clock() -> float`, `any_server_alive()` and
    `lease_active()` are injectable for tests.
    """

    def __init__(
        self,
        base_dir: Path,
        *,
        idle_timeout_sec: int = 300,
        hang_after_s: float = DEFAULT_HANG_AFTER_S,
        kill_hung: bool = True,
        launch: Callable[[ServerSlot], subprocess.Popen] | None = None,
        clock: Callable[[], float] = time.time,
        any_server_alive: Callable[[], bool] = _any_server_alive,
        preempt: bool = True,
        preempt_after_s: float = DEFAULT_PREEMPT_AFTER_S,
        lease_active: Callable[[], bool] | None = None,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.queue_dir = lanes.backfill_queue_dir(self.base_dir)
        self.live_queue_dir = lanes.live_queue_dir(self.base_dir)
        self.claims_dir = self.base_dir / "claims"
        self.preempt = preempt
        self.preempt_after_s = preempt_after_s
        self._lease_active = lease_active or (
            lambda: lanes.live_lease_active(self.base_dir)
        )
        self._last_preempt = float("-inf")
        self.idle_timeout_sec = idle_timeout_sec
        self.hang_after_s = hang_after_s
        self.kill_hung = kill_hung
        self.slots = [ServerSlot(sid, dev) for sid, dev in SERVERS]
        self._launch = launch or self._launch_matlab
        self._clock = clock
        self._any_server_alive = any_server_alive
        self._clean_exit_pending_consolidation = False
        self._last_orphan_sweep = float("-inf")
        for d in (
            self.queue_dir,
            self.live_queue_dir,
            self.base_dir / "completed",
            self.base_dir / "failed",
            self.claims_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    # --- one supervision pass -------------------------------------------------

    def tick(self) -> None:
        now = self._clock()
        for slot in self.slots:
            if slot.proc is not None and slot.proc.poll() is not None:
                self._on_exit(slot, now)

        if not any(s.proc is not None for s in self.slots):
            if now - self._last_orphan_sweep >= ORPHAN_SWEEP_EVERY_S:
                self._last_orphan_sweep = now
                self._reclaim_unowned()
            if self._clean_exit_pending_consolidation:
                self._clean_exit_pending_consolidation = False
                self._run_consolidations()

        # A fresh live lease starts servers even before its first ticket
        # lands: Matlab plus its parpool take about a minute to come up.
        if (
            has_claimable_work(self.live_queue_dir)
            or has_claimable_work(self.queue_dir)
            or self._lease_active()
        ):
            for slot in self.slots:
                if slot.proc is None and now >= slot.next_launch_at:
                    print(
                        f"🚀 Launching server {slot.server_id} "
                        f"on GPU {slot.cuda_device}...",
                        flush=True,
                    )
                    slot.proc = self._launch(slot)

        for slot in self.slots:
            if (
                slot.proc is not None
                and now - slot.last_hang_check >= HANG_CHECK_EVERY_S
            ):
                slot.last_hang_check = now
                self._check_hang(slot, now)

        if self.preempt:
            self._maybe_preempt(now)

        self._write_status(now)

    # --- exits and claims -----------------------------------------------------

    def _on_exit(self, slot: ServerSlot, now: float) -> None:
        rc = slot.proc.returncode
        slot.proc = None
        slot.idle_s = None
        slot.hang_flagged_ticket = None
        if slot.preempting:
            # Our own kill, to hand the GPU to live work: not a failure, so no
            # backoff, and the backfill ticket's requeue doesn't count.
            slot.preempting = False
            slot.next_launch_at = now
            self._requeue_claim_of(slot, reason="preempted for live work", count=False)
            print(
                f"🔁 Server {slot.server_id} preempted; relaunching for live work.",
                flush=True,
            )
            return
        self._requeue_claim_of(
            slot, reason=f"server {slot.server_id} exited with rc={rc}"
        )
        if rc == 0:
            slot.consecutive_failures = 0
            slot.next_launch_at = now
            self._clean_exit_pending_consolidation = True
            print(
                f"🛑 Server {slot.server_id} spun down after "
                f"{self.idle_timeout_sec}s idle.",
                flush=True,
            )
            return
        # run_petakit_server.m wraps all per-job work in try/catch and exits 0
        # on its idle timeout, so a nonzero exit means Matlab itself died:
        # license/module failure at launch, OOM, a crash, or our own kill.
        slot.consecutive_failures += 1
        backoff = _next_backoff_sec(slot.consecutive_failures)
        slot.next_launch_at = now + backoff
        print(
            f"❌ Server {slot.server_id} exited with rc={rc}. Relaunching it alone in "
            f"{backoff:.0f}s (consecutive failures: {slot.consecutive_failures}); "
            "other servers are unaffected.",
            flush=True,
        )

    def _claim_path(self, slot: ServerSlot) -> Path:
        return self.claims_dir / f"S{slot.server_id}.json"

    def _read_claim(self, slot: ServerSlot) -> Claim | None:
        """The ticket this server holds, or None. Records written before
        lanes existed have no "queue" field and came from the backfill queue."""
        path = self._claim_path(slot)
        try:
            rec = json.loads(path.read_text())
            lane = str(rec.get("queue") or lanes.BACKFILL_QUEUE_NAME)
            if lane not in (lanes.LIVE_QUEUE_NAME, lanes.BACKFILL_QUEUE_NAME):
                lane = lanes.BACKFILL_QUEUE_NAME
            return Claim(str(rec["ticket"]), path.stat().st_mtime, lane)
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return None

    def _requeue_claim_of(
        self, slot: ServerSlot, reason: str, count: bool = True
    ) -> None:
        claim = self._read_claim(slot)
        self._claim_path(slot).unlink(missing_ok=True)
        if claim is None:
            return
        active = claim.active_path(self.base_dir)
        if active.exists():
            outcome = requeue_claim(active, reason, count=count)
            print(f"♻️  {claim.ticket}: {outcome} ({reason})", flush=True)

    def _reclaim_unowned(self) -> None:
        """With no server process alive at all, nothing can legitimately hold
        a claim, so every `.active_` ticket is an orphan. Also finishes any
        requeue interrupted between its write and its rename."""
        if self._any_server_alive():
            return  # e.g. a server launched by hand outside this supervisor
        for lane_dir in (self.live_queue_dir, self.queue_dir):
            for staging in lane_dir.glob(".requeue_*.json"):
                os.replace(staging, lane_dir / staging.name.removeprefix(".requeue_"))
            for active in sorted(lane_dir.glob(".active_*.json")):
                outcome = requeue_claim(active, "orphaned claim: no server was running")
                print(f"♻️  {active.name}: {outcome} (orphaned claim)", flush=True)
        for rec in self.claims_dir.glob("S*.json"):
            rec.unlink(missing_ok=True)

    # --- hang detection -------------------------------------------------------

    def _check_hang(self, slot: ServerSlot, now: float) -> None:
        claim = self._read_claim(slot)
        if claim is None:
            slot.idle_s = None
            return
        ticket, claimed_at = claim.ticket, claim.claimed_at
        try:
            data_dir = json.loads(claim.active_path(self.base_dir).read_text())[
                "dataDir"
            ]
        except (OSError, ValueError, KeyError, TypeError):
            return
        last_write = max(claimed_at, newest_mtime(Path(data_dir)) or 0.0)
        slot.idle_s = now - last_write
        if slot.idle_s < self.hang_after_s or slot.hang_flagged_ticket == ticket:
            return
        slot.hang_flagged_ticket = ticket
        print(
            f"⚠️  Server {slot.server_id} looks hung on {ticket}: nothing written under "
            f"{data_dir} for {slot.idle_s / 60:.0f} min.",
            flush=True,
        )
        if self.kill_hung:
            print(
                f"🔪 Killing server {slot.server_id} so its GPU comes back; "
                "the ticket is requeued.",
                flush=True,
            )
            self._kill(slot)

    # --- preemption -----------------------------------------------------------

    def _maybe_preempt(self, now: float) -> None:
        """Kill one server that is working on a backfill ticket, so live work
        gets a GPU, when either

        - there is live work (a live session's lease, or a live ticket) and
          no server can take it -- every one is on backfill. Right away: a
          relaunch plus the session warm-up take ~20 s, about what the first
          volume takes to arrive, while backfill tickets run for minutes; or
        - a live ticket has waited `preempt_after_s` and more are waiting than
          the idle or starting servers can absorb (escalation to a second GPU).

        At most one per `preempt_after_s`."""
        if now - self._last_preempt < self.preempt_after_s:
            return
        waited = oldest_claimable_age(self.live_queue_dir, now)
        live_wanted = waited is not None or self._lease_active()
        if live_wanted and not self._live_capacity():
            reason = "Live work with every GPU on backfill"
        elif waited is not None and waited >= self.preempt_after_s:
            # A running server with no claim is idle or still starting and
            # will take a live ticket next; only preempt for live tickets
            # beyond what those can absorb.
            free = sum(
                1
                for slot in self.slots
                if slot.proc is not None
                and not slot.preempting
                and self._read_claim(slot) is None
            )
            waiting = sum(
                1
                for p in self.live_queue_dir.glob("*.json")
                if not p.name.startswith(".")
            )
            if waiting <= free:
                return
            reason = f"Live ticket waiting {waited:.0f}s"
        else:
            return
        for slot in self.slots:
            if slot.proc is None or slot.preempting:
                continue
            claim = self._read_claim(slot)
            if claim is None or claim.lane != lanes.BACKFILL_QUEUE_NAME:
                continue
            print(
                f"⏩ {reason}: preempting server {slot.server_id} "
                f"(backfill {claim.ticket}).",
                flush=True,
            )
            slot.preempting = True
            self._last_preempt = now
            self._kill(slot)
            return

    def _live_capacity(self) -> bool:
        """A server that can take live work: running, and idle, starting, or
        on live work already."""
        for slot in self.slots:
            if slot.proc is None or slot.preempting:
                continue
            claim = self._read_claim(slot)
            if claim is None or claim.lane != lanes.BACKFILL_QUEUE_NAME:
                return True
        return False

    def _kill(self, slot: ServerSlot) -> None:
        """Terminate the server's whole process tree: the bash wrapper,
        Matlab, and its parpool workers (all carry its PETAKIT_SERVER_ID)."""
        for sig, wait_s in ((signal.SIGTERM, KILL_GRACE_S), (signal.SIGKILL, 5.0)):
            pids = _server_pids(slot.server_id)
            if slot.proc is not None and slot.proc.poll() is None:
                pids.append(slot.proc.pid)
            if not pids:
                return
            for pid in set(pids):
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass
            deadline = time.time() + wait_s
            while time.time() < deadline and _server_pids(slot.server_id):
                time.sleep(0.5)

    # --- process launch, consolidation, status --------------------------------

    def _launch_matlab(self, slot: ServerSlot) -> subprocess.Popen:
        env = os.environ.copy()
        env["PETAKIT_IDLE_TIMEOUT"] = str(self.idle_timeout_sec)
        env["PETAKIT_SERVER_ID"] = slot.server_id
        # Matlab's gpuDevice index is 1-based within CUDA_VISIBLE_DEVICES,
        # which exposes exactly one physical GPU to each server.
        env["PETAKIT_GPU_ID"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = slot.cuda_device
        env["PETAKIT_CPUS"] = "10"  # Limit workers to prevent GPU OOM
        # Use bash to load the matlab module so licensing works correctly
        cmd_str = (
            "module load matlab/R2024b && "
            f"matlab -nodisplay -sd {OPYM_DIR} -batch run_petakit_server"
        )
        return subprocess.Popen(["bash", "-c", cmd_str], env=env)  # nosec B603

    def _run_consolidations(self) -> None:
        print("\n🔗 Checking for pending OME-Zarr consolidations...", flush=True)
        try:
            n = run_pending_consolidations(self.base_dir)
            print(
                f"✅ Consolidated {n} dataset(s) into OME-Zarr"
                if n
                else "   No pending consolidations found",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - never let this kill the supervisor
            print(f"⚠️  Consolidation error: {exc}", flush=True)

    def _write_status(self, now: float) -> None:
        servers = []
        for slot in self.slots:
            claim = self._read_claim(slot) if slot.proc is not None else None
            servers.append(
                {
                    "server_id": slot.server_id,
                    "cuda_device": slot.cuda_device,
                    "running": slot.proc is not None,
                    "pid": slot.proc.pid if slot.proc is not None else None,
                    "consecutive_failures": slot.consecutive_failures,
                    "next_launch_at": slot.next_launch_at,
                    "ticket": claim.ticket if claim else None,
                    "lane": claim.lane if claim else None,
                    "claimed_at": claim.claimed_at if claim else None,
                    "idle_s": slot.idle_s,
                    "suspected_hang": slot.hang_flagged_ticket is not None,
                }
            )
        tmp = self.base_dir / ".supervisor_status.json.tmp"
        status = {
            "updated_at": now,
            "live_lease_active": self._lease_active(),
            "live_queued": sum(
                1
                for p in self.live_queue_dir.glob("*.json")
                if not p.name.startswith(".")
            ),
            "backfill_queued": sum(
                1 for p in self.queue_dir.glob("*.json") if not p.name.startswith(".")
            ),
            "servers": servers,
        }
        tmp.write_text(json.dumps(status, indent=1))
        os.replace(tmp, self.base_dir / "supervisor_status.json")


def process_queue(idle_timeout_sec: int = 300, poll_interval: int = 2):
    """Supervise the servers until interrupted. Hang handling is set by
    OPYM_SERVE_HANG_MIN (default 60) and OPYM_SERVE_KILL_HUNG (default 1);
    live preemption by OPYM_LIVE_PREEMPT (default 1) and
    OPYM_LIVE_PREEMPT_AFTER_S (default 30)."""
    hang_after_s = (
        float(os.environ.get("OPYM_SERVE_HANG_MIN", DEFAULT_HANG_AFTER_S / 60)) * 60
    )
    kill_hung = os.environ.get("OPYM_SERVE_KILL_HUNG", "1").strip() not in (
        "0",
        "false",
        "",
    )
    preempt = os.environ.get("OPYM_LIVE_PREEMPT", "1").strip() not in ("0", "false", "")
    preempt_after_s = float(
        os.environ.get("OPYM_LIVE_PREEMPT_AFTER_S", DEFAULT_PREEMPT_AFTER_S)
    )
    sup = ServerSupervisor(
        BASE_DIR,
        idle_timeout_sec=idle_timeout_sec,
        hang_after_s=hang_after_s,
        kill_hung=kill_hung,
        preempt=preempt,
        preempt_after_s=preempt_after_s,
    )

    print("=" * 60)
    print(" 🚀 OPYM PetaKit GPU Supervisor Initialized")
    print("=" * 60)
    print(f" 📂 Live Queue:      {sup.live_queue_dir} (claimed first)")
    print(f" 📂 Backfill Queue:  {sup.queue_dir}")
    print(
        " 🖥️  Servers:         "
        + ", ".join(f"{s.server_id}->GPU {s.cuda_device}" for s in sup.slots)
    )
    print(f" ⏱️  Idle Timeout:    {idle_timeout_sec} seconds")
    print(f" 🔍 Polling Rate:    Every {poll_interval} seconds")
    print(
        f" ⚠️  Failure Backoff: {FAILURE_BACKOFF_BASE_SEC}s-{FAILURE_BACKOFF_CAP_SEC}s "
        "(exponential, per server)"
    )
    print(
        f" 🧊 Hang Handling:   flag after {hang_after_s / 60:.0f} min idle, "
        f"kill={'on' if kill_hung else 'off'}"
    )
    print(
        f" ⏩ Live Preemption: {'on' if preempt else 'off'}: at once when live "
        f"work has no GPU, a second one after {preempt_after_s:.0f}s"
    )
    print(f" 🔧 Backend Script:  {OPYM_DIR}/run_petakit_server.m")
    print("=" * 60, flush=True)

    try:
        while True:
            sup.tick()
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        print("\n🛑 Supervisor shut down.", flush=True)


def main():
    # 3600 seconds = 1 hour of idle time before a server releases its GPU
    process_queue(idle_timeout_sec=3600)


if __name__ == "__main__":
    main()
