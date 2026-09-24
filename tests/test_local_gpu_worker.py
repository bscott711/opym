"""Tests for local_gpu_worker.py's per-server supervisor.

Guards the failure modes that stalled production:
- Matlab dying at launch (e.g. a broken license checkout) must back off,
  not hot-loop.
- One dead server must be relaunched on its own; the old watchdog waited on
  both, and on 2026-09-23 a dead server 1 plus a hung server 2 meant 43 hours
  of no GPU work.
- A dead or hung server's claimed ticket must go back in the queue, with a
  cap so a poisonous ticket can't loop forever.
"""

from __future__ import annotations

import json
import os

from opym import local_gpu_worker
from opym.local_gpu_worker import (
    MAX_REQUEUES,
    ServerSupervisor,
    _next_backoff_sec,
    has_claimable_work,
    requeue_claim,
)


class TestNextBackoffSec:
    def test_first_failure_returns_base(self):
        assert _next_backoff_sec(1, base=15, cap=300) == 15

    def test_doubles_each_consecutive_failure(self):
        assert [_next_backoff_sec(n, base=15, cap=300) for n in range(1, 5)] == [
            15,
            30,
            60,
            120,
        ]

    def test_clamps_at_cap(self):
        assert _next_backoff_sec(6, base=15, cap=300) == 300
        assert _next_backoff_sec(20, base=15, cap=300) == 300


class _FakeProc:
    _next_pid = 1000

    def __init__(self):
        self.returncode = None
        _FakeProc._next_pid += 1
        self.pid = _FakeProc._next_pid

    def poll(self):
        return self.returncode

    def exit(self, rc):
        self.returncode = rc


class _Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def _make_supervisor(tmp_path, **kwargs):
    launched: list[tuple[str, _FakeProc]] = []

    def launch(slot):
        proc = _FakeProc()
        launched.append((slot.server_id, proc))
        return proc

    clock = kwargs.pop("clock", _Clock())
    sup = ServerSupervisor(
        tmp_path / "petakit_jobs",
        launch=launch,
        clock=clock,
        any_server_alive=kwargs.pop("any_server_alive", lambda: False),
        **kwargs,
    )
    return sup, launched, clock


def _ticket(queue_dir, name, data_dir="/nonexistent", **extra):
    queue_dir.mkdir(parents=True, exist_ok=True)
    path = queue_dir / name
    path.write_text(
        json.dumps({"jobType": "deskew", "dataDir": str(data_dir), **extra})
    )
    return path


def _claim(sup, server_id, ticket_name):
    """What run_petakit_server.m does on a claim: rename to .active_ and
    record it in claims/S<id>.json."""
    os.replace(sup.queue_dir / ticket_name, sup.queue_dir / f".active_{ticket_name}")
    (sup.claims_dir / f"S{server_id}.json").write_text(
        json.dumps({"server_id": server_id, "ticket": ticket_name, "pid": 1})
    )


def test_orphaned_claims_are_not_work(tmp_path):
    queue = tmp_path / "queue"
    _ticket(queue, ".active_old.json")
    _ticket(queue, ".requeue_mid.json")
    assert not has_claimable_work(queue)
    _ticket(queue, "new.json")
    assert has_claimable_work(queue)


def test_launches_every_server_when_work_exists_and_none_without(tmp_path):
    sup, launched, _ = _make_supervisor(tmp_path)
    sup.tick()
    assert launched == []
    _ticket(sup.queue_dir, "job1.json")
    sup.tick()
    assert sorted(sid for sid, _ in launched) == ["1", "2"]
    sup.tick()
    assert len(launched) == 2  # already running, not relaunched


def test_dead_server_is_relaunched_alone_and_its_claim_requeued(tmp_path):
    sup, launched, clock = _make_supervisor(tmp_path)
    _ticket(sup.queue_dir, "a.json")
    _ticket(sup.queue_dir, "b.json")
    sup.tick()
    procs = dict(launched)
    _claim(sup, "1", "a.json")
    _claim(sup, "2", "b.json")

    procs["1"].exit(137)  # e.g. OOM-killed mid-job
    sup.tick()

    # Server 1's ticket is back in the queue with its requeue recorded;
    # server 2 and its claim are untouched.
    requeued = json.loads((sup.queue_dir / "a.json").read_text())
    assert requeued["requeueCount"] == 1
    assert not (sup.queue_dir / ".active_a.json").exists()
    assert (sup.queue_dir / ".active_b.json").exists()
    assert not (sup.claims_dir / "S1.json").exists()
    assert procs["2"].poll() is None

    # Server 1 comes back only after its own backoff; server 2 is never relaunched.
    assert len(launched) == 2
    clock.t += _next_backoff_sec(1)
    sup.tick()
    assert [sid for sid, _ in launched] == ["1", "2", "1"]


def test_backoff_grows_per_server_and_resets_on_clean_exit(tmp_path):
    sup, launched, clock = _make_supervisor(tmp_path)
    _ticket(sup.queue_dir, "job.json")

    def server1_launch_times():
        return [t for sid, t in launch_log if sid == "1"]

    launch_log: list[tuple[str, float]] = []
    real_launch = sup._launch

    def logging_launch(slot):
        launch_log.append((slot.server_id, clock.t))
        return real_launch(slot)

    sup._launch = logging_launch
    t0 = clock.t
    for _ in range(80):  # 80 s of 1 s ticks
        sup.tick()
        for sid, proc in launched:
            if sid == "1" and proc.poll() is None:
                proc.exit(1)  # server 1 keeps dying at launch
        clock.t += 1
    assert [t - t0 for t in server1_launch_times()] == [0, 16, 47]
    assert sup.slots[0].consecutive_failures == 3

    # A clean exit (idle timeout) resets the streak.
    clock.t += 300
    sup.tick()
    live = [p for sid, p in launched if sid == "1" and p.poll() is None]
    live[0].exit(0)
    sup.tick()
    assert sup.slots[0].consecutive_failures == 0


def test_requeue_cap_moves_ticket_to_failed(tmp_path):
    queue = tmp_path / "queue"
    _ticket(queue, "job.json")
    for n in range(MAX_REQUEUES):
        os.replace(queue / "job.json", queue / ".active_job.json")
        assert requeue_claim(queue / ".active_job.json", f"crash {n}") == "requeued"
    ticket = json.loads((queue / "job.json").read_text())
    assert ticket["requeueCount"] == MAX_REQUEUES
    assert [h["reason"] for h in ticket["requeueHistory"]] == ["crash 0", "crash 1"]

    os.replace(queue / "job.json", queue / ".active_job.json")
    assert requeue_claim(queue / ".active_job.json", "crash again") == "failed"
    assert not (queue / "job.json").exists()
    assert "crash again" in (tmp_path / "failed" / "job.json.log").read_text()


def test_unreadable_claim_goes_straight_to_failed(tmp_path):
    queue = tmp_path / "queue"
    queue.mkdir()
    (queue / ".active_bad.json").write_text("{not json")
    assert requeue_claim(queue / ".active_bad.json", "server died") == "failed"
    assert (tmp_path / "failed" / "bad.json").exists()


def test_orphans_reclaimed_only_when_no_server_process_is_alive(tmp_path):
    alive = {"value": True}
    sup, launched, clock = _make_supervisor(
        tmp_path, any_server_alive=lambda: alive["value"]
    )
    _ticket(sup.queue_dir, ".active_orphan.json")

    sup.tick()  # a server launched by hand might own it
    assert (sup.queue_dir / ".active_orphan.json").exists()

    alive["value"] = False
    clock.t += local_gpu_worker.ORPHAN_SWEEP_EVERY_S
    sup.tick()
    assert (sup.queue_dir / "orphan.json").exists()
    assert not (sup.queue_dir / ".active_orphan.json").exists()


def test_interrupted_requeue_is_completed_by_the_orphan_sweep(tmp_path):
    sup, _, _ = _make_supervisor(tmp_path)
    _ticket(sup.queue_dir, ".requeue_half.json")
    sup.tick()
    assert (sup.queue_dir / "half.json").exists()


def test_hung_server_is_killed_and_its_ticket_requeued(tmp_path, monkeypatch):
    sup, launched, clock = _make_supervisor(tmp_path, hang_after_s=600, kill_hung=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    stale = data_dir / "Decon" / "mask.zarr"
    stale.parent.mkdir()
    stale.write_text("x")
    _ticket(sup.queue_dir, "stuck.json", data_dir=data_dir)
    sup.tick()
    procs = dict(launched)
    _claim(sup, "2", "stuck.json")

    # Claimed and last wrote 20 minutes ago.
    old = clock.t - 1200
    os.utime(stale, (old, old))
    os.utime(sup.claims_dir / "S2.json", (old, old))

    killed = []
    monkeypatch.setattr(
        sup,
        "_kill",
        lambda slot: (killed.append(slot.server_id), procs[slot.server_id].exit(-15)),
    )
    clock.t += local_gpu_worker.HANG_CHECK_EVERY_S
    sup.tick()
    assert killed == ["2"]
    status = json.loads((sup.base_dir / "supervisor_status.json").read_text())
    assert status["servers"][1]["suspected_hang"] is True

    sup.tick()  # reaps the killed server
    assert json.loads((sup.queue_dir / "stuck.json").read_text())["requeueCount"] == 1


def test_busy_server_is_not_flagged(tmp_path):
    sup, launched, clock = _make_supervisor(tmp_path, hang_after_s=600, kill_hung=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _ticket(sup.queue_dir, "busy.json", data_dir=data_dir)
    sup.tick()
    _claim(sup, "1", "busy.json")
    os.utime(sup.claims_dir / "S1.json", (clock.t - 5000, clock.t - 5000))
    (data_dir / "frame_T009.tif").write_text("x")
    os.utime(
        data_dir / "frame_T009.tif", (clock.t - 30, clock.t - 30)
    )  # wrote 30 s ago

    clock.t += local_gpu_worker.HANG_CHECK_EVERY_S
    sup.tick()
    assert sup.slots[0].hang_flagged_ticket is None


def test_consolidation_runs_once_after_servers_spin_down_cleanly(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        local_gpu_worker,
        "run_pending_consolidations",
        lambda base: calls.append(base) or 0,
    )
    sup, launched, _ = _make_supervisor(tmp_path)
    _ticket(sup.queue_dir, "job.json")
    sup.tick()
    (sup.queue_dir / "job.json").unlink()  # a server finished it
    for _, proc in launched:
        proc.exit(0)
    sup.tick()
    sup.tick()
    assert len(calls) == 1
