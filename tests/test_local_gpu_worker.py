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


def _claim(sup, server_id, ticket_name, lane="queue"):
    """What run_petakit_server.m does on a claim: rename to .active_ in its
    lane and record it in claims/S<id>.json."""
    lane_dir = sup.base_dir / lane
    os.replace(lane_dir / ticket_name, lane_dir / f".active_{ticket_name}")
    (sup.claims_dir / f"S{server_id}.json").write_text(
        json.dumps(
            {"server_id": server_id, "ticket": ticket_name, "queue": lane, "pid": 1}
        )
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


# --- priority lanes -------------------------------------------------------


def test_live_claim_of_a_dead_server_is_requeued_into_the_live_lane(tmp_path):
    sup, launched, _ = _make_supervisor(tmp_path)
    _ticket(sup.live_queue_dir, "LIVE_t000.json")
    sup.tick()
    procs = dict(launched)
    _claim(sup, "1", "LIVE_t000.json", lane="queue_live")
    procs["1"].exit(137)
    sup.tick()
    assert (sup.live_queue_dir / "LIVE_t000.json").exists()
    assert not (sup.queue_dir / "LIVE_t000.json").exists()


def test_claim_record_without_a_queue_field_is_the_backfill_lane(tmp_path):
    """Records written by servers from before lanes existed."""
    sup, launched, _ = _make_supervisor(tmp_path)
    _ticket(sup.queue_dir, "old.json")
    sup.tick()
    os.replace(sup.queue_dir / "old.json", sup.queue_dir / ".active_old.json")
    (sup.claims_dir / "S2.json").write_text(json.dumps({"ticket": "old.json"}))
    dict(launched)["2"].exit(1)
    sup.tick()
    assert (sup.queue_dir / "old.json").exists()


def test_live_lease_starts_servers_before_any_ticket(tmp_path):
    lease = {"active": False}
    sup, launched, _ = _make_supervisor(tmp_path, lease_active=lambda: lease["active"])
    sup.tick()
    assert launched == []
    lease["active"] = True
    sup.tick()
    assert sorted(sid for sid, _ in launched) == ["1", "2"]


def _busy_with_backfill_and_live_waiting(tmp_path, waited_s, **kwargs):
    sup, launched, clock = _make_supervisor(tmp_path, preempt_after_s=30, **kwargs)
    _ticket(sup.queue_dir, "bf_a.json")
    _ticket(sup.queue_dir, "bf_b.json")
    sup.tick()
    procs = dict(launched)
    _claim(sup, "1", "bf_a.json")
    _claim(sup, "2", "bf_b.json")
    live = _ticket(sup.live_queue_dir, "LIVE_t000.json")
    os.utime(live, (clock.t - waited_s, clock.t - waited_s))
    killed = []

    def fake_kill(slot):
        killed.append(slot.server_id)
        procs[slot.server_id].exit(-15)

    sup._kill = fake_kill
    return sup, launched, clock, procs, killed


def test_no_preemption_before_a_live_ticket_has_waited_long_enough(tmp_path):
    sup, _, _, _, killed = _busy_with_backfill_and_live_waiting(tmp_path, waited_s=10)
    sup.tick()
    assert killed == []


def test_waiting_live_ticket_preempts_one_backfill_server(tmp_path):
    sup, launched, clock, procs, killed = _busy_with_backfill_and_live_waiting(
        tmp_path, waited_s=31
    )
    sup.tick()
    assert killed == ["1"]  # one server only

    sup.tick()  # reap: requeued without counting, relaunched with no backoff
    requeued = json.loads((sup.queue_dir / "bf_a.json").read_text())
    assert "requeueCount" not in requeued
    assert requeued["requeueHistory"][-1]["counted"] is False
    assert sup.slots[0].consecutive_failures == 0
    assert [sid for sid, _ in launched] == ["1", "2", "1"]

    # Server 1 is relaunching (no claim yet) and will take the waiting live
    # ticket, so server 2 keeps its backfill work even after the cooldown.
    clock.t += 40
    sup.tick()
    assert killed == ["1"]

    # Live tickets pile up beyond what the starting server can take: escalate
    # to the second GPU, but not within the cooldown of the last preemption.
    for t in (1, 2):
        extra = _ticket(sup.live_queue_dir, f"LIVE_t00{t}.json")
        os.utime(extra, (clock.t - 60, clock.t - 60))
    sup._last_preempt = clock.t - 10
    sup.tick()
    assert killed == ["1"]
    clock.t += 30
    sup.tick()
    assert killed == ["1", "2"]


def test_servers_on_live_work_are_never_preempted(tmp_path):
    sup, launched, clock = _make_supervisor(tmp_path, preempt_after_s=30)
    _ticket(sup.live_queue_dir, "LIVE_t000.json")
    _ticket(sup.live_queue_dir, "LIVE_t001.json")
    sup.tick()
    _claim(sup, "1", "LIVE_t000.json", lane="queue_live")
    _claim(sup, "2", "LIVE_t001.json", lane="queue_live")
    waiting = _ticket(sup.live_queue_dir, "LIVE_t002.json")
    os.utime(waiting, (clock.t - 120, clock.t - 120))
    killed = []
    sup._kill = lambda slot: killed.append(slot.server_id)
    sup.tick()
    assert killed == []


def test_preemption_can_be_turned_off(tmp_path):
    sup, _, _, _, killed = _busy_with_backfill_and_live_waiting(
        tmp_path, waited_s=300, preempt=False
    )
    sup.tick()
    assert killed == []


def test_status_reports_lanes(tmp_path):
    sup, _, _ = _make_supervisor(tmp_path, lease_active=lambda: True)
    _ticket(sup.live_queue_dir, "LIVE_t000.json")
    _ticket(sup.queue_dir, "bf.json")
    sup.tick()
    status = json.loads((sup.base_dir / "supervisor_status.json").read_text())
    assert status["live_lease_active"] is True
    assert status["live_queued"] == 1
    assert status["backfill_queued"] == 1
