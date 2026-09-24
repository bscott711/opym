# Ruff style: Compliant
"""Write-behind drain: copies one completed session from a fast local
staging root (intended to be a RAM disk / tmpfs, e.g. ``/dev/shm``) to its
declared final destination on GPFS, in the background, without blocking the
receiver's ingest path.

Exists because reading input for deskew/decon straight off ``/dev/shm``
(the same RAM disk PetaKit5D's own local job queue already lives on -- see
``local_gpu_worker.py``, ``run_petakit_server.m``) avoids a GPFS read on the
processing-critical path entirely, while GPFS remains the durable system of
record once drained. See ``opym.stream.receiver``'s module docstring for how
this plugs in -- staging is opt-in (``OPYM_STREAM_STAGE_ROOT``); this module
does nothing when it's unset.

Verification is a full file-count + byte-count comparison between source and
destination trees (not a spot check) before the drained copy is atomically
renamed into place and the staging copy becomes eligible for eviction. A
failed drain is logged loudly and retried on the next receiver restart is
NOT automatic -- see ``DrainPool._drain_one``'s docstring; this deliberately
does not swallow failures the way ``run_napari_opym.py``'s
``_write_behind_worker`` does (``except Exception: pass`` there silently
loses data and orphans the staging copy -- see the design plan this module
implements, "Stage 3").
"""

from __future__ import annotations

import json
import logging
import queue
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Keep a drained (GPFS-verified) session's RAM-disk copy around for this
# long after drain, so a still-running deskew/decon pass reads it from RAM
# rather than racing eviction -- see module docstring. Not a correctness
# dependency (the GPFS copy is already durable the moment drain succeeds),
# purely a latency optimization for whatever is currently reading the
# staging copy.
DEFAULT_RETENTION_S = 4 * 3600.0
# 200 of tmpfs's 252 GiB on Argus today (see the ok-is-there-something-
# hazy-finch plan's Stage-3 capacity note) -- leaves headroom for
# PetaKit5D's own /dev/shm/petakit_jobs queue and other tmpfs users.
DEFAULT_HIGH_WATER_BYTES = 200 * (1024**3)
DEFAULT_DRAIN_WORKERS = 4


@dataclass
class DrainJob:
    """One completed session's staged artifacts, ready to copy to GPFS.

    A session is NOT one contiguous directory tree: each channel's zarr
    store is a top-level sibling (``<write_root>/<base_name>_<channel
    name>.ome.zarr``, see ``rawmirror.store_path_for_channel``) and
    ``decon_stage/`` (when decon is enabled) is separately nested under
    ``<write_root>/<base_name>/decon_stage``. ``items`` is the full list of
    ``(stage_path, dest_path)`` pairs that together make up this session --
    built by the receiver from ``SessionState.channel_store_paths`` plus,
    when present, ``decon_stage_dir``.
    """

    session_id: str
    items: list[tuple[Path, Path]]


@dataclass
class _DrainedEntry:
    stage_dirs: list[Path]
    drained_at: float
    size_bytes: int


class DrainPool:
    """A small pool of background threads draining `DrainJob`s to GPFS.

    Not started automatically -- callers (`StreamReceiver`) own the
    lifecycle via `start()`/`stop()`, mirroring `ArgusTunnelManager` /
    `AsyncWriter`'s own explicit-lifecycle pattern elsewhere in this
    codebase. Multiple worker threads share one queue; GPFS was measured
    (Phase 0, Stage 1B) to handle several concurrent writers without a
    meaningful per-stream penalty, so this does not need per-job isolation.
    """

    def __init__(
        self,
        num_workers: int = DEFAULT_DRAIN_WORKERS,
        retention_s: float = DEFAULT_RETENTION_S,
        high_water_bytes: int = DEFAULT_HIGH_WATER_BYTES,
        manifest_dir: Path | None = None,
    ) -> None:
        """`manifest_dir` (the receiver passes `<stage root>/.drain_manifests`)
        makes retention survive a restart: each drained session is recorded
        there until its staging copies are evicted, and `start()` reloads the
        records. Without it, retention lives only in this process, and on
        2026-09-24 one opym-receive restart left 110 GB of drained copies on
        /dev/shm with nothing to evict them."""
        self._manifest_dir = Path(manifest_dir) if manifest_dir else None
        self._num_workers = num_workers
        self._retention_s = retention_s
        self._high_water_bytes = high_water_bytes
        self._queue: queue.Queue[DrainJob] = queue.Queue()
        self._drained: dict[str, _DrainedEntry] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        if self._threads:
            return
        self._load_manifests()
        for i in range(self._num_workers):
            t = threading.Thread(
                target=self._worker_loop, name=f"OpymDrainWorker-{i}", daemon=True
            )
            t.start()
            self._threads.append(t)

    def stop(self, timeout: float = 5.0) -> None:
        """Signal workers to stop after their current job; does not wait
        for the queue to drain -- callers that need every enqueued job
        flushed before shutdown should drain the queue themselves first."""
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=timeout)
        self._threads = []

    def enqueue(self, job: DrainJob) -> None:
        self._queue.put(job)

    @property
    def pending_count(self) -> int:
        return self._queue.qsize()

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                job = self._queue.get(timeout=1.0)
            except queue.Empty:
                self._sweep_retention()
                continue
            self._drain_one(job)
            self._sweep_retention()

    def _drain_one(self, job: DrainJob) -> None:
        """Copies every ``(stage_path, dest_path)`` in `job.items` to a
        ``.draining`` temp path, verifies file-count and byte-count match
        exactly for EACH, and only once every item in the session has been
        copied and verified does it atomically rename all of them into
        place -- so `opym-backfill --watch`'s discovery scan never sees a
        partially-drained dataset (some channel stores present, others not).

        On any failure, logs at ERROR (never silently drops -- see module
        docstring) and leaves every staging copy untouched so nothing is
        lost; this receiver process does not currently persist a retry
        queue across restarts (matching the existing RESUME design's own v1
        degradation noted in `opym.stream.protocol`), so a failed drain
        needs operator attention (check `~/projects/opym_local` receiver
        logs).
        """
        pending_renames: list[tuple[Path, Path]] = []  # (tmp_dest, dest_path)
        total_bytes = 0
        try:
            for stage_path, dest_path in job.items:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_dest = dest_path.with_name(dest_path.name + ".draining")
                if tmp_dest.exists():
                    shutil.rmtree(tmp_dest)
                shutil.copytree(stage_path, tmp_dest)

                src_count, src_bytes = _tree_stats(stage_path)
                dst_count, dst_bytes = _tree_stats(tmp_dest)
                if src_count != dst_count or src_bytes != dst_bytes:
                    raise RuntimeError(
                        f"Drain verification failed for session "
                        f"{job.session_id} item {stage_path}: source has "
                        f"{src_count} files/{src_bytes} bytes, copy has "
                        f"{dst_count} files/{dst_bytes} bytes"
                    )
                pending_renames.append((tmp_dest, dest_path))
                total_bytes += src_bytes

            for tmp_dest, dest_path in pending_renames:
                if dest_path.exists():
                    shutil.rmtree(dest_path)
                tmp_dest.replace(dest_path)
        except Exception:
            logger.exception(
                "Drain FAILED for session %s -- staging copies on /dev/shm "
                "retained untouched; this will NOT auto-retry, needs "
                "operator attention. Items: %s",
                job.session_id,
                [str(s) for s, _ in job.items],
            )
            return

        with self._lock:
            self._drained[job.session_id] = _DrainedEntry(
                stage_dirs=[s for s, _ in job.items],
                drained_at=time.monotonic(),
                size_bytes=total_bytes,
            )
        self._write_manifest(job.session_id, [s for s, _ in job.items], total_bytes)
        logger.info(
            "Drained session %s: %d item(s), %.2f GB, verified byte-for-byte",
            job.session_id,
            len(job.items),
            total_bytes / 1e9,
        )

    def release(self, stage_paths: list[Path]) -> None:
        """Evicts now, instead of when retention expires, every drained
        session whose staging copies include any of `stage_paths`, so a new
        session can take over that name in the (flat, shared) staging root.

        Safe to do early: a drained session's GPFS copy is already verified
        byte-for-byte, and nothing but the receiver itself reads the staging
        root. A path that is still draining, or whose drain failed, is not in
        `_drained` and is left alone -- the caller treats it as taken.
        """
        wanted = {Path(p) for p in stage_paths}
        with self._lock:
            hits = [
                (sid, e)
                for sid, e in self._drained.items()
                if wanted & set(e.stage_dirs)
            ]
            for sid, _ in hits:
                del self._drained[sid]
        for sid, entry in hits:
            self._evict(sid, entry)

    def _sweep_retention(self) -> None:
        now = time.monotonic()
        to_evict: list[tuple[str, _DrainedEntry]] = []
        with self._lock:
            total = sum(e.size_bytes for e in self._drained.values())
            over_budget = total > self._high_water_bytes
            for session_id, entry in sorted(
                self._drained.items(), key=lambda kv: kv[1].drained_at
            ):
                age = now - entry.drained_at
                if age >= self._retention_s or over_budget:
                    to_evict.append((session_id, entry))
                    total -= entry.size_bytes
                    over_budget = total > self._high_water_bytes
            # Claim them while still holding the lock. Every idle worker
            # sweeps about once a second, so otherwise several of them rmtree
            # the same session at once; `_evict` hands back any it can't remove.
            for session_id, _ in to_evict:
                del self._drained[session_id]

        for session_id, entry in to_evict:
            self._evict(session_id, entry)

        if over_budget:
            logger.warning(
                "RAM disk staging high-water mark (%.1f GB) exceeded even "
                "after evicting every eligible drained session -- drain "
                "and/or downstream processing is falling behind ingest",
                self._high_water_bytes / 1e9,
            )

    def _evict(self, session_id: str, entry: _DrainedEntry) -> None:
        failed = False
        for stage_dir in entry.stage_dirs:
            try:
                shutil.rmtree(stage_dir)
            except FileNotFoundError:
                pass  # already gone is what eviction wants
            except Exception:
                logger.exception(
                    "Failed to evict RAM disk staging copy for session %s "
                    "at %s -- leaving it in place, will retry next sweep",
                    session_id,
                    stage_dir,
                )
                failed = True
        if failed:
            with self._lock:
                self._drained.setdefault(session_id, entry)
            return
        if self._manifest_dir is not None:
            (self._manifest_dir / f"{session_id}.json").unlink(missing_ok=True)
        logger.debug(
            "Evicted RAM disk staging copies for %s (%d item(s))",
            session_id,
            len(entry.stage_dirs),
        )

    # --- restart-safe retention ------------------------------------------------

    def _write_manifest(
        self, session_id: str, stage_dirs: list[Path], size: int
    ) -> None:
        if self._manifest_dir is None:
            return
        try:
            self._manifest_dir.mkdir(parents=True, exist_ok=True)
            path = self._manifest_dir / f"{session_id}.json"
            tmp = path.with_name(f".{path.name}.tmp")
            tmp.write_text(
                json.dumps(
                    {
                        "stage_dirs": [str(d) for d in stage_dirs],
                        "drained_at": time.time(),
                        "size_bytes": size,
                    }
                )
            )
            tmp.replace(path)
        except OSError:
            logger.exception("Could not record drain manifest for %s", session_id)

    def _load_manifests(self) -> None:
        """Re-adopt sessions a previous process drained but hadn't evicted
        yet, keeping their original drain time for the retention clock."""
        if self._manifest_dir is None or not self._manifest_dir.is_dir():
            return
        now_wall, now_mono = time.time(), time.monotonic()
        for path in self._manifest_dir.glob("*.json"):
            try:
                rec = json.loads(path.read_text())
                entry = _DrainedEntry(
                    stage_dirs=[Path(d) for d in rec["stage_dirs"]],
                    drained_at=now_mono - max(0.0, now_wall - float(rec["drained_at"])),
                    size_bytes=int(rec["size_bytes"]),
                )
            except (OSError, ValueError, KeyError, TypeError):
                logger.warning("Ignoring unreadable drain manifest %s", path)
                continue
            with self._lock:
                self._drained.setdefault(path.stem, entry)
        if self._drained:
            logger.info(
                "Re-adopted %d drained session(s) awaiting RAM-disk eviction",
                len(self._drained),
            )


def _tree_stats(root: Path) -> tuple[int, int]:
    count = 0
    total = 0
    for p in root.rglob("*"):
        if p.is_file():
            count += 1
            total += p.stat().st_size
    return count, total
