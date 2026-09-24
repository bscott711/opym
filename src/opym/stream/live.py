# Ruff style: Compliant
"""Live lane: decon -> deskew/rotate each streamed timepoint as soon as all of
its channels have landed, instead of after the whole acquisition drains.

Driven by `opym.stream.receiver` when OPYM_LIVE_LANE=1 and decon is enabled.
Per session:

- Dispatcher. When every channel of timepoint t has been staged (the
  receiver's `decon_stage/` TIFFs, already in PetaKit5D's layout), queue a
  'live' ticket in `queue_live/` (see run_live_frames.m). The ticket takes
  its parameters from `opym.decon_config.deskew_decon_kwargs`, the same
  source the backfill uses; the live path is bit-identical to the batch one.
  At most `max_outstanding` tickets are in flight. When more timepoints are
  ready than free slots, they are grouped, up to `max_batch` per ticket, so
  a backlog drains in fewer, larger tickets.
- Finisher. When a ticket completes, each frame's DSR TIFF and MIP are copied
  to the dataset's final DSR directory on GPFS (the one the backfill would
  write, `<leaf>/decon_stage/Decon/DSR_decon`) on a small thread pool, off
  the receiver's socket loop, then removed from the RAM disk. Copies land
  under a temp name and are renamed into place, so readers never see a
  partial file. A failed ticket is retried once, then its timepoints are
  marked failed.
- Viewer store. Each finished timepoint is also written into the dataset's
  pyramidal OME-Zarr (`<leaf>/viewer/<base>_dsr.ome.zarr`, the path and
  layout the backfill's viewer export uses; see opym.ome_zarr_writer), so
  napari (opym.live_view) can follow the acquisition. `live_latest.json` in
  the jobs dir points at the newest session's store.
- Hand-off. `.live_status.json` in that DSR directory says whether the
  session is running (heartbeat), complete or failed, with the provenance it
  was produced under. The backfill reads it (`read_live_status`) before
  submitting a batch deskew: complete + matching provenance means done,
  running + fresh means wait, and anything else means reprocess in batch.
  Registry writes stay in the backfill.

The lane creates the dataset's GPFS leaf directory itself, before anything
else writes there. `opym.utils.resolve_output_base` treats a leaf directory
that doesn't exist yet as unwritable, so creating it first is what makes the
lane and the backfill resolve the same output path.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from opym import lanes, ome_zarr_writer
from opym.decon_config import (
    DSR_INTERP_METHOD,
    decon_params_fingerprint,
    deskew_decon_kwargs,
    dsr_dir_name_for,
)
from opym.petakit import submit_live_frames_job
from opym.utils import resolve_output_base

logger = logging.getLogger(__name__)

LIVE_STATUS_NAME = ".live_status.json"
LIVE_LATEST_NAME = "live_latest.json"
# A "running" status older than this is treated as abandoned (receiver
# crashed): the backfill then reprocesses the dataset in batch.
LIVE_STATUS_STALE_S = 300.0
HEARTBEAT_S = 30.0
MAX_ATTEMPTS = 2


def live_dsr_dir(dest_leaf: Path, psf: Path) -> Path:
    """The final DSR directory for a streamed zarr dataset: the same path
    `bioimaging.backfill.pipeline` derives (zarr_deskew_data_dir ->
    dsr_output_dir) once `dest_leaf` exists."""
    return (
        resolve_output_base(dest_leaf) / "decon_stage" / "Decon" / dsr_dir_name_for(psf)
    )


def live_viewer_store(dest_leaf: Path) -> Path:
    """The napari store for a streamed dataset: where the backfill's viewer
    export writes it (`resolve_output_base(leaf)/viewer/<leaf name>_dsr...`)."""
    dest_leaf = Path(dest_leaf)
    return resolve_output_base(dest_leaf) / "viewer" / f"{dest_leaf.name}_dsr.ome.zarr"


def channel_label(name: str) -> str:
    """`GFP_488` -> `GFP 488`, matching the backfill export's labels."""
    parts = name.split("_")
    if len(parts) >= 2 and parts[-1].isdigit():
        return f"{parts[-2]} {parts[-1]}"
    return name


def read_live_status(dsr_dir: Path) -> dict | None:
    try:
        return json.loads((Path(dsr_dir) / LIVE_STATUS_NAME).read_text())
    except (OSError, ValueError):
        return None


def live_status_is_fresh(status: dict, now: float | None = None) -> bool:
    now = time.time() if now is None else now
    return now - float(status.get("updated_at", 0)) <= LIVE_STATUS_STALE_S


@dataclass
class _Ticket:
    name: str
    timepoints: list[int]
    attempt: int


@dataclass
class LiveSession:
    session_id: str
    base_name: str
    num_timepoints: int
    n_channels: int
    frames_dir: Path  # the receiver's decon_stage/ (RAM disk when staging)
    work_dir: Path  # <stage leaf>/live
    dsr_dir: Path  # final DSR dir on GPFS
    z_step_um: float
    zarr_path: Path  # napari store on GPFS
    channel_labels: list[str]
    staged: dict[int, set[int]] = field(default_factory=dict)  # t -> cidx staged
    ready: list[int] = field(default_factory=list)
    tickets: dict[str, _Ticket] = field(default_factory=dict)
    copying: dict[int, Future] = field(default_factory=dict)
    done: set[int] = field(default_factory=set)
    failed: set[int] = field(default_factory=set)
    ended: bool = False
    started_at: float = field(default_factory=time.time)
    first_done_at: float | None = None
    last_status_write: float = 0.0
    zarr_lock: threading.Lock = field(default_factory=threading.Lock)
    zarr_created: bool = False

    @property
    def decon_dir(self) -> Path:
        return self.work_dir / "Decon"

    def frame(self, t: int, cidx: int) -> Path:
        return self.frames_dir / f"{self.base_name}_C{cidx}_T{t:03d}.tif"

    def settled(self) -> bool:
        """Every fully staged timepoint is done or failed, nothing in flight."""
        full = {t for t, cs in self.staged.items() if len(cs) == self.n_channels}
        return (
            self.ended
            and not self.ready
            and not self.tickets
            and not self.copying
            and full <= (self.done | self.failed)
        )


class LiveLane:
    def __init__(
        self,
        psf: Path,
        *,
        jobs: Path | None = None,
        max_outstanding: int = 4,
        max_batch: int = 8,
        copy_workers: int = 2,
        clock=time.time,
    ) -> None:
        self.psf = Path(psf)
        self._jobs = jobs
        self.max_outstanding = max_outstanding
        self.max_batch = max_batch
        self._clock = clock
        self.sessions: dict[str, LiveSession] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(
            max_workers=copy_workers, thread_name_prefix="OpymLiveCopy"
        )

    @property
    def jobs(self) -> Path:
        return self._jobs or lanes.jobs_dir()

    def close(self) -> None:
        self._pool.shutdown(wait=True)

    def busy(self) -> bool:
        return bool(self.sessions)

    # --- receiver hooks -------------------------------------------------------

    def start_session(
        self,
        session_id: str,
        *,
        base_name: str,
        num_timepoints: int,
        n_channels: int,
        frames_dir: Path,
        stage_leaf: Path,
        dest_leaf: Path,
        z_step_um: float,
        channel_labels: list[str] | None = None,
    ) -> LiveSession:
        # Create the GPFS leaf before resolving the output path; see module
        # docstring. An unwritable raw root falls back to the mirror, the
        # same way the backfill's resolve_output_base does.
        try:
            Path(dest_leaf).mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        session = LiveSession(
            session_id=session_id,
            base_name=base_name,
            num_timepoints=num_timepoints,
            n_channels=n_channels,
            frames_dir=Path(frames_dir),
            work_dir=Path(stage_leaf) / "live",
            dsr_dir=live_dsr_dir(Path(dest_leaf), self.psf),
            z_step_um=z_step_um,
            zarr_path=live_viewer_store(Path(dest_leaf)),
            channel_labels=list(channel_labels or [f"C{c}" for c in range(n_channels)]),
        )
        session.dsr_dir.mkdir(parents=True, exist_ok=True)
        (session.dsr_dir / "MIPs").mkdir(exist_ok=True)
        self.sessions[session_id] = session
        self._write_status(session, "running")
        self._write_latest(session)
        logger.info(
            "Live lane: session %s (%s) -> %s", session_id, base_name, session.dsr_dir
        )
        return session

    def frame_staged(self, session_id: str, t: int, cidx: int) -> None:
        session = self.sessions.get(session_id)
        if session is None:
            return
        cs = session.staged.setdefault(t, set())
        if cidx in cs:
            return
        cs.add(cidx)
        if len(cs) == session.n_channels:
            session.ready.append(t)

    def end_session(self, session_id: str) -> None:
        session = self.sessions.get(session_id)
        if session is not None:
            session.ended = True

    # --- one pass, from the receiver's poll loop ------------------------------

    def pump(self) -> None:
        for session in list(self.sessions.values()):
            self._reap(session)
            self._dispatch(session)
            self._finish_copies(session)
            now = self._clock()
            if session.settled():
                self._finalize(session)
            elif now - session.last_status_write >= HEARTBEAT_S:
                self._write_status(session, "running")

    def _in_flight(self) -> int:
        return sum(len(s.tickets) for s in self.sessions.values())

    def _dispatch(self, session: LiveSession) -> None:
        free = self.max_outstanding - self._in_flight()
        if free <= 0 or not session.ready:
            return
        session.ready.sort()
        per_ticket = min(self.max_batch, max(1, math.ceil(len(session.ready) / free)))
        while session.ready and free > 0:
            group, session.ready = (
                session.ready[:per_ticket],
                session.ready[per_ticket:],
            )
            self._submit(session, group, attempt=1)
            free -= 1

    def _submit(
        self, session: LiveSession, timepoints: list[int], attempt: int
    ) -> None:
        frames = [
            session.frame(t, c) for t in timepoints for c in range(session.n_channels)
        ]
        patterns = [f"_C{c}_T" for c in range(session.n_channels)]
        ticket = submit_live_frames_job(
            frames,
            session.decon_dir,
            session.frame(0, 0),  # erodeByFTP: the session's first C0 frame
            [self.psf] * session.n_channels,
            patterns,
            session.z_step_um,
            ticket_name=session.base_name,
            queue_dir=lanes.live_queue_dir(self.jobs),
            **deskew_decon_kwargs(self.psf),
        )
        session.tickets[ticket.name] = _Ticket(ticket.name, list(timepoints), attempt)

    def _reap(self, session: LiveSession) -> None:
        for name, ticket in list(session.tickets.items()):
            if (self.jobs / "completed" / name).exists():
                del session.tickets[name]
                for t in ticket.timepoints:
                    session.copying[t] = self._pool.submit(self._copy_out, session, t)
            elif (self.jobs / "failed" / name).exists():
                del session.tickets[name]
                if ticket.attempt < MAX_ATTEMPTS:
                    logger.warning(
                        "Live ticket %s failed; retrying timepoints %s",
                        name,
                        ticket.timepoints,
                    )
                    self._submit(session, ticket.timepoints, attempt=ticket.attempt + 1)
                else:
                    logger.error(
                        "Live ticket %s failed twice; timepoints %s left to the "
                        "batch backfill",
                        name,
                        ticket.timepoints,
                    )
                    session.failed.update(ticket.timepoints)

    def _copy_out(self, session: LiveSession, t: int) -> None:
        """Copy timepoint t's DSR frames and MIPs to GPFS, then free the RAM
        disk copies. Runs on the copy pool."""
        src_dsr = session.decon_dir / dsr_dir_name_for(self.psf)
        for c in range(session.n_channels):
            fsname = f"{session.base_name}_C{c}_T{t:03d}"
            self._write_viewer_store(session, t, c, src_dsr / f"{fsname}.tif")
            for rel in (f"{fsname}.tif", f"MIPs/{fsname}_MIP_z.tif"):
                src, dst = src_dsr / rel, session.dsr_dir / rel
                tmp = dst.with_name(f".{dst.name}.copying")
                shutil.copyfile(src, tmp)
                os.replace(tmp, dst)
                src.unlink()
            # The first C0 frame builds the session's erosion mask; keep it
            # until the session is finalized.
            if not (t == 0 and c == 0):
                session.frame(t, c).unlink(missing_ok=True)

    def _write_viewer_store(
        self, session: LiveSession, t: int, c: int, src: Path
    ) -> None:
        import tifffile

        vol = tifffile.imread(src)
        with session.zarr_lock:
            if not session.zarr_created:
                session.zarr_path.parent.mkdir(parents=True, exist_ok=True)
                ome_zarr_writer.create_store(
                    session.zarr_path,
                    n_t=session.num_timepoints,
                    n_c=session.n_channels,
                    shape_zyx=vol.shape,
                    dtype=vol.dtype,
                    channel_labels=session.channel_labels,
                )
                session.zarr_created = True
        ome_zarr_writer.write_timepoint(session.zarr_path, t, c, vol)

    def _write_viewer_progress(self, session: LiveSession, state: str) -> None:
        if not session.zarr_created:
            return
        done = [[t, c] for t in sorted(session.done) for c in range(session.n_channels)]
        try:
            ome_zarr_writer.write_progress(
                session.zarr_path,
                n_t=session.num_timepoints,
                n_c=session.n_channels,
                done=done,
                state=state,
            )
        except OSError:
            logger.exception(
                "Live lane: could not update %s progress", session.zarr_path
            )

    def _write_latest(self, session: LiveSession) -> None:
        """Point `naparym-live` (no arguments) at the newest session."""
        path = self.jobs / LIVE_LATEST_NAME
        tmp = path.with_name(f".{LIVE_LATEST_NAME}.tmp")
        try:
            tmp.write_text(
                json.dumps(
                    {
                        "session_id": session.session_id,
                        "base_name": session.base_name,
                        "store": str(session.zarr_path),
                        "dsr_dir": str(session.dsr_dir),
                        "started_at": session.started_at,
                    }
                )
            )
            os.replace(tmp, path)
        except OSError:
            logger.exception("Live lane: could not write %s", path)

    def _finish_copies(self, session: LiveSession) -> None:
        for t, fut in list(session.copying.items()):
            if not fut.done():
                continue
            del session.copying[t]
            try:
                fut.result()
            except Exception:
                logger.exception(
                    "Live lane: copying timepoint %d of %s to GPFS failed",
                    t,
                    session.base_name,
                )
                session.failed.add(t)
                continue
            session.done.add(t)
            if session.first_done_at is None:
                session.first_done_at = self._clock()
            self._write_status(session, "running")
            self._write_viewer_progress(session, "running")

    def _finalize(self, session: LiveSession) -> None:
        full = sorted(
            t for t, cs in session.staged.items() if len(cs) == session.n_channels
        )
        state = "complete" if full and not session.failed else "failed"
        self._write_status(session, state)
        self._write_viewer_progress(session, state)
        # The staged TIFFs aren't drained in live mode, so nothing else would
        # ever evict leftovers (e.g. a failed timepoint's). A batch fallback
        # rebuilds them from the raw stores on GPFS.
        shutil.rmtree(session.work_dir, ignore_errors=True)
        shutil.rmtree(session.frames_dir, ignore_errors=True)
        try:
            session.work_dir.parent.rmdir()  # the stage leaf, if now empty
        except OSError:
            pass
        del self.sessions[session.session_id]
        logger.info(
            "Live lane: session %s %s, %d/%d timepoint(s) processed live",
            session.session_id,
            state,
            len(session.done),
            len(full),
        )

    def _write_status(self, session: LiveSession, state: str) -> None:
        now = self._clock()
        status = {
            "state": state,
            "session_id": session.session_id,
            "base_name": session.base_name,
            "updated_at": now,
            "started_at": session.started_at,
            "first_done_at": session.first_done_at,
            "n_channels": session.n_channels,
            "declared_timepoints": session.num_timepoints,
            "timepoints_done": sorted(session.done),
            "timepoints_failed": sorted(session.failed),
            "decon_psf": str(self.psf),
            "decon_params": decon_params_fingerprint(),
            "interp_method": DSR_INTERP_METHOD,
        }
        path = session.dsr_dir / LIVE_STATUS_NAME
        tmp = path.with_name(f".{LIVE_STATUS_NAME}.tmp")
        try:
            tmp.write_text(json.dumps(status, indent=1))
            os.replace(tmp, path)
            session.last_status_write = now
        except OSError:
            logger.exception("Live lane: could not write %s", path)
