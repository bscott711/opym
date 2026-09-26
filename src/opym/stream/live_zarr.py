# Ruff style: Compliant
"""One-format live lane: every streamed (t, c) volume goes raw OME-Zarr ->
GPU (decon -> deskew/rotate, in memory) -> processed OME-Zarr, with no
intermediate files and nothing written to GPFS before the viewer has it.

Selected by `OPYM_LIVE_FORMAT=zarr` (read when the receiver creates its
lane); the TIFF lane (opym.stream.live.LiveLane) stays the default until
this one is proven in production. Per session:

- Dispatch. Each (t, c) gets its own 'live_zarr' ticket (run_live_zarr.m)
  the moment its last plane is in the raw store, so both GPUs share one
  timepoint's channels. Nothing is dispatched before (0, 0) is staged: the
  edge-erosion mask is built from channel 0's first timepoint.
- View. The ticket reads the (t, c) straight from the raw store on the RAM
  disk and, before any encoding, drops the full-resolution result into
  `buffers/T<t>_C<c>.npy` (uncompressed, memory-mapped by naparym-live).
  It then writes every pyramid level and the Z-MIP into the processed store
  on the RAM disk (`<view root>/<session>/<base>_dsr.ome.zarr`, bioformats2raw
  layout 3 with OME-XML, see opym.ome_zarr_writer), and the lane records the
  (t, c) in that store's progress file.
- Archive. Each finished (t, c)'s chunk files are then copied, unchanged, to
  the same store on GPFS (`<leaf>/viewer/<base>_dsr.ome.zarr`, where the
  backfill's viewer export puts it): no re-encoding. A timepoint is done
  when every channel is archived; `.live_status.json` (the backfill's
  hand-off, as for the TIFF lane) and the GPFS store's progress follow.
- RAM disk. Only the newest `BUFFER_KEEP_T` timepoints keep their
  uncompressed buffers; older ones are read from the store. A session's
  view directory stays until a later session starts (so the last
  acquisition remains viewable), then goes.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from opym import lanes
from opym.decon_config import XY_PIXEL_SIZE_UM, deskew_decon_kwargs
from opym.ome_zarr_writer import (
    ProcessedArrays,
    create_processed_store,
    write_progress,
)
from opym.petakit import submit_live_zarr_job
from opym.stream import trace
from opym.stream.live import (
    LIVE_LATEST_NAME,
    LiveLane,
    live_dsr_dir,
    live_qc_dir,
    live_viewer_store,
)
from opym.utils import dsr_shape_zyx

logger = logging.getLogger(__name__)

LIVE_FORMAT_ENV_VAR = "OPYM_LIVE_FORMAT"
VIEW_ROOT_ENV_VAR = "OPYM_LIVE_VIEW_ROOT"
DEFAULT_VIEW_ROOT = Path("/dev/shm/opym_live_view")
# Newest timepoints whose uncompressed view buffers stay on the RAM disk
# (about 1 GB per channel each at 161 planes).
BUFFER_KEEP_T = 3
SHEET_ANGLE_DEG = 60.0


def buffer_name(t: int, c: int) -> str:
    """A (t, c) view buffer's file name (naparym-live reads the same)."""
    return f"T{t:04d}_C{c}.npy"


def parse_buffer_name(name: str) -> tuple[int, int] | None:
    if not (name.startswith("T") and name.endswith(".npy") and "_C" in name):
        return None
    try:
        t, c = name[1:-4].split("_C")
        return int(t), int(c)
    except ValueError:
        return None


def live_format() -> str:
    """ "zarr" for this lane, else "tiff" (the default for now)."""
    return (
        "zarr" if os.environ.get(LIVE_FORMAT_ENV_VAR, "").strip() == "zarr" else "tiff"
    )


def view_root() -> Path:
    return Path(os.environ.get(VIEW_ROOT_ENV_VAR, "").strip() or DEFAULT_VIEW_ROOT)


@dataclass
class _ZTicket:
    name: str
    t: int
    c: int
    attempt: int


@dataclass
class ZarrLiveSession:
    session_id: str
    base_name: str
    num_timepoints: int
    n_channels: int
    z_step_um: float
    channel_labels: list[str]
    raw_arrays: list[Path]  # each channel's raw (T, Z, Y, X) array, by cidx
    work_dir: Path  # <stage leaf>/live/<session>: decon cache (PSF, OMW, mask)
    view_dir: Path  # <view root>/<session> on the RAM disk
    view_store: Path  # processed store the GPU writes; naparym-live reads
    archive_store: Path  # the same store on GPFS
    dsr_dir: Path  # where .live_status.json goes (the backfill's hand-off)
    qc_dir: Path
    view_arrays: ProcessedArrays = None  # type: ignore[assignment]
    staged: dict[int, set[int]] = field(default_factory=dict)  # t -> cidx staged
    ready: list[tuple[int, int]] = field(default_factory=list)
    tickets: dict[str, _ZTicket] = field(default_factory=dict)
    buffered: set[tuple[int, int]] = field(default_factory=set)
    viewed: set[tuple[int, int]] = field(default_factory=set)  # store written
    archiving: dict[tuple[int, int], object] = field(default_factory=dict)
    archived: set[tuple[int, int]] = field(default_factory=set)
    done: set[int] = field(default_factory=set)  # every channel archived
    failed: set[int] = field(default_factory=set)
    ended: bool = False
    superseded: bool = False
    started_at: float = 0.0
    first_done_at: float | None = None
    last_status_write: float = 0.0

    @property
    def decon_dir(self) -> Path:
        return self.work_dir / "Decon"

    @property
    def buffers_dir(self) -> Path:
        return self.view_dir / "buffers"

    def buffer(self, t: int, c: int) -> Path:
        return self.buffers_dir / buffer_name(t, c)

    def settled(self) -> bool:
        """Every fully staged timepoint is archived or failed, nothing in flight."""
        if self.superseded:
            return not self.tickets and not self.archiving
        full = {t for t, cs in self.staged.items() if len(cs) == self.n_channels}
        return (
            self.ended
            and not self.ready
            and not self.tickets
            and not self.archiving
            and full <= (self.done | self.failed)
        )


class ZarrLiveLane(LiveLane):
    """The one-format lane; the receiver hooks match LiveLane's, except
    start_session, which takes the raw arrays instead of a frames dir."""

    def start_session(
        self,
        session_id: str,
        *,
        base_name: str,
        num_timepoints: int,
        n_channels: int,
        raw_arrays: list[Path],
        raw_shape_zyx: tuple[int, int, int],
        stage_leaf: Path,
        dest_leaf: Path,
        z_step_um: float,
        channel_labels: list[str] | None = None,
        time_interval_s: float | None = None,
    ) -> ZarrLiveSession:
        dest_leaf = Path(dest_leaf)
        try:
            dest_leaf.mkdir(parents=True, exist_ok=True)  # see live.py's docstring
        except OSError:
            pass
        archive_store = live_viewer_store(dest_leaf)
        view_dir = view_root() / session_id
        session = ZarrLiveSession(
            session_id=session_id,
            base_name=base_name,
            num_timepoints=num_timepoints,
            n_channels=n_channels,
            z_step_um=z_step_um,
            channel_labels=list(channel_labels or [f"C{c}" for c in range(n_channels)]),
            raw_arrays=[Path(p) for p in raw_arrays],
            work_dir=Path(stage_leaf) / "live" / session_id,
            view_dir=view_dir,
            view_store=view_dir / archive_store.name,
            archive_store=archive_store,
            dsr_dir=live_dsr_dir(dest_leaf, self.psf),
            qc_dir=live_qc_dir(dest_leaf),
            started_at=self._clock(),
        )
        for other in self.sessions.values():
            if other.dsr_dir == session.dsr_dir and not other.superseded:
                self._supersede(other, by=session_id)
        self._evict_views(keep={session_id, *self.sessions})

        shape = dsr_shape_zyx(
            tuple(raw_shape_zyx), z_step_um, XY_PIXEL_SIZE_UM, SHEET_ANGLE_DEG
        )
        store_kw = {
            "n_t": num_timepoints,
            "n_c": n_channels,
            "shape_zyx": shape,
            "channel_labels": session.channel_labels,
            "time_interval_s": time_interval_s or None,
        }
        session.buffers_dir.mkdir(parents=True, exist_ok=True)
        session.view_arrays = create_processed_store(session.view_store, **store_kw)
        create_processed_store(session.archive_store, **store_kw)
        session.dsr_dir.mkdir(parents=True, exist_ok=True)
        self.sessions[session_id] = session
        self._write_status(session, "running")
        self._write_latest(session)
        logger.info(
            "Live lane (zarr): session %s (%s) -> %s, archived to %s",
            session_id,
            base_name,
            session.view_store,
            session.archive_store,
        )
        return session

    def frame_staged(self, session_id: str, t: int, cidx: int, raw=None) -> None:
        """(t, cidx) is complete in the raw store. `raw` is that volume, for QC."""
        session = self.sessions.get(session_id)
        if session is None:
            return
        cs = session.staged.setdefault(t, set())
        if cidx in cs:
            return
        cs.add(cidx)
        session.ready.append((t, cidx))
        if self.qc and raw is not None and not session.superseded:
            self._submit_qc(session, t, cidx, raw)

    # --- pump steps ---------------------------------------------------------

    def _dispatch(self, session: ZarrLiveSession) -> None:
        if session.superseded or not session.ready:
            return
        if 0 not in session.staged or 0 not in session.staged[0]:
            return  # the erosion mask comes from (0, 0)
        free = self.max_outstanding - self._in_flight()
        session.ready.sort()
        while session.ready and free > 0:
            t, c = session.ready.pop(0)
            self._submit(session, t, c, attempt=1)
            free -= 1

    def _submit(self, session: ZarrLiveSession, t: int, c: int, attempt: int) -> None:
        ticket = submit_live_zarr_job(
            session.raw_arrays[c],
            t,
            c,
            mask_store=session.raw_arrays[0],
            levels=session.view_arrays.levels,
            mip=session.view_arrays.mip,
            psf_path=self.psf,
            decon_dir=session.decon_dir,
            z_step_um=session.z_step_um,
            ticket_name=f"{session.base_name}_T{t:04d}_C{c}",
            queue_dir=lanes.live_queue_dir(self.jobs),
            view_npy=session.buffer(t, c),
            **deskew_decon_kwargs(self.psf),
        )
        session.tickets[ticket.name] = _ZTicket(ticket.name, t, c, attempt)
        trace.record(
            "ticket",
            jobs=self._jobs,
            session_id=session.session_id,
            ticket=ticket.name,
            timepoints=[t],
            c=c,
            attempt=attempt,
        )

    def _reap(self, session: ZarrLiveSession) -> None:
        for name, ticket in list(session.tickets.items()):
            tc = (ticket.t, ticket.c)
            if tc not in session.buffered and session.buffer(*tc).exists():
                session.buffered.add(tc)
                trace.record(
                    "view_buffer",
                    jobs=self._jobs,
                    session_id=session.session_id,
                    t=ticket.t,
                    c=ticket.c,
                )
            if (self.jobs / "completed" / name).exists():
                del session.tickets[name]
                trace.record(
                    "ticket_done",
                    jobs=self._jobs,
                    session_id=session.session_id,
                    ticket=name,
                    timepoints=[ticket.t],
                    c=ticket.c,
                )
                if session.superseded:
                    continue
                session.buffered.add(tc)
                session.viewed.add(tc)
                self._write_progress(session, session.view_store, session.viewed)
                if self._all_channels(session, session.viewed, ticket.t):
                    trace.record(
                        "view_ready",
                        jobs=self._jobs,
                        session_id=session.session_id,
                        t=ticket.t,
                    )
                session.archiving[tc] = self._pool.submit(self._archive, session, *tc)
            elif (self.jobs / "failed" / name).exists():
                del session.tickets[name]
                if session.superseded:
                    continue
                if ticket.attempt < 2:
                    logger.warning(
                        "Live zarr ticket %s failed; retrying T=%d C=%d",
                        name,
                        ticket.t,
                        ticket.c,
                    )
                    self._submit(session, ticket.t, ticket.c, ticket.attempt + 1)
                else:
                    logger.error(
                        "Live zarr ticket %s failed twice; T=%d left to the "
                        "batch backfill",
                        name,
                        ticket.t,
                    )
                    session.failed.add(ticket.t)
        self._trim_buffers(session)

    def _archive(self, session: ZarrLiveSession, t: int, c: int) -> None:
        """Copy (t, c)'s chunk files, unchanged, from the RAM-disk store to
        the GPFS one. Each file lands under a temp name and is renamed into
        place, and its size is checked; the pixels are reproducible from
        the raw store (itself drained byte-for-byte) should that ever
        matter. Runs on the copy pool."""
        arrays = [*session.view_arrays.levels, session.view_arrays.mip]
        for arr in arrays:
            src_dir = arr / str(t) / str(c)
            if not src_dir.is_dir():
                continue  # every chunk was all-zero (never written)
            dst_dir = session.archive_store / arr.relative_to(session.view_store)
            dst_dir = dst_dir / str(t) / str(c)
            for src in src_dir.rglob("*"):
                if not src.is_file():
                    continue
                dst = dst_dir / src.relative_to(src_dir)
                dst.parent.mkdir(parents=True, exist_ok=True)
                tmp = dst.with_name(f".{dst.name}.copying")
                shutil.copyfile(src, tmp)
                if tmp.stat().st_size != src.stat().st_size:
                    raise OSError(f"short copy of {src} -> {tmp}")
                os.replace(tmp, dst)

    def _finish_copies(self, session: ZarrLiveSession) -> None:
        changed = False
        for tc, fut in list(session.archiving.items()):
            if not fut.done():
                continue
            del session.archiving[tc]
            try:
                fut.result()
            except Exception:
                logger.exception(
                    "Live lane (zarr): archiving T=%d C=%d of %s to GPFS failed",
                    *tc,
                    session.base_name,
                )
                session.failed.add(tc[0])
                continue
            session.archived.add(tc)
            changed = True
            if self._all_channels(session, session.archived, tc[0]):
                session.done.add(tc[0])
                if session.first_done_at is None:
                    session.first_done_at = self._clock()
        if changed:
            self._write_progress(session, session.archive_store, session.archived)
            self._write_status(session, "running")

    def _finalize(self, session: ZarrLiveSession) -> None:
        shutil.rmtree(session.work_dir, ignore_errors=True)
        if session.superseded:
            shutil.rmtree(session.view_dir, ignore_errors=True)
            del self.sessions[session.session_id]
            logger.info(
                "Live lane (zarr): superseded session %s closed", session.session_id
            )
            return
        full = sorted(
            t for t, cs in session.staged.items() if len(cs) == session.n_channels
        )
        state = "complete" if full and not session.failed else "failed"
        self._write_status(session, state)
        self._write_progress(session, session.view_store, session.viewed, state)
        self._write_progress(session, session.archive_store, session.archived, state)
        for d in (session.work_dir.parent, session.work_dir.parent.parent):
            try:
                d.rmdir()  # only if now empty
            except OSError:
                pass
        del self.sessions[session.session_id]
        logger.info(
            "Live lane (zarr): session %s %s, %d/%d timepoint(s) processed live",
            session.session_id,
            state,
            len(session.done),
            len(full),
        )

    # --- helpers ------------------------------------------------------------

    @staticmethod
    def _all_channels(session: ZarrLiveSession, pairs: set, t: int) -> bool:
        return all((t, c) in pairs for c in range(session.n_channels))

    def _write_progress(
        self,
        session: ZarrLiveSession,
        store: Path,
        pairs: set[tuple[int, int]],
        state: str = "running",
    ) -> None:
        try:
            write_progress(
                store,
                n_t=session.num_timepoints,
                n_c=session.n_channels,
                done=[list(tc) for tc in sorted(pairs)],
                state=state,
            )
        except OSError:
            logger.exception("Live lane (zarr): could not update %s progress", store)

    def _trim_buffers(self, session: ZarrLiveSession) -> None:
        """Keep uncompressed buffers only for the newest BUFFER_KEEP_T
        timepoints, and only once their store copy exists (older timepoints
        are read from the store). Unlinking a buffer naparym-live still has
        mapped is safe: the mapping outlives the name."""
        viewed_t = sorted({t for t, _ in session.viewed})
        keep = set(viewed_t[-BUFFER_KEEP_T:])
        for t, c in list(session.viewed):
            if t not in keep:
                session.buffer(t, c).unlink(missing_ok=True)

    def _evict_views(self, keep: set[str]) -> None:
        """Remove the view directories of sessions this lane isn't running,
        except the newest finished one (still worth looking at)."""
        root = view_root()
        if not root.is_dir():
            return
        others = [d for d in root.iterdir() if d.is_dir() and d.name not in keep]
        others.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        for d in others[1:]:
            shutil.rmtree(d, ignore_errors=True)

    def _write_latest(self, session: ZarrLiveSession) -> None:
        """Point naparym-live at the newest session: the RAM-disk store and
        its buffers first, the GPFS store for later."""
        path = self.jobs / LIVE_LATEST_NAME
        tmp = path.with_name(f".{LIVE_LATEST_NAME}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(
                json.dumps(
                    {
                        "session_id": session.session_id,
                        "base_name": session.base_name,
                        "store": str(session.archive_store),
                        "view_store": str(session.view_store),
                        "buffers_dir": str(session.buffers_dir),
                        "dsr_dir": str(session.dsr_dir),
                        "started_at": session.started_at,
                        "qc_dir": str(session.qc_dir) if self.qc else None,
                        "num_timepoints": session.num_timepoints,
                        "n_channels": session.n_channels,
                        "channel_labels": session.channel_labels,
                        "z_step_um": session.z_step_um,
                    }
                )
            )
            os.replace(tmp, path)
        except OSError:
            logger.exception("Live lane (zarr): could not write %s", path)
