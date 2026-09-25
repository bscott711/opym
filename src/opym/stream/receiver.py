# Ruff style: Compliant
"""
ROUTER-side server for the real-time frame-streaming protocol (see
`opym.stream.protocol`). Writes each incoming frame directly into a
GPFS-resident raw OME-Zarr mirror store (`opym.stream.rawmirror`) -- the
exact same directory-per-channel layout a completed Globus transfer already
leaves behind for the newer pymmcore-based MDA writer. That means
`opym.discovery` and the existing `opym-backfill --watch` loop discover a
streamed dataset, and submit its one deskew/decon ticket, with no code
changes on the discovery/backfill side: this module's whole job is making a
live acquisition indistinguishable, once `SESSION_END` lands, from one that
finished uploading over Globus.

When decon is enabled on this host (`OPYM_DECON_PSF` set -- the same switch
`bioimaging.backfill.pipeline.resolve_decon_psf` reads), each frame is also
written straight to `decon_stage/` in PetaKit5D's required `(ny, nx, nz)`
TIFF layout (`opym.utils.write_decon_staged_tiff`, shared with
`bioimaging.backfill.pipeline.build_decon_staging_dir`). This is a pure
pre-computation: it makes the batch pass's own staging step a no-op (every
destination file already exists) rather than a second, divergent decon
path -- decon parameters themselves (PSF, wiener_alpha, edge_erosion) are
still resolved and the ticket still submitted entirely by the batch
backfill driver, unchanged.

There is deliberately no `opym.consolidate` step here anymore: each
channel's frames are written straight into their final-shaped store as they
arrive, not into small per-(t,c) outputs that need stitching afterward.

RAM-disk staging (opt-in, `OPYM_STREAM_STAGE_ROOT`): when set, every write
this module makes (raw mirror stores AND decon-stage TIFFs) goes to
`<stage_root>/<base_name>/...` instead of `<raw_root>/<base_name>/...` --
intended to be a tmpfs mount (e.g. `/dev/shm`), so PetaKit5D's local GPU
pipeline (itself already tmpfs-native -- see `local_gpu_worker.py`,
`run_petakit_server.m`) reads its input with zero GPFS read latency. `Session
State.raw_root` keeps its exact original meaning (the client's declared
final destination) unchanged; a background `opym.stream.drain.DrainPool`
copies each completed session from the staging root to `raw_root`,
verifies it byte-for-byte, and only then makes it visible there -- see
`_finalize_session`. Unset (the default, and what every existing test and
the current `opym-receive.service` use), this module behaves exactly as
before: every write goes straight to `raw_root`, synchronously, no staging,
no drain, no behavior change at all.

Live QC (`MSG_QC`): a client that lists "qc" in SESSION_START's `accepts`
is sent each new verdict the live QC service writes for its session
(`<leaf>/qc/qc_latest.json`, see `opym.stream.live`), header only. Clients
that don't ask never see the message type.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import zmq

from opym import lanes
from opym.decon_config import resolve_decon_psf
from opym.stream import drain, live, rawmirror
from opym.stream.live import LiveLane
from opym.stream.protocol import (
    MSG_ACK,
    MSG_FRAME,
    MSG_QC,
    MSG_RESUME,
    MSG_SESSION_END,
    MSG_SESSION_START,
    pack_message,
    unpack_message,
)
from opym.utils import write_decon_staged_tiff

logger = logging.getLogger(__name__)

DEFAULT_BIND_ADDR = "tcp://127.0.0.1:5555"
ACK_EVERY_N_FRAMES = 10
ACK_EVERY_SEC = 2.0
IDLE_TIMEOUT_SEC = 600.0
POLL_TIMEOUT_MS = 500
QC_CHECK_SEC = 0.5
# `_resolve_base_name` tries `name`, `name_001`, ... up to this suffix. A
# thousand earlier acquisitions under one name means something is wrong
# with naming, not a genuine collision; reject the session and say so.
_MAX_NAME_SUFFIX = 999

# Per-session staging timings are appended here as JSON lines (see
# `_write_stage_profile`), beside run_petakit_server.m's per-ticket
# profiling/S<id>.jsonl; PETAKIT_JOBS_DIR is the same override both honor.
_JOBS_DIR_ENV_VAR = "PETAKIT_JOBS_DIR"
_DEFAULT_JOBS_DIR = "/dev/shm/petakit_jobs"

# Same environment variable bioimaging.backfill.pipeline.resolve_decon_psf
# reads to decide whether the batch pass decons at all. Read directly
# (rather than importing that function) to keep the one-way dependency
# direction CLAUDE.md documents: opym_local must not import from
# bioimaging. Duplicating a single env-var name is a far smaller coupling
# than that import would be, and getting this wrong costs nothing but a
# missed pre-staging optimization -- build_decon_staging_dir's own
# skip-if-exists check makes it harmless either way (see module docstring).
_DECON_PSF_ENV_VAR = "OPYM_DECON_PSF"

# See module docstring's "RAM-disk staging" section. Read fresh per session
# (not captured once at receiver construction), matching `_decon_enabled`'s
# own per-session-read pattern -- both are toggled per-test via monkeypatch,
# and neither is expected to change mid-process in production.
_STAGE_ROOT_ENV_VAR = "OPYM_STREAM_STAGE_ROOT"


# Holding the live lease (opym.lanes) makes the GPU servers and the backfill
# yield to this acquisition. Off by default: until live tickets exist
# (Phase 2), holding it would only park the backfill with the GPUs idle.
_LIVE_LANE_ENV_VAR = "OPYM_LIVE_LANE"


def _live_lane_enabled() -> bool:
    return os.environ.get(_LIVE_LANE_ENV_VAR, "").strip() in ("1", "true", "yes")


def _decon_enabled() -> bool:
    return bool(os.environ.get(_DECON_PSF_ENV_VAR, "").strip())


def _holds_files(path: Path) -> bool:
    """True if any regular file exists anywhere under `path`. An empty
    directory tree (e.g. a staging leaf whose files the live lane already
    removed) doesn't block reusing its name."""
    return path.is_dir() and any(p.is_file() for p in path.rglob("*"))


def _stage_root_from_env() -> Path | None:
    val = os.environ.get(_STAGE_ROOT_ENV_VAR, "").strip()
    return Path(val) if val else None


def _requested_output_format(header: dict[str, Any], session_id: str) -> str | None:
    """SESSION_START's optional `output_format`. An unrecognized value is
    logged and ignored (falls back to the backfill default) rather than
    rejecting the session -- losing a whole acquisition over a display
    preference would be the wrong trade."""
    fmt = header.get("output_format")
    if fmt is None or fmt in rawmirror.OUTPUT_FORMATS:
        return fmt
    logger.warning(
        "Session %s: ignoring unknown output_format %r (expected one of %s)",
        session_id,
        fmt,
        rawmirror.OUTPUT_FORMATS,
    )
    return None


def _estimate_session_bytes(header: dict[str, Any]) -> int:
    """Rough pre-flight size estimate from SESSION_START's declared shape --
    used only to guard against starting a session RAM-disk staging can't
    possibly fit, not an exact accounting (per-FRAME `shape_zyx` may differ
    slightly per the protocol, see `protocol.py`)."""
    nz, ny, nx = header["shape_zyx"]
    itemsize = np.dtype(header["dtype"]).itemsize
    return nz * ny * nx * itemsize * header["num_timepoints"] * len(header["channels"])


@dataclass
class SessionState:
    session_id: str
    identity: bytes
    base_name: str
    raw_root: Path
    """The client's declared FINAL destination -- unchanged in meaning from
    before RAM-disk staging existed. Only used directly as a write target
    when staging is off; when staging is on, it's the drain worker's
    destination instead (see `dest_leaf_dir`)."""
    write_root: Path
    """Where this session's raw mirror stores and decon-stage TIFFs are
    ACTUALLY written: `raw_root` with staging off, or the staging root with
    it on. See module docstring's "RAM-disk staging" section."""
    dtype: str
    shape_zyx: tuple[int, int, int]
    num_timepoints: int
    channels: list[int]
    channel_names: list[str]
    channel_store_paths: dict[int, Path]
    # Position of each channel's store path in the whole session's
    # lexicographically-sorted store-path list -- this is exactly the
    # `cidx` `opym.discovery.discover_leaf_datasets` will later assign via
    # `sorted(members)`, so decon-stage filenames written here match what
    # `build_decon_staging_dir` would independently produce byte-for-byte
    # (see its skip-if-exists check in `write_decon_staged_tiff`).
    channel_cidx: dict[int, int]
    z_step_um: float
    decon_enabled: bool
    output_format: str | None = None
    """The client's requested final-output format (one of
    `rawmirror.OUTPUT_FORMATS`), recorded on each raw store; None leaves it
    to the backfill's own default."""
    channel_arrays: dict[int, Any] = field(default_factory=dict)
    received_pairs: set[tuple[int, int]] = field(default_factory=set)
    processed_frame_indices: set[int] = field(default_factory=set)
    ack_floor: int = -1
    frames_since_ack: int = 0
    last_ack_time: float = field(default_factory=time.monotonic)
    last_activity: float = field(default_factory=time.monotonic)
    ended: bool = False
    # Seconds spent writing each frame's raw store / staged decon TIFF, on
    # this poll loop -- ingest keeps pace only while their sum per timepoint
    # stays under the acquisition interval.
    live: bool = False
    """Processed timepoint by timepoint by the live lane (opym.stream.live)."""
    accepts_qc: bool = False
    """The client listed "qc" in SESSION_START's `accepts`: forward MSG_QC."""
    qc_seq_sent: int = -1
    qc_mtime: float = 0.0
    qc_checked_at: float = 0.0
    stage_raw_s: list[float] = field(default_factory=list)
    stage_tiff_s: list[float] = field(default_factory=list)
    staged_bytes: int = 0
    started_at: float = field(default_factory=time.time)

    @property
    def leaf_dir(self) -> Path:
        """Where this session actually lives right now: `write_root /
        base_name`. Matches `opym.discovery.LeafDataset.leaf_dir` exactly
        when staging is off (`write_root is raw_root`); with staging on,
        this is the RAM-disk copy, and `dest_leaf_dir` is the GPFS one."""
        return self.write_root / self.base_name

    @property
    def dest_leaf_dir(self) -> Path:
        """The drain worker's destination: `raw_root / base_name`. Equal to
        `leaf_dir` (and unused) when staging is off."""
        return self.raw_root / self.base_name

    @property
    def decon_stage_dir(self) -> Path:
        """Matches `zarr_deskew_data_dir`'s decon branch in
        `bioimaging.backfill.pipeline`."""
        return self.leaf_dir / "decon_stage"


class StreamReceiver:
    """Owns the ROUTER socket and every active session's state.

    One instance is a single long-running process (the `opym-receive`
    entry point), analogous to `local_gpu_worker.py`'s watchdog for the
    batch path -- but this one talks to the network instead of polling a
    queue directory.
    """

    def __init__(
        self,
        bind_addr: str = DEFAULT_BIND_ADDR,
        ack_every_n_frames: int = ACK_EVERY_N_FRAMES,
        ack_every_sec: float = ACK_EVERY_SEC,
        idle_timeout_sec: float = IDLE_TIMEOUT_SEC,
        drain_workers: int = drain.DEFAULT_DRAIN_WORKERS,
        drain_retention_s: float = drain.DEFAULT_RETENTION_S,
        drain_high_water_bytes: int = drain.DEFAULT_HIGH_WATER_BYTES,
    ) -> None:
        self.bind_addr = bind_addr
        self.ack_every_n_frames = ack_every_n_frames
        self.ack_every_sec = ack_every_sec
        self.idle_timeout_sec = idle_timeout_sec
        self.sessions: dict[str, SessionState] = {}
        # Always constructed (cheap: idle threads blocked on a queue read)
        # so staging can be toggled per-session via the env var without
        # needing the receiver process restarted -- mirrors `_decon_enabled`
        # being read fresh per SESSION_START rather than cached at startup.
        stage_root = _stage_root_from_env()
        self._drain_pool = drain.DrainPool(
            num_workers=drain_workers,
            retention_s=drain_retention_s,
            high_water_bytes=drain_high_water_bytes,
            manifest_dir=stage_root / ".drain_manifests" if stage_root else None,
        )
        self._drain_pool.start()
        self._lease = lanes.LeaseKeeper()
        # Created on the first live session (see _maybe_start_live), so the
        # env switches stay readable per session like the others here.
        self._live: LiveLane | None = None

        self._ctx = zmq.Context.instance()
        self._socket = self._ctx.socket(zmq.ROUTER)
        # Required for the RESUME/reconnect design this protocol depends on
        # (see protocol.py's module docstring): by default a ROUTER socket
        # REJECTS a new connection that presents an identity already
        # associated with another (possibly just-dropped) connection,
        # rather than handing the identity over to it -- which silently
        # breaks exactly the "fresh TCP connection, same identity" reconnect
        # this receiver is built around. Confirmed via a real flaky
        # reconnect-drops-RESUME failure without this set.
        self._socket.setsockopt(zmq.ROUTER_HANDOVER, 1)
        self._socket.bind(self.bind_addr)
        self._poller = zmq.Poller()
        self._poller.register(self._socket, zmq.POLLIN)

    def close(self) -> None:
        self._lease.update([])
        if self._live is not None:
            self._live.close()
        self._socket.close(linger=0)
        # Deliberately not draining the queue first: shutdown must not block
        # on GPFS copies. Any not-yet-drained session's staging copy stays
        # on /dev/shm, untouched, and a restarted receiver process re-drains
        # it once it's told to (not automatic today -- see drain.py's
        # `_drain_one` docstring, same v1 scope as the RESUME design).
        self._drain_pool.stop()

    def __enter__(self) -> StreamReceiver:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def run_forever(self) -> None:
        logger.info("Stream receiver listening on %s", self.bind_addr)
        try:
            while True:
                self._run_once()
        except KeyboardInterrupt:
            logger.info("Stream receiver shutting down.")

    def _run_once(self) -> None:
        events = dict(self._poller.poll(timeout=POLL_TIMEOUT_MS))
        if self._socket in events:
            self._handle_incoming()
        self._flush_pending_acks()
        self._sweep_idle_sessions()
        # Held while any session is open; refreshed every few seconds so it
        # goes stale (and frees the GPUs) within a minute if we crash.
        if self._live is not None:
            self._live.pump()
            self._forward_qc()
        # The lease also covers live work still in flight after SESSION_END.
        busy = set(self.sessions) | set(self._live.sessions if self._live else ())
        self._lease.update(busy if _live_lane_enabled() else ())

    def _handle_incoming(self) -> None:
        identity, *rest = self._socket.recv_multipart()
        try:
            msg_type, session_id, header, payload = unpack_message(rest)
        except ValueError as exc:
            logger.warning("Dropping malformed message from %r: %s", identity, exc)
            return

        # The DEALER client is required to set zmq.IDENTITY == session_id
        # (see protocol.py) -- trust the transport-level identity for
        # routing/session lookup and treat a mismatched header session_id
        # as a misbehaving client rather than silently using either one.
        if identity != session_id.encode("utf-8"):
            logger.warning(
                "Dropping message: socket identity %r does not match header "
                "session_id %r",
                identity,
                session_id,
            )
            return

        if msg_type == MSG_SESSION_START:
            self._handle_session_start(identity, session_id, header)
        elif msg_type == MSG_FRAME:
            self._handle_frame(session_id, header, payload)
        elif msg_type == MSG_SESSION_END:
            self._handle_session_end(session_id, header)
        elif msg_type == MSG_RESUME:
            self._handle_resume(identity, session_id)
        else:  # pragma: no cover -- unpack_message already validates this
            logger.warning("Unhandled message type %r", msg_type)

    def _handle_session_start(
        self, identity: bytes, session_id: str, header: dict[str, Any]
    ) -> None:
        # A malformed header here must not take down every other session
        # this process is holding -- log and drop rather than let a
        # KeyError/ValueError propagate out of _run_once.
        try:
            raw_root = Path(header["raw_root"])
            base_name = header["base_name"]
            channels = list(header["channels"])
            channel_names = list(header.get("channel_names") or [])
            if len(channel_names) != len(channels):
                raise ValueError(
                    f"channel_names (got {len(channel_names)}) must have exactly "
                    f"one entry per channel (got {len(channels)}) -- see "
                    "STREAMING_PROTOCOL.md's SESSION_START fields"
                )

            stage_root = _stage_root_from_env()
            write_root = stage_root if stage_root is not None else raw_root
            if stage_root is not None:
                # mkdir first -- disk_usage needs an existing path, and this
                # root is otherwise only created lazily on a channel store's
                # first write (rawmirror.create_channel_store).
                stage_root.mkdir(parents=True, exist_ok=True)
                estimated_bytes = _estimate_session_bytes(header)
                free_bytes = shutil.disk_usage(stage_root).free
                # Require 2x headroom, not just enough to fit exactly -- the
                # staging root is shared with every other concurrently
                # staging session (and PetaKit5D's own /dev/shm usage), and
                # a session that starts right at the edge would starve
                # whichever one grows next. See module docstring.
                if estimated_bytes * 2 > free_bytes:
                    raise ValueError(
                        f"staging root {stage_root} has {free_bytes / 1e9:.1f} GB "
                        f"free, need >= {estimated_bytes * 2 / 1e9:.1f} GB "
                        f"(2x this session's estimated {estimated_bytes / 1e9:.1f} "
                        "GB) -- rejecting rather than risking a mid-session "
                        "tmpfs overflow"
                    )

            base_name = self._resolve_base_name(
                session_id, base_name, channel_names, raw_root, write_root
            )
            channel_store_paths = {
                c: rawmirror.store_path_for_channel(write_root, base_name, name)
                for c, name in zip(channels, channel_names)
            }
            sorted_paths = sorted(channel_store_paths.values())
            channel_cidx = {
                c: sorted_paths.index(p) for c, p in channel_store_paths.items()
            }

            session = SessionState(
                session_id=session_id,
                identity=identity,
                base_name=base_name,
                raw_root=raw_root,
                write_root=write_root,
                dtype=header["dtype"],
                shape_zyx=tuple(header["shape_zyx"]),
                num_timepoints=header["num_timepoints"],
                channels=channels,
                channel_names=channel_names,
                channel_store_paths=channel_store_paths,
                channel_cidx=channel_cidx,
                z_step_um=header["z_step_um"],
                decon_enabled=_decon_enabled(),
                output_format=_requested_output_format(header, session_id),
                accepts_qc="qc" in (header.get("accepts") or ()),
            )
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("Rejecting SESSION_START for %s: %s", session_id, exc)
            return

        self.sessions[session_id] = session
        self._maybe_start_live(session)
        logger.info(
            "Session %s started: base_name=%s grid=%dT x %dC -> %s "
            "(decon_enabled=%s, staging=%s)",
            session_id,
            session.base_name,
            session.num_timepoints,
            len(session.channels),
            session.write_root,
            session.decon_enabled,
            session.write_root != session.raw_root,
        )
        self._send_ack(session)

    def _resolve_base_name(
        self,
        session_id: str,
        requested: str,
        channel_names: list[str],
        raw_root: Path,
        write_root: Path,
    ) -> str:
        """The name this session is written under: `requested`, or
        `requested_001`, `_002`, ... when an earlier acquisition already
        used it (pymmcore's own `_001` convention).

        Two sessions must never share storage. On 2026-09-24 a 100-timepoint
        Cell_001 started three minutes after a 1-timepoint test of the same
        name: `create_channel_store` reopened the test's `[1, ...]` array,
        every frame was rejected, and the drain re-copied the stale test --
        the acquisition was lost. Had the shapes matched (Cell_004 was also
        started twice that day), the second run would have silently
        overwritten the first instead.

        A name is taken if one of its channel stores or its leaf directory
        exists under `raw_root` (an earlier acquisition landed there), or is
        still occupied under the staging root. Staging copies of already-
        drained sessions are released first (`DrainPool.release`), since the
        staging root is flat across every `raw_root` and a retained copy from
        another experiment folder would otherwise force a needless rename.
        Stores tagged with this `session_id` are this session's own -- a
        SESSION_START resent after a receiver restart -- and keep their name.
        """
        existing = self.sessions.get(session_id)
        if existing is not None:
            return existing.base_name
        staging = write_root != raw_root
        for n in range(_MAX_NAME_SUFFIX + 1):
            name = requested if n == 0 else f"{requested}_{n:03d}"
            stage_stores = [
                rawmirror.store_path_for_channel(write_root, name, ch)
                for ch in channel_names
            ]
            if any(rawmirror.read_session_id(p) == session_id for p in stage_stores):
                return name
            dest_stores = [
                rawmirror.store_path_for_channel(raw_root, name, ch)
                for ch in channel_names
            ]
            taken = any(p.exists() for p in dest_stores) or (raw_root / name).exists()
            if staging and not taken:
                stage_leaf = write_root / name
                self._drain_pool.release([*stage_stores, stage_leaf / "decon_stage"])
                taken = any(p.exists() for p in stage_stores) or _holds_files(
                    stage_leaf
                )
            if not taken:
                if n:
                    logger.warning(
                        "Session %s: base_name %r is already used by an earlier "
                        "acquisition under %s -- writing this one as %r instead",
                        session_id,
                        requested,
                        raw_root,
                        name,
                    )
                return name
        raise ValueError(
            f"base_name {requested!r} and all of its _001.._{_MAX_NAME_SUFFIX:03d} "
            f"variants are already used under {raw_root}"
        )

    def _maybe_start_live(self, session: SessionState) -> None:
        """Hand a session to the live lane when it's enabled (OPYM_LIVE_LANE=1)
        and decon is on: its per-timepoint input is the decon_stage/ TIFFs
        this receiver already writes.

        Single-timepoint sessions go through it too, not just time-lapses:
        a quick alignment/test snap benefits from showing up in naparym-live
        immediately just as much as a real acquisition does, and the live
        lane's own dispatch/finalize logic has no special-casing for T>1 to
        begin with -- this guard was the only place that assumed one.
        """
        if not (_live_lane_enabled() and session.decon_enabled):
            return
        psf = resolve_decon_psf()
        if psf is None:
            return
        if self._live is None:
            self._live = LiveLane(psf)
        self._live.start_session(
            session.session_id,
            base_name=session.base_name,
            num_timepoints=session.num_timepoints,
            n_channels=len(session.channels),
            frames_dir=session.decon_stage_dir,
            stage_leaf=session.leaf_dir,
            dest_leaf=session.dest_leaf_dir,
            z_step_um=session.z_step_um,
            channel_labels=[
                live.channel_label(name)
                for _, name in sorted(
                    zip(session.channels, session.channel_names),
                    key=lambda cn: session.channel_cidx[cn[0]],
                )
            ],
        )
        session.live = True

    def _handle_frame(
        self, session_id: str, header: dict[str, Any], payload: bytes | None
    ) -> None:
        session = self.sessions.get(session_id)
        if session is None:
            logger.warning(
                "FRAME for unknown session %s (no SESSION_START seen) -- dropped",
                session_id,
            )
            return
        if payload is None:
            logger.warning(
                "FRAME with no payload for session %s -- dropped", session_id
            )
            return

        session.last_activity = time.monotonic()
        t, c, frame_index = header["t"], header["c"], header["frame_index"]

        if frame_index not in session.processed_frame_indices:
            if (t, c) not in session.received_pairs:
                try:
                    self._stage_frame(session, header, payload)
                except Exception:
                    logger.exception(
                        "Failed to stage (t=%d, c=%d) for session %s -- "
                        "dropping this frame; the client's retry buffer will "
                        "resend it on the next ACK-driven resume.",
                        t,
                        c,
                        session_id,
                    )
                    return
                session.received_pairs.add((t, c))
            else:
                logger.debug(
                    "Duplicate frame (t=%d, c=%d) for session %s already staged "
                    "-- skipping restage, still acking",
                    t,
                    c,
                    session_id,
                )
            session.processed_frame_indices.add(frame_index)
            self._advance_ack_floor(session)
            session.frames_since_ack += 1

        if (
            session.frames_since_ack >= self.ack_every_n_frames
            or time.monotonic() - session.last_ack_time >= self.ack_every_sec
        ):
            self._send_ack(session)

    def _stage_frame(
        self, session: SessionState, header: dict[str, Any], payload: bytes
    ) -> None:
        t, c = header["t"], header["c"]
        shape_zyx = tuple(header.get("shape_zyx") or session.shape_zyx)
        dtype = header.get("dtype") or session.dtype

        raw = np.frombuffer(payload, dtype=np.dtype(dtype)).reshape(shape_zyx)
        t_start = time.perf_counter()

        arr = session.channel_arrays.get(c)
        if arr is None:
            # Lazily created on this channel's first frame, using THIS
            # frame's own shape/dtype -- shape_zyx can legitimately differ
            # per channel (e.g. two cameras with different crops), so a
            # single session-wide array shape can't be assumed up front.
            arr = rawmirror.create_channel_store(
                session.channel_store_paths[c],
                num_timepoints=session.num_timepoints,
                shape_zyx=shape_zyx,
                dtype=dtype,
                z_step_um=session.z_step_um,
                output_format=session.output_format,
                session_id=session.session_id,
            )
            session.channel_arrays[c] = arr
        rawmirror.write_timepoint(arr, t, raw)
        t_raw = time.perf_counter()

        if session.decon_enabled:
            dst = self._decon_stage_path(session, c, t)
            dst.parent.mkdir(parents=True, exist_ok=True)
            write_decon_staged_tiff(raw, dst)
            if session.live:
                self._live.frame_staged(
                    session.session_id, t, session.channel_cidx[c], raw=raw
                )
        session.stage_raw_s.append(t_raw - t_start)
        session.stage_tiff_s.append(time.perf_counter() - t_raw)
        session.staged_bytes += raw.nbytes

        logger.debug(
            "Staged T=%d C=%d for session %s -> %s",
            t,
            c,
            session.session_id,
            session.channel_store_paths[c],
        )

    def _decon_stage_path(self, session: SessionState, c: int, t: int) -> Path:
        """Matches `build_decon_staging_dir`'s own naming exactly, so its
        skip-if-exists check treats this file as already done -- for a
        session the live lane never touches. A *live* session's staged
        frames are read back by their live ticket, which names them
        `_C{cidx}_T{ttt}.tif` unconditionally (`LiveSession.frame`,
        `opym.stream.live`) -- there is no single-timepoint special case on
        that side, and there cannot be one here either, or the live lane's
        `imfinfo` on the erosion-mask frame fails outright (confirmed on
        Argus 2026-09-25 once single-timepoint sessions started going
        live: `num_timepoints > 1` alone used to be equivalent to "am I
        live", back when the live lane skipped every T<=1 session; once it
        stopped skipping them, the two diverged).
        """
        if session.num_timepoints > 1 or session.live:
            cidx = session.channel_cidx[c]
            return session.decon_stage_dir / f"{session.base_name}_C{cidx}_T{t:03d}.tif"
        store_name = session.channel_store_paths[c].name.removesuffix(".zarr")
        return session.decon_stage_dir / f"{store_name}.tif"

    def _advance_ack_floor(self, session: SessionState) -> None:
        processed = session.processed_frame_indices
        while (session.ack_floor + 1) in processed:
            processed.discard(session.ack_floor + 1)
            session.ack_floor += 1

    def _send_ack(self, session: SessionState) -> None:
        header = {"through_frame_index": session.ack_floor}
        self._socket.send_multipart(
            [session.identity, *pack_message(MSG_ACK, session.session_id, header)]
        )
        session.frames_since_ack = 0
        session.last_ack_time = time.monotonic()

    def _forward_qc(self) -> None:
        """Send each open session's newest QC verdict to its client, once."""
        now = time.monotonic()
        for session in self.sessions.values():
            if not (session.accepts_qc and session.live) or session.ended:
                continue
            if now - session.qc_checked_at < QC_CHECK_SEC:
                continue
            session.qc_checked_at = now
            path = self._live.qc_latest_path(session.session_id)
            try:
                mtime = path.stat().st_mtime
                if mtime == session.qc_mtime:
                    continue
                record = json.loads(path.read_text())
            except (AttributeError, OSError, ValueError):
                continue
            session.qc_mtime = mtime
            seq = record.get("seq", -1)
            if record.get("session_id") != session.session_id:
                continue
            if not isinstance(seq, int) or seq <= session.qc_seq_sent:
                continue
            self._socket.send_multipart(
                [session.identity, *pack_message(MSG_QC, session.session_id, record)]
            )
            session.qc_seq_sent = seq

    def _handle_session_end(self, session_id: str, header: dict[str, Any]) -> None:
        session = self.sessions.get(session_id)
        if session is None:
            logger.warning("SESSION_END for unknown session %s -- ignored", session_id)
            return
        self._finalize_session(session, reason=header.get("reason", "complete"))

    def _finalize_session(self, session: SessionState, reason: str) -> None:
        if session.ended:
            return
        session.ended = True
        logger.info(
            "Session %s ended (%s): %d/%d (t,c) pairs received for %s under %s",
            session.session_id,
            reason,
            len(session.received_pairs),
            session.num_timepoints * len(session.channels),
            session.base_name,
            session.write_root,
        )
        _write_stage_profile(session, reason)
        if session.live:
            self._live.end_session(session.session_id)
        # Bug fix: a client's `wait_for_all_acked()` (its durability signal --
        # see protocol.py's ACK docs) previously had no way to learn about
        # this session's LAST batch of frames if SESSION_END arrived before
        # the next periodic ACK was due (ack_every_n_frames / ack_every_sec).
        # Confirmed on the real rig: the client would time out waiting for an
        # ACK that was simply never going to come, even though every frame
        # had already been durably staged. Send one final ACK at the true
        # ack_floor before the session state (and the identity needed to
        # reach the client) is gone.
        self._send_ack(session)

        if session.write_root != session.raw_root:
            # Staging was active for this session -- hand its completed
            # artifacts to the background drain pool. A session is NOT one
            # contiguous directory: each channel's store is a top-level
            # sibling under write_root, and decon_stage/ (if present) is
            # separately nested under write_root/base_name -- see
            # `opym.stream.drain.DrainJob`'s docstring. Only items that
            # actually got written are included (a channel declared in
            # SESSION_START but never sent, or decon_stage when zero frames
            # arrived, simply won't exist).
            items = [
                (
                    session.channel_store_paths[c],
                    rawmirror.store_path_for_channel(
                        session.raw_root, session.base_name, name
                    ),
                )
                for c, name in zip(session.channels, session.channel_names)
            ]
            # A live session's staged TIFFs are consumed (and deleted) by the
            # live lane, which writes its DSR output into the GPFS
            # decon_stage/ itself; draining them would copy a derivable
            # intermediate and, worse, rmtree+replace that live output.
            if session.decon_enabled and not session.live:
                items.append(
                    (session.decon_stage_dir, session.dest_leaf_dir / "decon_stage")
                )
            items = [(s, d) for s, d in items if s.exists()]
            # Never drain a store another session wrote. `_resolve_base_name`
            # already keeps sessions apart; this is the backstop for the
            # 2026-09-24 failure, where a session that staged nothing still
            # re-drained an earlier session's same-named store over GPFS.
            foreign = [
                s
                for s, _ in items
                if s.name.endswith(".ome.zarr")
                and rawmirror.read_session_id(s) != session.session_id
            ]
            for s in foreign:
                logger.error(
                    "Session %s: not draining %s -- it was written by session %s",
                    session.session_id,
                    s,
                    rawmirror.read_session_id(s),
                )
            items = [(s, d) for s, d in items if s not in foreign]

            if items:
                # Enqueue is a cheap, non-blocking `queue.put`; the actual
                # GPFS copy happens on a drain worker thread, off this poll
                # loop, so a slow/loaded GPFS write can never delay ingest
                # for the NEXT session.
                self._drain_pool.enqueue(
                    drain.DrainJob(session_id=session.session_id, items=items)
                )
                logger.info(
                    "Session %s: queued %d item(s) for drain to %s",
                    session.session_id,
                    len(items),
                    session.raw_root,
                )
        # No consolidation step: see module docstring. With staging off,
        # `opym-backfill --watch` (already running independently) discovers
        # this dataset directly under raw_root on its own next pass -- with
        # staging on, it discovers it once the drain above lands it there --
        # either way nothing else needs triggering here.
        del self.sessions[session.session_id]

    def _handle_resume(self, identity: bytes, session_id: str) -> None:
        session = self.sessions.get(session_id)
        if session is None:
            # Receiver process restarted, or this session never actually
            # reached SESSION_START here -- reply -1 so the client resends
            # everything still in its local buffer. Acceptable v1
            # degradation: session state isn't persisted to disk, only kept
            # in-process (see docs/STREAMING_PROTOCOL.md).
            self._socket.send_multipart(
                [
                    identity,
                    *pack_message(MSG_ACK, session_id, {"through_frame_index": -1}),
                ]
            )
            return
        session.identity = (
            identity  # reconnect: new TCP connection, same identity value
        )
        self._send_ack(session)

    def _flush_pending_acks(self) -> None:
        """Time-based ack flush for sessions that went quiet mid-burst --
        `_handle_frame`'s inline check only fires on the next incoming
        frame, so a session that stops sending needs this to still get its
        final ack promptly instead of waiting for `idle_timeout_sec`.
        """
        now = time.monotonic()
        for session in self.sessions.values():
            if (
                session.frames_since_ack > 0
                and now - session.last_ack_time >= self.ack_every_sec
            ):
                self._send_ack(session)

    def _sweep_idle_sessions(self) -> None:
        now = time.monotonic()
        for session in list(self.sessions.values()):
            if now - session.last_activity >= self.idle_timeout_sec:
                self._finalize_session(session, reason="idle_timeout")


def _stage_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p95": None, "max": None}
    arr = np.asarray(values)
    return {
        "p50": round(float(np.percentile(arr, 50)), 4),
        "p95": round(float(np.percentile(arr, 95)), 4),
        "max": round(float(arr.max()), 4),
    }


def _write_stage_profile(session: SessionState, reason: str) -> None:
    """Log and append one JSON line of this session's staging timings to
    `<jobs dir>/profiling/receiver.jsonl`. Best effort: never raises."""
    try:
        record = {
            "session_id": session.session_id,
            "base_name": session.base_name,
            "reason": reason,
            "started_at": session.started_at,
            "ended_at": time.time(),
            "frames": len(session.stage_raw_s),
            "staged_gb": round(session.staged_bytes / 1e9, 3),
            "decon_staging": session.decon_enabled,
            "raw_write_s": _stage_stats(session.stage_raw_s),
            "tiff_write_s": _stage_stats(session.stage_tiff_s),
        }
        logger.info(
            "Session %s staging: %d frame(s), %.2f GB, raw write p95 %s s, "
            "decon TIFF write p95 %s s",
            session.session_id,
            record["frames"],
            record["staged_gb"],
            record["raw_write_s"]["p95"],
            record["tiff_write_s"]["p95"],
        )
        jobs_dir = Path(os.environ.get(_JOBS_DIR_ENV_VAR) or _DEFAULT_JOBS_DIR)
        prof_dir = jobs_dir / "profiling"
        prof_dir.mkdir(parents=True, exist_ok=True)
        with open(prof_dir / "receiver.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:  # noqa: BLE001 - profiling must never break ingest
        logger.debug("Could not write staging profile", exc_info=True)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    receiver = StreamReceiver()
    with receiver:
        receiver.run_forever()


if __name__ == "__main__":
    main()
