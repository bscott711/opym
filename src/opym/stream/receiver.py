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

Direct endpoint (opt-in, `OPYM_STREAM_DIRECT_BIND`): besides the loopback
endpoint the acquisition PC reaches through its SSH tunnel (~33 MB/s, the
biggest single delay in the live view on 2026-09-25), the receiver can listen
on the 10 GbE interface itself. That socket is plain TCP, not encrypted:
CurveZMQ measured 85 MB/s with pyzmq's bundled libzmq. It is locked down
instead. A ZAP IP allowlist (`OPYM_STREAM_ALLOW_IPS`) refuses every other
host, and sessions arriving on it may only write under
`OPYM_STREAM_RAW_ROOTS`. The site firewall admits only the acquisition PC
on top of that. The receiver refuses to bind it with either list empty.
Each session remembers the socket it is reached on, and follows its client
if it reconnects through the other one.

Slabs: a client that sees "slabs" in an ACK's `features` sends each volume
as z-slabs while it is still being acquired (FRAME headers with `z0`,
`nz`), instead of one message after the last plane. Each slab goes straight
into the raw store. The volume is staged for processing (decon TIFF, live
lane, QC) once its last plane lands, and its slabs are ACKed only then, so
a restart mid-volume makes the client resend the whole volume.

Compression: a client that sees "blosc" in `features` may send any FRAME's
payload as one blosc frame (header `codec: "blosc"`); it's decoded here
before anything else looks at it. Camera frames compress ~2.5x with lz4 +
bitshuffle, which is ~2.5x the throughput of a link that is the bottleneck
(the SSH tunnel: ~33 MB/s).

Links: a client that sees "links" may stream one session over several
connections at once (identities `<session_id>#<k>`, see protocol.py): one
SSH tunnel is capped at ~2 MB per round trip, so N tunnels move ~N times
as much. Nothing here depends on which link a message came in on.

Recovery ("resume"): a FRAME or RESUME for a session this process doesn't
know (it restarted, or idle-timed the session out) is answered with
`unknown_session`, and the client re-sends SESSION_START with
`resume_through`. The session then continues in its old stores -- on the
staging root if its copy is still there, else straight into raw_root -- and
goes to the batch pipeline rather than the live lane.

Paused runs: SESSION_END reason "paused" means the client gave up
streaming mid-run (it couldn't reach Argus for longer than its RAM buffer
holds). The partial copy is kept on staging for the usual retention but
never copied to raw_root, so the run's full local save can be sent there by
Globus instead.
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
from numcodecs import blosc as _blosc

from opym import lanes
from opym.decon_config import resolve_decon_psf
from opym.stream import drain, live, live_zarr, rawmirror, trace
from opym.stream.live import LiveLane
from opym.stream.live_zarr import ZarrLiveLane
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
POLL_TIMEOUT_MS = 100
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


# The direct (10 GbE) endpoint; see the module docstring.
_DIRECT_BIND_ENV_VAR = "OPYM_STREAM_DIRECT_BIND"
_ALLOW_IPS_ENV_VAR = "OPYM_STREAM_ALLOW_IPS"
_RAW_ROOTS_ENV_VAR = "OPYM_STREAM_RAW_ROOTS"
# Setting a ZAP domain is what makes libzmq consult the authenticator (here:
# the IP allowlist) for NULL-mechanism connections at all.
_ZAP_DOMAIN = b"opym-direct"
# Advertised in every ACK; clients only use a feature once they've seen it.
SERVER_FEATURES = ["slabs", "blosc", "links", "resume"]
# Unknown-session replies to FRAMEs are rate-limited per session: a client
# resending hundreds of MB of slabs would otherwise get one per slab.
_UNKNOWN_NOTICE_EVERY_S = 2.0


def _env_list(name: str) -> list[str]:
    return [v.strip() for v in os.environ.get(name, "").split(",") if v.strip()]


def _identity_belongs(identity: bytes, session_id: str) -> bool:
    """A link identity is the session_id (link 0) or `<session_id>#<k>`."""
    sid = session_id.encode("utf-8")
    if identity == sid:
        return True
    prefix, sep, k = identity.rpartition(b"#")
    return bool(sep) and prefix == sid and k.isdigit()


def _link_number(identity: bytes) -> int:
    """0 for the session_id identity, k for `<session_id>#<k>`."""
    _prefix, sep, k = identity.rpartition(b"#")
    return int(k) if sep and k.isdigit() else 0


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
class _SlabAssembly:
    """One (t, c) volume arriving as z-slabs (FRAME headers with `z0`)."""

    volume: np.ndarray
    z0s: set[int] = field(default_factory=set)
    planes: int = 0
    frame_indices: list[int] = field(default_factory=list)
    first_recv_s: float = 0.0
    raw_write_s: float = 0.0
    wire_bytes: int = 0
    links: set[int] = field(default_factory=set)
    """Which links (0 = the session_id identity) delivered its slabs."""
    dups: int = 0
    """Slabs that arrived again (resent) before the volume completed."""


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
    live_zarr: bool = False
    """...by the one-format lane (opym.stream.live_zarr): its GPU tickets read
    the raw store directly, so no decon TIFF is staged."""
    t_interval_s: float = 0.0
    accepts_qc: bool = False
    """The client listed "qc" in SESSION_START's `accepts`: forward MSG_QC."""
    sock: Any = None
    """The ROUTER socket this session's client is reached on right now
    (loopback behind the SSH tunnel, or the direct endpoint). ACKs and QC
    go back through it; it follows the client if it reconnects through the
    other one."""
    slabs: dict[tuple[int, int], _SlabAssembly] = field(default_factory=dict)
    """Volumes still arriving slab by slab, by (t, c)."""
    resumed: bool = False
    """Re-opened by a SESSION_START with `resume_through` (see the module
    docstring's "Recovery"); never live."""
    qc_seq_sent: int = -1
    qc_mtime: float = 0.0
    qc_checked_at: float = 0.0
    dup_slabs: int = 0
    """Slabs resent after their volume was already complete."""
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
        direct_bind: str | None = None,
        allow_ips: list[str] | tuple[str, ...] = (),
        raw_roots: list[Path | str] | tuple[Path | str, ...] = (),
    ) -> None:
        self.bind_addr = bind_addr
        self.direct_bind = direct_bind
        self.raw_roots = [Path(r).resolve() for r in raw_roots]
        self.ack_every_n_frames = ack_every_n_frames
        self.ack_every_sec = ack_every_sec
        self.idle_timeout_sec = idle_timeout_sec
        self.sessions: dict[str, SessionState] = {}
        self._unknown_noticed: dict[str, float] = {}
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

        self._direct_socket: zmq.Socket | None = None
        self._direct_ctx: zmq.Context | None = None
        self._auth = None
        if direct_bind:
            self._bind_direct(direct_bind, list(allow_ips))

    def _bind_direct(self, addr: str, allow_ips: list[str]) -> None:
        """Listen on the direct (10 GbE) endpoint; see the module docstring.
        Its own context, so its ZAP authenticator can't collide with another
        receiver's in the same process (tests)."""
        if not allow_ips or not self.raw_roots:
            logger.error(
                "Not binding the direct endpoint %s: it needs both %s and %s "
                "set. Unrestricted, it would accept any host on the network "
                "and let it write anywhere this service can.",
                addr,
                _ALLOW_IPS_ENV_VAR,
                _RAW_ROOTS_ENV_VAR,
            )
            return
        from zmq.auth.thread import ThreadAuthenticator

        self._direct_ctx = zmq.Context()
        self._auth = ThreadAuthenticator(self._direct_ctx)
        self._auth.start()
        self._auth.allow(*allow_ips)
        sock = self._direct_ctx.socket(zmq.ROUTER)
        sock.setsockopt(zmq.ROUTER_HANDOVER, 1)
        sock.zap_domain = _ZAP_DOMAIN
        sock.bind(addr)
        self._direct_socket = sock
        self._poller.register(sock, zmq.POLLIN)
        logger.info(
            "Direct endpoint %s: allowing %s, writes under %s",
            addr,
            ", ".join(allow_ips),
            ", ".join(map(str, self.raw_roots)),
        )

    def _raw_root_allowed(self, raw_root: Path) -> bool:
        resolved = Path(raw_root).resolve()
        return any(resolved.is_relative_to(root) for root in self.raw_roots)

    def close(self) -> None:
        self._lease.update([])
        if self._live is not None:
            self._live.close()
        self._socket.close(linger=0)
        if self._direct_socket is not None:
            self._direct_socket.close(linger=0)
        if self._auth is not None:
            self._auth.stop()
        if self._direct_ctx is not None:
            self._direct_ctx.term()
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
        logger.info(
            "Stream receiver listening on %s%s",
            self.bind_addr,
            f" and {self.direct_bind} (direct)" if self._direct_socket else "",
        )
        try:
            while True:
                self._run_once()
        except KeyboardInterrupt:
            logger.info("Stream receiver shutting down.")

    def _run_once(self) -> None:
        events = dict(self._poller.poll(timeout=POLL_TIMEOUT_MS))
        for sock in (self._socket, self._direct_socket):
            if sock is not None and sock in events:
                self._handle_incoming(sock)
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

    def _handle_incoming(self, sock: zmq.Socket) -> None:
        identity, *rest = sock.recv_multipart()
        try:
            msg_type, session_id, header, payload = unpack_message(rest)
        except ValueError as exc:
            logger.warning("Dropping malformed message from %r: %s", identity, exc)
            return

        # The DEALER client is required to set zmq.IDENTITY == session_id,
        # or "<session_id>#<k>" for its extra links (see protocol.py) --
        # treat a mismatched header session_id as a misbehaving client
        # rather than silently using either one.
        if not _identity_belongs(identity, session_id):
            logger.warning(
                "Dropping message: socket identity %r does not match header "
                "session_id %r",
                identity,
                session_id,
            )
            return

        if msg_type == MSG_SESSION_START:
            self._handle_session_start(sock, identity, session_id, header)
            return
        session = self.sessions.get(session_id)
        if session is not None:
            # Answer on the link (and socket) heard from last: it just
            # proved it works, and a link or route that died stops being
            # used as soon as any other one delivers something.
            session.sock, session.identity = sock, identity
        elif msg_type == MSG_FRAME:
            self._notice_unknown(sock, identity, session_id)
            return
        elif msg_type == MSG_SESSION_END:
            # Already closed (the client missed the final ACK and resent
            # SESSION_END): confirm it again.
            self._send_unknown(sock, identity, session_id)
            return
        if msg_type == MSG_FRAME:
            self._handle_frame(session_id, header, payload)
        elif msg_type == MSG_SESSION_END:
            self._handle_session_end(session_id, header)
        elif msg_type == MSG_RESUME:
            self._handle_resume(sock, identity, session_id)
        else:  # pragma: no cover -- unpack_message already validates this
            logger.warning("Unhandled message type %r", msg_type)

    def _handle_session_start(
        self,
        sock: zmq.Socket,
        identity: bytes,
        session_id: str,
        header: dict[str, Any],
    ) -> None:
        existing = self.sessions.get(session_id)
        if existing is not None and not existing.ended:
            # A resend (the client re-synced after a link hiccup): the
            # session is intact here, so just re-ACK where it stands.
            existing.sock, existing.identity = sock, identity
            logger.info(
                "Session %s: SESSION_START for an open session -- re-ACKing", session_id
            )
            self._send_ack(existing)
            return
        # A malformed header here must not take down every other session
        # this process is holding -- log and drop rather than let a
        # KeyError/ValueError propagate out of _run_once.
        try:
            raw_root = Path(header["raw_root"])
            if sock is self._direct_socket and not self._raw_root_allowed(raw_root):
                raise ValueError(
                    f"raw_root {raw_root} is outside {_RAW_ROOTS_ENV_VAR} "
                    f"({', '.join(map(str, self.raw_roots))}), which is all "
                    "the direct endpoint may write to"
                )
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
            resume_through = header.get("resume_through")
            if resume_through is not None:
                resume_through = int(resume_through)
                write_root = self._resume_write_root(
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
                sock=sock,
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
                t_interval_s=float(header.get("t_interval_s") or 0.0),
                ack_floor=-1 if resume_through is None else resume_through,
                resumed=resume_through is not None,
            )
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("Rejecting SESSION_START for %s: %s", session_id, exc)
            return

        self.sessions[session_id] = session
        self._unknown_noticed.pop(session_id, None)
        if session.resumed:
            logger.warning(
                "Session %s resumed after frame %d (this receiver restarted or "
                "idle-timed it out): continuing in %s; the batch pipeline, not "
                "the live lane, processes it",
                session_id,
                session.ack_floor,
                session.leaf_dir,
            )
        else:
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
        Stores tagged with this `session_id` -- staged or already drained --
        are this session's own (a SESSION_START resent after a receiver
        restart or idle timeout), and keep their name.
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
            dest_stores = [
                rawmirror.store_path_for_channel(raw_root, name, ch)
                for ch in channel_names
            ]
            if any(
                rawmirror.read_session_id(p) == session_id
                for p in (*stage_stores, *dest_stores)
            ):
                return name
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

    def _resume_write_root(
        self,
        session_id: str,
        base_name: str,
        channel_names: list[str],
        raw_root: Path,
        write_root: Path,
    ) -> Path:
        """Where a resumed session keeps writing: into its staging copy if
        that still exists (taking it back out of drain retention so it
        isn't evicted mid-write; its own drain later replaces the GPFS copy
        with it, a superset), else straight into its drained GPFS stores. A
        fresh staging copy would hold only the frames after the resume, and
        draining it would replace the complete GPFS copy with it."""
        if write_root == raw_root:
            return write_root
        stage_stores = [
            rawmirror.store_path_for_channel(write_root, base_name, ch)
            for ch in channel_names
        ]
        self._drain_pool.reclaim(
            [*stage_stores, write_root / base_name / "decon_stage"]
        )
        if any(rawmirror.read_session_id(p) == session_id for p in stage_stores):
            return write_root
        dest_stores = [
            rawmirror.store_path_for_channel(raw_root, base_name, ch)
            for ch in channel_names
        ]
        if any(rawmirror.read_session_id(p) == session_id for p in dest_stores):
            return raw_root
        return write_root

    def _notice_unknown(
        self, sock: zmq.Socket, identity: bytes, session_id: str
    ) -> None:
        """Tell a client its session is unknown here (at most every
        `_UNKNOWN_NOTICE_EVERY_S`), so it can resume it."""
        now = time.monotonic()
        if now - self._unknown_noticed.get(session_id, -1e9) < _UNKNOWN_NOTICE_EVERY_S:
            return
        self._unknown_noticed[session_id] = now
        logger.warning(
            "FRAME for unknown session %s -- asking the client to resume it",
            session_id,
        )
        self._send_unknown(sock, identity, session_id)

    @staticmethod
    def _send_unknown(sock: zmq.Socket, identity: bytes, session_id: str) -> None:
        header = {
            "through_frame_index": -1,
            "unknown_session": True,
            "server_time_s": time.time(),
            "features": SERVER_FEATURES,
        }
        sock.send_multipart([identity, *pack_message(MSG_ACK, session_id, header)])

    def _maybe_start_live(self, session: SessionState) -> None:
        """Hand a session to the live lane when it's enabled (OPYM_LIVE_LANE=1)
        and decon is on.

        The lane is created on the first live session, as the one-format
        lane (opym.stream.live_zarr) when OPYM_LIVE_FORMAT=zarr, else the
        TIFF lane (opym.stream.live), whose per-timepoint input is the
        decon_stage/ TIFFs this receiver writes.

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
            if live_zarr.live_format() == "zarr":
                self._live = ZarrLiveLane(psf)
            else:
                self._live = LiveLane(psf)
        by_cidx = sorted(session.channels, key=lambda c: session.channel_cidx[c])
        labels = [
            live.channel_label(session.channel_names[session.channels.index(c)])
            for c in by_cidx
        ]
        common = {
            "base_name": session.base_name,
            "num_timepoints": session.num_timepoints,
            "n_channels": len(session.channels),
            "stage_leaf": session.leaf_dir,
            "dest_leaf": session.dest_leaf_dir,
            "z_step_um": session.z_step_um,
            "channel_labels": labels,
        }
        if isinstance(self._live, ZarrLiveLane):
            self._live.start_session(
                session.session_id,
                raw_arrays=[session.channel_store_paths[c] / "p0" for c in by_cidx],
                raw_shape_zyx=session.shape_zyx,
                time_interval_s=session.t_interval_s,
                **common,
            )
            session.live_zarr = True
        else:
            self._live.start_session(
                session.session_id, frames_dir=session.decon_stage_dir, **common
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

        recv_s = time.time()
        session.last_activity = time.monotonic()
        t, c, frame_index = header["t"], header["c"], header["frame_index"]
        wire_bytes = len(payload)
        if header.get("codec") == "blosc":
            try:
                payload = _blosc.decompress(payload)
            except Exception:
                logger.exception(
                    "Undecodable blosc FRAME (t=%d, c=%d) for session %s -- "
                    "dropped; the client resends unACKed frames on resume.",
                    t,
                    c,
                    session_id,
                )
                return
        elif header.get("codec") not in (None, "raw"):
            logger.warning(
                "FRAME with unknown codec %r for session %s -- dropped",
                header.get("codec"),
                session_id,
            )
            return

        if "z0" in header:
            self._handle_slab(session, header, payload, recv_s, wire_bytes)
        elif frame_index not in session.processed_frame_indices:
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
                _trace_frame(
                    session, header, len(payload), recv_s, wire_bytes=wire_bytes
                )
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

    def _handle_slab(
        self,
        session: SessionState,
        header: dict[str, Any],
        payload: bytes,
        recv_s: float,
        wire_bytes: int | None = None,
    ) -> None:
        """One z-slab `[z0, z0 + n)` of volume (t, c): written into the raw
        store now, staged for processing once the volume's last plane lands.
        Its frame_index is ACKed only then (see the module docstring)."""
        t, c, frame_index, z0 = (
            header["t"],
            header["c"],
            header["frame_index"],
            header["z0"],
        )
        session.frames_since_ack += 1  # keeps ACKs flowing mid-volume
        if (
            frame_index <= session.ack_floor
            or frame_index in session.processed_frame_indices
        ):
            return
        if (t, c) in session.received_pairs:
            # The whole volume is already staged (a resend after a reconnect).
            session.dup_slabs += 1
            session.processed_frame_indices.add(frame_index)
            self._advance_ack_floor(session)
            return
        dtype = np.dtype(header.get("dtype") or session.dtype)
        slab = np.frombuffer(payload, dtype=dtype).reshape(header["shape_zyx"])
        asm = session.slabs.get((t, c))
        if asm is None:
            nz = int(header["nz"])
            asm = _SlabAssembly(
                volume=np.empty((nz, *slab.shape[1:]), dtype=dtype),
                first_recv_s=recv_s,
            )
            session.slabs[(t, c)] = asm
        asm.frame_indices.append(frame_index)
        asm.links.add(_link_number(session.identity))
        if z0 in asm.z0s:
            asm.dups += 1
            return  # a resend of a slab already written
        asm.wire_bytes += len(payload) if wire_bytes is None else wire_bytes
        try:
            t_start = time.perf_counter()
            arr = self._raw_array(session, c, asm.volume.shape, dtype.str)
            rawmirror.write_planes(arr, t, z0, slab)
            asm.raw_write_s += time.perf_counter() - t_start
        except Exception:
            logger.exception(
                "Failed to write slab z0=%d of (t=%d, c=%d) for session %s -- "
                "dropping it; the client resends unACKed slabs on resume.",
                z0,
                t,
                c,
                session.session_id,
            )
            asm.frame_indices.remove(frame_index)
            return
        asm.volume[z0 : z0 + slab.shape[0]] = slab
        asm.z0s.add(z0)
        asm.planes += slab.shape[0]
        if asm.planes < asm.volume.shape[0]:
            return

        del session.slabs[(t, c)]
        try:
            self._stage_volume(session, t, c, asm.volume, raw_write_s=asm.raw_write_s)
        except Exception:
            logger.exception(
                "Failed to stage (t=%d, c=%d) for session %s after its last "
                "slab -- the client resends it on the next resume.",
                t,
                c,
                session.session_id,
            )
            return
        session.received_pairs.add((t, c))
        session.processed_frame_indices.update(asm.frame_indices)
        self._advance_ack_floor(session)
        _trace_frame(
            session,
            header,
            asm.volume.nbytes,
            recv_s,
            first_recv_s=asm.first_recv_s,
            slabs=len(asm.z0s),
            wire_bytes=asm.wire_bytes,
            links=sorted(asm.links),
            dup_slabs=asm.dups,
        )

    def _raw_array(
        self, session: SessionState, c: int, shape_zyx: tuple[int, ...], dtype: str
    ):
        """Channel c's raw store array, created on its first frame with THAT
        frame's volume shape -- shape_zyx can legitimately differ per channel
        (e.g. two cameras with different crops), so a single session-wide
        array shape can't be assumed up front."""
        arr = session.channel_arrays.get(c)
        if arr is None:
            arr = rawmirror.create_channel_store(
                session.channel_store_paths[c],
                num_timepoints=session.num_timepoints,
                shape_zyx=tuple(shape_zyx),
                dtype=dtype,
                z_step_um=session.z_step_um,
                output_format=session.output_format,
                session_id=session.session_id,
            )
            session.channel_arrays[c] = arr
        return arr

    def _stage_frame(
        self, session: SessionState, header: dict[str, Any], payload: bytes
    ) -> None:
        """A whole (t, c) volume in one FRAME: raw store, then processing."""
        t, c = header["t"], header["c"]
        shape_zyx = tuple(header.get("shape_zyx") or session.shape_zyx)
        dtype = header.get("dtype") or session.dtype
        raw = np.frombuffer(payload, dtype=np.dtype(dtype)).reshape(shape_zyx)
        t_start = time.perf_counter()
        arr = self._raw_array(session, c, shape_zyx, dtype)
        rawmirror.write_timepoint(arr, t, raw)
        self._stage_volume(
            session, t, c, raw, raw_write_s=time.perf_counter() - t_start
        )

    def _stage_volume(
        self,
        session: SessionState,
        t: int,
        c: int,
        raw: np.ndarray,
        *,
        raw_write_s: float,
    ) -> None:
        """Volume (t, c) is complete in the raw store: stage it for decon and
        hand it to the live lane."""
        t_start = time.perf_counter()
        if session.decon_enabled:
            if not session.live_zarr:
                dst = self._decon_stage_path(session, c, t)
                dst.parent.mkdir(parents=True, exist_ok=True)
                write_decon_staged_tiff(raw, dst)
            if session.live:
                self._live.frame_staged(
                    session.session_id, t, session.channel_cidx[c], raw=raw
                )
        session.stage_raw_s.append(raw_write_s)
        session.stage_tiff_s.append(time.perf_counter() - t_start)
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

    def _send_ack(self, session: SessionState, ended: bool = False) -> None:
        header = {
            "through_frame_index": session.ack_floor,
            "server_time_s": time.time(),
            "features": SERVER_FEATURES,
        }
        if ended:
            header["ended"] = True
        session.sock.send_multipart(
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
            session.sock.send_multipart(
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
        # reach the client) is gone; `ended` tells the client its
        # SESSION_END landed.
        self._send_ack(session, ended=True)

        if reason == "paused":
            self._retain_paused(session)
        elif session.write_root != session.raw_root:
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

    def _retain_paused(self, session: SessionState) -> None:
        """The client gave up streaming this run mid-way (SESSION_END reason
        "paused"): its full local save goes to Argus by Globus, into the
        same raw_root, so the partial streamed copy must not land there."""
        if session.write_root == session.raw_root:
            logger.error(
                "Session %s was PAUSED by the client after %d (t,c) pair(s); "
                "staging is off, so its partial raw copy is already in %s -- "
                "remove it before sending the run by Globus",
                session.session_id,
                len(session.received_pairs),
                session.dest_leaf_dir,
            )
            return
        stage_paths = [*session.channel_store_paths.values(), session.decon_stage_dir]
        self._drain_pool.retain(session.session_id, stage_paths)
        logger.error(
            "Session %s was PAUSED by the client (link down longer than its "
            "buffer) after %d of %d (t,c) pair(s): the partial copy stays on "
            "staging and is NOT copied to %s -- send this run by Globus",
            session.session_id,
            len(session.received_pairs),
            session.num_timepoints * len(session.channels),
            session.raw_root,
        )

    def _handle_resume(
        self, sock: zmq.Socket, identity: bytes, session_id: str
    ) -> None:
        session = self.sessions.get(session_id)
        if session is None:
            # Receiver process restarted, or idle-timed this session out, or
            # it never reached SESSION_START here: flag it, so the client
            # re-sends SESSION_START with resume_through and then everything
            # still in its buffer (see the module docstring's "Recovery").
            self._send_unknown(sock, identity, session_id)
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


# Client-clock times a FRAME header may carry (see protocol.py), traced as
# sent; the report converts them with the frame's clock_offset_s.
_CLIENT_TIME_FIELDS = (
    "acq_first_s",
    "acq_last_s",
    "queued_s",
    "sent_s",
    "clock_offset_s",
)


def _trace_frame(
    session: SessionState,
    header: dict[str, Any],
    nbytes: int,
    recv_s: float,
    **extra: Any,
) -> None:
    """One `frame` trace line per newly staged (t, c): when its FRAME (or
    its last slab) was received and when staging finished, plus the client's
    own timestamps (a slab carries its volume's)."""
    trace.record(
        "frame",
        session_id=session.session_id,
        base_name=session.base_name,
        t=header["t"],
        c=header["c"],
        cidx=session.channel_cidx.get(header["c"]),
        bytes=nbytes,
        recv_s=recv_s,
        staged_s=time.time(),
        **{k: header[k] for k in _CLIENT_TIME_FIELDS if k in header},
        **extra,
    )


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
            "dup_slabs_after_complete": session.dup_slabs,
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
    receiver = StreamReceiver(
        direct_bind=os.environ.get(_DIRECT_BIND_ENV_VAR, "").strip() or None,
        allow_ips=_env_list(_ALLOW_IPS_ENV_VAR),
        raw_roots=_env_list(_RAW_ROOTS_ENV_VAR),
    )
    with receiver:
        receiver.run_forever()


if __name__ == "__main__":
    main()
