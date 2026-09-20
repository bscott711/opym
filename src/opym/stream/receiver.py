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
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import zmq

from opym.stream import rawmirror
from opym.stream.protocol import (
    MSG_ACK,
    MSG_FRAME,
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

# Same environment variable bioimaging.backfill.pipeline.resolve_decon_psf
# reads to decide whether the batch pass decons at all. Read directly
# (rather than importing that function) to keep the one-way dependency
# direction CLAUDE.md documents: opym_local must not import from
# bioimaging. Duplicating a single env-var name is a far smaller coupling
# than that import would be, and getting this wrong costs nothing but a
# missed pre-staging optimization -- build_decon_staging_dir's own
# skip-if-exists check makes it harmless either way (see module docstring).
_DECON_PSF_ENV_VAR = "OPYM_DECON_PSF"


def _decon_enabled() -> bool:
    return bool(os.environ.get(_DECON_PSF_ENV_VAR, "").strip())


@dataclass
class SessionState:
    session_id: str
    identity: bytes
    base_name: str
    raw_root: Path
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
    channel_arrays: dict[int, Any] = field(default_factory=dict)
    received_pairs: set[tuple[int, int]] = field(default_factory=set)
    processed_frame_indices: set[int] = field(default_factory=set)
    ack_floor: int = -1
    frames_since_ack: int = 0
    last_ack_time: float = field(default_factory=time.monotonic)
    last_activity: float = field(default_factory=time.monotonic)
    ended: bool = False

    @property
    def leaf_dir(self) -> Path:
        """Matches `opym.discovery.LeafDataset.leaf_dir` for this session's
        eventual dataset: `raw_root / base_name`."""
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
    ) -> None:
        self.bind_addr = bind_addr
        self.ack_every_n_frames = ack_every_n_frames
        self.ack_every_sec = ack_every_sec
        self.idle_timeout_sec = idle_timeout_sec
        self.sessions: dict[str, SessionState] = {}

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
        self._socket.close(linger=0)

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
            channel_store_paths = {
                c: rawmirror.store_path_for_channel(raw_root, base_name, name)
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
                dtype=header["dtype"],
                shape_zyx=tuple(header["shape_zyx"]),
                num_timepoints=header["num_timepoints"],
                channels=channels,
                channel_names=channel_names,
                channel_store_paths=channel_store_paths,
                channel_cidx=channel_cidx,
                z_step_um=header["z_step_um"],
                decon_enabled=_decon_enabled(),
            )
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("Rejecting SESSION_START for %s: %s", session_id, exc)
            return

        self.sessions[session_id] = session
        logger.info(
            "Session %s started: base_name=%s grid=%dT x %dC -> %s (decon_enabled=%s)",
            session_id,
            session.base_name,
            session.num_timepoints,
            len(session.channels),
            session.raw_root,
            session.decon_enabled,
        )
        self._send_ack(session)

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
            )
            session.channel_arrays[c] = arr
        rawmirror.write_timepoint(arr, t, raw)

        if session.decon_enabled:
            dst = self._decon_stage_path(session, c, t)
            dst.parent.mkdir(parents=True, exist_ok=True)
            write_decon_staged_tiff(raw, dst)

        logger.debug(
            "Staged T=%d C=%d for session %s -> %s",
            t,
            c,
            session.session_id,
            session.channel_store_paths[c],
        )

    def _decon_stage_path(self, session: SessionState, c: int, t: int) -> Path:
        """Matches `build_decon_staging_dir`'s own naming exactly, so its
        skip-if-exists check treats this file as already done."""
        if session.num_timepoints > 1:
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
            session.raw_root,
        )
        # No consolidation step: see module docstring. `opym-backfill
        # --watch` (already running independently) discovers this dataset
        # on its own next pass -- nothing to trigger here.
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


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    receiver = StreamReceiver()
    with receiver:
        receiver.run_forever()


if __name__ == "__main__":
    main()
