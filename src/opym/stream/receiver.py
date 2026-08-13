# Ruff style: Compliant
"""
ROUTER-side server for the real-time frame-streaming protocol (see
`opym.stream.protocol`). Bridges incoming frames into the existing
`/dev/shm/petakit_jobs` ticket pipeline -- `opym.petakit`,
`opym.local_gpu_worker`, and `opym.consolidate` are unmodified; this module
only produces the same staged-zarr + JSON-ticket shape the batch path
already produces (see `bioimaging/CLAUDE.md`'s pipeline diagram), per frame
instead of per dataset.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import zarr
import zmq

from opym.petakit import QUEUE_DIR, submit_pipeline_job
from opym.stream.protocol import (
    MSG_ACK,
    MSG_FRAME,
    MSG_RESUME,
    MSG_SESSION_END,
    MSG_SESSION_START,
    pack_message,
    unpack_message,
)
from opym.utils import orient_zyx_for_dsr

logger = logging.getLogger(__name__)

DEFAULT_BIND_ADDR = "tcp://127.0.0.1:5555"
DEFAULT_SHM_DIR = Path("/dev/shm/opym_jobs")
ACK_EVERY_N_FRAMES = 10
ACK_EVERY_SEC = 2.0
IDLE_TIMEOUT_SEC = 600.0
POLL_TIMEOUT_MS = 500


@dataclass
class SessionState:
    session_id: str
    identity: bytes
    base_name: str
    output_dir: Path
    dtype: str
    shape_zyx: tuple[int, int, int]
    num_timepoints: int
    channels: list[int]
    channel_names: list[str]
    z_step_um: float
    xy_pixel_size: float
    sheet_angle_deg: float
    t_interval_s: float
    interp_method: str
    rl_method: str
    iterations: int | None
    psf_paths: list[str]
    dz_psf: float | None
    received_pairs: set[tuple[int, int]] = field(default_factory=set)
    processed_frame_indices: set[int] = field(default_factory=set)
    ack_floor: int = -1
    frames_since_ack: int = 0
    last_ack_time: float = field(default_factory=time.monotonic)
    last_activity: float = field(default_factory=time.monotonic)
    ended: bool = False

    def expected_zarr_names(self) -> list[str]:
        return [
            f"{self.base_name}_T{t:04d}_C{c}.zarr"
            for t in range(self.num_timepoints)
            for c in self.channels
        ]


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
        shm_dir: Path = DEFAULT_SHM_DIR,
        queue_dir: Path = QUEUE_DIR,
        ack_every_n_frames: int = ACK_EVERY_N_FRAMES,
        ack_every_sec: float = ACK_EVERY_SEC,
        idle_timeout_sec: float = IDLE_TIMEOUT_SEC,
    ) -> None:
        self.bind_addr = bind_addr
        self.shm_dir = Path(shm_dir)
        self.queue_dir = Path(queue_dir)
        self.ack_every_n_frames = ack_every_n_frames
        self.ack_every_sec = ack_every_sec
        self.idle_timeout_sec = idle_timeout_sec
        self.sessions: dict[str, SessionState] = {}

        self.shm_dir.mkdir(parents=True, exist_ok=True)

        self._ctx = zmq.Context.instance()
        self._socket = self._ctx.socket(zmq.ROUTER)
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
        output_dir = Path(header["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        session = SessionState(
            session_id=session_id,
            identity=identity,
            base_name=header["base_name"],
            output_dir=output_dir,
            dtype=header["dtype"],
            shape_zyx=tuple(header["shape_zyx"]),
            num_timepoints=header["num_timepoints"],
            channels=list(header["channels"]),
            channel_names=list(header.get("channel_names") or []),
            z_step_um=header["z_step_um"],
            xy_pixel_size=header["xy_pixel_size"],
            sheet_angle_deg=header["sheet_angle_deg"],
            t_interval_s=header["t_interval_s"],
            interp_method=header.get("interp_method", "cubic"),
            rl_method=header.get("rl_method", "simple"),
            iterations=header.get("iterations"),
            psf_paths=list(header.get("psf_paths") or []),
            dz_psf=header.get("dz_psf"),
        )
        self.sessions[session_id] = session
        logger.info(
            "Session %s started: base_name=%s grid=%dT x %dC -> %s",
            session_id,
            session.base_name,
            session.num_timepoints,
            len(session.channels),
            output_dir,
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
                    self._stage_and_ticket(session, header, payload)
                except Exception:
                    logger.exception(
                        "Failed to stage/ticket (t=%d, c=%d) for session %s -- "
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

    def _stage_and_ticket(
        self, session: SessionState, header: dict[str, Any], payload: bytes
    ) -> None:
        t, c = header["t"], header["c"]
        shape_zyx = tuple(header.get("shape_zyx") or session.shape_zyx)
        dtype = header.get("dtype") or session.dtype

        raw = np.frombuffer(payload, dtype=np.dtype(dtype)).reshape(shape_zyx)
        volume = orient_zyx_for_dsr(raw)

        name = f"{session.base_name}_T{t:04d}_C{c}.zarr"
        shm_path = self.shm_dir / name
        zarr.save_array(str(shm_path), volume, chunks=volume.shape)

        output_file = session.output_dir / name
        submit_pipeline_job(
            output_file=output_file,
            shm_path=shm_path,
            psf_paths=session.psf_paths or None,
            z_step_um=session.z_step_um,
            xy_pixel_size=session.xy_pixel_size,
            sheet_angle_deg=session.sheet_angle_deg,
            interp_method=session.interp_method,
            iterations=session.iterations,
            rl_method=session.rl_method,
            dz_psf=session.dz_psf,
            queue_dir=self.queue_dir,
        )
        logger.debug(
            "Staged + ticketed T=%d C=%d for session %s", t, c, session.session_id
        )

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
        sidecar = session.output_dir / ".opym_consolidate.json"
        sidecar.write_text(
            json.dumps(
                {
                    "expected_zarrs": session.expected_zarr_names(),
                    "base_name": session.base_name,
                    "z_step_um": session.z_step_um,
                    "xy_pixel_um": session.xy_pixel_size,
                    "t_interval_s": session.t_interval_s,
                    "channel_names": session.channel_names,
                },
                indent=2,
            )
        )
        logger.info(
            "Session %s ended (%s): %d/%d (t,c) pairs received, sidecar written to %s",
            session.session_id,
            reason,
            len(session.received_pairs),
            session.num_timepoints * len(session.channels),
            sidecar,
        )
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
