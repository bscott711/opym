# Ruff style: Compliant
"""
Acquisition-side client for the real-time frame-streaming protocol (see
`opym.stream.protocol` and `opym_local/docs/STREAMING_PROTOCOL.md`).

Two pieces:

- `StreamSender` -- the DEALER-socket sender core: SESSION_START/FRAME/
  SESSION_END, the bounded retry buffer, and ACK/RESUME-driven eviction.
  Transport-only; it has no opinion about where a volume comes from.
- `watch_and_stream` -- a bolt-on front end that polls a local per-channel
  `*.ome.zarr` store (the newer pymmcore-based MDA writer's own output) for
  newly-completed timepoints and feeds them to a `StreamSender`. Zero
  changes to the acquisition process: it only ever reads the same files the
  writer itself produces, so a crash or hang here cannot stall an MDA run.
  A future in-process pymmcore-plus `frameReady` hook (pushing volumes
  straight from memory, no local-disk round trip) shares `StreamSender` and
  is a strict latency improvement over this watcher, not a replacement for
  it -- keep this as the fallback path.

This module is intentionally pure-Python-portable (pyzmq, msgpack, numpy,
zarr) since it is meant to run on the Windows acquisition workstation, not
just on Argus.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import zarr
import zmq

from opym.discovery import find_channel_zarr_stores, parse_zarr_group_prefix
from opym.metadata import parse_zarr_z_step_from_store
from opym.stream.protocol import (
    MSG_ACK,
    MSG_FRAME,
    MSG_RESUME,
    MSG_SESSION_END,
    MSG_SESSION_START,
    pack_message,
    unpack_message,
)

logger = logging.getLogger(__name__)

DEFAULT_SNDHWM = 50  # a few seconds' worth of volumes; see docs/STREAMING_PROTOCOL.md
DEFAULT_POLL_TIMEOUT_MS = 200


class StreamSender:
    """Owns one DEALER connection for one session.

    Not thread-safe -- pyzmq sockets aren't, and this protocol has no
    concurrent-sender case (one acquisition run == one session == one
    socket). `watch_and_stream` and any future frameReady hook both drive
    one `StreamSender` from a single thread/loop.
    """

    def __init__(
        self,
        connect_addr: str,
        session_id: str | None = None,
        sndhwm: int = DEFAULT_SNDHWM,
    ) -> None:
        self.session_id = session_id or str(uuid.uuid4())
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.DEALER)
        # MUST be set before connect() -- see protocol.py's module docstring
        # for why this is the whole mechanism behind resumable reconnects.
        self._sock.setsockopt(zmq.IDENTITY, self.session_id.encode("utf-8"))
        self._sock.setsockopt(zmq.SNDHWM, sndhwm)
        self._sock.connect(connect_addr)
        self._poller = zmq.Poller()
        self._poller.register(self._sock, zmq.POLLIN)

        # frame_index -> (header, payload), pruned as ACKs advance
        # through_frame_index past an entry. Bounded implicitly by sndhwm:
        # once the socket blocks, the caller's own send loop stalls before
        # this buffer can grow unbounded.
        self._retry_buffer: dict[int, tuple[dict[str, Any], bytes]] = {}
        self.through_frame_index: int = -1

    def close(self) -> None:
        self._sock.close(linger=0)

    def __enter__(self) -> StreamSender:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def session_start(self, header: dict[str, Any]) -> None:
        self._sock.send_multipart(
            pack_message(MSG_SESSION_START, self.session_id, header)
        )
        self._await_ack()

    def send_frame(self, header: dict[str, Any], volume: np.ndarray) -> None:
        """`header` must include `frame_index`; `volume` is sent
        C-contiguous exactly as given -- callers are responsible for any
        reshaping/dtype match with what SESSION_START declared.
        """
        payload = np.ascontiguousarray(volume).tobytes()
        frame_index = header["frame_index"]
        self._retry_buffer[frame_index] = (header, payload)
        self._send_frame_wire(header, payload)
        self.drain_acks()

    def send_slab(
        self, header: dict[str, Any], volume: np.ndarray, z0: int, planes: int
    ) -> None:
        """Send planes `[z0, z0 + planes)` of `volume` as one slab FRAME (see
        the receiver's module docstring). `header` is the volume's FRAME
        header; its `frame_index` must be this slab's own."""
        slab = np.ascontiguousarray(volume[z0 : z0 + planes])
        slab_header = {
            **header,
            "z0": z0,
            "nz": volume.shape[0],
            "shape_zyx": list(slab.shape),
        }
        self.send_frame(slab_header, slab)

    def _send_frame_wire(self, header: dict[str, Any], payload: bytes) -> None:
        self._sock.send_multipart(
            pack_message(MSG_FRAME, self.session_id, header, payload)
        )

    def session_end(self, reason: str = "complete") -> None:
        self._sock.send_multipart(
            pack_message(MSG_SESSION_END, self.session_id, {"reason": reason})
        )

    def resume(self, timeout_ms: int = 5000) -> None:
        """Call right after constructing a fresh `StreamSender` with the
        SAME `session_id` as a previous, now-dead connection (e.g. after a
        network blip). Blocks up to `timeout_ms` for the server's ACK, then
        resends every buffered frame above `through_frame_index`, in order.
        """
        self._sock.send_multipart(pack_message(MSG_RESUME, self.session_id, {}))
        if not self._await_ack(timeout_ms=timeout_ms):
            logger.warning(
                "No RESUME ack within %d ms for session %s -- resending whole "
                "retry buffer anyway",
                timeout_ms,
                self.session_id,
            )
        self._evict_acked()
        for frame_index in sorted(self._retry_buffer):
            header, payload = self._retry_buffer[frame_index]
            self._send_frame_wire(header, payload)

    def drain_acks(self, timeout_ms: int = 0) -> None:
        """Non-blocking (by default) drain of any pending ACKs, evicting
        retry-buffer entries at or below the new `through_frame_index`.
        """
        while dict(self._poller.poll(timeout=timeout_ms)):
            self._handle_one_ack()
            timeout_ms = 0  # only block on the first iteration, if at all

    def wait_for_all_acked(self, timeout_s: float = 30.0) -> bool:
        """Blocks (polling in short bursts) until every frame currently in
        the retry buffer has been acked, or `timeout_s` elapses.

        The receiver acks periodically (every ~10 frames or ~2s), not
        immediately per frame, so a single bounded `drain_acks` call right
        after the last `send_frame`/`session_end` can return before the
        receiver has actually finished staging everything it was sent --
        exactly the ack-is-the-durability-signal contract this method
        exists to wait out (see docs/STREAMING_PROTOCOL.md's ACK section).
        Returns False on timeout with frames still unacked; callers should
        treat that as "session ended, but not everything was confirmed
        durable" rather than a hard failure -- the frames are still in
        `self._retry_buffer`, so a subsequent `resume()` (e.g. from a
        supervising process that restarts this session) can still recover
        them.
        """
        deadline = time.monotonic() + timeout_s
        while self.pending_count > 0 and time.monotonic() < deadline:
            self.drain_acks(timeout_ms=200)
        return self.pending_count == 0

    def _await_ack(self, timeout_ms: int = 5000) -> bool:
        if not dict(self._poller.poll(timeout=timeout_ms)):
            return False
        self._handle_one_ack()
        return True

    def _handle_one_ack(self) -> None:
        msg_type, session_id, header, _ = unpack_message(self._sock.recv_multipart())
        if msg_type != MSG_ACK or session_id != self.session_id:
            logger.warning(
                "Unexpected message on session %s: %r", self.session_id, msg_type
            )
            return
        self.through_frame_index = header["through_frame_index"]
        self._evict_acked()

    def _evict_acked(self) -> None:
        for frame_index in [
            fi for fi in self._retry_buffer if fi <= self.through_frame_index
        ]:
            del self._retry_buffer[frame_index]

    @property
    def pending_count(self) -> int:
        return len(self._retry_buffer)


# --- Watcher front end ----------------------------------------------------


@dataclass
class ChannelWatchState:
    c: int
    store: Path
    pixel_dir: Path
    nz: int
    next_t: int = 0


def _z_planes_written(pixel_dir: Path, t: int) -> tuple[int, float]:
    """Number of z-plane subdirectories written for timepoint `t`, and the
    mtime of their most recently modified chunk file -- used to judge
    whether `t` is done arriving rather than merely started (see
    `write_timepoint`'s note on this same per-z-chunk non-atomicity in
    `opym.stream.rawmirror`).
    """
    t_dir = pixel_dir / str(t)
    if not t_dir.is_dir():
        return 0, 0.0
    z_dirs = [e for e in t_dir.iterdir() if e.is_dir() and e.name.isdigit()]
    if not z_dirs:
        return 0, 0.0
    files = [f for d in z_dirs for f in d.rglob("*") if f.is_file()]
    newest_mtime = max((f.stat().st_mtime for f in files), default=0.0)
    return len(z_dirs), newest_mtime


def watch_and_stream(
    local_store_paths: dict[int, Path],
    channel_names: dict[int, str],
    *,
    base_name: str,
    raw_root: str,
    connect_addr: str,
    num_timepoints: int | None = None,
    z_step_um: float | None = None,
    session_id: str | None = None,
    poll_interval_s: float = 1.0,
    stability_polls: int = 2,
    idle_exit_after_s: float = 1800.0,
) -> None:
    """Polls each channel's local `*.ome.zarr` store for newly-completed
    timepoints and streams them, in whatever order they finish, until every
    channel reaches `num_timepoints` or `idle_exit_after_s` passes with no
    new timepoint from any channel.

    `local_store_paths`/`channel_names` are keyed by the same channel index
    `c` this session will use on the wire -- `channel_names[c]` must be the
    "<ChannelName>_<Wavelength>" suffix (e.g. "GFP_488") the receiver will
    use to build the remote store name; it need not match the local store's
    own name.

    A `(t, c)` pair is sent once its z-plane count reaches the channel's
    declared Z depth AND its newest chunk file's mtime is at least
    `stability_polls * poll_interval_s` seconds old -- an age check, not a
    poll-count check, so a channel that's merely catching up on an
    already-fully-written backlog (e.g. the watcher started mid-acquisition)
    doesn't wait out an artificial N-poll delay per timepoint, while a
    timepoint genuinely still being written never looks done just because
    the outer loop happened to spin a few times quickly. Same "don't trust
    a store until it stops changing" caution
    `bioimaging.backfill.pipeline._zarr_store_is_ready` already applies to
    Globus transfers, applied here to a live writer instead.
    """
    if z_step_um is None:
        # num_timepoints, if also unset, is resolved per-channel below from
        # each store's own declared shape instead -- only z_step_um has one
        # reliable source (the first store's own "z" coordinate array).
        first_store = next(iter(local_store_paths.values()))
        z_step_um = parse_zarr_z_step_from_store(first_store) or 0.3

    channels = sorted(local_store_paths)
    states: dict[int, ChannelWatchState] = {}
    declared_nt = num_timepoints
    for c in channels:
        store = local_store_paths[c]
        pixel_dir = store / _dataset_path(store)
        arr = zarr.open(str(pixel_dir), mode="r")
        nz = arr.shape[1] if arr.ndim == 4 else arr.shape[0]
        if declared_nt is None:
            declared_nt = arr.shape[0] if arr.ndim == 4 else 1
        states[c] = ChannelWatchState(c=c, store=store, pixel_dir=pixel_dir, nz=nz)

    session_header = {
        "base_name": base_name,
        "raw_root": raw_root,
        "dtype": str(zarr.open(str(states[channels[0]].pixel_dir), mode="r").dtype),
        "shape_zyx": [
            states[channels[0]].nz,
            *_yx_shape(states[channels[0]].pixel_dir),
        ],
        "num_timepoints": declared_nt,
        "channels": channels,
        "channel_names": [channel_names[c] for c in channels],
        "z_step_um": z_step_um,
    }

    sender = StreamSender(connect_addr, session_id=session_id)
    frame_index = 0
    last_progress = time.monotonic()
    try:
        sender.session_start(session_header)
        logger.info(
            "Session %s started: base_name=%s, %d channels, declared T=%s",
            sender.session_id,
            base_name,
            len(channels),
            declared_nt,
        )
        min_stable_age_s = stability_polls * poll_interval_s
        while True:
            made_progress = False
            all_done = True
            for c in channels:
                st = states[c]
                if declared_nt is not None and st.next_t >= declared_nt:
                    continue
                all_done = False
                if _poll_channel_once(st, min_stable_age_s):
                    arr = zarr.open(str(st.pixel_dir), mode="r")
                    volume = np.asarray(arr[st.next_t])
                    header = {
                        "t": st.next_t,
                        "c": c,
                        "frame_index": frame_index,
                        "timestamp": time.time(),
                        "camera_id": c,
                        "shape_zyx": list(volume.shape),
                        "dtype": str(volume.dtype),
                    }
                    sender.send_frame(header, volume)
                    logger.debug(
                        "Sent T=%d C=%d (frame_index=%d)", st.next_t, c, frame_index
                    )
                    frame_index += 1
                    st.next_t += 1
                    made_progress = True

            if all_done:
                break
            if made_progress:
                last_progress = time.monotonic()
            elif time.monotonic() - last_progress > idle_exit_after_s:
                logger.warning(
                    "No new timepoints from any channel for %.0fs -- ending session "
                    "as incomplete (client_abort)",
                    idle_exit_after_s,
                )
                sender.session_end(reason="client_abort")
                return
            else:
                time.sleep(poll_interval_s)

        sender.session_end(reason="complete")
        if not sender.wait_for_all_acked(timeout_s=30.0):
            logger.warning(
                "Session %s: %d frame(s) still unacked after session_end -- "
                "receiver may not have finished staging everything",
                sender.session_id,
                sender.pending_count,
            )
        logger.info(
            "Session %s complete: %d frames sent", sender.session_id, frame_index
        )
    finally:
        sender.close()


def _poll_channel_once(st: ChannelWatchState, min_stable_age_s: float) -> bool:
    """True iff `st.next_t` has all its z-planes written and its newest
    chunk file is at least `min_stable_age_s` old. Does not sleep --
    callers pace their own poll loop.
    """
    n_planes, newest_mtime = _z_planes_written(st.pixel_dir, st.next_t)
    if n_planes < st.nz or newest_mtime <= 0.0:
        return False
    return (time.time() - newest_mtime) >= min_stable_age_s


def _dataset_path(store: Path, default: str = "p0") -> str:
    import json

    try:
        attrs = json.loads((store / ".zattrs").read_text())
        return attrs["multiscales"][0]["datasets"][0]["path"]
    except Exception:  # noqa: BLE001 - malformed/unexpected attrs, fall back
        return default


def _yx_shape(pixel_dir: Path) -> tuple[int, int]:
    arr = zarr.open(str(pixel_dir), mode="r")
    return tuple(arr.shape[-2:])


def discover_local_session(
    leaf_dir: Path,
) -> dict[str, dict[int, Path] | dict[int, str] | str]:
    """Convenience helper for a one-off CLI/manual run: groups every
    `*.ome.zarr` store directly under `leaf_dir` the same way
    `opym.discovery.group_channel_zarr_stores` would, and returns the
    `local_store_paths`/`channel_names`/`base_name` `watch_and_stream`
    needs, for the first (or only) group found.

    Only useful when there's exactly one sample under `leaf_dir` -- a
    directory holding several samples (see `LeafDataset`'s docstring) needs
    its own `watch_and_stream` call per group; this just picks one so a
    manual replay/smoke-test doesn't need to hand-write the grouping.
    """
    stores = find_channel_zarr_stores(leaf_dir)
    if not stores:
        raise FileNotFoundError(f"No *.ome.zarr stores found directly under {leaf_dir}")
    base_name = parse_zarr_group_prefix(stores[0])
    group = [s for s in stores if parse_zarr_group_prefix(s) == base_name]
    local_store_paths = dict(enumerate(sorted(group)))
    channel_names = {
        c: p.name.removesuffix(".ome.zarr").removeprefix(f"{base_name}_")
        for c, p in local_store_paths.items()
    }
    return {
        "local_store_paths": local_store_paths,
        "channel_names": channel_names,
        "base_name": base_name,
    }


def main() -> None:
    import argparse

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Watch a local pymmcore MDA zarr store and stream it to opym-receive."
        )
    )
    parser.add_argument(
        "leaf_dir", type=Path, help="Local directory holding *.ome.zarr stores"
    )
    parser.add_argument("--connect", default="tcp://127.0.0.1:5555")
    parser.add_argument(
        "--raw-root", required=True, help="Remote GPFS raw_root to declare"
    )
    parser.add_argument("--poll-interval", type=float, default=1.0)
    args = parser.parse_args()

    session = discover_local_session(args.leaf_dir)
    watch_and_stream(
        session["local_store_paths"],
        session["channel_names"],
        base_name=session["base_name"],
        raw_root=args.raw_root,
        connect_addr=args.connect,
        poll_interval_s=args.poll_interval,
    )


if __name__ == "__main__":
    main()
