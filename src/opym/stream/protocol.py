# Ruff style: Compliant
"""
Wire protocol for the real-time frame-streaming receiver.

This module IS the spec: an acquisition-side client (in any language/stack)
only needs to reproduce the message layout and header schemas documented
here -- see `opym_local/docs/STREAMING_PROTOCOL.md` for the narrative
version plus a worked example.

Transport: ZeroMQ DEALER (client) <-> ROUTER (server), one TCP connection.
The DEALER socket's `zmq.IDENTITY` MUST be set to the session's `session_id`
(UTF-8 bytes) before connecting -- this is what lets a reconnect (a fresh
TCP connection) be recognized as the same logical session on the server
side, which is the whole mechanism behind RESUME/ACK-based resumability.

Every message is a ZMQ multipart message of the form (after pyzmq's ROUTER
auto-prepended identity frame, which callers don't construct by hand):

    [msg_type, session_id, msgpack_header, raw_payload?]

`raw_payload` is present only for `MSG_FRAME` (the pixel buffer); every
other message type is exactly 3 parts. ZMQ frames each part itself, so no
additional length-prefixing is needed anywhere in this protocol.
"""

from __future__ import annotations

from typing import Any

import msgpack

MSG_SESSION_START = b"SESSION_START"
MSG_FRAME = b"FRAME"
MSG_SESSION_END = b"SESSION_END"
MSG_ACK = b"ACK"
MSG_RESUME = b"RESUME"

_VALID_TYPES = frozenset(
    {MSG_SESSION_START, MSG_FRAME, MSG_SESSION_END, MSG_ACK, MSG_RESUME}
)

# --- SESSION_START header fields ---------------------------------------
#
# Sent once, before any MSG_FRAME, to declare the acquisition this session
# will stream. The receiver writes each channel directly into a raw OME-Zarr
# mirror store on GPFS (opym.stream.rawmirror) -- the same layout a
# completed Globus transfer already produces -- rather than submitting a
# processing ticket itself; opym-backfill's normal discovery/deskew/decon
# path picks the finished dataset up from there unmodified. So only what
# the raw-mirror writer itself needs is REQUIRED:
#
#   base_name        str    -- the sample/grouping prefix, e.g. "Cell_001".
#                               Each channel's raw store is named
#                               "<base_name>_<channel_name>.ome.zarr" --
#                               must match opym.discovery's
#                               "<prefix>_<Channel>_<Wavelength>" convention
#                               (see channel_names below) so the finished
#                               store groups back under this same prefix.
#   raw_root         str    -- GPFS directory to write each channel's raw
#                               "<base_name>_<channel_name>.ome.zarr" store
#                               into -- what a Globus transfer would drop
#                               the session into today, e.g.
#                               ".../DataUpload/<session>/"
#   dtype            str    -- numpy dtype string for every frame's raw
#                               payload, e.g. "uint16"
#   shape_zyx        [int, int, int]  -- raw per-volume (Z, Y, X) shape;
#                               a per-FRAME shape_zyx overrides this for
#                               that channel (see FRAME fields below)
#   num_timepoints   int    -- declared T count, used as the raw store's
#                               array shape. May exceed what actually gets
#                               written (aborted acquisition) -- handled
#                               downstream exactly like a Globus-landed
#                               dataset (see channel_store_timepoints in
#                               bioimaging/backfill/pipeline.py)
#   channels         [int]  -- channel indices this session will send
#   channel_names    [str]  -- REQUIRED, one per `channels` entry, in the
#                               "<ChannelName>_<Wavelength>" shape
#                               opym.discovery expects, e.g. "GFP_488"
#   z_step_um        float  -- written into each store's own "z" coordinate
#                               array, the only z-step source
#                               opym.metadata.parse_zarr_z_step_from_store
#                               trusts (see its docstring)
#
# Decon parameters (PSF, wiener_alpha, edge_erosion, rl_method) are NOT
# part of this handshake -- they're resolved server-side by the batch
# backfill driver's own fixed configuration (OPYM_DECON_PSF env var,
# DECON_WIENER_ALPHA/DECON_EDGE_EROSION constants), same as for a
# Globus-landed dataset. When that env var is set on the receiver's host,
# every frame is ALSO pre-staged as a decon-ready TIFF
# (opym.utils.write_decon_staged_tiff) so the batch pass's own staging
# step is a no-op by the time it runs -- purely a latency optimization,
# not a correctness dependency; a client sending sheet_angle_deg,
# xy_pixel_size, t_interval_s, psf_paths, etc. is harmless (ignored) but
# no longer required, since STREAMING_PROTOCOL.md's worked example still
# shows them for provenance/logging purposes.
#
# --- FRAME header fields -------------------------------------------------
#
#   t, c             int    -- this volume's own acquisition indices, not
#                               stream position. Frames may arrive with any
#                               (t, c) order; the receiver stages/tickets by
#                               these indices, so out-of-order delivery
#                               across timepoints/channels is always safe.
#   frame_index      int    -- monotonic counter assigned once per real
#                               acquired frame (NOT per send attempt) by the
#                               client, stable across reconnects. Used only
#                               for ACK/RESUME bookkeeping, never for
#                               staging/output naming.
#   timestamp        float  -- acquisition epoch seconds
#   camera_id        str | int | None
#   shape_zyx        [int, int, int]  -- redundant with SESSION_START but
#                               makes each frame self-describing.
#   dtype            str
#
# --- SESSION_END header fields -------------------------------------------
#
#   reason           str    -- "complete" | "idle_timeout" | "client_abort"
#
# --- ACK header fields (server -> client, unsolicited or RESUME reply) ---
#
#   through_frame_index  int  -- highest frame_index N such that every
#                               frame_index in [0, N] has been durably
#                               written into its channel's raw mirror store
#                               on GPFS. -1 if nothing has been processed
#                               yet for this session. The client's local
#                               retry buffer only needs to keep frames with
#                               frame_index > through_frame_index.
#
# --- RESUME header fields (client -> server, right after reconnecting) ---
#
#   (none required -- session_id in the envelope is enough; the server
#   replies with a fresh ACK for that session_id)


def pack_message(
    msg_type: bytes,
    session_id: str,
    header: dict[str, Any],
    payload: bytes | None = None,
) -> list[bytes]:
    """Builds the multipart frame list for one protocol message.

    `payload` should be given (and non-None) only for MSG_FRAME -- every
    other message type is header-only.
    """
    if msg_type not in _VALID_TYPES:
        raise ValueError(f"Unknown message type: {msg_type!r}")
    parts = [
        msg_type,
        session_id.encode("utf-8"),
        msgpack.packb(header, use_bin_type=True),
    ]
    if payload is not None:
        parts.append(payload)
    return parts


def unpack_message(
    parts: list[bytes],
) -> tuple[bytes, str, dict[str, Any], bytes | None]:
    """Inverse of `pack_message`.

    `parts` is the message as received AFTER stripping any transport-level
    identity frame (pyzmq's `router.recv_multipart()` prepends one -- the
    caller strips `parts[0]` before calling this).
    """
    if len(parts) not in (3, 4):
        raise ValueError(f"Expected 3 or 4 message parts, got {len(parts)}")
    msg_type, session_id_bytes, header_bytes, *rest = parts
    if msg_type not in _VALID_TYPES:
        raise ValueError(f"Unknown message type: {msg_type!r}")
    header = msgpack.unpackb(header_bytes, raw=False)
    payload = rest[0] if rest else None
    return msg_type, session_id_bytes.decode("utf-8"), header, payload
