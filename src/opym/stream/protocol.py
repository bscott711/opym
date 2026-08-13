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
# will stream. Fields map directly onto opym.petakit.submit_pipeline_job's
# parameters -- the receiver has no other source for them, so everything
# that function needs must be declared upfront:
#
#   base_name        str    -- output basename; final zarrs are named
#                               "<base_name>_T{t:04d}_C{c}.zarr"
#   output_dir       str    -- GPFS directory PetaKit5D writes final
#                               Decon/DSR output into (submit_pipeline_job's
#                               `output_file.parent`), e.g. ".../Decon"
#   dtype            str    -- numpy dtype string for every frame's raw
#                               payload, e.g. "uint16"
#   shape_zyx        [int, int, int]  -- raw per-volume (Z, Y, X) shape
#                               every MSG_FRAME payload will match
#   num_timepoints   int    -- total T count (for the (T, C) grid)
#   channels         [int]  -- channel indices this session will send
#   channel_names    [str]  -- human-readable names, same order as `channels`
#   z_step_um        float
#   xy_pixel_size    float
#   sheet_angle_deg  float
#   t_interval_s     float  -- seconds between timepoints (consolidate metadata)
#   interp_method    str    -- default "cubic"
#   rl_method        str    -- default "simple"
#   iterations       int | None
#   psf_paths        [str] | None -- omit/empty to skip decon (deskew only)
#   dz_psf           float | None -- the PSF's own z-step, microns. If
#                               omitted while psf_paths is non-empty,
#                               submit_pipeline_job falls back to reading it
#                               from the first PSF file's ImageJ 'spacing'
#                               tag (same as the batch path) -- but that read
#                               happens on the hot per-frame path here, so
#                               sending it explicitly avoids a per-frame file
#                               open and avoids a hard failure if the PSF
#                               file lacks that tag.
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
#                               staged to /dev/shm AND ticketed. -1 if
#                               nothing has been processed yet for this
#                               session. The client's local retry buffer
#                               only needs to keep frames with
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
