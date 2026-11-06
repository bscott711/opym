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

Links: once an ACK advertises "links", a client may send the same session
over more connections at once, one DEALER per link, with identity
`"<session_id>#<k>"` for link k >= 1 (link 0 keeps the bare session_id and
is the one that sends SESSION_START). One SSH tunnel carries ~33 MB/s
however fast the network is (sshd's 2 MB window per round trip), so N
tunnels carry about N times that. Any message may arrive on any link, in
any order across links; the server keys everything by the header's
session_id, dedupes by frame_index / (t, c) / z0, and sends ACKs and QC
back on whichever link it heard from last.

Every message is a ZMQ multipart message of the form (after pyzmq's ROUTER
auto-prepended identity frame, which callers don't construct by hand):

    [msg_type, session_id, msgpack_header, raw_payload?]

`raw_payload` is present only for `MSG_FRAME` (the pixel buffer); every
other message type is exactly 3 parts. ZMQ frames each part itself, so no
additional length-prefixing is needed anywhere in this protocol.
"""

from __future__ import annotations

import math
import numbers
from typing import Any, Literal, TypedDict

# --- Wire constants -------------------------------------------------------
#
# The addresses, limits and texts both ends of the wire depend on. The
# receivers (opym.stream.receiver, backfill_receiver, linkbench) and the
# acquisition-side senders all take them from here. All three bind loopback;
# a client reaches them through an SSH tunnel.

STREAM_PORT = 5555  # opym-receive: the live frame stream
BACKFILL_PORT = 5556  # opym-receive-backfill: finished OME-Zarr stores
LINKBENCH_PORT = 5599  # opym.stream.linkbench: the link throughput sink
DEFAULT_BIND_ADDR = f"tcp://127.0.0.1:{STREAM_PORT}"
BACKFILL_BIND_ADDR = f"tcp://127.0.0.1:{BACKFILL_PORT}"
LINKBENCH_BIND_ADDR = f"tcp://127.0.0.1:{LINKBENCH_PORT}"

# A session with no new data for this long is closed (unless its client
# heartbeats; see receiver.STALL_MAX_SEC).
IDLE_TIMEOUT_SEC = 600.0
# The `rejected` text of an ACK that says a session was closed for good. The
# rig matches IDLE_CLOSED_PREFIX as a string to tell "closed, stop" from
# "rejected, retry": rewording the start of this text turns "closed" into
# endless retries.
IDLE_CLOSED_PREFIX = "closed on Argus"
IDLE_CLOSED_REASON = "closed on Argus after 10 min with no new data; not resuming it"
# Advertised in every ACK; clients only use a feature once they've seen it.
SERVER_FEATURES = ["slabs", "blosc", "links", "resume", "heartbeat", "client_log"]

MSG_SESSION_START = b"SESSION_START"
MSG_FRAME = b"FRAME"
MSG_SESSION_END = b"SESSION_END"
MSG_ACK = b"ACK"
MSG_RESUME = b"RESUME"
MSG_QC = b"QC"
MSG_PREPARE = b"PREPARE"
MSG_HEARTBEAT = b"HEARTBEAT"
MSG_CLIENT_LOG = b"CLIENT_LOG"

_VALID_TYPES = frozenset(
    {
        MSG_SESSION_START,
        MSG_FRAME,
        MSG_SESSION_END,
        MSG_ACK,
        MSG_RESUME,
        MSG_QC,
        MSG_PREPARE,
        MSG_HEARTBEAT,
        MSG_CLIENT_LOG,
    }
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
#   output_format    str    -- OPTIONAL, one of "tiff", "ome-zarr", "both"
#                               (rawmirror.OUTPUT_FORMATS): the format the
#                               processed DSR result is kept in. Recorded on
#                               each raw store's .zattrs["opym"]; an unknown
#                               value is logged and ignored. Omitted -> the
#                               backfill's OPYM_OUTPUT_FORMAT default.
#   accepts          [str]  -- OPTIONAL server->client message types this
#                               client understands beyond ACK. Only "qc" is
#                               defined (MSG_QC below). A client that omits
#                               it is never sent anything but ACKs.
#   resume_through   int    -- OPTIONAL, only once an ACK advertised
#                               "resume": this SESSION_START re-opens a
#                               session the server no longer knows (it was
#                               restarted, or idle-timed the session out),
#                               and every frame_index <= resume_through was
#                               already ACKed. The server reopens the
#                               session's stores (tagged with its
#                               session_id), starts its ACK floor there, and
#                               leaves the rest of the session to the batch
#                               pipeline (no live lane: its state is gone).
#                               A SESSION_START for a session still open is
#                               just re-ACKed.
#
# Decon parameters (PSF, wiener_alpha, edge_erosion, rl_method) are NOT
# part of this handshake -- they're resolved server-side by the batch
# backfill driver's own fixed configuration (OPYM_DECON_PSF env var,
# DECON_WIENER_ALPHA/DECON_EDGE_EROSION constants), same as for a
# Globus-landed dataset. When that env var is set on the receiver's host,
# every frame is ALSO pre-staged as a decon-ready TIFF
# (opym.utils.write_decon_staged_tiff) so the batch pass's own staging
# step is a no-op by the time it runs -- purely a latency optimization,
# not a correctness dependency. A client sending sheet_angle_deg,
# xy_pixel_size, psf_paths, etc. is harmless (ignored) and not required;
# STREAMING_PROTOCOL.md's worked example still shows them for provenance.
# t_interval_s IS read (receiver.py, _handle_session_start): it is the
# nominal time step handed to the live view lane.
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
#   timestamp        float  -- acquisition epoch seconds (when the volume's
#                               first plane was acquired). A value outside
#                               [1e9, 1e11) is not accepted as a time; the
#                               receiver uses acq_first_s instead
#                               (frame_epoch_s below).
#   camera_id        str | int | None
#   shape_zyx        [int, int, int]  -- redundant with SESSION_START but
#                               makes each frame self-describing.
#   dtype            str
#
# OPTIONAL slab fields -- only once an ACK has advertised "slabs" in its
# `features`. The FRAME then carries planes [z0, z0 + shape_zyx[0]) of
# volume (t, c) rather than the whole volume, so a volume streams while it
# is still being acquired:
#
#   z0               int    -- first plane of this slab
#   nz               int    -- the whole volume's plane count
#   shape_zyx        [int, int, int]  -- THIS slab's shape
#
# Each slab has its own frame_index. The receiver ACKs a volume's slabs
# only once every plane of it has landed, so after a receiver restart the
# client resends the whole partial volume.
#
# OPTIONAL compression -- only once an ACK has advertised "blosc":
#
#   codec            str    -- "blosc": the payload is one blosc (v1) frame
#                               holding the raw bytes (lz4 + bitshuffle
#                               compresses camera frames ~2.5x); absent or
#                               "raw": the raw bytes themselves. shape_zyx
#                               and dtype always describe the DECODED data.
#
# OPTIONAL latency-trace fields (opym.stream.trace; all epoch seconds on the
# CLIENT's clock). The receiver records them as sent; opym-live-trace
# converts them to server time by adding clock_offset_s:
#
#   acq_first_s      float  -- first plane of this volume acquired
#   acq_last_s       float  -- last plane acquired (volume complete)
#   queued_s         float  -- volume handed to the sender
#   sent_s           float  -- this FRAME handed to the socket
#   clock_offset_s   float  -- server clock minus client clock, measured by
#                               the client from SESSION_START -> first ACK
#                               (server_time_s at the round trip's midpoint)
#
# --- SESSION_END header fields -------------------------------------------
#
#   reason           str    -- "complete" | "idle_timeout" | "client_abort"
#                               | "paused": the client stopped streaming
#                               mid-run (it could not reach the server for
#                               longer than its RAM buffer holds). The run's
#                               partial copy is NOT copied to raw_root, so
#                               the full local save can be sent by Globus
#                               into the same place.
#                               | "client_error": the client's sender failed;
#                               `detail` says how.
#   detail           str    -- OPTIONAL, a human-readable why, shown with
#                               the session's end everywhere (opym.stream.outcome).
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
#   features         [str]  -- OPTIONAL, what this receiver supports beyond
#                               the base protocol: "slabs", "blosc",
#                               "links", "resume", "heartbeat", "client_log".
#   unknown_session  bool   -- OPTIONAL, true when the server has no open
#                               session with this session_id (it answers a
#                               RESUME, or a FRAME at most every 2 s, this
#                               way). A client that saw "resume" re-sends
#                               SESSION_START with resume_through, then
#                               everything unACKed. through_frame_index is
#                               -1 and means nothing here. A SESSION_END for
#                               a session already closed is answered this
#                               way too (it confirms the SESSION_END).
#   rejected         str    -- OPTIONAL, with unknown_session: the server
#                               turned this SESSION_START down (e.g. its RAM
#                               disk has no room for the session yet), and
#                               why. The client retries with backoff.
#   closed           bool   -- OPTIONAL, with unknown_session: the server
#                               closed this session for good (it sent no new
#                               data for its idle timeout) and will not
#                               resume it. Retrying is pointless: the client
#                               stops streaming the run and says so; its
#                               local save is the copy to send.
#   ended            bool   -- OPTIONAL, true on the final ACK, sent once
#                               the server has closed the session: it
#                               confirms SESSION_END. A client that got no
#                               confirmation may resend SESSION_END.
#   server_time_s        float -- OPTIONAL, the server's epoch seconds when
#                               it sent this ACK; lets a client estimate the
#                               clock offset for the trace fields above.
#
# --- RESUME header fields (client -> server, right after reconnecting) ---
#
#   (none required -- session_id in the envelope is enough; the server
#   replies with a fresh ACK for that session_id)
#
# --- QC header fields (server -> client, only if SESSION_START accepts "qc") --
#
# The live QC service's verdict on one timepoint, forwarded as written
# (header only, at most one per new verdict). Advisory: the client shows it;
# nothing on the server waits for a reply. Core fields:
#
#   seq              int    -- increases with every verdict in the session
#   session_id       str
#   t                int    -- timepoint judged
#   stage            str    -- "raw" (from the raw frames, ~1 s after they
#                               land) or "dsr" (from the deskewed volume,
#                               adds the cell's bounding box)
#   verdict          str    -- "ok" | "warn" | "act" | "no_cell"
#   flags            [str]  -- e.g. "clipped_depth_high", "drift_exit_soon",
#                               "defocus", "bleaching"
#   advice           [dict] -- {"action", "axis", "direction", "amount_um",
#                               "when": "now" | "next_run", "text"}
#   metrics          dict   -- per-channel numbers behind the verdict
#
# Clients must ignore fields they don't know; new ones may be added.

# --- PREPARE header fields (client -> server, before a run) --------------
#
# An acquisition with this plan is being set up (e.g. the MDA was configured
# with streaming on): the server warms a GPU server for its shape, so the
# first timepoint costs what the others do. Sent on its own connection, with
# a fresh id as `session_id` (and identity); it opens no session, and nothing
# answers it. Send it again whenever the plan changes; the warm-up holds for
# about 10 minutes after the last one (opym.lanes.WARM_LEASE_MAX_AGE_S).
#
#   shape_zyx        [int, int, int]  -- planned raw per-volume (Z, Y, X)
#   z_step_um        float  -- planned z step
#   dtype            str    -- optional, e.g. "uint16"
#   num_timepoints   int    -- optional, for the log
#   channel_names    [str]  -- optional, for the log


# --- HEARTBEAT header fields (client -> server, once "heartbeat" is seen) --
#
# Every few seconds while a run is open, on any link that is up. It is how
# the server tells a client that is alive but can't send (or is between
# timepoints of a slow time-lapse) from one that is gone: a session whose
# client heartbeats is not closed as idle (up to a long cap), and one that
# is closed says what its client last reported. All fields optional:
#
#   state            str    -- "acquiring" | "finishing"
#   links_up         int    -- links whose connection is up
#   links_total      int
#   unacked_bytes    int    -- what the client holds unACKed
#   unacked_volumes  int
#   submitted        int    -- volumes (or slabs) acquired so far
#   acked_through    int
#   stalled_s        float  -- seconds since an ACK last moved it forward
#                               while it had data in flight
#   last_error       str
#
# --- CLIENT_LOG header fields (client -> server, once "client_log" is seen) --
#
# The client's own stream log, so a post-mortem never needs the acquisition
# PC: the server appends the lines to its rig log (one file per day). Sent
# on any up link, for any session id (it only labels the lines):
#
#   lines            [str]  -- formatted log lines, oldest first
#   host             str    -- OPTIONAL, the client's host name


OutputFormat = Literal["tiff", "ome-zarr", "both"]
"""Format the processed (DSR) result is kept in on Argus.

Matches ``opym.stream.rawmirror.OUTPUT_FORMATS``: ``"tiff"`` keeps the
OME-TIFF frames (ChimeraX), ``"ome-zarr"`` replaces them with one OME-Zarr
(napari), ``"both"`` keeps both.
"""


class _SessionStartOptional(TypedDict, total=False):
    # Server -> client message types this client understands beyond ACK.
    # ["qc"] asks for MSG_QC; a receiver without live QC just never sends it.
    accepts: list[str]
    # Only when re-opening a session Argus answered ``unknown_session`` for
    # (it restarted or timed the session out): every frame_index up to this
    # was already ACKed. Needs the "resume" feature.
    resume_through: int


class SessionStartHeader(_SessionStartOptional):
    """``SESSION_START`` header -- sent once, before any ``FRAME`` message.

    Fields match ``opym.stream.receiver.StreamReceiver._handle_session_start``
    (Argus side, canonical -- read directly from Argus, not from this
    file's own history). The receiver writes each channel straight into a
    raw OME-Zarr mirror store under ``raw_root`` (``opym.stream.rawmirror``);
    it does not submit a processing ticket itself, and does not read decon
    parameters (PSF, wiener_alpha, edge_erosion, rl_method, iterations) from
    this handshake at all -- those are resolved entirely server-side from
    the ``OPYM_DECON_PSF`` env var, same as a batch/Globus-landed dataset.
    An earlier draft of this header targeted a since-rewritten receiver that
    took an ``output_dir`` and consumed PSF paths directly; sending that
    shape to the current receiver has ``raw_root`` missing, which the
    receiver rejects outright (every frame silently dropped as an unknown
    session) -- see ``opym.stream.protocol``'s own module docstring for the
    field-by-field spec this must keep matching.
    """

    base_name: str
    raw_root: str
    dtype: str
    shape_zyx: list[int]
    num_timepoints: int
    channels: list[int]
    channel_names: list[str]
    z_step_um: float
    xy_pixel_size: float
    t_interval_s: float
    output_format: OutputFormat


class _FrameTraceFields(TypedDict, total=False):
    # Latency trace (opym.stream.trace on Argus). Epoch seconds on THIS
    # machine's clock; Argus converts them with clock_offset_s.
    acq_first_s: float
    acq_last_s: float
    queued_s: float
    sent_s: float
    clock_offset_s: float


class _FrameSlabFields(TypedDict, total=False):
    # Only once an ACK advertised "slabs": this FRAME carries planes
    # [z0, z0 + shape_zyx[0]) of an nz-plane volume.
    z0: int
    nz: int


class _FrameCodecFields(TypedDict, total=False):
    # Only once an ACK advertised "blosc": "blosc" means the payload is one
    # blosc frame of the raw bytes (lz4 + bitshuffle). shape_zyx and dtype
    # always describe the decoded data.
    codec: str


class FrameHeader(_FrameTraceFields, _FrameSlabFields, _FrameCodecFields):
    """``FRAME`` header -- one per acquired volume, sent alongside its bytes.

    ``t``/``c`` are this volume's own acquisition indices, not send order --
    frames may arrive in any ``(t, c)`` order and the receiver stages/tickets
    by those indices, not by arrival order. ``frame_index`` is unrelated: a
    monotonic counter assigned once per real acquired volume (not per send
    attempt), used only for ``ACK``/``RESUME`` bookkeeping.
    """

    t: int
    c: int
    frame_index: int
    timestamp: float
    camera_id: str | int | None
    shape_zyx: list[int]
    dtype: str


class QCHeader(TypedDict, total=False):
    """``QC`` header -- server -> client, only if ``accepts`` has ``"qc"``.

    The Argus live QC service's verdict on one timepoint, forwarded as it
    wrote it (``opym.stream.protocol`` documents the fields). Advisory only:
    nothing on Argus waits for a reply. There are two per timepoint:
    ``stage == "raw"`` about a second after the frames land (coverage, drift,
    focus, signal), then ``"dsr"`` once deskewed (adds the cell's box).
    Unknown fields must be ignored; new ones may be added.
    """

    seq: int
    session_id: str
    t: int
    stage: str
    verdict: Literal["ok", "warn", "act", "no_cell"]
    flags: list[str]
    advice: list[dict[str, Any]]
    metrics: dict[str, Any]


class _SessionEndOptional(TypedDict, total=False):
    # Why, in words: shown with the run's end on Argus (naparym-live, the
    # dashboard). Set for "client_error" and "paused".
    detail: str


class SessionEndHeader(_SessionEndOptional):
    """``SESSION_END`` header."""

    # "paused": this run stopped streaming mid-way (Argus unreachable longer
    # than the send buffer holds); Argus won't keep its partial copy.
    # "client_error": this app's sender failed (``detail`` says how).
    reason: Literal[
        "complete", "idle_timeout", "client_abort", "paused", "client_error"
    ]


class _AckOptional(TypedDict, total=False):
    # Argus's clock when it sent the ACK: lets the client estimate its clock
    # offset for the FRAME trace fields.
    server_time_s: float
    # What this receiver supports beyond the base protocol: "slabs",
    # "blosc", "links", "resume", "heartbeat", "client_log".
    features: list[str]
    # Argus has no open session with this id: re-send SESSION_START with
    # resume_through, then everything unACKed.
    unknown_session: bool
    # The final ACK, sent once Argus has closed the session: it confirms
    # SESSION_END.
    ended: bool
    # Argus dropped frames it couldn't stage (its RAM disk was below its
    # floor) and has room again: resend everything unACKed now.
    resend: bool
    # Sent with unknown_session when Argus turned the SESSION_START down
    # (e.g. its RAM disk is short): why, for the status label. The client
    # keeps retrying with backoff; space may come back once a drain ends.
    rejected: str
    # With unknown_session: Argus closed this session for good (no new data
    # for its idle timeout) and won't resume it. Retrying is pointless; the
    # run stops streaming and its local save is the copy to send.
    closed: bool


class AckHeader(_AckOptional):
    """``ACK`` header -- server -> client, unsolicited, not per-frame.

    ``through_frame_index`` is the highest ``frame_index`` such that every
    frame in ``[0, through_frame_index]`` has been durably staged *and*
    ticketed on the Argus side. ``-1`` means nothing processed yet.
    """

    through_frame_index: int


class ResumeHeader(TypedDict):
    """``RESUME`` header -- client -> server, right after reconnecting.

    No fields required: ``session_id`` in the message envelope is enough for
    the server to look up (or fail to find) this session's state.
    """


class _PrepareOptional(TypedDict, total=False):
    dtype: str
    num_timepoints: int
    channel_names: list[str]


class PrepareHeader(_PrepareOptional):
    """``PREPARE`` header -- client -> server, while an MDA is being set up.

    Argus warms a GPU server for this plan so the run's first timepoint costs
    what the others do. Sent on a connection of its own with a fresh id (no
    session is opened, nothing answers); send it again when the plan
    changes. The warm-up holds for ~10 min after the last one (see
    ``opym.stream.protocol`` and ``opym.lanes.WARM_LEASE_MAX_AGE_S``).
    """

    shape_zyx: list[int]
    z_step_um: float


class HeartbeatHeader(TypedDict, total=False):
    """``HEARTBEAT`` header -- client -> server every few seconds.

    Only once an ACK advertised ``"heartbeat"``. Tells Argus this app is
    alive even while nothing new reaches it (a slow time-lapse, or a link
    that can't carry data), and what it holds: Argus then keeps the session
    open, and says why when it does close it.
    """

    state: Literal["acquiring", "finishing"]
    links_up: int
    links_total: int
    unacked_bytes: int
    unacked_volumes: int
    submitted: int
    acked_through: int
    stalled_s: float
    last_error: str


class ClientLogHeader(TypedDict, total=False):
    """``CLIENT_LOG`` header -- this app's own stream log lines.

    Appended to the day's rig log on Argus (once an ACK advertised
    ``"client_log"``).
    """

    lines: list[str]
    host: str


MessageHeader = (
    SessionStartHeader
    | FrameHeader
    | SessionEndHeader
    | AckHeader
    | ResumeHeader
    | QCHeader
    | PrepareHeader
    | HeartbeatHeader
    | ClientLogHeader
)


# --- FRAME time -------------------------------------------------------------

# A FRAME `timestamp` must be epoch seconds. This window is the same one
# opym.timeline uses to tell a real time from a missing one (1e9 s is
# 2001-09-09). Its top rejects epoch milliseconds (1.8e12 today); the bottom
# rejects "milliseconds since the run started" (a run would have to last 11.6
# days to reach 1e9 ms) and 0.0.
EPOCH_MIN_S = 1e9
EPOCH_MAX_S = 1e11


def epoch_s_or_none(value: object) -> float | None:
    """``value`` as a float if finite and inside the epoch-seconds window, else None.

    Accepts Python and numpy int/float scalars (numpy registers them as Real).
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    v = float(value)
    return v if math.isfinite(v) and EPOCH_MIN_S <= v < EPOCH_MAX_S else None


def frame_epoch(header: dict[str, Any]) -> tuple[float | None, str]:
    """The acquisition time of a FRAME's volume, and where it came from.

    Returns ``(epoch_s, source)`` with source ``"timestamp"``,
    ``"acq_first_s"`` or ``"none"``. Receivers use the source to log once per
    session when the fallback is in use.
    """
    ts = epoch_s_or_none(header.get("timestamp"))
    if ts is not None:
        return ts, "timestamp"
    first = epoch_s_or_none(header.get("acq_first_s"))
    if first is not None:
        return first, "acq_first_s"
    return None, "none"


def frame_epoch_s(header: dict[str, Any]) -> float | None:
    """The FRAME's acquisition time in epoch seconds, or None.

    ``timestamp`` is epoch seconds (STREAMING_PROTOCOL.md). The acquisition
    PC used to send milliseconds since the run started there (always 0.0 in
    some paths), which are not a time of day. A value outside
    [EPOCH_MIN_S, EPOCH_MAX_S) is therefore not trusted: the header's
    ``acq_first_s`` (epoch seconds of the volume's first plane, sent by every
    rig since 2026-09-25) is used instead, and if that is unusable too the
    result is None and the caller records no time. Both new and old senders
    work with both new and old receivers.
    """
    return frame_epoch(header)[0]


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
    import msgpack  # lazy: the module imports with numpy only; packing needs [stream]

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
    import msgpack  # lazy, see pack_message

    if len(parts) not in (3, 4):
        raise ValueError(f"Expected 3 or 4 message parts, got {len(parts)}")
    msg_type, session_id_bytes, header_bytes, *rest = parts
    if msg_type not in _VALID_TYPES:
        raise ValueError(f"Unknown message type: {msg_type!r}")
    header = msgpack.unpackb(header_bytes, raw=False)
    payload = rest[0] if rest else None
    return msg_type, session_id_bytes.decode("utf-8"), header, payload
