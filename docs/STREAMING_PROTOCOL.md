# Real-time frame-streaming protocol (acquisition workstation -> Argus)

This is the handoff spec for whoever implements the Windows/Python
acquisition-side client. The receiver (`opym-receive`, implemented in
`opym.stream.receiver.StreamReceiver`) is the counterpart on Argus. The
canonical machine-readable version of everything below is
`opym_local/src/opym/stream/protocol.py` — if this doc and that file ever
disagree, the code wins.

## Why this exists

Argus's existing pipeline (see the root `bioimaging/CLAUDE.md`) is entirely
batch/file-based: it discovers *finished* acquisitions on a GPFS mount. This
protocol is the first real-time ingress into that same pipeline — frames land
in `/dev/shm/opym_jobs/` and get ticketed into `/dev/shm/petakit_jobs/queue/`
exactly like the batch path already does, just one frame at a time as they
arrive instead of one dataset at a time after the fact. Everything downstream
of the ticket queue (the MATLAB PetaKit5D watchdog, Decon/DSR, OME-Zarr
consolidation) is unmodified and doesn't know the difference.

## Transport

**ZeroMQ, `DEALER` (client) <-> `ROUTER` (server).** Not PUSH/PULL — PUSH is
one-way and can't carry acks back, which this protocol needs for resumable
reconnects. DEALER/ROUTER is still fully async (no request/response lockstep
like REQ/REP), but the server can push ack messages back on the same
connection whenever it wants.

Argus is not directly reachable from the acquisition workstation — it's only
reached via SSH/DCV. The receiver binds to `127.0.0.1` on Argus (never a
public interface), and the workstation reaches it through a local SSH port
forward over whatever SSH connection is already in use:

```
ssh -L 5555:127.0.0.1:5555 <user>@<argus-host>
```

Then the client connects its DEALER socket to `tcp://127.0.0.1:5555` (on the
*workstation's* side of the tunnel).

**Critical: set the DEALER socket's identity before connecting.**

```python
import zmq

ctx = zmq.Context()
sock = ctx.socket(zmq.DEALER)
sock.setsockopt(zmq.IDENTITY, session_id.encode("utf-8"))  # before connect()
sock.connect("tcp://127.0.0.1:5555")
```

This identity is how the server recognizes a reconnect as the *same*
logical session rather than a brand-new one — a fresh TCP connection still
presents the same identity, so `session_id` must be stable for the lifetime
of one MDA run (mint a fresh UUID per run, not per connection).

Set a bounded high-water mark on the socket (`zmq.SNDHWM`, a few seconds'
worth of volumes) — once the receiver can't keep up, `send()` blocks. That's
the entire flow-control mechanism; there's no separate credit/backpressure
message.

## Message format

Every message is a ZMQ multipart message:

```
[msg_type, session_id, msgpack_header, raw_payload?]
```

`raw_payload` is present only on `FRAME` messages. ZMQ frames each part
itself — there's no additional length-prefixing to implement. Headers are
`msgpack`-packed dicts (`msgpack.packb(header, use_bin_type=True)`).

## Message types

### 1. `SESSION_START` — sent once, before any frames

```python
header = {
    "base_name": "cell_042",
    "output_dir": "/mmfs2/scratch/.../cell_042/Decon",  # GPFS, final Decon/DSR output
    "dtype": "uint16",
    "shape_zyx": [64, 512, 2048],   # per-volume raw (Z, Y, X)
    "num_timepoints": 200,
    "channels": [0, 1],
    "channel_names": ["mScarlet_561", "GFP_488"],
    "z_step_um": 0.3,
    "xy_pixel_size": 0.116,
    "sheet_angle_deg": 60.0,
    "t_interval_s": 2.5,
    "interp_method": "cubic",       # optional, default "cubic"
    "rl_method": "simple",          # optional, default "simple"
    "iterations": None,             # optional
    "psf_paths": ["/mmfs2/.../PSF/561_psf.tif", "/mmfs2/.../PSF/488_psf.tif"],
    "dz_psf": 0.1,                  # PSF's own z-step; see note below
}
```

If your acquisition engine is pymmcore-plus/useq-schema based (Argus's
existing MDA writer already is — see `opym.metadata.parse_mda_settings`),
the natural source for most of these fields is the `useq.MDASequence` you
already have in memory at the start of an MDA run.

Omit or empty `psf_paths` to get deskew-only (no deconvolution). `dz_psf` is
technically optional if `psf_paths` is given — the receiver falls back to
reading it from the PSF file's own ImageJ `spacing` tag — but that read
happens per-frame on the hot path here, so send it explicitly if you have
it.

**`sheet_angle_deg` is not a free parameter — use `60.0`.** That's the
validated production value for this OPM, used as the default everywhere
else in this codebase (`opym.petakit`, `run_petakit_server.m`,
`run_napari_opym.py`, every `psf_tools/*` script) and explicitly called out
as such in `bioimaging/psf_tools/omw_rl_comparison.py`'s
`--sheet-angle-deg` help text. Every other value in the example above is
just illustrative; this one isn't.

**Multi-camera acquisitions — there is no separate camera axis, by
design.** `channels`/`c` is the only channel-identity key the receiver
addresses staging and tickets by (`camera_id` in the `FRAME` header below
is descriptive metadata only — it is never used for staging, ticketing, or
dedup). This OPM is always dual-camera, so resolve each physical
camera's output into its own distinct `c` index client-side, exactly like
the existing pre-cropped zarr batch writer already does (see
`opym.discovery`'s `KIND_ZARR_PRECROPPED` docstring: the newer
pymmcore-based MDA writer "crop[s] and split[s] one-per-channel at capture
time," and everything downstream — PetaKit5D tickets, `consolidate.py` —
only ever sees flat channel indices, never a camera dimension). A
2-camera, single-excitation session declares e.g.
`"channels": [0, 1], "channel_names": ["Cam0_mScarlet_561",
"Cam1_GFP_488"]` and sends each camera's frames under its own `c`; a
setup with more excitations/sub-channels per camera just extends the same
flat list further. There's nothing to add to the protocol for this —
`SessionState.channels` in `receiver.py` is already an arbitrary-length
list with no hardcoded channel-count assumption anywhere in the receiver.

The server replies with an `ACK` (`through_frame_index: -1`) once the
session is registered.

### 2. `FRAME` — one per acquired volume

```python
header = {
    "t": 17, "c": 0,
    "frame_index": 34,          # monotonic per real acquired frame, stable
                                  # across reconnects — NOT the same as t/c
    "timestamp": 1755000000.123,
    "camera_id": 0,
    "shape_zyx": [64, 512, 2048],
    "dtype": "uint16",
}
payload = volume.tobytes()      # C-contiguous (Z, Y, X)
```

`t`/`c` are your acquisition's own indices, not send order — frames may
arrive in any `(t, c)` order (e.g. the two cameras finishing a Z-stack at
different times) and the receiver stages/tickets by those indices, not by
arrival order. Resending an already-staged `(t, c)` (e.g. after a
reconnect) is a safe no-op, not a duplicate ticket. `camera_id` is carried
along for logging/debugging only — see the multi-camera note above:
`c` (matched against `SESSION_START`'s `channels`) is what actually
identifies which camera+excitation this volume belongs to.

### 3. `SESSION_END`

```python
header = {"reason": "complete"}  # or "client_abort"
```

Triggers the `.opym_consolidate.json` sidecar write that the existing
`opym.consolidate.run_pending_consolidations()` (already polled every
watchdog cycle) picks up to stitch the final OME-Zarr. If you go silent for
10 minutes without sending this, the receiver does it for you with
`reason: "idle_timeout"` and consolidates whatever arrived.

### 4. `ACK` — server -> client, unsolicited, not per-frame

```python
header = {"through_frame_index": 33}
```

Sent periodically (every ~10 frames or ~2s, whichever first) once frames up
to `through_frame_index` are durably staged *and* ticketed. `-1` means
nothing processed yet. **This is your resumability signal**: keep a bounded
local ring buffer of sent-but-unacked frames, and drop entries once their
`frame_index <= through_frame_index`.

### 5. `RESUME` — client -> server, right after reconnecting

```python
header = {}
```

Send this immediately after a fresh `connect()` on the same `session_id`
identity. The server replies with an `ACK` reflecting whatever it actually
has for that session (`-1` if it doesn't recognize the session at all — e.g.
the receiver process itself restarted; session state isn't persisted to
disk in v1). Resend everything in your local buffer with
`frame_index > through_frame_index`, in order.

## Worked example (client-side pseudocode)

```python
import zmq
from opym.stream.protocol import (
    MSG_ACK, MSG_FRAME, MSG_SESSION_END, MSG_SESSION_START, pack_message,
)

sock = ctx.socket(zmq.DEALER)
sock.setsockopt(zmq.IDENTITY, session_id.encode())
sock.setsockopt(zmq.SNDHWM, 50)
sock.connect("tcp://127.0.0.1:5555")

sock.send_multipart(pack_message(MSG_SESSION_START, session_id, session_header))

for volume, t, c, frame_index in acquire():
    header = {"t": t, "c": c, "frame_index": frame_index, ...}
    sock.send_multipart(pack_message(MSG_FRAME, session_id, header, volume.tobytes()))
    retry_buffer[frame_index] = (header, volume)
    # drain any pending ACKs non-blockingly, evict retry_buffer entries <= through_frame_index

sock.send_multipart(pack_message(MSG_SESSION_END, session_id, {"reason": "complete"}))
```

On reconnect: create a new DEALER socket with the *same* identity, connect,
send `RESUME`, wait for the `ACK`, then resend everything left in
`retry_buffer` above the acked index.

## What the receiver does with each frame

(For context — not something the client needs to implement.)

1. Reorient `(Z, Y, X)` -> PetaKit5D's `(ny, nx, nz)` layout
   (`opym.utils.orient_zyx_for_dsr`, same transform the batch path applies).
2. `zarr.save_array()` to `/dev/shm/opym_jobs/<base_name>_T{t:04d}_C{c}.zarr`.
3. `opym.petakit.submit_pipeline_job()` — writes a JSON ticket into
   `/dev/shm/petakit_jobs/queue/`, exactly like the batch path.
4. On `SESSION_END`, writes `<output_dir>/.opym_consolidate.json` with the
   full expected `(T, C)` grid so `run_pending_consolidations()` can stitch
   the final OME-Zarr once the MATLAB watchdog has processed every ticket.
