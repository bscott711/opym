# Real-time frame-streaming protocol (acquisition workstation -> Argus)

This is the handoff spec for whoever implements the Windows/Python
acquisition-side client. The receiver (`opym-receive`, implemented in
`opym.stream.receiver.StreamReceiver`) is the counterpart on Argus. The
canonical machine-readable version of everything below is
`opym_local/src/opym/stream/protocol.py` — if this doc and that file ever
disagree, the code wins.

A reference sender (`opym.stream.client.StreamSender`, the DEALER-socket
mechanics below already implemented) and a bolt-on local-store watcher
front end (`opym.stream.client.watch_and_stream` / the `opym-stream-watch`
entry point) already exist and are pure-Python-portable (no MATLAB/GPU
deps) — a custom client can import and reuse `StreamSender` directly
instead of reimplementing this protocol's send/ack/retry mechanics from
scratch. The watcher polls a local pymmcore MDA zarr store for newly
completed timepoints; a future in-process `frameReady` hook would share the
same `StreamSender` for lower latency, pushing volumes straight from
memory instead of polling disk.

## Why this exists

Argus's existing pipeline (see the root `bioimaging/CLAUDE.md`) is entirely
batch/file-based: it discovers *finished* acquisitions on a GPFS mount, today
after a Globus transfer lands them there. This protocol is a real-time
replacement for that transfer step, not for the pipeline itself: each
channel's frames are written directly into a GPFS-resident raw OME-Zarr
mirror store (`opym.stream.rawmirror`) in the exact same directory-per-channel
layout a completed Globus transfer already produces. `opym-backfill --watch`
(already running independently, polling every 120s) discovers the finished
dataset on its own next pass and submits its one deskew/decon ticket exactly
as it would for a Globus-landed dataset — nothing downstream of this protocol
changes or needs to know the data arrived over the network instead.

When decon is enabled on Argus (the `OPYM_DECON_PSF` env var the batch
backfill driver already reads), the receiver *also* pre-stages each frame as
a decon-ready TIFF, so that step is a no-op by the time the batch pass gets
to it. This is a latency optimization only — decon parameters themselves
(PSF, wiener_alpha, edge_erosion) are resolved entirely server-side by the
batch driver's own fixed configuration, not by anything this protocol's
client declares.

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
    "raw_root": "/mmfs2/scratch/.../DataUpload/20260920-session/",  # GPFS -- what
                                       # Globus would drop this session into today
    "dtype": "uint16",
    "shape_zyx": [64, 512, 2048],      # per-volume raw (Z, Y, X)
    "num_timepoints": 200,
    "channels": [0, 1],
    "channel_names": ["mScarlet_561", "GFP_488"],  # see convention note below
    "z_step_um": 0.3,
}
```

If your acquisition engine is pymmcore-plus/useq-schema based (Argus's
existing MDA writer already is — see `opym.metadata.parse_mda_settings`),
the natural source for most of these fields is the `useq.MDASequence` you
already have in memory at the start of an MDA run.

**`channel_names[i]` must be the `"<ChannelName>_<Wavelength>"` suffix
`opym.discovery` expects** (e.g. `"GFP_488"`, `"mScarlet_561"`) — the
receiver names each channel's raw store
`"<base_name>_<channel_names[i]>.ome.zarr"`, and that exact naming is what
lets `opym.discovery.group_channel_zarr_stores` re-group the finished stores
back under `base_name` during backfill discovery, same as it would for a
Globus-landed acquisition.

These are the only fields the receiver's raw-mirror write path actually
needs. Decon config (PSF, `wiener_alpha`, `edge_erosion`, `rl_method`) is
**not** part of this handshake — see "Why this exists" above. A client may
still send `sheet_angle_deg`, `xy_pixel_size`, `t_interval_s`, `psf_paths`,
etc. for its own logging/provenance; the receiver ignores anything it
doesn't need.

**If you do send `sheet_angle_deg`, it is not a free parameter — use
`60.0`.** That's the validated production value for this OPM, used as the
default everywhere else in this codebase (`opym.petakit`,
`run_petakit_server.m`, `run_napari_opym.py`, every `psf_tools/*` script) and
explicitly called out as such in
`bioimaging/psf_tools/omw_rl_comparison.py`'s `--sheet-angle-deg` help text.

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

Optional latency-trace fields, all epoch seconds on the client's own clock:
`acq_first_s` / `acq_last_s` (first / last plane of the volume acquired),
`queued_s` (handed to the sender), `sent_s` (handed to the socket) and
`clock_offset_s` (server minus client clock, estimated from the round trip
of `SESSION_START` and the first `ACK`'s `server_time_s`). The receiver
records them in `profiling/live_trace.jsonl`; `opym-live-trace` turns them
into per-hop latencies, including the time on the wire.

#### Slabs (stream a volume while it's being acquired)

Once an `ACK` has listed `"slabs"` in its `features`, a client may send each
volume as several `FRAME`s, each carrying a run of planes:

```python
header = {
    "t": 17, "c": 0, "frame_index": 71,
    "z0": 24, "nz": 161,              # planes [24, 36) of a 161-plane volume
    "shape_zyx": [12, 490, 1458],     # THIS slab's shape
    "dtype": "uint16", "timestamp": ..., "camera_id": 0,
}
payload = volume[24:36].tobytes()
```

The receiver writes each slab into the raw store as it arrives, and stages
the volume once its last plane has landed. A volume's slabs are ACKed only
at that point, so a receiver restart mid-volume makes the client resend the
whole volume. ACKs keep arriving meanwhile (with an unchanged
`through_frame_index`), so a slow volume never looks like a dead link.
About 16 MB per slab keeps the per-message overhead negligible.

### 3. `SESSION_END`

```python
header = {"reason": "complete"}  # or "client_abort"
```

Just marks the session finished on the receiver and logs how many `(t, c)`
pairs arrived — there is nothing to stitch or consolidate (see "Why this
exists" above): each channel's frames were already written straight into
their final-shaped raw mirror store as they arrived. `opym-backfill --watch`
discovers the dataset on its own; nothing needs to be told to look at it. If
you go silent for 10 minutes without sending this, the receiver finalizes
the session for you with `reason: "idle_timeout"` — whatever arrived by then
is still a valid (if truncated) dataset, handled the same way an aborted
Globus-landed acquisition already is (`channel_store_timepoints` in
`bioimaging/backfill/pipeline.py`).

### 4. `ACK` — server -> client, unsolicited, not per-frame

```python
header = {"through_frame_index": 33, "server_time_s": 1755000000.456}
```

`features` lists what this receiver supports beyond the base protocol;
only `"slabs"` is defined so far. A client must not send slabs to a
receiver that hasn't advertised them.

`server_time_s` (the server's clock when it sent the ACK) is informational:
clients use it only to estimate their clock offset for the trace fields.

Sent periodically (every ~10 frames or ~2s, whichever first) once frames up
to `through_frame_index` are durably written into their channel's raw mirror
store on GPFS. `-1` means nothing processed yet. **This is your resumability
signal**: keep a bounded local ring buffer of sent-but-unacked frames, and
drop entries once their `frame_index <= through_frame_index`.

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

### 6. `QC` — server -> client, only if the client asked for it

Opt in by adding `"accepts": ["qc"]` to `SESSION_START`. A client that doesn't
list it is never sent anything but `ACK`, so older clients are unaffected
(the reference `unpack_message` rejects message types it doesn't know).

```python
header = {
    "seq": 41, "session_id": "...", "t": 20, "stage": "raw",
    "verdict": "act",                       # ok | warn | act | no_cell
    "flags": ["clipped_depth_high", "drift_exit_soon"],
    "advice": [{"action": "focus_offset", "axis": "depth", "direction": "+",
                "amount_um": 3.0, "when": "now", "text": "..."}],
    "metrics": {...},
}
```

One message per new verdict from the live QC service (CORE's `celldet-live-qc`),
forwarded unchanged. It is advisory: the server does not wait for a reply.
There are two verdicts per timepoint. `stage: "raw"` arrives about a second
after the timepoint's frames land and covers coverage, drift, focus and signal.
`stage: "dsr"` follows the deskewed volume and adds the cell's bounding box.
Ignore fields you don't recognize.

Where it comes from: with `OPYM_LIVE_QC=1` the live lane writes projections of
every raw frame to `<leaf>/qc/rawproj/<base>_C<c>_T<ttt>.npz`
(`opym.stream.qcproj`). The QC service reads those and writes
`<leaf>/qc/live_qc.jsonl` (every verdict) and `<leaf>/qc/qc_latest.json` (the
newest). The receiver forwards the newest one.

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

(For context — not something the client needs to implement. See
`opym.stream.receiver`'s module docstring for the full rationale.)

1. On a channel's first frame, creates
   `<raw_root>/<base_name>_<channel_names[i]>.ome.zarr` — zarr v2, one
   z-plane per chunk, plus a `z` coordinate array derived from `z_step_um`
   (`opym.stream.rawmirror.create_channel_store`) — the same layout
   `opym.discovery` and `opym.metadata.parse_zarr_z_step_from_store` already
   read from a Globus-landed acquisition.
2. Writes the raw `(Z, Y, X)` volume into that store at index `t`, exactly
   as received — **no** `orient_zyx_for_dsr` rotation; that's applied
   downstream by the batch path's own deskew-mirror step, not here.
3. If decon is enabled on this host (`OPYM_DECON_PSF` set), also stages the
   same volume as a decon-ready TIFF (`opym.utils.write_decon_staged_tiff`,
   the `orient_zyx_for_decon_tiff` rotation) into
   `<raw_root>/<base_name>/decon_stage/`, named to exactly match what
   `bioimaging.backfill.pipeline.build_decon_staging_dir` would independently
   produce, so that step's skip-if-exists check treats it as already done.
4. Submits no ticket and stages nothing under `/dev/shm/` — `opym-backfill
   --watch` finds the dataset through its normal GPFS walk once
   `SESSION_END` (or the idle timeout) finalizes it, and submits the one
   deskew/decon ticket for the whole thing, unmodified.

## Direct endpoint (10 GbE, no SSH tunnel)

The SSH tunnel tops out around 33 MB/s. That is ~13 s to move one
161-plane, 2-channel timepoint, and was the largest delay in the live view on
2026-09-25. `opym-receive` can also listen on the 10 GbE interface itself:

| env var | example | meaning |
|---|---|---|
| `OPYM_STREAM_DIRECT_BIND` | `tcp://137.216.250.14:5556` | the direct endpoint |
| `OPYM_STREAM_ALLOW_IPS` | `10.x.y.z` | comma-separated; every other host is refused (ZMQ ZAP) |
| `OPYM_STREAM_RAW_ROOTS` | `/mmfs1/scratch/jacks.local/microscopy` | comma-separated; sessions on this endpoint may only write under these |

The receiver will not bind the endpoint unless both allowlists are set. The
link is plain TCP, not encrypted: CurveZMQ reached only 85 MB/s with pyzmq's
bundled libzmq, versus ~3.9 GB/s plain. The site firewall admits only the
acquisition PC on this port. The loopback (SSH tunnel) endpoint is
unchanged, and it stays the client's fallback. A session follows its client
if it reconnects through the other endpoint.
