# The live view pipeline

This document follows one timepoint from the camera on the acquisition PC
to the screen on Argus. It covers what runs where, the choices behind each
step, how fast each step is, and what keeps it working when something fails.

It describes the one-format live path (`OPYM_LIVE_FORMAT=zarr`), in
production since 2026-09-25, with 4 stream links since 2026-09-26. The wire
protocol itself is in [STREAMING_PROTOCOL.md](STREAMING_PROTOCOL.md). An illustrated
version of this page (private artifact): https://claude.ai/artifact/LMYs4A8zu5bqdZQUA85RVQ

## The short answer: planes stream, processing waits for the volume

The channels are acquired one after the other. Each timepoint is the whole
488 nm stack (GFP, Camera-1), then the whole 561 nm stack (mScarlet,
Camera-2). The planes of each stack leave the PC while that stack is still
being acquired:

- pymmcore-gui packs each channel's planes into slabs of about 16 MB. A
  490 × 1458 uint16 plane is 1.43 MB, so a slab is about 11 planes.
- Each slab is compressed (blosc lz4 with bitshuffle, about 2.5×) and sent
  on whichever of the 4 links is free, as soon as it fills.
- Argus writes each slab straight into the raw OME-Zarr on the RAM disk.
- When a stack's last plane is exposed, only its final slab (about 6 MB on
  the wire) is still in flight. That is why a full 161-plane volume is on
  Argus 0.09 s (p50) / 0.22 s (p95) after its last plane.

Deconvolution and deskew/rotate need a channel's complete volume, so a
channel's GPU work starts when its own last slab lands. GFP does not wait
for mScarlet: its ticket is queued the moment the 488 stack is complete, and
it is processed while the 561 stack is still being acquired. naparym-live
shows a timepoint once every channel is processed, so the critical path is
the last channel's: from the last 561 plane to the screen.

Real timing (2026-09-02 runs, 301 planes, 10 ms exposure, hardware
triggered, `frame_meta` in the MDA save): 488 stack 10.7 s, 0.83 s gap,
561 stack 10.7 s, a timepoint every 23.1 s. That is about 36 ms per plane,
so a 161-plane stack takes about 6 s. GFP is processed about one stack
before mScarlet, then waits for it.

## The path

```
Acquisition PC (Windows, pymmcore-gui)
  488 stack (Camera-1), then 561 stack (Camera-2): planes of 490 × 1458 uint16
  VolumeAssembler ──> 16 MB z-slabs ──> blosc lz4 ──> 4 SSH links (1 ssh process each)
                                                          │  ~63 MB/s per link
Argus (micro001)                                          v
  opym-receive ──> raw OME-Zarr on /dev/shm/opym_stream_stage (1 chunk per plane)
     │   a channel's volume complete: ACK, QC projections, its own live ticket
     v
  queue_live/ (claimed before the backfill queue; LIVE_LEASE.json holds the backfill off)
     │
  opym-serve: 2 MATLAB PetaKit servers, one per RTX PRO 6000; each ticket goes to the first free one
     read (cpp-zarr N-D mex) -> OMW decon, 2 iterations (GPU) -> linear deskew/rotate
     -> opymWriteLiveOutputs: view buffer .npy FIRST, then the processed store
     │
     ├──> buffers/T<tttt>_C<c>.npy on /dev/shm (uncompressed 8-bit, 419 × 1458 × 833, 0.51 GB;
     │        each channel's shift in buffers/display_C<c>.json)
     │        └──> naparym-live (mmap, full resolution, 3D, 8-bit) ──> DCV ──> your screen
     └──> processed OME-Zarr on /dev/shm (bioformats2raw layout 3: 3 levels + Z-MIP)
              └──> archived per (t, c) to <leaf>/viewer/<base>_dsr.ome.zarr on GPFS

Side paths
  live QC: raw projections -> CORE celldet-live-qc -> verdict -> MSG_QC to pymmcore-gui
  backfill hand-off: .live_status.json says "done live", so the batch pass skips it
  raw drain: at SESSION_END the raw store is copied to GPFS and verified byte for byte
```

Code, in path order:

| Step | Code |
|---|---|
| Slabs, compression, links, resume | `pymmcore-gui/src/pymmcore_gui/_argus_stream/_session.py`, `_tunnel.py` |
| Receive, raw store, ACK, resume | `opym/stream/receiver.py`, `opym/stream/rawmirror.py` |
| Live lane: tickets, reap, archive, hand-off | `opym/stream/live_zarr.py` (on top of `opym/stream/live.py`) |
| Priority and lease | `opym/lanes.py` |
| Server supervisor: relaunch, requeue, preempt | `opym/local_gpu_worker.py` (`opym-serve`) |
| GPU work | `opym/run_petakit_server.m` -> `opym/run_live_zarr.m` |
| Viewer | `opym/live_view.py` (`naparym-live`) |
| Per-hop trace | `opym/stream/trace.py`, `opym/stream/trace_report.py` (`opym-live-trace`) |

## One timepoint, hop by hop

Shape: 161 planes × 490 × 1458, 2 channels, dz 0.5 µm. The table follows
the last channel (561), from its last plane: GFP was processed about one
stack earlier and is already on screen (about 6.5 s ahead of the full
timepoint).

**Measured 2026-09-27** on the test stack (`scripts/live_bench`, run
`lv_fft`): 30 timepoints replayed from Argus at the rig's pace (13.2 s per
timepoint), 8-bit view buffers, FFT-friendly decon, test naparym-live on
DCV. The wire hop is from the PC replays (2026-09-26); on Argus it is ~0.01 s.

| After the last plane | p50 (p95) | What happens |
|---|---|---|
| Last slab on the wire | 0.09 s (0.22 s), PC over 4 links | The final ~6 MB leaves on a free link. |
| Staged, ticket queued | 0.01 s (0.02 s) | The receiver writes the slab, marks (t, 561) complete, and the lane writes its `live_zarr` ticket in the same loop pass. |
| Claimed | 0.02 s (0.03 s) | Servers poll `queue_live/` every 0.02 s while a lease is held. |
| Read, decon, DSR, view buffer | 0.34 s (0.40 s) | read 0.04, OMW decon 0.19 (2 iterations, GPU, z padded 161 → 162), linear DSR 0.03, 8-bit view buffer 0.07. |
| Viewer notices | 0.06 s (0.09 s) | naparym-live polls every 0.1 s. |
| Built | 0.08 s (0.11 s) | The layers stay; the time slider moves and the new channel is re-read in place from the mapped buffer. |
| Painted | 0.08 s (0.09 s) | One 0.5 GB 8-bit texture uploaded. |
| **Last plane → on screen** | **0.60 s (0.65 s)** | Flat: −3.6 ms per timepoint over the run; 60/60 volumes bit-identical, processed and archived. |

Off the view path, after the buffer: the pyramid and Z-MIP are written into
the processed store (0.15 s), the lane archives the chunks to GPFS (0.41 s),
and the live QC verdict reaches pymmcore-gui about 0.9 s after the last
plane.

**Cold start.** If the servers have been idle for an hour
(`PETAKIT_IDLE_TIMEOUT`, 3600 s in production), the first timepoint takes
about 20 s: about 10 s to boot MATLAB and about 9 s to build the PSF cache
and compile. The session warm-up (`<jobs>/live_warmup.json`) and the shared
PSF cache (`<jobs>/psf_cache/`) make every timepoint after that cost what a
steady one does.

## Key choices

| Choice | Why | Rejected |
|---|---|---|
| **Stream slabs during the stack** | Only the last slab is left to send after the last plane. | Sending each volume after its last plane: a whole channel (about 90 MB compressed) still in flight, about 0.35 s on 4 links; about 7 s uncompressed on the old single tunnel. |
| **4 SSH links, each its own ssh process** | sshd's fixed 2 MB channel window caps one link at about 63 MB/s. Separate processes scale linearly (4 links: 255 MB/s). | 4 links in one ssh process (110 MB/s). 8 links (p95 0.39 s, worse than 4). |
| **blosc lz4 + bitshuffle on the wire** | Camera frames compress about 2.5×, and the PC compresses at 1.3 GB/s. | Uncompressed. |
| **Direct 10 GbE (plain TCP, locked to the PC)**: built, not enabled | Would remove SSH entirely. Waits on the PC's static IP and a firewall rule. | CurveZMQ encryption: 85 MB/s with pyzmq's bundled libzmq. |
| **RAM disk (/dev/shm) for everything on the view path** | No GPFS read or write before the view. GPFS gets its copy in the background. | Writing to GPFS first: about 10 s of copy-out before the view (2026-09-25). |
| **One format: OME-Zarr, bioformats2raw layout 3 with OME-XML** | One store serves the viewer, QC, the backfill and the archive. | Intermediate TIFFs (TIFF export exists on demand). |
| **PetaKit kept, bit-identical to the batch path** | The live output is the archive; there is no second pass. A GPU test checks it. | A Python/torch rewrite. |
| **Linear deskew/rotate everywhere** | 0.07 s per frame on the mex path. Correlation 0.9997 with cubic, peaks about 3% softer. | Cubic: 13.6 s per frame on the CPU. |
| **One ticket per (t, c)** | Each channel is processed as soon as its own stack lands: GFP while the 561 stack is still being acquired. | One ticket per timepoint, which would hold GFP until mScarlet is in. |
| **Live queue first, lease, preemption** | Streaming always has a GPU. A backfill server is killed and its ticket requeued when live work has none. | Waiting for backfill tickets, which run for minutes. |
| **View buffer before the store** | The viewer maps the uncompressed volume; it never waits on encoding. | The viewer decoding 1 GB per channel from the compressed store every timepoint. |
| **Full resolution in the viewer** | napari's 3D view only ever renders a multiscale's coarsest level. | Multiscale layers (a quarter of the resolution). |
| **8-bit display, written by the GPU server** | Half the bytes to write, map and upload: last plane → painted 0.73 → 0.63 s p50, scrubbing ~35% faster. Side by side on DCV, 16-bit and 8-bit looked the same: the processed data spans ~60–800 counts, and each channel keeps ≥ 120 grey levels below its display window (a per-session power-of-two shift, `display_C<c>.json`). The stores stay 16-bit. | Converting in the viewer (paint 0.13 → 0.07 s, but the conversion cost as much on the UI thread). |
| **FFT-friendly decon sizes** | cuFFT is slow on large primes (161 = 7 × 23): padding to 162 takes the decon from 0.21 to 0.19 s. Differences ≤ 14 counts at a few voxels; live and batch both pad. | Unpadded (the archives before 2026-09-27). |
| **No VRAM texture cache by default** (`--vram-gb`) | Measured: a cached timepoint still costs ~0.1 s per step (after the upload, not paging), and once dozens of textures exist, misses slow from 0.14 to ~0.4 s. | On by default (the fixes and a residency draw are in; it stays opt-in). |
| **Session warm-up and shared PSF cache; no GPU reset between live tickets** | The first timepoint costs what the others do; cached OTFs and FFT plans are reused. | Resetting the GPU after each ticket (about 0.3 s, and loses the caches). |
| **Claims by one rename(2)** | Atomic between two servers. MATLAB's `movefile` lost the winner's claim in 7 of 150 contested claims. | `movefile`. |
| **Channel 2 in magenta** | The lab's standard: green/magenta, never red/green. | Red. |

## Speed history

| Date | Last plane → view | Largest hop |
|---|---|---|
| 2026-09-25 morning | about 28 s (about 2.2 timepoints behind) | Wire: 13 s through one SSH tunnel at 31–35 MB/s. GPFS copy-out: 10 s. MATLAB: 3 s. napari: 1.5 s. |
| 2026-09-25 evening | 0.62–0.71 s to *viewable* (buffers and store written) | One-format lane and perf round: GPU time per (t, c) 3.4 s → 0.6–1.0 s, server boot 60 s → 10 s. |
| 2026-09-26 | wire 0.09 s p50 / 0.22 s p95 | 4 parallel links. Every replayed volume bit-identical. |
| 2026-09-26 | 2.41 s p50 / 2.98 s p95 to *painted* (PC replay, 100 T) | The viewer, ~80%: layers rebuilt every timepoint. Reworked (in-place layers, per-channel follow): 0.73 s p50 / 0.82 s p95 (local replays; the report counted this wrong until 2026-09-27, see opym#34). |
| 2026-09-27 | **0.60 s p50 / 0.65 s p95** to *painted* | 8-bit view buffers from the GPU server (−0.10 s) and FFT-friendly decon (−0.015 s per ticket). |

## When something fails

| Failure | What happens now |
|---|---|
| A link drops | The client resends what that link carried on the others. |
| The receiver restarts mid-run | The client resumes the session (`resume_through`). The live lane picks up where it was: volumes already on disk count as received, a half-received one is finished from the planes already written, and processed ones aren't redone. Tested: receiver SIGKILLed mid-run, back in 10 s, 60/60 volumes bit-identical, 30/30 processed live. |
| The PC can't reach Argus for longer than its RAM buffer | The run is marked "paused"; the full local save goes to GPFS by Globus instead. |
| SESSION_END | Confirmed by an "ended" ACK, so a run never waits out the receiver's idle timeout. |
| A live ticket fails | Retried once, then that timepoint is left to the batch backfill. |
| A MATLAB server dies | The supervisor requeues its ticket on the next pass (2 s) and relaunches it after 15 s. Tested: SIGKILL mid-ticket, no timepoint delayed. |
| A MATLAB server hangs on a live ticket | Killed 30 s after its claim (`OPYM_SERVE_LIVE_DEADLINE_S`) and the ticket requeued for the other GPU. Tested: SIGSTOP mid-ticket, that one timepoint shown 30 s late, the next on time. Backfill tickets keep the 60 min `OPYM_SERVE_HANG_MIN` check. |
| Both GPUs are on backfill when a session starts | One backfill server is killed at once and its ticket requeued (not counted against its cap), and its half-written outputs removed; the other finishes its ticket, then takes live work. A PREPARE while the MDA is set up does the first part before the run starts. Tested (R6, a real 6-minute backfill ticket on both GPUs): first timepoint 4.1 s without PREPARE, 0.76 s with; the live queue never above 1; both backfill tickets completed afterwards. |
| The RAM disk runs short | Below twice `OPYM_STREAM_STAGE_FLOOR_GB` (20) free, drained sessions are evicted early, oldest first, then finished view stores (never the newest one). Below the floor, frames are held back unACKed; once space is back the receiver asks the client to resend them at once (an ACK with `resend`), or the client pauses if space never returns. Tested (R5): eviction with a session running, 20/20 untouched; hold-back released 40 ms after space returned, 40/40 bit-identical. |
| The live lane's work is lost with the receiver | `.live_status.json` goes stale after 300 s, and the backfill reprocesses the dataset in batch. |
| naparym-live is (re)started mid-session | It follows the newest session and reads timepoints whose buffers are gone from the store. Tested: SIGKILL and relaunch mid-run, showing the newest timepoint with the whole time axis within 15 s; a new session is followed on its own. |

Known gaps:

- A scrub step to a timepoint already in memory still takes ~0.1 s after
  its upload is done (napari's re-slice and draw), even from a texture in
  VRAM.
- The paused path (the client's buffer overflows while the RAM disk stays
  full) is unit-tested only.
- `test_deskew_only_regression` fails: its no-decon peaks don't match the
  decon-included golden, and its own golden was never committed.

## Measuring it

`opym-live-trace [--session ID | --base NAME] [--jobs DIR] [--out DIR]` joins the receiver's and
lane's trace (`profiling/live_trace.jsonl`), the GPU servers' profiles
(`profiling/S<id>.jsonl`) and naparym-live's events
(`profiling/live_view.jsonl`) into p50/p95 per hop.

Without the microscope, stream a saved run from S:/ through the real sender
and links on the PC:

```
uv run python -m pymmcore_gui._argus_stream.replay "S:\...\<run>_GFP_488.ome.zarr" --links 4
```

Point it at a test receiver, never production, unless everyone agrees.
