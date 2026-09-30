# Changelog

All notable changes to `opym` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [Semantic Versioning](https://semver.org/) (pre-1.0: minor releases may
change behaviour).

## [Unreleased]

## [0.4.0] - 2026-09-30

The headline of this release is the **live lane**: a streamed acquisition is now
received, deconvolved, deskewed and shown in napari one timepoint at a time,
keeping pace with the microscope (about 0.6 s from the last plane of a
timepoint to its painted frame on the test stack). Everything else is the
supporting work: a real-time stream receiver, one processed OME-Zarr format,
a supervised two-GPU server pool, and a batch backfill that the live lane
hands off to.

Numbers in parentheses are pull requests. Work from July to mid-September 2026
predates the pull-request workflow in this repo (it was committed directly to
`main`, or merged in bulk by #5), so it is described by theme.

### Added

**Real-time stream receiver** (`opym.stream`)
- `opym-receive`: a ZeroMQ ROUTER server that takes frames from the
  acquisition PC and writes each channel straight into a raw OME-Zarr mirror
  store, so the backfill discovers a streamed dataset exactly like a transferred
  one. Sessions survive reconnects (DEALER identity plus `RESUME`/`ACK`), accept
  duplicate and out-of-order frames, and are confirmed at the end by a final
  `ended` ACK (#4, #5). Protocol spec: `docs/STREAMING_PROTOCOL.md`.
- `opym-stream-watch` and `opym.stream.client.StreamSender`: a portable
  (pure Python) sender, and a watcher that streams a local pymmcore MDA zarr
  store timepoint by timepoint (#5).
- Wire features an ACK advertises and a client may then use: `slabs` (send a
  volume as z-slabs while it is still being acquired) (#18), `blosc`
  (lz4+bitshuffle payloads, about 2.5x on camera frames) (#20), `links`
  (several parallel connections per session, past the single-SSH-tunnel
  bandwidth cap) and `resume` (re-open a session after a receiver restart)
  (#24).
- `PREPARE` message: the client announces a planned shape while the MDA is being
  set up, and the receiver warms a GPU server for it so the first timepoint
  costs what the others do (#32).
- `QC` message: live quality-control verdicts forwarded to clients that opt in
  (`accepts: ["qc"]`) (#10).
- Optional direct 10 GbE endpoint (`OPYM_STREAM_DIRECT_BIND`), refused unless an
  IP allowlist and a raw-root allowlist are both set (#18).
- Opt-in RAM-disk staging (`OPYM_STREAM_STAGE_ROOT`) with a background drain
  that copies each finished session to its final location and verifies it byte
  for byte before making it visible (#4). A free-space floor
  (`OPYM_STREAM_STAGE_FLOOR_GB`, default 20) evicts drained sessions early and
  holds frames back rather than filling the RAM disk; the receiver asks the
  client to resend as soon as there is room again (#33, #36).
- Paused runs: a client that cannot reach the receiver for longer than its
  buffer holds ends the session as `paused`, and the partial copy is not
  promoted to the final location (#24).
- `python -m opym.stream.linkbench`: a sink for benchmarking the links from the
  acquisition PC (#24).
- End-to-end latency tracing (`opym.stream.trace`) and the `opym-live-trace`
  report: per-hop p50/p95 from last plane acquired to frame painted (#17, #26,
  #34).

**Live lane** (per-timepoint decon + deskew, ahead of the backfill)
- Priority lanes (`opym.lanes`): a `queue_live/` claimed before the backfill
  `queue/`, a live lease that holds the backfill off while an acquisition is
  open, and preemption of a backfill server when live work has no GPU (#6).
- The live lane: each streamed timepoint is deconvolved and deskewed/rotated as
  soon as its frames land, with output bit-identical to the batch path (#7).
  Deconvolution and deskew settings now live in one place
  (`opym.decon_config`) shared by the live and batch paths, and deskew
  interpolation is linear (the CPU cubic path was 13.6 s per frame).
- One-format live pipeline (`OPYM_LIVE_FORMAT=zarr`): one ticket per
  (timepoint, channel) reads the raw OME-Zarr, deconvolves and deskews in memory
  and writes a processed OME-Zarr in bioformats2raw layout 3 (series 0: DSR
  pyramid, series 1: Z-MIP, with OME-XML). No intermediate TIFFs. Adds the
  `ome-types` dependency and MATLAB mex readers/writers for N-D zarr (#19).
- Session warm-up and a shared PSF cache, so the first timepoint is not slower
  than the rest (#20, #32).
- Live QC hooks (`OPYM_LIVE_QC=1`): raw-projection sidecars for the QC
  service, its verdicts forwarded to the client, and drawn in the viewer (#10).
- 8-bit view buffers written by the GPU server (`OPYM_LIVE_VIEW_BITS`, default
  8); the processed stores stay 16-bit (#37).

**`naparym-live` viewer** (`opym.live_view`)
- Follows a live acquisition in napari: a 3-D view that opens immediately,
  waits for a session instead of exiting, follows whichever session is newest,
  and shows single-timepoint snaps too (#9, #11, #12).
- Full resolution, read straight from memory-mapped view buffers on the RAM
  disk, with timepoints prefetched around the time slider and each channel
  shown as it lands (#22, #28, #29). 8-bit display by default (#37).
- Live QC overlay: each timepoint's cell box coloured by verdict (#10).
- Experimental GPU texture cache, off by default (`--vram-gb`) (#30).
- `build_parser()` / `open_viewer()` split so another host can reuse the viewer
  window (#41).

**GPU server pool and batch pipeline**
- A supervisor (`opym-serve`) that keeps each GPU's MATLAB server alive on its
  own: independent relaunch with exponential backoff, a dead server's ticket
  requeued (at most twice, then `failed/`), orphaned claims swept, hung servers
  detected and killed, and per-ticket timings written to `profiling/`. Live
  tickets that are claimed for more than 30 s are treated as hung and the ticket
  is requeued for the other GPU (#5, #33).
- Test stacks that run beside production: `PETAKIT_JOBS_DIR` and
  `OPYM_SERVE_SERVERS` isolate the queues and name the servers, and
  `scripts/live_bench` measures the live view end to end and injects faults
  (#26, #27, #31, #34, #35).
- Bulk backfill building blocks: recursive dataset discovery, a SQLite status
  registry with triage flags and expected/actual timepoint counts, ROI
  auto-detection, pre-cropped per-channel zarr acquisitions, and output
  mirroring when a raw directory is not writable (`OPYM_OUTPUT_MIRROR_ROOT`)
  (#5).
- Decon re-enabled in the deskew/decon job pipeline: OMW (Wiener-Butterworth)
  back-projector with `wiener_alpha`, `otf_cum_thresh`, `hann_win_bounds` and
  damp-factor controls, per-channel PSFs, edge erosion, and a provenance
  fingerprint that records which decon settings produced an output (#5).
- A `pipeline_batch` job type that amortises GPU-lock and dispatch cost across
  frames sharing a PSF, and a golden-reference regression test for the GPU
  pipeline.

### Changed

- Decon runs on FFT-friendly sizes: each axis is padded to a size whose prime
  factors are 2, 3, 5 or 7 (161 to 162 planes), then cropped back. Live and
  batch both pad; differences from unpadded output are at most 14 counts at a
  few voxels (#38).
- Session names: two streamed sessions can never share storage (#8). A name
  reused within one experiment folder now keeps the microscope's name: the
  earlier run is moved, not deleted, to `.superseded/` (which discovery skips)
  and forgotten in the registry, and only falls back to a `_001` suffix while
  that run may still be in use (#42). Staging is namespaced per destination
  folder.
- Session admission: a session is accepted when the RAM disk has room for
  itself plus the floor (previously twice its size), and one that is turned
  down is answered with the reason so the client can retry (#40).
- Channel 2 is magenta, not red, in OME-Zarr metadata (#23).
- Consolidated OME-Zarr stores are physically (T, C, Z, Y, X) on disk, not just
  labelled that way; `scripts/fix_ome_zarr_axes.py` migrates older stores.
- Default `rl_method` for decon jobs is `omw` (was `simple`).
- Live lane polling is tighter: servers poll the live queue every 20 ms while a
  live lease is held, the receiver pumps the lane every 100 ms, and
  `naparym-live` polls every 100 ms (#20).
- New dependencies: `pyzmq`, `msgpack` (stream), `ome-types>=0.6` (OME-XML), and
  a `zarr<3` pin.
- `opym-live-trace` times a timepoint by its last channel on screen and adds
  per-channel rows (#34).
- `docs/live-view-pipeline.md` walks one timepoint from camera to screen with
  measured timings and the reasons behind each choice (#25, #39).

### Fixed

- Deskew geometry for pre-cropped zarr datasets: they were deskewed with
  transposed axes and a default z step while reporting success. The input axis
  order is now `zxy` and the z step is read from each store's own `z`
  coordinate array.
- `rl_method='simple'` matched nothing in PetaKit5D's decon switch and wrote
  empty volumes. It is normalised to `simplified`, and unknown names raise. The
  standalone decon job type also never received its PSF.
- A `rmdirs` shim for a PetaKit5D typo that made decon re-runs fail when an
  eroded-mask directory already existed.
- Stream receiver: a final ACK is sent on `SESSION_END` (#4); reconnects with an
  existing identity were dropped about half the time until `ROUTER_HANDOVER`
  was set; a receiver restart mid-run no longer wedges the client or the live
  view (#24, #33); a single-timepoint live session named its staged file wrongly
  (#13); drained sessions are evicted by exactly one worker (#5).
- Two acquisitions were lost when they reused the name of an earlier test:
  the receiver reopened the test's store, rejected every frame, and the drain
  re-copied the stale test (#8).
- Live lane races and crashes found in the first production sessions: two
  servers building the same erosion mask, and a same-name re-run deleting
  another run's staged frames (#9); `naparym-live` failing on a store that did
  not exist yet (#14), and showing blank or stale data for newly written
  timepoints (#15, #16).
- Tickets are claimed with one `rename(2)`. MATLAB's `movefile` lost the
  winner's claim in 7 of 150 contested claims (#21).
- A killed backfill ticket's half-written outputs are swept before it is
  requeued (#35).
- Backfill and GPU pipeline: stalled workers on large single-timepoint
  reference projections, empty working-directory detection, uncompressed
  (`compressor: null`) zarr reads, discovery that wrongly required
  `MDA_settings.yaml`, a stale master file after re-discovery, a fire-and-forget
  GPFS copy that could truncate output, and the watchdog counting orphaned
  `.active_` tickets as new work. MATLAB launch failures back off instead of
  hot-looping.
- The GPU tests ran the shared PetaKit5D functions instead of opym's `patches/`
  that the servers use; the MATLAB engine now resolves the patched copies (#38).

## Before 0.4.0

`opym` began in September 2025 as a Python package for cropping and
deskewing dual-camera OPM data: interactive ROI selection in a notebook, a
command-line interface for batch cropping, and job tickets for PetaKit5D
(#1). Through mid-2026 it grew PSF extraction and averaging tools, optional
deconvolution, a MATLAB PetaKit5D job server with GPU support, galvo-scan
deskew defaults (60 degree sheet angle), and OME-Zarr consolidation. Version
0.3.0 (July 2026) closed with the unified GPU pipeline decon/DSR fix (#2).
