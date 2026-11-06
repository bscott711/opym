# opym (repo `opym_local`, GitHub `bscott711/opym`)

OPM (oblique plane / light-sheet) processing library. Data flow:
rig PC (pymmcore-gui sender) -> `opym-receive` (raw OME-Zarr) -> live lane / SSH backfill -> PetaKit5D on MATLAB
(deskew, decon) -> processed OME-Zarr `viewer/<base>_dsr.ome.zarr` -> naparym-live, CORE live QC, opym-dashboard.
Read [README.md](README.md) first, then `docs/` (`STREAMING_PROTOCOL.md`, `live-view-pipeline.md`, `SSH_BACKFILL.md`).
Review and plan: [docs/review/2026-11-review.md](docs/review/2026-11-review.md), [docs/ROADMAP.md](docs/ROADMAP.md).

## Run and test

- Use bioimaging's venv, FROM THE REPO ROOT ONLY:
  `../bioimaging/.venv/bin/python -m pytest -m "not gpu"` (389 pass). From a worktree use
  `~/projects/science/bioimaging/.venv/bin/python`. This repo's own `.venv` is stale (numcodecs 0.16 vs zarr 2;
  `ipywidgets` missing); re-sync with `uv sync`.
- Never import test modules from another directory: the conftest isolation is skipped and the lane tests act on
  production paths (one such run deleted the RAM-disk view caches in `/dev/shm/opym_live_view`).
- Do not export `OPYM_STREAM_STAGE_ROOT`, `OPYM_LIVE_VIEW_ROOT` or `PETAKIT_JOBS_DIR` pointing at production while
  testing. With `OPYM_STREAM_STAGE_ROOT` set, 26 stream tests fail.
- GPU tests (`-m gpu`) need a free GPU (`nvidia-smi` first; the two GPUs are shared) and MATLAB needs a licence.
- Lint: `just format`; ruff and mypy are not clean today (see the review).

## Deploy

- Production imports THIS checkout (editable) through `bioimaging`. Pulling main is a partial deploy: pull and
  restart the `opym-*` user units together (`opym-receive`, `-receive-backfill`, `-serve`, `-backfill`), and only
  when asked. Units live in `bioimaging/deploy/`, installed in `~/.config/systemd/user`.
- Units set `OPYM_LIVE_FORMAT=zarr` (default is the old TIFF lane), `OPYM_LIVE_LANE=1`, `OPYM_LIVE_QC=1`,
  `OPYM_DECON_PSF`, `OPYM_STREAM_STAGE_ROOT=/dev/shm/opym_stream_stage`, `OPYM_BACKFILL_REGISTRY_PATH`, the
  `OPYM_SERVE_*` knobs, and `OPYM_BACKFILL_RAW_ROOTS` / `_SETTLE_SECONDS` for the backfill receiver.
  Full table: README "Environment variables".
- Ports: 5555 receiver, 5556 backfill receiver (both loopback). Queues, leases and claims: `/dev/shm/petakit_jobs`.
- naparym-live is a separate deployed copy pinned to an opym git rev; changing `live_view.py` or the display record
  needs a pin bump there.

## Gotchas

- Channel 2 is magenta everywhere (viewers, OME metadata, movies); never red-green.
- The rig sender lives in pymmcore-gui (`_argus_stream/`). Its `backfill_send.py` is the sender the rig runs; opym's
  `stream/backfill_client.py` is a diverged copy used by tests. `bfill_run.py` must stay identical in both repos,
  and `bfill.py` must match pymmcore-gui's `_bfill.py`.
- Times: the rig sends ms since run start, not epoch seconds, so streamed stores have no real times yet (ING-F1).
  `opym-add-times` is dry run unless `--apply`.
- Handedness: `DSR_nodecon` (plain transpose) is a y-mirror of decon and live output (PROC-F1). Undecided; do not
  move masks between them.
- sqrt8 contract with naparym-live: 8-bit view buffers hold `round(255*sqrt(v/vmax))`, `vmax` = 2^k-1 fixed per
  channel and session by the first volume; encode (MATLAB LUT), decode (GPU shader) and `DisplayEncoding` must agree.
- The receiver's idle-close text ("closed on Argus ...", `receiver._IDLE_CLOSED_REASON`) is string-matched by the rig;
  do not reword it.
- Do not delete datasets under `/mmfs1` or `/mmfs2` (the PI decides). Tests and scripts must not write there.
- Do not run `launch_petakit_servers.sh` (starts unsupervised servers on the production queue).

## Workflow

Branch, test, PR into `bscott711/opym` main (never an upstream), merge it yourself, then pull main. Deploy only when
the task says so. No time or effort estimates in plans or docs.
