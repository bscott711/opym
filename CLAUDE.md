# opym (repo `opym_local`, GitHub `bscott711/opym`)

OPM (oblique plane / light-sheet) processing library. Data flow:
rig PC (pymmcore-gui sender) -> `opym-receive` (raw OME-Zarr) -> live lane / SSH backfill -> PetaKit5D on MATLAB
(deskew, decon) -> processed OME-Zarr `viewer/<base>_dsr.ome.zarr` -> naparym-live, CORE live QC, opym-dashboard.
Read [README.md](README.md) first, then `docs/` (`STREAMING_PROTOCOL.md`, `live-view-pipeline.md`, `SSH_BACKFILL.md`).
Review and plan: [docs/review/2026-11-review.md](docs/review/2026-11-review.md), [docs/ROADMAP.md](docs/ROADMAP.md).

## Run and test

- Use bioimaging's venv, FROM THE REPO ROOT ONLY:
  `../bioimaging/.venv/bin/python -m pytest -m "not gpu"` (780 pass once the hygiene and handedness-prep branches are both merged; 768 on the hygiene branch alone). From a worktree use
  `~/projects/science/bioimaging/.venv/bin/python`. This repo's own `.venv` is stale (numcodecs 0.16 vs zarr 2;
  `ipywidgets` missing); re-sync with `uv sync`.
- Never import test modules from another directory: the conftest isolation is skipped and the lane tests act on
  production paths (one such run deleted the RAM-disk view caches in `/dev/shm/opym_live_view`).
- `tests/conftest.py` redirects `PETAKIT_JOBS_DIR`, `OPYM_LIVE_VIEW_ROOT`, `OPYM_STREAM_STAGE_ROOT` and the registry path into
  a tmp dir and fails the session if one resolves under a production path (#79), so an exported variable no longer breaks
  tests. Still do not point them at production.
- GPU tests (`-m gpu`) need a free GPU (`nvidia-smi` first; the two GPUs are shared) and MATLAB needs a licence.
- Lint: `just format`; ruff and mypy are not clean (E501 and format drift remain, see docs/ROADMAP.md item 8 for the counts).
  Keep import order and whitespace clean on lines you touch.
- Light base: `import opym.<contract module>` must work with numpy only. A new contract module goes in `LIGHT_MODULES`
  (`tests/test_light_base.py`); heavy imports go inside functions. See docs/adr/0001-light-base.md.

## Deploy

- Production imports THIS checkout (editable) through `bioimaging`. Pulling main is a partial deploy: pull and
  restart the `opym-*` user units together (`opym-receive`, `-receive-backfill`, `-serve`, `-backfill`), and only
  when asked. Units live in `bioimaging/deploy/`, installed in `~/.config/systemd/user`.
- Units set `OPYM_LIVE_FORMAT=zarr` (default is the old TIFF lane), `OPYM_LIVE_LANE=1`, `OPYM_LIVE_QC=1`,
  `OPYM_DECON_PSF`, `OPYM_STREAM_STAGE_ROOT=/dev/shm/opym_stream_stage`, `OPYM_BACKFILL_REGISTRY_PATH`, the
  `OPYM_SERVE_*` knobs, and `OPYM_BACKFILL_RAW_ROOTS` / `_SETTLE_SECONDS` for the backfill receiver.
  Full table: README "Environment variables".
- Ports: 5555 receiver, 5556 backfill receiver (both loopback). Queues, leases and claims: `/dev/shm/petakit_jobs`.
- Consumers (naparym-live, opym-dashboard, opym-viewer, CORE, pymmcore-gui, chimerax-ngff, cell-cinema, argus-desktop,
  channelaligner) pin opym by SHA; bioimaging keeps the editable `../opym_local` path. A merged change reaches a consumer
  only when its pin is bumped and it is redeployed.

## Gotchas

- Channel 2 is magenta everywhere (viewers, OME metadata, movies); never red-green.
- opym is the source of truth for its contracts: other repos import them (`opym.stream.protocol`, `bfill`, `store`,
  `display`, `timeline`, `registry`, ...; table in README "Contract modules") instead of copying them. To change a contract,
  change it here with its golden vectors (`opym.testing`), then bump each consumer's pin. The rig's sender is
  `opym.stream.backfill_client`; pymmcore-gui keeps a thin wrapper around it.
- Times: one rule, `opym.stream.protocol.frame_epoch_s`: FRAME `timestamp` is epoch seconds (pymmcore-gui #65, on the fork's
  main, makes the rig send `acq_first_s`); the old ms form is still accepted through `acq_first_s`. Stores written before have relative time
  only. `opym-add-times` is dry run unless `--apply`.
- Handedness: decided 2026-11-05, canonical = rot90 (decon, live, measured PSF, TIFF cropper). `zarr_input=True`
  (`zxy`) is its y-mirror and the backfill no longer uses it (bioimaging PR #39: deskew-only stages rot90 too). The
  roughly 44 zarr datasets processed deskew-only 2026-09-24..28 are still the old mirrored handedness and have NOT been
  re-run (user decides; no deletions by us): do not move masks from those to other products. Stamping the convention
  in store `.zattrs` is still open (PROC-F1).
- sqrt8 contract with naparym-live: 8-bit view buffers hold `round(255*sqrt(v/vmax))`, `vmax` = 2^k-1 fixed per
  channel and session by the first volume; encode (MATLAB LUT), decode (GPU shader) and `DisplayEncoding` must agree.
- The receiver's idle-close text ("closed on Argus ...", `protocol.IDLE_CLOSED_REASON`) is string-matched by the rig;
  do not reword it.
- 8-bit viewers use opym's sqrt rule (`opym.display`); ChimeraX stays linear by the user's rule. The legacy `FF3D3D` red
  is remapped to magenta in the reader (`normalize_color`), never by rewriting stores.
- Do not delete datasets under `/mmfs1` or `/mmfs2` (the PI decides). Tests and scripts must not write there.

## Workflow

Branch, test, PR into `bscott711/opym` main (never an upstream), merge it yourself, then pull main. Deploy only when
the task says so. No time or effort estimates in plans or docs.
