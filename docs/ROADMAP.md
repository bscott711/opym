# opym roadmap

Ordered from the 2026-11 review ([docs/review/2026-11-review.md](review/2026-11-review.md)); finding ids refer to it.
Scope: S one file or two, low risk; M several files in one subsystem; L several files or repos, or touches live
behaviour. Items 1 to 8 are in priority order, but items without a dependency can run in parallel when their files
do not overlap.

| # | Item | Scope | Depends on | Unblocks |
|---|---|---|---|---|
| 1 | Real acquisition times end to end | L (cross-repo: pymmcore-gui) | none | 5 (partly), cell-cinema time stamps, grant evidence |
| 2 | Receiver hardening | M | none | 5 |
| 3 | Handedness decision, then consistency | M | user decision | any mask or ROI reuse across products |
| 4 | Test isolation and view eviction safety | S | none | safe test runs for 2 and 5 |
| 5 | Live lane liveness | M | 2 (same file), 4 | none |
| 6 | Backfill sender single source, orphan sweep, other BF P2s | L (cross-repo) | none | none |
| 7 | Processing P2s | M | 3 for the dz and handedness items | none |
| 8 | Release hygiene | M | 1 to 7 for the 0.5.0 cut | a clean release |

## 1. Real acquisition times end to end
- What: the receiver takes the epoch from `acq_first_s` (+ `clock_offset_s`), or the sender sends epoch seconds in
  `timestamp`; refuse values below 1e9 with a log line. Add a contract test that feeds the real sender's FRAME header.
  `opym-add-times` derives relative seconds from the ms values already stored.
- Why: ING-F1, PROC-F8. The rig sends ms since run start; 0 of 432 processed stores carry real times.
- Unblocks: cell-cinema real time stamps; the acquisition-times claims in the grant evidence.
- Decision: none. The origin of existing ms values is unknown, so existing stores get relative time only.

## 2. Receiver hardening
- What: discard partial slab volumes at finalize; catch-all per message; durable drain manifest with retry and a
  writability check of `raw_root`; path-component validation for names and `session_id`; `pump()` guard (LIVE-02).
- Why: ING-F2, F3, F4, F5, LIVE-02. These are the paths that lose or corrupt data or take the receiver down.
- Unblocks: item 5.
- Decision: the trust model for ING-F5 (loopback on a shared host vs direct endpoint) sets its priority.

## 3. Handedness decision, then consistency
- What: the user picks the canonical handedness using a known asymmetric structure or the rig geometry. Then align
  the no-decon zarr path (flip, or stage like the decon path) and stamp the choice in `.zattrs`.
- Why: PROC-F1. `DSR_nodecon` is a y-mirror of every decon and live product.
- Depends on: an open user decision. Reprocessing of the roughly 44 no-decon zarr datasets follows the decision.

## 4. Test isolation and view eviction safety
- What: autouse conftest isolation for `OPYM_LIVE_VIEW_ROOT` and `OPYM_STREAM_STAGE_ROOT`; a state check in
  `_evict_views`, as in `evict_finished_views`.
- Why: LIVE-07, TEST-01. A test module imported outside the conftest removed production view caches during this review.

## 5. Live lane liveness
- What: finalize a session that cannot dispatch after `ended` (LIVE-01); keep drained raw stores while live tickets
  need them (LIVE-04, ING-F8); include the view store in the room estimate (LIVE-05); cancel old archives on
  supersede (LIVE-06); ticket dedupe after resume (LIVE-09).
- Why: LIVE-01, 04, 05, 06, 09, ING-F8. A stuck session holds the live lease and the backfill off its dataset.
- Depends on: 2 and 4 (same files, safer tests).

## 6. Backfill sender single source and orphan sweep
- What: one tested home for the sender (BF-02); sweep partial outputs when reclaiming orphans (BF-01); then the BF
  P2s: marker removal before validation (BF-03), repair of mismatched copies (BF-04), give-up policy (BF-05),
  per-lane orphan reclaim (BF-06, 07), supervisor tick off the GPFS walk (BF-08), consolidation gap (BF-09), channel
  regex (BF-10).
- Why: BF-01 and BF-02 are P1. BF-01 is local to `local_gpu_worker.py` and can ship first.
- Cross-repo: pymmcore-gui (sender), bioimaging (give-up policy, regex).

## 7. Processing P2s
- What: atomic ticket writes (PROC-F3); read the PSF z step and send `dz_psf` (PROC-F4); decide the PSF resample
  step for the `pipeline` route (PROC-F5); refuse `psf` with `zarr_input` (PROC-F2); skip-stale-DSR flag (PROC-F11);
  decon fingerprint covers all knobs and the PSF (PROC-F12); refuse to overwrite a complete processed store
  (PROC-F9); TIFF path refuses to guess dz (PROC-F10).
- Depends on: 3 for PROC-F2 and F5, which are geometry questions.

## 8. Release hygiene
- Cut 0.5.0: `[Unreleased]` in CHANGELOG is large (61 commits past v0.4.0). Bump after items 1 and 2.
- Re-sync opym's own `.venv` (`uv sync`; the lock already pins zarr 2.18.7 since #54, LIVE-19); lazy-import `batch` so a base install can
  `import opym` (BF-12).
- Lint and type debt: ruff 547 errors (497 E501), 25 files unformatted, mypy 197 errors. Confirm first whether they
  predate recent work.
- Remove root strays: `autofix_long_lines.py`, `test_load.py` (a bare `pytest` at the root collects it),
  `launch_petakit_servers.sh` (starts unsupervised servers on the production queue).
- Fix `scripts/live_bench/stack.sh` (uses `~/projects/CORE`; the live QC step is silently skipped).
- Fix or retire `Notebooks/OPM_Interactive_Petakit_Tuner.ipynb` (calls `opym.run_llsm_petakit_processing`, removed).
- `pyproject.toml` description is stale (D08); `scipy` is imported but not declared.
- Stale source docstrings (ING-F15) and the Globus wording in the `receiver.py` log text (D16).
