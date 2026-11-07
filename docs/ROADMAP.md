# opym roadmap

Ordered from the 2026-11 review ([docs/review/2026-11-review.md](review/2026-11-review.md)); finding ids refer to it, and DUP-ids refer to the duplication audit ([docs/review/2026-11-duplication.md](review/2026-11-duplication.md)).
Scope: S one file or two, low risk; M several files in one subsystem; L several files or repos, or touches live
behaviour. Items 1 to 8 are in priority order, but items without a dependency can run in parallel when their files
do not overlap.

| # | Item | Scope | Depends on | Unblocks | State (2026-11-07) |
|---|---|---|---|---|---|
| 1 | Real acquisition times end to end | L (cross-repo: pymmcore-gui) | none | 5 (partly), cell-cinema time stamps, grant evidence | done and deployed: opym #83, pymmcore-gui #65 |
| 2 | Receiver hardening | M | none | 5 | done: #86 |
| 3 | Handedness decision, then consistency | M | user decision | any mask or ROI reuse across products | decided; 3a and 3b are the user's |
| 4 | Test isolation and view eviction safety | S | none | safe test runs for 2 and 5 | done: #79 |
| 5 | Live lane liveness | M | 2 (same file), 4 | none | done: #89 (LIVE-09 C++ pid deferred) |
| 6 | Backfill sender single source, orphan sweep, other BF P2s | L (cross-repo) | none | none | done: #83, #87 (BF-04, 05, 10 deferred) |
| 7 | Processing P2s | M | 3 for the dz and handedness items | none | done: #88 (PROC-F2, F5 deferred) |
| 8 | Release hygiene | M | 1 to 7 for the release cut | a clean release | done except the release cut and the dev venv re-sync |
| 9 | Consolidation: shared code instead of mirrors | M (cross-repo) | none | safe edits to any shared contract | done: see the DUP table in the duplication audit |

Merged PRs of the 2026-11-06/07 run (all `bscott711/opym` unless named): #79 test isolation, #80 light base
([ADR 0001](adr/0001-light-base.md)), #81 read-only readers, #82 display/store/live_status, #83 stream wire, FRAME epoch time and
THE sender, #84 and #85 contract gaps, #86 receiver hardening, #87 backfill, #88 processing, #89 live liveness. Consumers:
bioimaging #44, opym-dashboard #23, opym-viewer #5, CORE #26, pymmcore-gui #65 (fork main), naparym-live #8, cell-cinema #14,
argus-desktop #12, chimerax-ngff #15, channelaligner #3.

**Deployed 2026-11-07 (about 09:00 Argus time)**, in the order of the consumers' READMEs and [ADR 0001](adr/0001-light-base.md):
opym_local `b9189d2`, bioimaging `d4bfdd9`, CORE `6d42533`, opym-dashboard `2eab996`, opym-viewer `e8df7ee`, argus-desktop `f17b972`,
channelaligner `9209119`, cell-cinema `2a24a8f`, chimerax-ngff `dd4b0a6`, naparym-live `e7ad74b`; the 8 units are active.
The `opym-*` units were restarted at the deploy, so `opym-serve` runs the edited `.m` files. The rig PC runs pymmcore-gui
`feat/piezo-camera-only` at `09180ac` (= fork `main` = tag `v0.3.0`, which pins opym `b9189d2`). opym `main` is two docs commits
ahead of the production checkout (#92). Not done: `opym-add-times --apply` on production, the `.zattrs` handedness stamp (3b),
and the opym release tag. Other docs point here for these SHAs.

## 1. Real acquisition times end to end
- Status: DONE in code (opym #83, pymmcore-gui #65; deployed 2026-11-07). One shared rule, `opym.stream.protocol.frame_epoch_s`:
  `timestamp` in [1e9, 1e11) is used; else `acq_first_s` in that window (logged once per session); else no time and one WARNING
  per session. The rig sends `timestamp = acq_first_s` (epoch s). Every old/new combination works; deploy order is receiver first,
  then the rig. cell-cinema reads what opym writes (DUP-16). Existing stores keep relative time only; `opym-add-times --apply`
  has not been run on production (the `--apply` run is still open).
- What: the receiver takes the epoch from `acq_first_s` (+ `clock_offset_s`), or the sender sends epoch seconds in
  `timestamp`; refuse values below 1e9 with a log line. Add a contract test that feeds the real sender's FRAME header.
  `opym-add-times` derives relative seconds from the ms values already stored. cell-cinema's `rawtimes.py` copy reads
  the same fields (DUP-16).
- Why: ING-F1, PROC-F8. The rig sends ms since run start; 0 of 432 processed stores carry real times.
- Unblocks: cell-cinema real time stamps; the acquisition-times claims in the grant evidence.
- Decision: none. The origin of existing ms values is unknown, so existing stores get relative time only.

## 2. Receiver hardening
- Status: DONE (#86): partial slab volumes discarded at finalize (ING-F2), catch-all per message (ING-F3), durable drain with
  retry and a `raw_root` writability check (ING-F4), path-component validation on both endpoints (ING-F5, so the trust-model
  decision was not needed), `pump()` guard (LIVE-02).
- What: discard partial slab volumes at finalize; catch-all per message; durable drain manifest with retry and a
  writability check of `raw_root`; path-component validation for names and `session_id`; `pump()` guard (LIVE-02).
- Why: ING-F2, F3, F4, F5, LIVE-02. These are the paths that lose or corrupt data or take the receiver down.
- Unblocks: item 5.
- Decision: the trust model for ING-F5 (loopback on a shared host vs direct endpoint) sets its priority.

## 3. Handedness decision, then consistency
- Decided 2026-11-05 (user): canonical = rot90 (decon, live, measured PSF, TIFF cropper). Done: bioimaging PR #39 makes
  the deskew-only zarr path stage rot90 TIFFs like decon (`zarr_input=False`), so new output matches.
- Resolved 2026-11-07 (a): the "~44 still mirrored" datasets no longer exist in the registry. Every served product of the 948 datasets is rot90
  products (735 `Decon/DSR_decon` finished on/after 2026-09-20, 139 live-lane viewer stores); 35 leaves keep an orphan
  `zarr_mirror/DSR_nodecon` that nothing reads, for the PI to delete (never us). No re-run needed. Remaining hazard: masks made on
  pre-2026-09-20 mirrored outputs must not be applied to current outputs.
- Open: (b) Stamp the convention in store `.zattrs`.
- Why: PROC-F1.
- Status: 3a closed; 3b is the user's. The prepared re-run steps and a read-only lister are in
  [docs/handedness-rerun.md](handedness-rerun.md).

## 4. Test isolation and view eviction safety
- Status: DONE (#79): the conftest redirects the four production paths into a per-session tmp dir and fails the session if one
  resolves under a production path; `_evict_views` keeps unfinished views.
- What: autouse conftest isolation for `OPYM_LIVE_VIEW_ROOT` and `OPYM_STREAM_STAGE_ROOT`; a state check in
  `_evict_views`, as in `evict_finished_views`.
- Why: LIVE-07, TEST-01. A test module imported outside the conftest removed production view caches during this review.

## 5. Live lane liveness
- Status: DONE (#89): LIVE-01, LIVE-04/ING-F8, LIVE-05 (degrades to "receiving without a live view" when only the view store
  does not fit), LIVE-06, LIVE-09 dedupe. Deferred: the pid in the buffer temp name (LIVE-09 second half), a C++ change in
  `src/opym/patches/cpp-zarr/mexSrc/opymwriteliveoutputsmex.cpp` that needs a mex rebuild.
- What: finalize a session that cannot dispatch after `ended` (LIVE-01); keep drained raw stores while live tickets
  need them (LIVE-04, ING-F8); include the view store in the room estimate (LIVE-05); cancel old archives on
  supersede (LIVE-06); ticket dedupe after resume (LIVE-09).
- Why: LIVE-01, 04, 05, 06, 09, ING-F8. A stuck session holds the live lease and the backfill off its dataset.
- Depends on: 2 and 4 (same files, safer tests).

## 6. Backfill sender single source and orphan sweep
- Status: DONE except BF-04, BF-05, BF-10 (see Deferred). BF-02 and DUP-13: the rig's sender is THE sender, now
  `opym.stream.backfill_client`, with pymmcore-gui keeping a thin wrapper (opym #83, pymmcore-gui #65, bioimaging #44 drops its
  script). BF-01, 03, 06, 07, 08, 09: #87.
- What: one tested home for the sender (BF-02, see DUP-13); sweep partial outputs when reclaiming orphans (BF-01); then the BF
  P2s: marker removal before validation (BF-03), repair of mismatched copies (BF-04), give-up policy (BF-05),
  per-lane orphan reclaim (BF-06, 07), supervisor tick off the GPFS walk (BF-08), consolidation gap (BF-09), channel
  regex (BF-10).
- Why: BF-01 and BF-02 are P1. BF-01 is local to `local_gpu_worker.py` and can ship first.
- Cross-repo: pymmcore-gui (sender), bioimaging (give-up policy, regex).

## 7. Processing P2s
- Status: DONE except PROC-F2 and PROC-F5 (see Deferred): PROC-F3, F4, F9, F10, F11, F12 in #88. F12 (full decon fingerprint)
  is opt-in (`OPYM_FULL_DECON_FINGERPRINT=1`) because turning it on makes every existing registry row mismatch once; F10 is opt-in
  (`strict=True`) and bioimaging has to pass it; F4's edited `.m` files are loaded since the `opym-serve` restart at the 2026-11-07 deploy.
- What: atomic ticket writes (PROC-F3); read the PSF z step and send `dz_psf` (PROC-F4); decide the PSF resample
  step for the `pipeline` route (PROC-F5); refuse `psf` with `zarr_input` (PROC-F2); skip-stale-DSR flag (PROC-F11);
  decon fingerprint covers all knobs and the PSF (PROC-F12); refuse to overwrite a complete processed store
  (PROC-F9); TIFF path refuses to guess dz (PROC-F10).
- Depends on: 3 for PROC-F2 and F5, which are geometry questions.

## 8. Release hygiene
- Release cut: 0.5.0 was cut 2026-11-05 (CHANGELOG, `pyproject.toml`). `[Unreleased]` now holds #79 to #89. No version bump and
  no tag were made in this run; the next release is the user's call.
- Re-sync opym's own `.venv` (`uv sync`; the lock pins zarr 2.18.7 since #54, LIVE-19): NOT done (it is a production checkout;
  tests use bioimaging's venv). Lazy-import so a base install can `import opym` (BF-12): done in #80.
- Lint and type debt, measured with the pre-commit pins (ruff 0.5.5 `check src tests`, mypy 1.11.0
  `--ignore-missing-imports src/opym`) on a `git archive` of each revision:

  | revision | ruff errors | of which E501 | ruff format would change | mypy errors |
  |---|---|---|---|---|
  | `ae1846e` (before the run) | 547 | 497 | 25 files | 158 in 19 files |
  | `origin/main` `32dccb1` (end of the run) | 439 | 377 | 24 files | 173 in 20 files |
  | after the hygiene change | 386 (377 E501, 8 UP038, 1 UP027) | 377 | 24 files | 134 in 12 files |

  So the debt predates the run (the review's 547 is reproduced at `ae1846e`) and the ruff total fell during it (547 to 439, E501
  497 to 377; the cause was not examined). The mypy counts used bioimaging's venv for third-party types, so they are not comparable with the review's 197, which
  used the pre-commit hook environment; they rose from 158 to 173 over the run (not broken down), then fell to 134 after the type-only fixes in the hygiene change. The hygiene change fixed every
  import-order, whitespace and similar finding (I001, W293, W291, UP015, F841 in `profile_report.py`) and set ruff `src` so the
  0.5.5 pin no longer reports opym as third-party. What is left is E501, `ruff format` drift, UP038/UP027 (rules removed in
  current ruff) and mypy; no mass reformat.
- Root strays `autofix_long_lines.py`, `test_load.py`, `launch_petakit_servers.sh`: removed (nothing referenced them in opym,
  bioimaging origin/main or the user units). `scripts/live_bench/stack.sh` finds CORE at `~/projects/science/CORE` (override
  `LB_CORE`). `Notebooks/OPM_Interactive_Petakit_Tuner.ipynb` moved to `Notebooks/legacy/` with a README line.
- `pyproject.toml` description (D08) and the undeclared `scipy`: fixed in #80. Stale source docstrings (ING-F15), the
  `roi_detect.py` comment and the Globus wording in the receiver text (D16, #86): fixed.

## 9. Consolidation (from docs/review/2026-11-duplication.md)
- Status: DONE under the user's decision of 2026-11-06 that opym is the source of truth and mirrors are replaced by
  `import opym...`, which supersedes the audit's "contract tests instead of imports". Contract tests stay as regression guards
  for the shared code. The state of each DUP-id and its PR is in the "Outcome (2026-11-07)" table of the audit.
- Item 9 remainder (DUP-07): `PSFAverager` and `scripts/generate_synthetic_psfs.py` are marked legacy and the rot90 PSF
  requirement is documented; nothing deleted.

## Deferred (with reasons)
- BF-04 (repair of mismatched copies): the fix is in the sender (`stream/backfill_client.py`) and is a design choice (resend the
  mismatching `t` once, or back off). The receiver already reports `mismatch: [t...]`.
- BF-05 (give-up policy): needs a registry schema change (a `first_failed_at` column, mirrored in opym-dashboard) and a policy choice.
- BF-10 (channel-suffix regex, three copies): changing the discovery grouping changes `dataset_key`s already in the registry.
- PROC-F2 (refuse `psf` with `zarr_input`) and PROC-F5 (PSF resample step for the `pipeline` route): geometry questions that wait
  for the handedness decision (item 3).
- LIVE-09, pid in the buffer temp name: C++ mex change, needs a rebuild.
- `raw_root` refinement (ING-F4, optional): with staging on, reject only permanent errors at SESSION_START. #86 answers any
  OSError there with `rejected`; left as is, an open call for the user.
