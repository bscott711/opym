# opym duplication and divergence audit (2026-11-06)

Rev: opym `eb55bfd`; bioimaging `01db054`; naparym-live `0d128ce`; opym-dashboard `f9cf089`; opym-viewer `fde6803`;
pymmcore-gui `85800c3`; CRISPy `7e9ee06`; channelaligner `c473cf8`; peach-pit-plugin `38ad2a4`; CORE `de6dd12`;
chimerax-ngff `c1c56a7`; cell-cinema `1c7799b`; argus-desktop `92f688a`. Read-only audit; no data, registry or rig PC
was touched. Order of work: [docs/ROADMAP.md](../ROADMAP.md) item 9. Finding ids (ING-, BF-, CL-) are from the
[2026-11 review](2026-11-review.md) and bioimaging's review.

## Verdict

1. There is one opym repository (GitHub `bscott711/opym`, no forks, no `opym-local` repo); the local directory is named `opym_local`. No second clone exists on Argus. "opym vs opym-local" is the name mismatch, not two repos.
2. What exists instead is more than a dozen hand mirrors of opym's contracts in 9 other repos (wire protocol, backfill wire format, layout-3 store, timeline, registry, colours), deliberately kept dependency-free; almost all agree today, none is guarded by a test.
3. Real divergence that matters: rig FRAME `timestamp` in ms vs epoch s (ING-F1, no real times in stores), the duplicate `opym` / `opym-backfill-send` console scripts, the diverged backfill sender (BF-02), opym-viewer's stale display rule and `resolve_output_base`, and the legacy `FF3D3D` red channel colour that only opym's live viewer remaps.

Scope letters: S one or two files, M one subsystem, L cross-repo or live behaviour (as in the roadmap).

## Pairs

Verdict words: in sync, equivalent, diverged, superseded, no overlap. "Pin+document" means keep the copy and guard it with a test or a note.

| # | opym side | copy / consumer | relation | divergence today | verdict | recommendation (scope) |
|---|---|---|---|---|---|---|
| DUP-01 | `src/opym` at main `eb55bfd` | naparym-live installed opym, pinned `ef9480c` (`pyproject.toml:64`, `uv.lock:1335,1490`) | installed snapshot of a git rev | byte-identical to `git archive ef9480c`; 40 commits behind main; `live_view.py`, `lanes.py`, `ome_zarr_writer.py`, `stream/trace.py` unchanged since, so nothing it imports differs | in sync, stale but harmless | pin+document; bump with the next naparym-live release (S) |
| DUP-02 | opym git revs | uv git cache `~/.cache/uv/git-v0/checkouts/`: 5 opym revs (`c551397`, `40cc55b`, `5341c30`, `91f2cc8`, `ef9480c`) | cache, not copies | each rev is a past naparym-live pin | in sync | leave; `uv cache prune` reclaims (S) |
| DUP-03 | `registry.py`, `utils.py:30-65`, `stream/live.py`, `timeline.py`, `backfill_runs.py` | opym-dashboard `registry_reader.py:23-49`, `live_reader.py:26-54`, `stream_status_reader.py:27-28`, `transfer_reader.py:32-33`, `mip_overlay.py:36,63-71`, `viewer_links.py:17,53` | hand mirror, no opym dependency by design (dashboard README:7) | STAGES, tables, columns, status values, `resolve_output_base` (incl. the new ancestor rule), stale and epoch constants all match; sync is by comment only | in sync | pin+document, contract test in opym (S) |
| DUP-04 | `live_view.py:140-149,267-300,356` (sqrt `DisplayEncoding`, opym 03b0beb, 2026-10-23); `utils.py:30-65` | opym-viewer `volumes.py:14-18,125-151` (`display_shift`, `to_8bit`); `registry.py:34-39` | hand mirror, no opym import in `src/` | 8-bit rule is the old linear shift, last viewer code change 2026-10-10; `resolve_output_base` is the old form (differs only when `leaf_dir` does not exist yet) | diverged | update `resolve_output_base` (S); display rule is a user decision (sqrt port M with visual check, or document own linear rule S) |
| DUP-05 | `pyproject.toml:77-79` `opym = "opym.cli:main"` | bioimaging `pyproject.toml:58` `opym = "run_pipeline_cli:main"` | same console-script name in two projects | `bioimaging/.venv/bin/opym` runs `opym.cli` today (mtime 2026-11-05 22:17, opym 0.5.0 install); a bioimaging reinstall flips it; bioimaging `README.md:47` and `CLAUDE.md` describe the other one; `opym <dir>` as documented there fails (CL-08) | diverged | rename bioimaging's script, fix its docs (S, needs a restart plan) |
| DUP-06 | `src/opym/cli.py` (253 lines) | bioimaging `run_pipeline_cli.py` (297 lines) | unrelated tools under one name | no shared subcommand; stale defaults: PSF `20260622` (`:120`), `--psf-dz-um` 0.1 (`:127`), `sys.path` to `projects/opym_local` (`:14`); still imported by `run_napari_opym.py:291,721`, `psf_tools/omw_rl_comparison.py:40` | diverged (hazard, not behaviour merge) | fix the name now (S); delete or keep is a user decision (M) |
| DUP-07 | `widgets/averager.py` (`PSFAverager`), `scripts/generate_synthetic_psfs.py` | bioimaging `psf_tools/extract_bead_psf.py` (`rotate_90=True`, `:635`), `make_synthetic_psfs.py`, `synthetic_opm_psf.py` | independent implementations | opym's averager saves no rot90 (`averager.py:228`) but decon needs it (`utils.py:214-219`); opym's Gaussian script is the older generation | superseded | retire or mark legacy, document the rot90 PSF requirement (S) |
| DUP-08 | `metadata.py:19-45,112`, `decon_config.py`, `ome_zarr_writer.py:47`, `consolidate.py:206`, `lanes.py:38`, `bfill.py:47-49`, `backfill_receiver.py:127-128` | bioimaging `run_pipeline_cli.py:30-35`, `backfill/pipeline.py:1507,648,735-753,710-716`, `mip_movie.py:42-47`; systemd units | constants read or typed in several places | z-step readers: 3 (defaults 1.0 / None / 0.3); `OPYM_ZARR_DEFAULT_Z_STEP` read only by the bioimaging backfill, the stream client falls back to 0.3 (`stream/client.py:326`); palettes: 3 (ch0 green and ch1 magenta agree; ch2 and ch3 differ); PSF and registry path typed by hand in 2-3 units (equal today); marker, lock and settle constants mirrored (identical) | diverged (z step, palettes), in sync (rest) | one z-step reader and one constants home (M); align palettes (S); marker/lock test (S) |
| DUP-09 | `registry.py:208-476` | bioimaging `tests/test_decon_staging.py:335-390` (fake registry); `scratch/check_auto_detect*.py` | test fake; scratch ancestors of `roi_detect.py` | fake can miss new registry methods; scratch files are never imported | superseded | thin subclass or contract test; delete scratch files (S) |
| DUP-10 | `src/opym/stream/protocol.py` | pymmcore-gui `_argus_stream/_protocol.py` | hand port, "the Argus copy is canonical" (`_protocol.py:3-10`) | 9 message types, framing, ports, ACK features and all constants match; the copy only adds TypedDicts | in sync (behaviour-identical) | golden-vector test per repo (S) |
| DUP-11 | `protocol.py:155`, `receiver.py:1467,1547`, `rawmirror.py:212`, `timeline.py:38-48` | rig `engine.py:843-845`, `_volume_assembler.py:135`, `_session.py:679` | FRAME `timestamp` | rig sends `runner_time_ms` (ms since run start); receiver and spec want epoch s; `_MIN_EPOCH_S = 1e9` discards it; 0 of 432 stores carry `opym_time` (ING-F1) | diverged, CONFIRMED | fix on both sides, then `opym-add-times` (S per side) |
| DUP-12 | `stream/bfill.py`, `stream/bfill_run.py` | rig `_bfill.py:1-7`, `_bfill_run.py` | verbatim mirrors | AST: 29 items identical (1 receiver-only function `chunk_count_for_volume`, `bfill.py:201`); `_bfill_run.py` differs by one `# ruff: noqa` line | in sync | golden-vector test (S) |
| DUP-13 | `stream/backfill_client.py` (775 lines) | rig `backfill_send.py` (1615 lines); `opym-backfill-send` in `pymmcore-gui/pyproject.toml:113` and `bioimaging/pyproject.toml:66` | same protocol, independent senders | 10 of 38 shared functions identical; rig has link handling and parallel reads, opym's has the Windows path bug (BF-02, `backfill_client.py:350,356`); two scripts with one name | diverged | rig sender canonical; mark opym's test-only; end-to-end test against `BackfillReceiver` (M, cross-repo) |
| DUP-14 | `stream/linkbench.py` (sink), `ome_zarr_writer.py:56-86` (`downsample2`) | rig `linkbench.py` (client); channelaligner `export.py:28-40` | complementary halves; one copied function | linkbench ports (5599), messages and `<q` seq agree; `downsample2` gives equivalent output (no test compares them) | in sync, equivalent | golden-vector test (S) |
| DUP-15 | `ome_zarr_writer.py:195-282` (`create_processed_store`), `stream/live_zarr.py:735-749` | CORE `cellcore/processed_store.py:1-27`, `celldet/liveqc.py:60-69,804-836`, `seglab/volumes.py:67-77` | layout-3 reader copy, "on purpose" | in agreement; CORE test fixture `test_liveqc.py:215-232` has no `multiscales`, `omero` or compressor, so a writer change that breaks `seglab.open_cell` is not caught | in sync | golden store from `create_processed_store` read by CORE tests (S) |
| DUP-16 | `timeline.py:34-50,114-125` | cell-cinema `rawtimes.py` | near line-for-line copy | bodies logically identical; inherits ING-F1 (falls back to raw `t` or the nominal interval) | in sync | shared test vector; fix the unit bug once, in opym (S) |
| DUP-17 | `ome_zarr_writer.py:47`; `live_view.py:70-76` (`FF3D3D` to magenta) | chimerax-ngff `ngff.py:34-36,496`; cell-cinema `datasets.py:89`; argus-desktop `napari_open.py:47,190-195` | store colour readers | stores written before 2026-09-26 (writer fix `9f1a7eb`) carry `FF3D3D`; only opym's live viewer remaps it; defaults for ch3 and ch4 differ (`00B3FF/FFC400` vs `00FFFF/FFFF00`) | diverged (by grep; on-disk stores not checked) | remap in `parse_color`, others follow, or rewrite old metadata (S) |
| DUP-18 | `ome_zarr_writer.py:145-180` (`create_store`) | chimerax-ngff `tools/tiff_series_to_ngff.py:13-14,44,148` | writer re-implementation | same codec, chunks, levels, separator; colours keyed by wavelength, per-data window, no layout-3, no `opym_time` | intentional differences | leave; add a comment that it is not opym-compatible (S) |
| DUP-19 | `metadata.py:249-294`, `core.py:34`, `roi_utils.py` | GitHub repos `opm_cropping`, `microscope-control`, `PixelPacker`, `ppk5d-env`, forks `LLSpy`, `pycudadecon` | old repos | `opm_cropping` is a pre-opym prototype (first opym commit 2025-09-12, cropper 2025-11-05/06), superseded. `microscope-control` is the ancestor of pymmcore-gui `asi_z_stack`, cited as reference `5f5f573` in 7 code comments: keep. `PixelPacker` is a separate product, no overlap. `ppk5d-env` pins opym 0.2.0 from `../software/opym` on the HPC: pending. `LLSpy`, `pycudadecon`: stale forks, no overlap | superseded / no overlap | user decides archive and delete, see Decisions (S) |

## Recommendations, in order

1. Fix ING-F1 on both sides (rig sends epoch `acq_first_s`; receiver refuses < 1e9) and run `opym-add-times`; cell-cinema needs no change because it reads what opym writes. S per side. Already roadmap item 1. Repos: opym, pymmcore-gui.
2. Contract tests instead of new dependencies: one golden-vector test per side for the stream protocol and `bfill` constants (opym + pymmcore-gui), a small golden layout-3 store generated by `create_processed_store` checked into opym test fixtures and read by CORE, cell-cinema, chimerax-ngff and argus-desktop tests, a timeline test vector shared with cell-cinema, a registry/`resolve_output_base` contract test for the dashboard and viewer copies. S per repo.
3. Console-script names: rename bioimaging's `opym` (and decide who owns `opym-backfill-send`) and fix bioimaging README/CLAUDE.md. S, bioimaging (needs a restart plan because units `uv run` re-sync the prod venv).
4. Colours: `FF3D3D -> FF00FF` remap in chimerax-ngff `parse_color` (cell-cinema and argus-desktop follow) or a one-time metadata rewrite of old stores; align `consolidate.py:206` and bioimaging `mip_movie.py` with `CHANNEL_COLORS`. S. Repos: chimerax-ngff, opym, bioimaging.
5. opym-viewer: update `resolve_output_base` to the current form (S); display rule is a user decision (match naparym-live sqrt: M with visual check; or document its own linear rule: S).
6. Backfill sender single source (roadmap item 6, BF-02): the rig's `backfill_send.py` is canonical; mark opym's `backfill_client.py` test-only and add an end-to-end test of the rig sender against `BackfillReceiver`. M, cross-repo (opym, pymmcore-gui, bioimaging).
7. One z-step reader (`opym.metadata.parse_z_step`) and one constants home (registry path, default z step) imported by bioimaging; retire the legacy CLI defaults (20260622 PSF, dz 0.1) or the CLI. M. Repos: opym, bioimaging.
8. Retire opym's `PSFAverager` and Gaussian `generate_synthetic_psfs.py` (or mark legacy and document the rot90 PSF requirement). S, opym.
9. naparym-live: bump the opym pin with the next naparym-live release, not before (no functional gap). S.

Not recommended: making CORE, dashboard, viewer, cell-cinema, chimerax-ngff or the rig import opym (different runtimes, `import opym` is heavy; the boundary is deliberate).

## Decisions for the user

- (a) Rename the local dir `opym_local` to `opym`: optional. It touches `bioimaging/pyproject.toml` (editable path `../opym_local`), bioimaging `.venv` (absolute paths), units run from bioimaging, `run_pipeline_cli.py:14` sys.path, docs and memory; it needs a recipe (new path, re-lock bioimaging, restart units together), not a move. Recommendation: leave the name, documented in CLAUDE.md line 1.
- (b) Old repos: archive `opm_cropping`; delete forks `LLSpy`, `pycudadecon`; keep `microscope-control` (cited by pymmcore-gui); `ppk5d-env`: does `../software/opym` 0.2.0 still exist on the Discovery HPC?
- (c) opym-viewer display rule: sqrt (match naparym-live) or its own linear rule.
- (d) May CORE ever depend on opym (today a stated rule: no).
- (e) Old red stores: remap in the readers, or rewrite their `.zattrs`.
- (f) Keep `run_pipeline_cli` at all.

## Pending

- Rig PC checkout and its `_argus_stream/` (tunnel down). On pymmcore-gui, `_argus_stream/` is identical on `main`, local `feat/mda-live-parity` and `origin/feat/mda-live-parity`.
- HPC `../software/opym`.
- How many pre-2026-09-26 stores on disk still carry `FF3D3D` (not checked: data trees off limits for this audit).
- Whether a bioimaging restart actually flips `.venv/bin/opym` (check `stat .venv/bin/opym` after the next restart).
- Production registry schema was checked from source, not from the live file.

## Doc and spec mismatches fixed in this PR

- `docs/STREAMING_PROTOCOL.md`: intro no longer points at `opym_local/src/...` (now `src/opym/stream/protocol.py` in `bscott711/opym`); says the rig GUI does not reuse `StreamSender` and carries a hand-ported copy, names the one known wire mismatch (FRAME `timestamp`, ING-F1); SESSION_END lists all reasons (`complete`, `idle_timeout`, `client_abort`, `paused`, `client_error`, and `stalled` from the server).
- `README.md`: command table note says which `opym` is on PATH depends on install order (CL-08); the `StreamSender` sentence says the rig GUI does not use it.
- `CLAUDE.md`: Gotchas bullet pointing at this audit before changing any mirrored contract.
- `docs/ROADMAP.md`: item 9, D-id cross-references in items 1 and 6.

Not fixed here (other repos or source code):

- bioimaging `README.md:47` and `CLAUDE.md` describe `opym` as `run_pipeline_cli`.
- `protocol.py:139` says `t_interval_s` is ignored, but the receiver reads it (`receiver.py:899`); source comment, left for a code PR.
- `opym-stream-watch` and `opym-backfill-send` are declared in bioimaging's pyproject, not opym's.
