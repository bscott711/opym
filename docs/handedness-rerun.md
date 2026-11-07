# Handedness: re-running the old deskew-only datasets (3a) and stamping the convention (3b)

## Resolved 2026-11-07 (3a not needed; 3b still open)

The lister returned 0 candidates on the 948-row registry: no deskew row records a
`zarr_mirror/DSR_nodecon` output. Read-only checks (registry `mode=ro`, `os.stat` and
directory listings only, no pixels read) show why: 735 rows end in `Decon/DSR_decon`
(all finished on/after 2026-09-20, after rot90 staging 3afa092 of 2026-09-19) and 139
are live-lane viewer stores (rot90 staging), so every served product is already rot90.
35 leaves still hold an orphan `zarr_mirror/DSR_nodecon` that no consumer reads
(dashboard, CORE, ChimeraX link code and `_dsr_dir_for` never look there); every viewer
store and MIP movie of those 35 is newer than the decon output. The "~44" figure below
was an estimate and has no registry equivalent.

- 3a (re-run) is not needed. The lister stays as a check: run it again if a deskew-only
  row ever points at `zarr_mirror/`.
- Remaining hazard: masks made on the pre-2026-09-20 mirrored outputs must not be applied
  to current outputs. The 35 orphan dirs are the PI's to delete, never us.
- 3b (stamping the convention in `.zattrs`) stays open as a proposal.

Everything below is the historical runbook, written before this check.

Prepared, not run. Nothing here has been executed against the production registry
or the data. Written against opym `32dccb1` and bioimaging `origin/main` `880c420`
(PR #39 is in it). ROADMAP item 3; finding PROC-F1 in `docs/review/2026-11-review.md`.

Decided 2026-11-05 (user): the canonical handedness is the rot90 one (decon, live
lane, measured PSF, TIFF cropper). Both items below are open and are the user's call.

## 3a. What the problem is

Until bioimaging PR #39, a zarr dataset deskewed WITHOUT a PSF was read through a
symlink mirror with `inputAxisOrder='zxy'` (a plain transpose) and written to
`<leaf>/zarr_mirror/DSR_nodecon`. That is the y-mirror of every decon and live
product. Since PR #39 the same dataset is staged as rot90 `(ny, nx, nz)` TIFFs in
`<leaf>/decon_stage/` and its deskew-only output goes to `<leaf>/decon_stage/DSR_nodecon`
(decon: `<leaf>/decon_stage/Decon/DSR_decon`). Nothing writes under `zarr_mirror/` any
more and `_dsr_dir_for` looks it up nowhere.

Roughly 44 datasets were processed deskew-only 2026-09-24..28, so their recorded
`deskew.output_path` is still the `zarr_mirror/DSR_nodecon` form. The figure 44 is
the repo docs' own estimate and has not been counted against the registry: the
lister below produces the real count.

## Prerequisite: an opym that has the registry reader

The lister and step 3 import `RegistryReader` and `registry_path` from `opym.registry`.
They exist from opym PR #81 (in `32dccb1`, which this runbook was written against). The
opym installed in production today is `ae1846e`: its `opym.registry` has only
`StatusRegistry`, so both commands exit with an ImportError (nothing is written). Run
them either after opym is deployed, or before that from a worktree of opym at
`origin/main` with `PYTHONPATH=<worktree>/src` set (the lister's `sys.path[0]` is
`scripts/`, which does not include `src`). Run step 3 only after the deploy, so the reset
goes through the same opym the service runs.

## 3a. Step 0: what the registry says (decides which path you are on)

The lister is a dry run (registry opened `mode=ro`, `os.stat` on five paths per
row, no directory walk, refuses any write-looking flag):

```
cd <a worktree of opym at origin/main>      # not the production checkout until opym is deployed
export PYTHONPATH=$PWD/src                  # makes the worktree's opym win over the installed one
PY=~/projects/science/bioimaging/.venv/bin/python
$PY scripts/handedness_rerun_candidates.py                          # table
$PY scripts/handedness_rerun_candidates.py --since 2026-09-24 --until 2026-09-28
$PY scripts/handedness_rerun_candidates.py --format csv > ~/handedness_manifest_$(date +%F).csv
```

Registry path: `$OPYM_BACKFILL_REGISTRY_PATH`, else the production default; override with
`--registry`. A row is a candidate when its `deskew` stage records a path ending in
`zarr_mirror/DSR_nodecon`. Save the CSV first: the registry overwrites `output_path`
when a re-run finishes, and the CSV is then the only index of the old outputs.

Columns that matter: `decon_psf` (empty = the registry says deskew-only), `deskew_status`,
the stat flags (old output still on disk, `viewer/`, `mip_movies/`, new output already
there) and `note`.

What the code does with the `decon_psf` column (bioimaging `submit_zarr_deskew_ticket`,
`decon_provenance_matches`):

- The service runs `--decon-psf .../20260910_averaged_psf.tif`. A dataset whose
  `deskew` is `done` is skipped only if its recorded PSF equals the current one. A row
  with `decon_psf` NULL therefore differs and WOULD be re-submitted by the running
  service, as DECON, rot90.
- The service log tail on 2026-11-07 (01:24..01:28) shows passes that submit nothing
  ("0 dataset(s) still awaiting deskew resolution"). So either the candidate rows have
  `decon_psf` set (the registry believes they are current although the pixels on disk
  are the mirrored deskew-only ones), or something else gates them. Read the
  lister's `decon_psf` column for the real answer before choosing a path; this note
  could not look at the registry.
- A deskew-only re-run (no PSF) is not compatible with the running service: after it,
  `decon_psf` is NULL, which differs from the service's PSF, so the next pass
  re-submits the dataset as decon and replaces the deskew-only result. Deskew-only
  would need the service stopped. The path below therefore re-runs under the
  service's own configuration (decon).

## 3a. The mechanism

There is no `--reprocess-handedness` switch and none is needed. The backfill
re-submits a zarr dataset whenever `is_stage_done(key, "deskew")` is false or the
provenance differs. The existing registry call for "this stage must run again" is
`StatusRegistry.reset_stage(key, "deskew")` (`opym.registry`): status `pending`, finish
time cleared, failure streak zeroed, `output_path` kept (the dashboard shows the dataset
as "reprocessing" with its old output). The next `--watch` pass (every 120 s) submits the
ticket; it resets `mip_encode` itself, and on completion records the new `output_path`,
`decon_psf`, `decon_params`, then encodes MIPs and the viewer export.

Not what you want here: `--retry-failed` (`clear_gave_up`) only clears the failure
streak of `failed` stages; `--reprocess-legacy-decon` only re-queues rows with a PSF and
a NULL `decon_params`. Neither touches a `done` row.

The in-flight cap (`OPYM_BACKFILL_MAX_INFLIGHT=4`) and the live-acquisition lease still
apply, so a batch of resets drains gradually and pauses while a live acquisition holds
the GPUs. Decon is a GPU run per dataset; check `nvidia-smi` and the rig schedule first.

## 3a. Exact steps (one pilot, then the rest)

Before anything: confirm the running service has PR #39.

```
git -C ~/projects/science/bioimaging merge-base --is-ancestor 827c4a8 HEAD && echo has-PR39
systemctl --user show opym-backfill -p ActiveEnterTimestamp
```
(`ActiveEnterTimestamp` must be after the pull; otherwise the running process still has
the old code and would write the mirrored output again.)

1. Manifest (above). Pick a pilot row, ideally one with `V` and `M` set.
2. Move the pilot's old products aside. A re-run does overwrite two things in place (see
   the table); move them first. The aside folder is inside `zarr_mirror/`, which
   discovery already prunes, so it is never mistaken for a dataset:

   ```
   B=<the dataset's leaf_dir, or its mirrored output base>   # old_output minus /zarr_mirror/DSR_nodecon
   A="$B/zarr_mirror/_pre_handedness_fix"; mkdir -p "$A"
   mv "$B/viewer"      "$A/viewer"
   mv "$B/mip_movies"  "$A/mip_movies"
   ```
   If the lister shows flag `N` or `D` for the row (a `decon_stage/` output already exists),
   the re-run removes that output first, so move `decon_stage/` aside as well
   (`mv "$B/decon_stage" "$A/decon_stage"`), or leave that row out of this batch.
   (`mv` within one filesystem is a rename: nothing is copied or deleted.)
3. Reset the deskew stage (this is the only registry write; it is yours to run):

   ```
   ~/projects/science/bioimaging/.venv/bin/python - <<'PY'
   from opym.registry import StatusRegistry, registry_path
   KEYS = ["<dataset_key from the manifest>"]
   with StatusRegistry(registry_path()) as reg:
       for k in KEYS:
           reg.reset_stage(k, "deskew")
   PY
   ```
4. Watch `~/projects/science/bioimaging/logs/opym-backfill.log` for the dataset key, then
   verify (below).
5. If the pilot is good, repeat steps 2-3 for the remaining keys in the manifest (a loop
   over the CSV's `dataset_key` and `old_output` columns). Run them in batches the in-flight
   cap can drain, or all at once and let it queue.

## 3a. What is overwritten and what is kept

The steps above delete nothing: they move `viewer/` and `mip_movies/` (and, for flag `N`/`D`
rows, `decon_stage/`) aside by rename, and reset one registry row. The re-run itself removes
only what the table marks REMOVED: a `decon_stage/` output left by an earlier run, and its
own staged intermediates. The old `zarr_mirror/` products are not touched. Checked in
bioimaging `backfill/` and `pipeline.py`:

| Item | After the re-run | Why |
|---|---|---|
| `<leaf>/zarr_mirror/DSR_nodecon/` (old DSR frames, `MIPs/`) | kept, untouched | the new run writes under `decon_stage/`; no code removes `zarr_mirror/` |
| `<leaf>/zarr_mirror/` symlink mirror | kept | no longer written, not removed |
| `<leaf>/decon_stage/Decon/`, `.../DSR_decon` | created; if present from an earlier run, REMOVED first (`_clean_stale_deskew_output`) | only touches the new location; for a candidate it should not exist (lister flag `D`) |
| `<leaf>/viewer/<name>_dsr.ome.zarr`, `viewer_export.json`, `.cxc` | OVERWRITTEN (`write_ome_zarr` removes the store and swaps in the new one) | same path as the old export: move aside first (step 2) |
| `<leaf>/mip_movies/` | OVERWRITTEN | same path |
| registry `deskew`: `output_path`, `decon_psf`, `decon_params`; `mip_encode` | REPLACED | the old `output_path` is lost: keep the CSV |
| registry `signal_flag`, `expected/actual_timepoints` | rewritten by `mip_encode` | |
| staged `decon_stage/*.tif` | created, then removed after MIP encode | reproducible intermediates (`_reap_decon_intermediates`) |

The registry then points at the new output; dashboard, viewer and CORE follow it.
Masks made on the old (mirrored) output must not be applied to the new one (CLAUDE.md).

## 3a. Verification (one dataset, ChimeraX, linear intensity)

A pure y-mirror is invisible in projections (the 2026-09-18 check used them), so
compare 3-D volumes. chimerax-ngff shows 16-bit data in 8-bit by default; use `bits 16`
and leave the image levels linear (no gamma stretch):

```
open <pilot>/viewer/<name>_dsr.ome.zarr format ngff bits 16                     # new, rot90 (#1)
open <pilot>/zarr_mirror/_pre_handedness_fix/viewer/<name>_dsr.ome.zarr format ngff bits 16   # old, mirrored (#2)
```
Expected: the new volume is the old one flipped along one lateral axis (the review calls
it the y-mirror) and sharper (decon). `volume flip #2 axis y modelId 3` should make #3
line up with #1 apart from the decon sharpening; if it does not, try the other lateral
axis and record which one it was. Then open a product that is already canonical for a
comparable object (a decon dataset of the same cell type, or the same acquisition's live
lane store if it has one) and check that the new volume has the same asymmetry as it
(for example the side the nucleus or a protrusion sits on relative to the coverslip),
not the old one's. Judge beside, not from numbers. Channel 2 stays magenta.

## 3a. Decisions for the user

1. Re-run at all, and which rows (all, or only `--since/--until` 2026-09-24..28, or
   only those with `decon_psf` empty).
2. Decon (what the service will produce) versus deskew-only (needs the service stopped;
   not prepared here).
3. The pilot dataset, and whether to wait for a free GPU window.
4. Whether to also keep the old `viewer/` and `mip_movies/` (step 2 does) or accept losing them.

## 3b. Stamping the convention in the store

Proposal (the decision is the user's):

- Key: root `.zattrs["opym_handedness"]` on processed OME-Zarr stores, next to
  `opym_time` (`opym.timeline.ATTR`).
- Value: `{"convention": "rot90", "version": 1}`. `convention` is the only field a reader
  needs; `version` lets the definition be tightened later without a new key. The
  absence of the key means "unknown / written before the stamp", which includes all
  mirrored stores. Old stores are not rewritten (same rule as the FF3D3D colour: the
  reader copes, the store stays).
- Writer: `opym.ome_zarr_writer`, in both `create_store` (legacy layout, used by the
  backfill's viewer export `write_ome_zarr` and by the live lane's older path) and
  `create_processed_store` (bioformats2raw layout of the live lane), via an optional
  `handedness: str | None = None` argument that sets `root.attrs["opym_handedness"]`.
  The writer cannot know the pixels' handedness, so it stamps only when the caller says
  so. Callers: the live lane (always rot90) and bioimaging `_export_for_viewers`, which
  passes it only when the DSR directory is under `decon_stage/` (never for a
  `zarr_mirror/` one, which is the case that would otherwise be stamped wrongly).
- Reader: a small `read_handedness(store) -> str | None` beside `timeline.read_times`,
  for CORE and the viewers to refuse or warn when a mask crosses stores with different
  or unknown conventions.
- Not proposed: stamping the existing `viewer/` stores after the fact. They are
  re-written by the re-run anyway; for the others the unstamped state is accurate.

Open for the user: the key name, whether `version` is wanted, and whether a stamped
legacy marker (`"convention": "zxy-mirror"`) should be written to the ~44 old stores,
which means touching their `.zattrs`.
