# SSH backfill: finished acquisitions from the acquisition PC to Argus

Lab notebook (before/now, choices with evidence): https://claude.ai/artifact/3KrSf6jB3zKo8Kedk1HTPv

Replaces the post-acquisition Globus web transfer for OME-Zarr acquisitions. The live stream
(`opym-receive`, port 5555, the live lane) is **not touched**: the backfill is a separate receiver process on its
own port and service, and it pauses while a live acquisition is running.

Why not Globus: Globus lands a store's ~16,000 small chunk files piecemeal at about 12 MB/s, and nothing tells the
pipeline when it has finished. On 2026-10-14 the backfill staged stores mid-copy, read unlanded chunks as zeros
(`fill_value: 0`) and produced all-zero decon output for `Cell_001`..`Cell_011` (bioimaging #20). The backfill here
ends every store with an explicit, verified completion marker instead.

## How it works

```
PC (Windows)                                              Argus (micro001)
opym-backfill-send  --ssh tunnel 127.0.0.1:5556-->  opym-receive-backfill  --> <raw root>/<same relative path>/*.ome.zarr
  scan *.ome.zarr under --map LOCAL                  inventory: absent | partial | marked | busy | conflict
  skip stores written to < --settle seconds ago      small files (.zarray, z coords, .zattrs) copied verbatim
  BF_INVENTORY  -> what do you hold?                 slabs of each missing timepoint written into the store
  BF_META       -> small files verbatim              BF_COMMIT: every volume READ BACK from disk, SHA-256 vs sender
  BF_VOLUME     -> slabs (blosc) + SHA-256 each      only then: root .zattrs, then .opym_complete.json
  BF_COMMIT     -> per-volume SHA-256 manifest
```

Real data (2026-10-15, two stores of `20260928-SVO-.../Cell_005`, 13 timepoints): every one of the 1,104 and 972 files
is **byte-identical** to the source (`diff -rq` finds only the marker), 2.95 GB of volumes crossed as 1.13 GB (blosc),
all 13 volumes verified by readback.

### Guarantees

* **Never supersedes or overwrites.** A store whose pixel array (shape, dtype, chunks) differs is `conflict`; any
  metadata file that exists and differs is a refusal. Nothing is moved to `.superseded/`.
* **Never writes into a store something else is writing.** Modified within `OPYM_BACKFILL_SETTLE_SECONDS` (900) and not
  locked by this PC's sender: `busy`. A lock held by a different sender id also blocks.
* **Live has priority.** The server holds every volume write and commit while `LIVE_LEASE.json` or `WARM_LEASE.json`
  (`opym.lanes`) is fresh; the sender also pauses on that flag and on `--pause-file`. The server enforces it even if a
  sender ignores the flag.
* **Complete means verified.** The marker (`.opym_complete.json`: format 1, state complete, written timepoints,
  manifest digest, `verified: readback-sha256`) is written last. Until then the store has no root `.zattrs`, so
  `_zarr_store_is_ready` and discovery treat it as not there yet. `bioimaging.backfill.pipeline` skips its 15 minute
  idle wait for a marked store.
* **Faithful to an aborted acquisition.** A store that declares 150 timepoints but wrote 6 is copied as 150-declared /
  6-written; planes whose chunks are absent on the PC stay absent. The marker records which timepoints exist.
* **Only allowed destinations.** `OPYM_BACKFILL_RAW_ROOTS` is an allowlist; `..`, absolute store paths and bad store
  names are rejected.
* Resumable: rerun after any failure and only missing timepoints are sent; a store that grew after it was marked is topped up.

## Argus

```bash
# unit lives in bioimaging/deploy/; not enabled by default
cp ~/projects/bioimaging/deploy/opym-receive-backfill.service ~/.config/systemd/user/
systemctl --user daemon-reload && systemctl --user enable --now opym-receive-backfill
tail -f ~/projects/bioimaging/logs/opym-receive-backfill.log
```
It binds 127.0.0.1:5556 only, at `Nice=15` / idle I/O class. Edit `OPYM_BACKFILL_RAW_ROOTS` to change where stores may land.

## Acquisition PC

Same environment as `opym-stream-watch` (opym with `zmq`, `msgpack`, `numcodecs`, `zarr`), and an `ssh` Host alias
`Argus` that logs in without a prompt (key or agent). One command does the rest:

```bat
opym-backfill-send --dry-run     :: one pass: what would be sent, per folder (sends nothing)
opym-backfill-send               :: S:\ -> Argus, forever: tunnels, scanning, sending, retrying
opym-backfill-send --install     :: ... and start it at every logon, hidden, restarted if it dies
opym-backfill-send --status      :: is it running, what is it doing, how did the last pass go
opym-backfill-send --stop        :: stop it           (--uninstall also removes the logon task)
```

**Mapping.** The drive root mirrors `/mmfs1/scratch/jacks.local/microscopy` (change with `--remote-root` or
`OPYM_BACKFILL_REMOTE_ROOT`): `S:\20261023_LCK-NG_JB\Cell_001\x.ome.zarr` goes to
`.../microscopy/20261023_LCK-NG_JB/Cell_001/x.ome.zarr`, the same place the live stream puts it. Giving a folder
(`opym-backfill-send S:\20261023_LCK-NG_JB`) maps the same way, relative to its drive, so it never matters whether
you point it at the drive or at one session. Every `*.ome.zarr` store up to 6 folders deep is found, including
sessions created after the sender started. Hidden folders, `$RECYCLE.BIN` and `System Volume Information` are
skipped, and a store whose name the receiver would refuse (letters, digits, space and `_ . + -` only) is reported
instead of silently ignored. The default drive is `S:\` on Windows; `OPYM_BACKFILL_LOCAL_ROOTS` or a folder
argument overrides it, and `--map LOCAL=REMOTE` still sets an explicit mapping.

**What happens to what is already on Argus.** Nothing is deleted or overwritten. A store with a completion marker
is skipped. One without (a Globus copy, or a live stream's own copy) has each timepoint counted as present when its
chunk-file count is right and no chunk is empty; only missing timepoints are sent. Either way the sender then
hashes every timepoint on the PC, Argus reads them all back, and the marker is written only if they match. A store
whose pixel array or metadata differs is a `conflict` or `failed`, reported and left alone; a store that matches in
shape but differs in content is flagged by that readback and is not re-sent on its own (fix or move it, then
`--rescan`). A store verified complete is remembered (`scan-state.json`, in the state folder) and skipped without
being read again until its newest modification time or timepoint count changes.

**Speed.** One SSH connection carries about 33-63 MB/s (see `STREAMING_PROTOCOL.md`, Links), so the sender opens
`--tunnels` (default 4) separate `ssh` processes, local ports `--base-port` (5656) + k, all to the receiver's port
5556, and spreads slabs across them; with 4 the live stream measured about 255 MB/s. Reading, hashing and
compressing the next slab runs on a worker thread while the tunnels send. A tunnel that dies is restarted with
backoff; a port that already has a listener is reused, not fought over. The live stream's own tunnels use local
ports 5555-5558 on this PC, so never forward 5556 (or anything else in that range) to the backfill receiver.

**Staying out of the live stream's way.** The sender runs at background CPU/IO priority (`--normal-priority` to
turn that off), its `ssh` processes at below-normal priority with no console window, and it pauses (nothing read,
nothing sent) for as long as Argus reports a live acquisition, the receiver also refusing writes while the lease
is held. `--pause-file C:\pause` stops it by hand; `--max-mbps` caps the rate; stores written to in the last
`--settle` seconds (900) are left alone. Only one sender can run per PC.

**Files.** `%LOCALAPPDATA%\opym\backfill-send\` (`$OPYM_BACKFILL_STATE_DIR` to move it): `backfill-send.log`
(rotating), `status.json` (what `--status` reads), `scan-state.json`, `task.xml`.

If you tunnel yourself (`ssh -N -L 5556:127.0.0.1:5556 Argus`), `--no-tunnels` uses that port, and `--connect
tcp://127.0.0.1:PORT` (repeatable) names several.

## Limits

* OME-Zarr stores only. OME-TIFF acquisitions still use their current route.
* Argus must allow the destination: `OPYM_BACKFILL_RAW_ROOTS` on `opym-receive-backfill` lists the folders it may
  write under (the microscopy folder and `DataUpload`).
* `--install`, the background priority and the hidden `ssh` windows are Windows-only; elsewhere run it in a
  terminal or under your own service manager.
* The sender must be able to read the stores while the acquisition software is closed or idle; `--settle` guards writes in progress.
* The receiver is single-threaded; one tunnel is the usual ceiling, not the receiver.
* Retire the registry rows of anything you later delete; the registry still lists every discovered copy.
