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

Same environment as `opym-stream-watch` (opym with `zmq`, `msgpack`, `numcodecs`, `zarr`).

```bat
:: one SSH tunnel (about 33-63 MB/s); add more tunnels on other ports for more throughput
ssh -N -L 5556:127.0.0.1:5556 <user>@argus.sdstate.edu

:: what would be sent (sends nothing)
opym-backfill-send --map D:\acquisitions=/mmfs1/scratch/SDSMT.LOCAL/bscott/DataUpload --dry-run

:: backfill everything missing, newest first, re-scan every 10 minutes
opym-backfill-send --map D:\acquisitions=/mmfs1/scratch/SDSMT.LOCAL/bscott/DataUpload --watch 600
```
The path below `D:\acquisitions` is kept below the remote root, so a session lands exactly where Globus would have put it.
Useful flags: `--settle 900` (skip stores written to recently), `--max-mbps`, `--pause-file C:\pause` (create it to
stop sending), `--order oldest`, repeated `--connect tcp://127.0.0.1:PORT` for several tunnels. Start it as a
Task Scheduler task at logon; it exits nonzero on nothing and logs one JSON line per folder.

## Limits

* OME-Zarr stores only. OME-TIFF acquisitions still use their current route.
* The sender must be able to read the stores while the acquisition software is closed or idle; `--settle` guards writes in progress.
* The receiver is single-threaded; one tunnel is the usual ceiling, not the receiver.
* Retire the registry rows of anything you later delete; the registry still lists every discovered copy.
