# live_bench: measure the live view end to end, beside production

An isolated copy of the live path: its own RAM-disk root (`/dev/shm/opym_lv`),
jobs dir, receiver port (5602), GPU server ids (`lv1`, `lv2`) and GPFS folder.
Production (`/dev/shm/petakit_jobs`, port 5555, servers `1`/`2`) is never touched.
It runs the code of the checkout it lives in, so a branch can be measured
before it is deployed. The GPUs are shared with production: run it while the
scope is off, and read `raw/sample.jsonl` (`prod_claims`) to see if production
was busy during a run.

```bash
scripts/live_bench/stack.sh up            # supervisor + receiver + live QC + sampler
scripts/live_bench/viewer.sh start        # naparym-live [TEST] on DCV :2
# either: a replay from the PC (pymmcore-gui, pacing from the MDA save's frame times)
#   uv run python -m pymmcore_gui._argus_stream.replay "S:\...\Cell_001_GFP_488.ome.zarr" --timepoints 10 --name lv_pc10
# or: from Argus, paced like the rig (488 stack, gap, 561 stack)
PYTHONPATH=src ../../bioimaging/.venv/bin/python scripts/live_bench/local_replay.py lv_local10 --timepoints 10
scripts/live_bench/viewer.sh shot lv_local10_end
scripts/live_bench/collect.sh lv_local10  # report, per-timepoint CSV, bit-identical check
scripts/live_bench/stack.sh down && scripts/live_bench/stack.sh clean
```

Results land in `~/projects/bioimaging/logs/live-view-2026-09-26/<run>/`:
`report.txt` (`opym-live-trace` per hop), `summary.json`, `timeline.csv`,
`verify.json`, the raw traces and GPU profiles, and screenshots.

## Fault tests (R1–R6)

Run each during a replay into the test stack; the pass bar is no manual
action and every (t, c) bit-identical, processed and archived (`collect.sh`).

| | Fault | How | Result (2026-09-26) |
|---|---|---|---|
| R1 | A server hangs on a live ticket | `inject_fault.py lv1 stop 80` beside a replay | killed at 32 s, ticket to the other GPU; one timepoint 30 s late |
| R2 | A server crashes | `inject_fault.py lv2 kill 240` | requeued on the next pass, no timepoint delayed |
| R3 | The receiver crashes | `kill -9` lv-receive's pid, then `stack.sh receiver`; drive the run with `gui_replay_direct.py` (only pymmcore-gui's session code resumes) | resumed live in 10 s, 60/60 bit-identical |
| R4 | naparym-live restarts; a new session | `viewer.sh start`, kill its pid mid-run, start again; run a second replay | reattached within 15 s; followed the new session |
| R5 | RAM-disk pressure | restart the receiver with a high floor, `LB_STAGE_FLOOR_GB=N stack.sh receiver`; for hold-back, a `fallocate` filler under `/dev/shm/opym_lv` and `gui_replay_direct.py` | eviction: 4 drained sessions oldest first, then finished view stores (never the newest), a running session untouched (20/20), evicted ones still 60/60 from GPFS; hold-back: 4 T held, 40/40 after space came back, but only once the client called its link stale (~3 min): the receiver now asks for a resend at once |
| R6 | A session starts while both GPUs run backfill | copy a dataset, `queue_backfill.py TICKET DATA` (each split ticket gets its own hard-linked data dir), wait for both claims, then replay (with and without `--prepare-s`) | one GPU preempted at once, the other joined live after its ticket; T0 last plane → painted 4.06 s without PREPARE (GFP 9.6 s), 0.76 s with; live queue ≤ 1; 60/60 both runs; the killed ticket's half-written outputs are now swept |

