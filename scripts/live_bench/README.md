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
