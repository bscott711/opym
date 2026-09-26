#!/usr/bin/env bash
# Collect one run's evidence into the results folder:
#   report.txt, summary.json, timeline.csv  (opym-live-trace)
#   verify.json                              (bit-identical, processed, archived)
#   the traces and GPU profiles, the replay report, the sampler window, shots
# usage: collect.sh RUN [--session ID]   (default: the newest session)
set -euo pipefail
source "$(dirname "$0")/env.sh"
RUN="$1"; shift || true
OUT="$RESULTS/$RUN"
mkdir -p "$OUT/raw"
cp "$JOBS"/profiling/*.jsonl "$OUT/raw/" 2>/dev/null || true
[ -f "$HOME/$RUN.replay.json" ] && cp "$HOME/$RUN.replay.json" "$OUT/"
cd "$WT"
PYTHONPATH="$WT/src" "$PY" -m opym.stream.trace_report --jobs "$JOBS" --out "$OUT" "$@" \
  | tee "$OUT/report.txt"
if [ -f "$HOME/$RUN.replay.json" ]; then
  PYTHONPATH="$WT/src" "$PY" "$LB_DIR/verify.py" "$RUN" "$@" | tee "$OUT/verify.json"
fi
[ -f "$LV/sample.jsonl" ] && cp "$LV/sample.jsonl" "$OUT/raw/"
if compgen -G "$LV/shots/*.png" >/dev/null; then mkdir -p "$OUT/shots" && mv "$LV"/shots/*.png "$OUT/shots/"; fi
echo "Collected into $OUT"
