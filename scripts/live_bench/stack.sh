#!/usr/bin/env bash
# The isolated live-view test stack, as transient user units:
#   lv-serve    the real GPU supervisor (opym.local_gpu_worker), servers lv1/lv2
#   lv-receive  a receiver on 127.0.0.1:$PORT with the one-format live lane
#   lv-qc       CORE's live QC service on the test jobs dir
#   lv-sample   GPU / RAM disk / memory / production-activity sampler
# usage: stack.sh up | down | status | clean
set -euo pipefail
source "$(dirname "$0")/env.sh"

common=(--user --collect -p "WorkingDirectory=$WT"
  --setenv=PYTHONPATH="$WT/src" --setenv=PETAKIT_JOBS_DIR="$JOBS"
  --setenv=PYTHONUNBUFFERED=1)

up() {
  if ss -ltn | grep -q "127.0.0.1:$PORT "; then
    echo "port $PORT is already in use" >&2; exit 1
  fi
  for u in $UNITS; do
    if systemctl --user is-active -q "$u"; then echo "$u already running" >&2; exit 1; fi
  done
  if [ -n "$(ls -A /dev/shm/petakit_jobs/claims 2>/dev/null)" ] \
     || [ -n "$(ls -A /dev/shm/petakit_jobs/queue_live 2>/dev/null)" ]; then
    echo "WARNING: production has claimed or live tickets right now; results may be contaminated" >&2
  fi
  mkdir -p "$JOBS" "$STAGE" "$VIEW" "$RAW_ROOT" "$LV/logs" "$LV/shots"
  systemd-run "${common[@]}" --unit=lv-serve \
    --setenv=OPYM_SERVE_SERVERS=lv1:0,lv2:1 \
    --setenv=OPYM_SERVE_HANG_MIN="${LB_HANG_MIN:-60}" --setenv=OPYM_SERVE_KILL_HUNG=1 \
    --setenv=OPYM_LIVE_PREEMPT=1 --setenv=OPYM_LIVE_PREEMPT_AFTER_S=30 \
    -p StandardOutput="append:$LV/logs/lv-serve.log" -p StandardError="append:$LV/logs/lv-serve.log" \
    "$PY" -m opym.local_gpu_worker
  systemd-run "${common[@]}" --unit=lv-receive \
    --setenv=OPYM_DECON_PSF="$PSF" --setenv=OPYM_STREAM_STAGE_ROOT="$STAGE" \
    --setenv=OPYM_LIVE_LANE=1 --setenv=OPYM_LIVE_QC=1 --setenv=OPYM_LIVE_FORMAT=zarr \
    --setenv=OPYM_LIVE_VIEW_ROOT="$VIEW" \
    -p StandardOutput="append:$LV/logs/lv-receive.log" -p StandardError="append:$LV/logs/lv-receive.log" \
    "$PY" "$LB_DIR/receiver.py" --bind "tcp://127.0.0.1:$PORT"
  if [ -x "$HOME/projects/CORE/.venv/bin/celldet-live-qc" ]; then
    systemd-run --user --collect --unit=lv-qc -p WorkingDirectory="$HOME/projects/CORE" \
      --setenv=CUDA_VISIBLE_DEVICES= --setenv=OMP_NUM_THREADS=4 --setenv=PYTHONUNBUFFERED=1 \
      -p StandardOutput="append:$LV/logs/lv-qc.log" -p StandardError="append:$LV/logs/lv-qc.log" \
      "$HOME/projects/CORE/.venv/bin/celldet-live-qc" --jobs "$JOBS"
  fi
  systemd-run "${common[@]}" --unit=lv-sample \
    -p StandardOutput="append:$LV/logs/lv-sample.log" -p StandardError="append:$LV/logs/lv-sample.log" \
    "$PY" "$LB_DIR/sample.py" --out "$LV/sample.jsonl"
  status
}

down() {
  for u in $UNITS; do systemctl --user stop "$u" 2>/dev/null || true; done
  # MATLAB servers live in lv-serve's cgroup and stop with it; double-check.
  for pid in $(grep -l "PETAKIT_JOBS_DIR=$JOBS" /proc/[0-9]*/environ 2>/dev/null | cut -d/ -f3); do
    kill "$pid" 2>/dev/null || true
  done
  status
}

status() {
  for u in $UNITS; do printf '%-11s %s\n' "$u" "$(systemctl --user is-active "$u" 2>/dev/null || true)"; done
  printf 'queue_live  %s   claims %s\n' "$(ls "$JOBS/queue_live" 2>/dev/null | wc -l)" \
    "$(ls "$JOBS/claims" 2>/dev/null | tr '\n' ' ')"
  df -h /dev/shm | tail -1
}

clean() {
  for u in $UNITS; do
    if systemctl --user is-active -q "$u"; then echo "stop the stack first" >&2; exit 1; fi
  done
  rm -rf "$LV" "$(dirname "$RAW_ROOT")"
  echo "removed $LV and $(dirname "$RAW_ROOT")"
}

"${1:-status}"
