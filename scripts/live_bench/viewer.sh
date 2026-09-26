#!/usr/bin/env bash
# A test naparym-live on the DCV display, following the test stack.
# usage: viewer.sh start | stop | shot NAME
set -euo pipefail
source "$(dirname "$0")/env.sh"
export DISPLAY="${LB_DISPLAY:-:2}"
export XAUTHORITY="${LB_XAUTHORITY:-/run/user/$(id -u)/dcv/bioimaging.xauth}"
case "${1:-}" in
  start)
    mkdir -p "$LV/logs" "$LV/shots"
    PETAKIT_JOBS_DIR="$JOBS" PYTHONPATH="$WT/src" setsid nohup "$PY" -m opym.live_view \
      --title "naparym-live [TEST]" >>"$LV/logs/viewer.log" 2>&1 &
    echo $! >"$LV/viewer.pid"; echo "test viewer pid $(cat "$LV/viewer.pid")" ;;
  stop)
    [ -f "$LV/viewer.pid" ] && kill "$(cat "$LV/viewer.pid")" 2>/dev/null || true
    rm -f "$LV/viewer.pid" ;;
  shot)
    "$PY" -c "from PIL import ImageGrab; ImageGrab.grab(xdisplay='$DISPLAY').save('$LV/shots/$2.png')"
    echo "$LV/shots/$2.png" ;;
  *) echo "usage: viewer.sh start | stop | shot NAME" >&2; exit 2 ;;
esac
