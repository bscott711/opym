"""Freeze (SIGSTOP) or kill (SIGKILL) a test-stack GPU server at the moment it
holds a live ticket: the R1 (hang) and R2 (crash) fault tests.

usage: inject_fault.py SERVER_ID {stop|kill} AFTER_S
  e.g. inject_fault.py lv1 stop 80   (the supervisor's live deadline must
  kill it within ~30 s and the other GPU take its ticket)
"""

import json
import os
import signal
import sys
import time
from pathlib import Path

sid, how, after = sys.argv[1], sys.argv[2], float(sys.argv[3])
jobs = Path(os.environ.get("PETAKIT_JOBS_DIR") or "/dev/shm/opym_lv/jobs")
time.sleep(after)


def matlab_pid() -> int | None:
    """The server's Matlab (run_petakit_server), by its environment."""
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            env = (d / "environ").read_bytes().split(b"\0")
            cmd = (d / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if (
            f"PETAKIT_SERVER_ID={sid}".encode() in env
            and f"PETAKIT_JOBS_DIR={jobs}".encode() in env
            and cmd[0].endswith(b"MATLAB")
            and b"run_petakit_server" in b" ".join(cmd)
        ):
            return int(d.name)
    return None


deadline = time.time() + 120
while time.time() < deadline:
    try:
        rec = json.loads((jobs / "claims" / f"S{sid}.json").read_text())
    except (OSError, ValueError):
        time.sleep(0.01)
        continue
    if rec.get("queue") == "queue_live" and (pid := matlab_pid()):
        os.kill(pid, signal.SIGSTOP if how == "stop" else signal.SIGKILL)
        print(
            f"{time.strftime('%H:%M:%S')} {how} server {sid} pid {pid} "
            f"holding {rec['ticket']}",
            flush=True,
        )
        break
    time.sleep(0.01)
else:
    print(f"no live claim by server {sid} within 120 s", flush=True)
