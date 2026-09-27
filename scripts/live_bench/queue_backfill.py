"""Queue real backfill work into the test stack for the R6 fault test (a
session starting while both GPUs run long backfill tickets): a production
ticket, pointed at a copy of its dataset, split into one ticket per channel
group so each GPU gets one.

usage: queue_backfill.py TICKET.json DATA_DIR [GROUP ...]
  GROUP is comma-separated channel patterns; default "_C0,_C1" "_C2,_C3".
  e.g. queue_backfill.py /dev/shm/petakit_jobs/completed/DESKEW_..._9bf2e7e3.json \\
       /mmfs2/scratch/SDSMT.LOCAL/bscott/opym_lv/bf/cell_1_1/processed_tiff_series_split
"""

import json
import os
import sys
import time
import uuid
from pathlib import Path

src, data = Path(sys.argv[1]), Path(sys.argv[2])
groups = [g.split(",") for g in (sys.argv[3:] or ["_C0,_C1", "_C2,_C3"])]
queue = Path(os.environ.get("PETAKIT_JOBS_DIR") or "/dev/shm/opym_lv/jobs") / "queue"
ticket = json.loads(src.read_text())
for k in ("requeueCount", "requeueHistory"):
    ticket.pop(k, None)
ticket["dataDir"] = str(data)
queue.mkdir(parents=True, exist_ok=True)
for group in groups:
    t = json.loads(json.dumps(ticket))
    t["parameters"]["channel_patterns"] = group
    name = f"DESKEW_bf_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}.json"
    tmp = queue / f".{name}.tmp"
    tmp.write_text(json.dumps(t))
    tmp.rename(queue / name)
    print("queued", name, group)
    time.sleep(0.01)
