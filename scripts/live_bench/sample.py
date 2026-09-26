"""Once a second: GPU use per device, RAM disk and memory, and whether
production is busy (a contaminated test window shows up here)."""

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

PROD = Path("/dev/shm/petakit_jobs")


def gpus() -> list[dict]:
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout
    rows = []
    for line in out.strip().splitlines():
        i, util, mem = (v.strip() for v in line.split(","))
        rows.append({"gpu": int(i), "util": int(util), "mem_mb": int(mem)})
    return rows


def mem_available_gb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1e6
    return float("nan")


def count(d: Path) -> int:
    try:
        return sum(1 for p in d.iterdir() if not p.name.startswith("."))
    except OSError:
        return 0


ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--every", type=float, default=1.0)
args = ap.parse_args()
while True:
    t0 = time.time()
    shm = shutil.disk_usage("/dev/shm")
    rec = {
        "at": t0,
        "gpus": gpus(),
        "shm_used_gb": round(shm.used / 1e9, 2),
        "mem_available_gb": round(mem_available_gb(), 2),
        "load1": os.getloadavg()[0],
        "prod_claims": count(PROD / "claims"),
        "prod_queue": count(PROD / "queue"),
        "prod_queue_live": count(PROD / "queue_live"),
    }
    with open(args.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    time.sleep(max(0.0, args.every - (time.time() - t0)))
