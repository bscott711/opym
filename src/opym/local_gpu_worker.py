# opym/local_gpu_worker.py
"""
Watchdog for PetaKit job queue.
Spins up the Matlab server when jobs are present and waits for it
to auto-shutdown on idle to release GPU resources.
"""

from __future__ import annotations

import os
import subprocess  # nosec B404
import time
import shutil
from pathlib import Path

from opym.consolidate import run_pending_consolidations

# Dynamically locate the opym installation directory
OPYM_DIR = Path(__file__).parent.resolve()

BASE_DIR = Path("/dev/shm/petakit_jobs")
QUEUE_DIR = BASE_DIR / "queue"

# If both Matlab servers exit nonzero (e.g. a license checkout failure kills
# Matlab before it ever reaches run_petakit_server.m), retrying on the normal
# short poll_interval would hot-loop launching Matlab -- and re-attempting a
# license checkout -- forever until whatever's broken gets fixed. Back off
# exponentially instead, capped, so a stuck license degrades gracefully
# rather than hammering the license server and spamming logs.
FAILURE_BACKOFF_BASE_SEC = 15
FAILURE_BACKOFF_CAP_SEC = 300


def _next_backoff_sec(
    consecutive_failures: int,
    base: float = FAILURE_BACKOFF_BASE_SEC,
    cap: float = FAILURE_BACKOFF_CAP_SEC,
) -> float:
    """Seconds to wait before the next launch attempt, given how many
    consecutive attempts have failed (1 = the first failure)."""
    return min(base * (2 ** (consecutive_failures - 1)), cap)


def _ensure_directories():
    """Ensures all necessary job directories exist."""
    for directory in (
        QUEUE_DIR,
        BASE_DIR / "completed",
        BASE_DIR / "failed",
    ):
        directory.mkdir(parents=True, exist_ok=True)


def process_queue(idle_timeout_sec: int = 300, poll_interval: int = 2):
    """
    Watches the queue. If jobs exist, launches the persistent Matlab server.
    The Matlab server handles the jobs and shuts itself down after `idle_timeout_sec`.
    """
    _ensure_directories()
    
    print("=" * 60)
    print(" 🚀 OPYM PetaKit GPU Watchdog Initialized")
    print("=" * 60)
    print(f" 📂 Queue Directory: {QUEUE_DIR}")
    print(f" ⏱️  Idle Timeout:    {idle_timeout_sec} seconds")
    print(f" 🔍 Polling Rate:    Every {poll_interval} seconds")
    print(
        f" ⚠️  Failure Backoff: {FAILURE_BACKOFF_BASE_SEC}s-{FAILURE_BACKOFF_CAP_SEC}s "
        "(exponential, if Matlab exits with an error)"
    )
    print(f" 🔧 Backend Script:  {OPYM_DIR}/run_petakit_server.m")
    print("=" * 60)
    print("👀 Listening for incoming jobs...\n")

    # Pass the timeout to Matlab via environment variables
    env = os.environ.copy()
    env["PETAKIT_IDLE_TIMEOUT"] = str(idle_timeout_sec)

    consecutive_failures = 0

    try:
        while True:
            # Check if there are any *claimable* JSON tickets in the queue.
            # Path.glob("*.json") also matches ".active_*.json" files (an
            # in-progress or orphaned ticket a MATLAB server already claimed
            # via movefile) -- unlike the shell, pathlib doesn't hide
            # dotfiles. run_petakit_server.m explicitly excludes those
            # (~startsWith(name, '.')) when deciding what it can pick up, so
            # mirror that here: an orphaned .active_ ticket left behind by a
            # crashed server shouldn't make the watchdog spin up fresh
            # servers that will just idle-timeout without touching it.
            if any(
                p for p in QUEUE_DIR.glob("*.json") if not p.name.startswith(".")
            ):
                print("\n🚀 Jobs detected. Spinning up PetaKit Matlab Server...")

                env1 = env.copy()
                env1["PETAKIT_SERVER_ID"] = "1"
                env1["PETAKIT_GPU_ID"] = "1"
                env1["CUDA_VISIBLE_DEVICES"] = "0"
                env1["PETAKIT_CPUS"] = "10"  # Limit workers to prevent GPU OOM
                
                env2 = env.copy()
                env2["PETAKIT_SERVER_ID"] = "2"
                env2["PETAKIT_GPU_ID"] = "1"  # Both use GPU index 1 because CUDA restricts visibility to 1 device
                env2["CUDA_VISIBLE_DEVICES"] = "1"
                env2["PETAKIT_CPUS"] = "10"  # Limit workers to prevent GPU OOM

                # Use bash to load the matlab module so licensing works correctly
                cmd_str = f"module load matlab/R2024b && matlab -nodisplay -sd {OPYM_DIR} -batch run_petakit_server"
                
                cmd = ["bash", "-c", cmd_str]

                print("➡️  Launching Server 1 on GPU 1...")
                p1 = subprocess.Popen(cmd, env=env1)
                
                print("➡️  Launching Server 2 on GPU 2...")
                p2 = subprocess.Popen(cmd, env=env2)

                # This will block until both Matlab scripts complete
                # their queues AND their timeouts
                p1.wait()
                p2.wait()

                # run_petakit_server.m wraps all per-job work in try/catch and
                # exits 0 on a clean idle-timeout `break`, so a nonzero exit
                # code here only happens when Matlab itself failed to get
                # running (license checkout, verify_mex(), a top-level crash)
                # -- never a job-level failure. Treat that as a launch failure
                # and back off instead of relaunching on the normal short
                # poll_interval.
                if p1.returncode != 0 or p2.returncode != 0:
                    consecutive_failures += 1
                    backoff = _next_backoff_sec(consecutive_failures)
                    print(
                        f"\n❌ Matlab server(s) exited with an error "
                        f"(server1={p1.returncode}, server2={p2.returncode}) -- "
                        "Matlab likely failed to start (check the log above for a "
                        "license/module error). The queued ticket(s) are untouched "
                        f"and will be retried automatically. Backing off {backoff:.0f}s "
                        f"before the next attempt (consecutive failures: {consecutive_failures})."
                    )
                    time.sleep(backoff)
                    continue

                consecutive_failures = 0

                print(
                    f"🛑 Matlab server spun down after {idle_timeout_sec}s "
                    "of inactivity. GPUs released."
                )

                print("\n🔗 Checking for pending OME-Zarr consolidations...")
                try:
                    n = run_pending_consolidations(BASE_DIR)
                    if n:
                        print(f"✅ Consolidated {n} dataset(s) into OME-Zarr")
                    else:
                        print("   No pending consolidations found")
                except Exception as exc:
                    print(f"⚠️  Consolidation error: {exc}")

                print(f"👀 Watchdog resuming listening on {QUEUE_DIR}...")

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        print("\n🛑 Watchdog gracefully shut down.")


def main():
    # 300 seconds = 5 minutes of idle time before releasing the GPU
    process_queue(idle_timeout_sec=3600)

if __name__ == "__main__":
    main()
