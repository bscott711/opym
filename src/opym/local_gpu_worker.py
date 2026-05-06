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
from pathlib import Path

PETAKIT_JOBS_DIR = Path.home() / "petakit_jobs"
QUEUE_DIR = PETAKIT_JOBS_DIR / "queue"


def _ensure_directories():
    """Ensures all necessary job directories exist."""
    for directory in (
        QUEUE_DIR,
        PETAKIT_JOBS_DIR / "completed",
        PETAKIT_JOBS_DIR / "failed",
    ):
        directory.mkdir(parents=True, exist_ok=True)


def process_queue(idle_timeout_sec: int = 300, poll_interval: int = 2):
    """
    Watches the queue. If jobs exist, launches the persistent Matlab server.
    The Matlab server handles the jobs and shuts itself down after `idle_timeout_sec`.
    """
    _ensure_directories()
    print(f"👀 Watchdog listening on {QUEUE_DIR}...")

    # Pass the timeout to Matlab via environment variables
    env = os.environ.copy()
    env["PETAKIT_IDLE_TIMEOUT"] = str(idle_timeout_sec)

    try:
        while True:
            # Check if there are any JSON files in the queue
            if any(QUEUE_DIR.glob("*.json")):
                print("\n🚀 Jobs detected. Spinning up PetaKit Matlab Server...")

                cmd = [
                    "matlab",
                    "-nodisplay",
                    "-batch",
                    "run_petakit_server",
                ]

                # This will block until the Matlab script completes
                # its queue AND its timeout
                subprocess.run(cmd, env=env, check=True)  # nosec B603

                print(
                    f"🛑 Matlab server spun down after {idle_timeout_sec}s "
                    "of inactivity. GPUs released."
                )
                print(f"👀 Watchdog resuming listening on {QUEUE_DIR}...")

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        print("\n🛑 Watchdog gracefully shut down.")


if __name__ == "__main__":
    # 300 seconds = 5 minutes of idle time before releasing the GPU
    process_queue(idle_timeout_sec=300)
