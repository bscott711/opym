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

# Dynamically locate the opym installation directory
OPYM_DIR = Path(__file__).parent.resolve()

BASE_DIR = Path("/dev/shm/petakit_jobs")
QUEUE_DIR = BASE_DIR / "queue"


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
    print(f" 🔧 Backend Script:  {OPYM_DIR}/run_petakit_server.m")
    print("=" * 60)
    print("👀 Listening for incoming jobs...\n")

    # Pass the timeout to Matlab via environment variables
    env = os.environ.copy()
    env["PETAKIT_IDLE_TIMEOUT"] = str(idle_timeout_sec)

    try:
        while True:
            # Check if there are any JSON files in the queue
            if any(QUEUE_DIR.glob("*.json")):
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

                print(
                    f"🛑 Matlab server spun down after {idle_timeout_sec}s "
                    "of inactivity. GPUs released."
                )
                print(f"👀 Watchdog resuming listening on {QUEUE_DIR}...")

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        print("\n🛑 Watchdog gracefully shut down.")


def main():
    # 300 seconds = 5 minutes of idle time before releasing the GPU
    process_queue(idle_timeout_sec=3600)

if __name__ == "__main__":
    main()
