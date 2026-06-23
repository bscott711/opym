#!/bin/bash
# launch_petakit_servers.sh
# Spawns two instances of the PetaKit server, each locked to a separate GPU.

echo "🚀 Launching PetaKit Multi-GPU Server Cluster..."

# Ensure we're in the right directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR/src/opym" || exit 1

# Launch Server 1 (GPU 1)
PETAKIT_GPU_ID=1 nohup matlab -nodisplay -nosplash -nodesktop -r "run_petakit_server" > server_gpu1.log 2>&1 &
PID1=$!
echo "✅ Server 1 (GPU 1) started with PID $PID1. Logging to src/opym/server_gpu1.log"

# Launch Server 2 (GPU 2)
PETAKIT_GPU_ID=2 nohup matlab -nodisplay -nosplash -nodesktop -r "run_petakit_server" > server_gpu2.log 2>&1 &
PID2=$!
echo "✅ Server 2 (GPU 2) started with PID $PID2. Logging to src/opym/server_gpu2.log"

echo "🎉 Cluster is now active and watching the queue in the background!"
