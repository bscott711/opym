# Shared settings for the isolated live-view test stack (sourced).
# Everything lives under its own RAM-disk root, jobs dir, port and GPFS
# folder; production (/dev/shm/petakit_jobs, port 5555) is never touched.
LB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT="$(cd "$LB_DIR/../.." && pwd)"                 # the opym checkout under test
PY="${LB_PY:-$HOME/projects/bioimaging/.venv/bin/python}"
LV="${LB_ROOT:-/dev/shm/opym_lv}"
JOBS="$LV/jobs"
STAGE="$LV/stage"
VIEW="$LV/view"
PORT="${LB_PORT:-5602}"                              # the replay's default --remote-port
RAW_ROOT="${LB_RAW_ROOT:-/mmfs2/scratch/SDSMT.LOCAL/bscott/opym_lv/raw}"
PSF="${LB_PSF:-/mmfs2/scratch/SDSMT.LOCAL/bscott/DataUpload/PSF/20260910_averaged_psf.tif}"
RESULTS="${LB_RESULTS:-$HOME/projects/bioimaging/logs/live-view-2026-09-26}"
UNITS="lv-serve lv-receive lv-qc lv-sample"
