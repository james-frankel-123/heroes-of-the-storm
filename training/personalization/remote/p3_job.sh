#!/bin/bash
# Launch a P3 job on a worker as a pause-aware hotsjob plain job.
# Usage: P3_CORES=28-31 P3_MEM=48G training/personalization/remote/p3_job.sh <host> <name> <command...>
#   CPU: pinned to P3_CORES (default 28-31; B dev jobs 32-33, GPU-side jobs 34-35) with 8 threads;
#   memory: per-process RLIMIT_DATA P3_MEM (default 48G); results to
#   personalization/results/oct26 (P3_RESULTS). GPU jobs pass P3_GPU=1.
set -euo pipefail
HOST=${1:?host}; NAME=${2:?name}; shift 2
CORES=${P3_CORES:-28-31}
THREADS=${P3_THREADS:-8}
MEM=${P3_MEM:-48G}
ENVV="P3_RESULTS=/home/max/hots/repo/training/personalization/results/oct26 OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS NUMBA_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS PYTHONUNBUFFERED=1"
CMD="$ENVV nice -n 19 taskset -c $CORES $*"
RUN_NAME="$NAME" HOTSJOB_FLAGS="--mem-max $MEM" "$(dirname "$0")/../../remote_workers/run_remote.sh" "$HOST" "$CMD"
