#!/bin/bash
# hotsjob entry for the worlds module on the 3090: CPU only (CUDA hidden unless WORLDS_GPU=1),
# paper-1 site env. Every stage skips finished outputs, so resume = re-run.
# Usage (from training/): bash worlds/jobs/run.sh <script.py> <args...>
set -eo pipefail
source ~/hots/env.sh
source ~/hots/td_env.sh
cd ~/hots/repo/training
[ "${WORLDS_GPU:-0}" = 1 ] && export CUDA_VISIBLE_DEVICES=0
export WORLDS_THREADS=${WORLDS_THREADS:-6}
exec env $P1ENV OMP_NUM_THREADS=$WORLDS_THREADS MKL_NUM_THREADS=$WORLDS_THREADS nice -n 10 python -u "worlds/$@"
