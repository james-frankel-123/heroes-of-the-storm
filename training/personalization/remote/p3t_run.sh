#!/bin/bash
# P3 tournament: run p3t stages in order on a worker (from ~/hots/repo/training).
# Usage: bash personalization/remote/p3t_run.sh p3t_pprior.py:targets:--lobbies:3000 p3t_tourney.py:run:--schedule:core
#   (one step per argument; ':' separates the step's own arguments)
set -uo pipefail
B=/home/max/hots/repo/training/personalization
export P3_CACHE=$B/cache_full P3_RESULTS=$B/results/rcf_full \
  P3_RCF_ROWS=$B/cache_full/merged_rows_full_1017.npz P3_RCF_KERNELS=$B/cache_full/hs_kernels.npz
for step in "$@"; do
  echo "=== ${step//:/ } $(date -Is)"
  # peak memory per stage (for host planning; the 3090 is out until its RAM is tested)
  TM=""; [ -x /usr/bin/time ] && TM="/usr/bin/time -f MAXRSS_KB=%M_SECS=%e"
  $TM python -u personalization/${step//:/ } || { echo "FAILED $step"; exit 1; }
done
echo "P3T RUN DONE"
