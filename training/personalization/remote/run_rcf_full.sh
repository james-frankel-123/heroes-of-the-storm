#!/bin/bash
# Rich-CF stages on the full_1017 panel (3090, from ~/hots/repo/training).
# Usage: bash personalization/remote/run_rcf_full.sh <p3_rcf.py args...>
set -euo pipefail
B=/home/max/hots/repo/training/personalization
export P3_CACHE=$B/cache_full P3_RESULTS=$B/results/rcf_full \
  P3_RCF_ROWS=$B/cache_full/merged_rows_full_1017.npz P3_RCF_KERNELS=$B/cache_full/hs_kernels.npz
python -u personalization/${RCF_SCRIPT:-p3_rcf.py} "$@"
