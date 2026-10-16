#!/bin/bash
# Several p3_c_tests.py predict arms in one job (from ~/hots/repo/training).
# Usage: bash personalization/remote/run_p3c_arms.sh <cache dir> <rows npz> NAME:flag,flag ...
#   e.g. A2S:--history=merged,--exp-counts=slots
set -euo pipefail
export P3_CACHE=${1:?cache}; ROWS=${2:?rows}; shift 2
for spec in "$@"; do
  name=${spec%%:*}; flags=${spec#*:}; [ "$flags" = "$spec" ] && flags=""
  python personalization/p3_c_tests.py predict --arm "$name" --rows "$ROWS" ${flags//,/ }
done
