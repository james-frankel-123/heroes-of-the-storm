#!/bin/bash
# Shrinkage study arms (no layer unless the flags ask). Usage: run_rcf_shrink_grid.sh NAME:flag,flag ...
set -uo pipefail
cd "$(dirname "$0")/../.."
. personalization/remote/rcf_env.sh "$PWD" rcf_shrink
PY="${RCF_PY:-python} -u personalization/p3_rcf.py"
for spec in "$@"; do
  name=${spec%%:*}; flags=${spec#*:}; [ "$flags" = "$spec" ] && flags=""
  [ -f $P3_RESULTS/rcf/p3c/${name}__merged.npz ] && { echo "skip $name"; continue; }
  echo "=== $name $flags $(date -Is)"
  $PY online --tag full_ms --hist merged --out $name ${flags//,/ } || echo "FAILED $name"
done
echo GRID DONE
