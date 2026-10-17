#!/bin/bash
# Rich-CF fits (p3_rcf.py fit) in one GPU job, one spec per argument.
# Usage (from ~/hots/repo/training): bash personalization/remote/run_rcf_fits.sh TAG:HIST:flag,flag ...
#   e.g. ard8:merged:--kmax=8,--ard
set -uo pipefail
for spec in "$@"; do
  IFS=: read -r tag hist flags <<< "$spec"
  if [ -f "personalization/results/rcf/params/$tag.npz" ]; then echo "skip $tag (done)"; continue; fi
  echo "=== $tag $hist $flags $(date -Is)"
  python -u personalization/p3_rcf.py fit --tag "$tag" --hist "$hist" ${flags//,/ } ${RCF_EXTRA:-} || echo "FAILED $tag"
done
