#!/bin/bash
# Local evaluation of rich-CF variants fitted on the 5090: fetch params, then
# online (V1/V2/OOT), Cassia and coverage. Usage: run_rcf_eval.sh HIST TAG ...
set -uo pipefail
HIST=$1; shift
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMBA_NUM_THREADS=4
PY="nice -n 19 taskset -c 48-51 /usr/bin/python3 personalization/p3_rcf.py"
P=personalization/results/rcf/params
mkdir -p $P
for tag in "$@"; do
  until ssh windows-5090-wsl "test -f ~/hots/repo/training/personalization/results/rcf/params/$tag.json"; do sleep 60; done
  rsync -t "windows-5090-wsl:hots/repo/training/personalization/results/rcf/params/${tag}*" $P/
  echo "=== $tag $(date -Is)"
  case "$tag" in *marg*) $PY heldout --tag $tag --hist $HIST ;; esac
  [ -f personalization/results/rcf/p3c/${tag}__$HIST.npz ] || $PY online --tag $tag --hist $HIST 2>&1 | grep -v "online walk"
  $PY cassia --tag $tag --hist $HIST
  $PY cover --tag $tag --hist $HIST
done
echo "ALL DONE"
