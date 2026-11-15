#!/bin/bash
# Paper-2 personal-term diagnosis (worlds/diag_p2.py): one leaf part swapped to the truth at a time, tune lobbies,
# f-cov then f-out (3090 GPU; resumable: every stage skips finished pair files).
set -eo pipefail
source ~/hots/env.sh
cd /home/max/hots/repo/training
export P3T_LOBBYSET=tune
R="bash worlds/jobs/run_p2.sh diag_p2.py"
for fam in ${DIAG_FAMS:-fc f}; do
  [ -f worlds/site/results/diag_p2_oracle_$fam.json ] || $R oracle --fam $fam --worlds 10
  $R queue --fam $fam --worlds ${DIAG_WORLDS:-8}
  $R score --fam $fam --worlds ${DIAG_WORLDS:-8}
done
echo CHAIN10_DONE
