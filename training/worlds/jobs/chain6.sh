#!/bin/bash
# Step 4c: family f bracket (f-out = v2 outcome-only, f-cov = + drafter-state covariate), resumable.
set -eo pipefail
R="bash worlds/jobs/run.sh"
cd ~/hots/repo/training
res=worlds/site/results
[ -f worlds/site/cache/player_fitcov.json ] || WORLDS_THREADS=4 $R player.py fitcov
[ -f $res/check_player_slope_cov.json ] || WORLDS_THREADS=4 $R player.py slopecov --n 200
export WORLDS_PROCS=${WORLDS_PROCS:-8} WORLDS_THREADS=1
$R rescore.py p1 --fams f,fc --n 1000
$R rescore.py p2 --fams fc --n 1000
$R rescore.py summary
echo CHAIN6_DONE
