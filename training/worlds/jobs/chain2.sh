#!/bin/bash
# Milestone-1 plausibility checks on the 3090 (resumable: finished outputs are skipped).
set -eo pipefail
R="bash worlds/jobs/run.sh"
cd ~/hots/repo/training
res=worlds/site/results
export WORLDS_PROCS=${WORLDS_PROCS:-16} WORLDS_THREADS=1
for f in a b c d e g; do [ -f $res/check_step1_${f}_default.json ] || $R checks.py step1 --fam $f --n 200 --dial default; done
for d in stress certify14; do [ -f $res/check_step1_a_$d.json ] || $R checks.py step1 --fam a --n 200 --dial $d; done
[ -f $res/check_logloss.json ] || WORLDS_THREADS=8 $R checks.py logloss --n 50
for f in a b c d; do [ -f $res/check_allowed_$f.json ] || $R checks.py allowed --fam $f --n 20; done
[ -f $res/check_pairs.json ] || WORLDS_PROCS=6 $R checks.py pairs --n 3
WORLDS_THREADS=16 $R recover.py --worlds 900,901,902
echo CHAIN2_DONE
