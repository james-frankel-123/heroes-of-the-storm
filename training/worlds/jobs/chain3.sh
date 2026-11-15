#!/bin/bash
# Rescoring (resumable). Waits for the checks chain first.
set -eo pipefail
source ~/hots/env.sh
cd ~/hots/repo/training
until python remote_workers/hotsjob.py state worlds_chain2b | grep -qE "completed|failed"; do sleep 60; done
R="bash worlds/jobs/run.sh"
export WORLDS_PROCS=${WORLDS_PROCS:-20} WORLDS_THREADS=1
$R rescore.py p1 --fams a,b,c,d,e,g --n 1000
$R rescore.py p2 --fams a,b,c,d,e,f --n 1000
echo CHAIN3_DONE
