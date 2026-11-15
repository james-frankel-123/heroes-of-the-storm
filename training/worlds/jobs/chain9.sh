#!/bin/bash
# Step 3: exploration simulations + value-of-information map (3090 CPU; resumable).
set -eo pipefail
R="bash worlds/jobs/run.sh"
cd ~/hots/repo/training
WORLDS_THREADS=1 $R explore.py queue --procs ${EXP_PROCS:-24} --worlds ${EXP_WORLDS:-10}
WORLDS_THREADS=1 $R explore.py summary
WORLDS_THREADS=1 $R explore.py voi_queue --procs 20 --worlds 10
WORLDS_THREADS=1 $R explore.py voi_summary
echo CHAIN9_DONE
