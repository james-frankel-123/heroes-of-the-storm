#!/bin/bash
# Step 4a: paper-1 truth-leaf search queue, then scoring (resumable).
set -eo pipefail
R="bash worlds/jobs/run.sh"
cd ~/hots/repo/training
WORLDS_THREADS=1 $R truthleaf.py queue --procs ${TL_PROCS:-24}
WORLDS_THREADS=4 $R truthleaf.py score
echo CHAIN4_DONE
