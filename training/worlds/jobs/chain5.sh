#!/bin/bash
# Step 4b: retrained arm, worlds 0-9 of family a (sequential; resumable by artifact).
set -eo pipefail
cd ~/hots/repo/training
WORLDS_GPU=1 WORLDS_THREADS=4 P1R_NPROC_OVERRIDE=8 bash worlds/jobs/run.sh retrain.py queue --wids ${RT_WIDS:-0,1,2,3,4,5,6,7,8,9}
echo CHAIN5_DONE
