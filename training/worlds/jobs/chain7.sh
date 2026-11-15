#!/bin/bash
# Step 4a, paper 2: truth-leaf drafting on the 3090 GPU, then scoring (resumable: finished settings skipped).
set -eo pipefail
cd /home/max/hots/repo/training
bash worlds/jobs/run_p2.sh tl_p2.py queue
bash worlds/jobs/run_p2.sh tl_p2.py score
echo CHAIN7_DONE
