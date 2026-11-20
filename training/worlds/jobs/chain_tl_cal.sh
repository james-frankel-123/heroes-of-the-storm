#!/bin/bash
# Truth-leaf regret under f-cal (tl_p2.py with TL_P2_PLAN=fcal; 3090 GPU, after rescore_cal.py headline):
# 20 worlds, TL-sel and TL-f25 vs pMCTS-cur both seats -> site/results/truthleaf_p2_cal.json
set -eo pipefail
cd /home/max/hots/repo/training
export TL_P2_PLAN=fcal TL_P2_WORLDS=${TL_P2_WORLDS:-20}
bash worlds/jobs/run_p2.sh tl_p2.py queue
bash worlds/jobs/run_p2.sh tl_p2.py score
echo CHAIN_TL_CAL_DONE
