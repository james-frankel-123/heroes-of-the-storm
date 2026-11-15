#!/bin/bash
# GPU queue after chain7: paper-2 truth-leaf against the second tune opponent (aMCTS-pp), same plan.
set -eo pipefail
source ~/hots/env.sh
cd /home/max/hots/repo/training
until python remote_workers/hotsjob.py state worlds_tl_p2 | grep -qE "completed|failed"; do sleep 60; done
TL_P2_OPP=aMCTS-pp bash worlds/jobs/run_p2.sh tl_p2.py queue
TL_P2_OPP=aMCTS-pp bash worlds/jobs/run_p2.sh tl_p2.py score
echo CHAIN8_DONE
