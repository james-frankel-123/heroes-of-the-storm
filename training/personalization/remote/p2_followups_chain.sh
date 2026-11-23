#!/bin/bash
# Paper 2 follow-ups, queued behind paper 1's 8-seed training: (1) split round 2 (p3t_split.py, P3T_SPLIT_ROUND=2:
# greedy's bans / no ten-player terms / greedy's candidate set with the drafter's weights), (2) truth-leaf under
# f-cal (worlds/jobs/chain_tl_cal.sh). Launch (main box): P3_CORES=16-35 P3_MEM=60G P3_GPU=1
#   training/personalization/remote/p3_job.sh windows-3090-wsl p2_followups bash personalization/remote/p2_followups_chain.sh
set -eo pipefail
cd /home/max/hots/repo/training
until grep -q "SEEDS8 TRAIN DONE" ~/hots/logs/seeds8_train.log 2>/dev/null || [ -f ~/hots/adhoc/p2_followups_go ]; do sleep 300; done
echo "$(date -Is) gate passed"
P3T_SPLIT_ROUND=2 bash personalization/remote/p3t_split_chain.sh
echo "$(date -Is) split round 2 done"
bash worlds/jobs/chain_tl_cal.sh
echo "P2 FOLLOWUPS DONE $(date -Is)"
