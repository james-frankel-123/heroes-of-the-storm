#!/bin/bash
# f-cal (player.PlayerWorldsCal) on the 3090, CPU only: the slope check (200 worlds) and paper 2's headline and
# tune rescoring (1,000 worlds, rescore_cal.py), for the slope-matched kcal (player_fitcal.json) and kcal = 1
# (tag _k1). New outputs only: check_player_slope_cal*.json, rescore_p2_<lset>_fcal*.npz, rescore_p2_cal.json.
# Launch (main box): P3_CORES=16-35 P3_MEM=40G training/personalization/remote/p3_job.sh windows-3090-wsl \
#   worlds_fcal bash worlds/jobs/chain_cal.sh
set -eo pipefail
cd /home/max/hots/repo/training
export WORLDS_PROCS=${WORLDS_PROCS:-6} WORLDS_THREADS=1
R="bash worlds/jobs/run.sh"
[ -f worlds/site/results/check_player_slope_cal.json ] || $R player.py slopecal --n 200
[ -f worlds/site/results/check_player_slope_cal_k1.json ] || $R player.py slopecal --n 200 --kcal 1.0 --tag _k1
$R rescore_cal.py p2 --n 1000
$R rescore_cal.py p2 --n 1000 --kcal 1.0 --tag _k1
echo CHAIN_CAL_DONE
