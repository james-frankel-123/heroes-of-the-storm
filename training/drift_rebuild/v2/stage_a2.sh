#!/usr/bin/env bash
# v2 stage A2 (after stage A): matched-clock statistics + feature passes.
cd /home/max/heroes-of-the-storm/training
until [ -f drift_v2/logs/stage_a.done ]; do sleep 60; done
R="nice -n 19 taskset -c 48-63 python3 -u drift_rebuild/v2/run.py"
L=drift_v2/logs
export OMP_NUM_THREADS=2 V2_WORKERS=12
$R drift2026/build_matched_clock_stats.py > $L/a5_matched_clock.log 2>&1
for p in decayedmc_prev decayedmcrev_prev; do
  $R drift2026/build_drift_features.py --only $p > $L/a3_feat_$p.log 2>&1
done
touch $L/stage_a2.done
