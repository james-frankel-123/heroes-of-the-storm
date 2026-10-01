#!/usr/bin/env bash
# v2 stage A: statistics and feature caches on site-tiered data.
set -e
cd /home/max/heroes-of-the-storm/training
R="nice -n 19 taskset -c 48-63 python3 -u drift_rebuild/v2/run.py"
L=drift_v2/logs
export OMP_NUM_THREADS=2 V2_WORKERS=12
$R drift2026/build_patch_stats.py > $L/a1_patch_stats.log 2>&1
$R drift2026/build_decayed_stats.py > $L/a2_decayed.log 2>&1
for p in cumulative_prev cutoff decayed90_prev decayed365_prev decayed90k100_prev; do
  $R drift2026/build_drift_features.py --only $p > $L/a3_feat_$p.log 2>&1
done
$R drift2026/build_drift_features.py --only cutoff_2.55.3.89754 --cutoff-at 2.55.3.89754 > $L/a3_feat_cutoff_c0.log 2>&1
for b in 2.55.14.95918 2.55.9.93613 2.55.4.91418 2.55.3.89754; do
  $R drift_rebuild/r1_oof_features.py --cutoff $b > $L/a4_oof_$b.log 2>&1
done
echo STAGE_A_DONE > $L/stage_a.done
