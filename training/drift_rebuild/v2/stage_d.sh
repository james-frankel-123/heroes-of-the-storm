#!/usr/bin/env bash
# v2 stage D: descriptive analyses that need only statistics (CPU).
cd /home/max/heroes-of-the-storm/training
R="nice -n 19 taskset -c 48-63 python3 -u drift_rebuild/v2/run.py"
L=drift_v2/logs
export OMP_NUM_THREADS=2 V2_WORKERS=8
$R drift2026/signal_decay.py > $L/d_signal_decay.log 2>&1
$R drift2026/w3_neverfielded.py > $L/d_neverfielded.log 2>&1
$R drift2026/w3_recovery.py --workers 8 > $L/d_recovery.log 2>&1
$R drift2026/w3_changepoints.py > $L/d_changepoints.log 2>&1
$R drift2026/blind_patch_multiscale.py > $L/d_blind.log 2>&1
touch $L/stage_d.done
