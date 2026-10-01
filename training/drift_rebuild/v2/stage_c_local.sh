#!/usr/bin/env bash
# v2 stage C (local CPU, after the value functions are fetched back):
# calibration, replay, detector replay arm/curves, partial refresh, nested
# selection, matched clock, decayed-aggregate z, misc artifacts.
cd /home/max/heroes-of-the-storm/training
R="nice -n 19 taskset -c 48-63 python3 -u drift_rebuild/v2/run.py"
L=drift_v2/logs
export OMP_NUM_THREADS=4 V2_WORKERS=12 CUDA_VISIBLE_DEVICES=""
$R drift_rebuild/r3_eval_vf.py > $L/c_r3.log 2>&1
$R drift_rebuild/audit_fixes/c2_calib_common.py > $L/c_c2.log 2>&1
$R drift_rebuild/r9_replay.py > $L/c_r9.log 2>&1
$R drift2026/summarize_w2.py --only c > $L/c_w2c.log 2>&1
$R drift2026/w2c_detector_arm.py --num-workers 12 > $L/c_detarm.log 2>&1
$R drift2026/detector_curves.py --num-workers 12 > $L/c_detcurves.log 2>&1
$R drift2026/phase_q5_partial_refresh.py --skip-sanity > $L/c_q5.log 2>&1
$R drift2026/w8c_nested_selection.py > $L/c_w8c.log 2>&1
$R drift2026/summarize_matched_clock.py > $L/c_q12.log 2>&1
$R drift_rebuild/audit_fixes/c3_decayed_z.py > $L/c_c3.log 2>&1
$R drift_rebuild/audit_fixes/c4_misc.py > $L/c_c4.log 2>&1
$R drift_rebuild/audit_fixes/c7_oof_missingness.py > $L/c_c7.log 2>&1
touch $L/stage_c.done
