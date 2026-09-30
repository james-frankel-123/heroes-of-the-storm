#!/usr/bin/env bash
# Staleness head-to-heads. Usage: run_h2h_batch_stale.sh s2oof_era S2   (or s1oof_era S1)
cd /home/max/heroes-of-the-storm/training
ARM=$1; TAG=$2
M=drift2026:w4_d2c_cumprev,w8_d2c_cumprev:15
python3 -u drift_rebuild/r7_h2h.py --a $M --b rebuild:r6_${ARM}:6 --out M_vs_${TAG} > drift_rebuild/logs/h2h_M_vs_${TAG}.log 2>&1 &
python3 -u drift_rebuild/r7_h2h.py --a rebuild:r6_uoof_rg:6 --b rebuild:r6_${ARM}:6 --out U_vs_${TAG} > drift_rebuild/logs/h2h_U_vs_${TAG}.log 2>&1 &
if [ "$TAG" = "S2" ]; then
python3 -u drift_rebuild/r7_h2h.py --a drift2026:w8_volmatch:5 --b rebuild:r6_${ARM}:6 --out Mvol_vs_${TAG} > drift_rebuild/logs/h2h_Mvol_vs_${TAG}.log 2>&1 &
fi
wait
