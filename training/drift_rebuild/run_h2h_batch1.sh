#!/usr/bin/env bash
# Head-to-heads that need only the leak-free unmaintained agents (r6_uoof_rg).
cd /home/max/heroes-of-the-storm/training
M=drift2026:w4_d2c_cumprev,w8_d2c_cumprev:15
python3 -u drift_rebuild/r7_h2h.py --a $M --b rebuild:r6_uoof_rg:6 --out M_vs_U > drift_rebuild/logs/h2h_M_vs_U.log 2>&1 &
python3 -u drift_rebuild/r7_h2h.py --a rebuild:r6_uoof_rg:6 --b drift2026:w4_d2b_allhist,w8_d2b_allhist:15 --out U_vs_Uleaky > drift_rebuild/logs/h2h_U_vs_Uleaky.log 2>&1 &
python3 -u drift_rebuild/r7_h2h.py --a drift2026:w8_champion:15 --b rebuild:r6_uoof_rg:6 --out Md90_vs_U > drift_rebuild/logs/h2h_Md90_vs_U.log 2>&1 &
wait
