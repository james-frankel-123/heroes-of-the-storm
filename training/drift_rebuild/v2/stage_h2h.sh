#!/usr/bin/env bash
# v2 head-to-heads (r7 protocol) for Secs. 5 and 8, then judge-free scoring (r8)
# and the structure-aware Net index (r15). Three matchups at a time, 4 threads
# each (12 cores; the judge seeds take 4 more). Run from training/ on a worker, under hotsjob with a cap.
set -uo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p drift_v2/logs/h2h
run() { nice -n 19 python3 -u drift_rebuild/v2/run.py drift_rebuild/r7_h2h.py --a "rebuild:$2" --b "rebuild:$3" --out "$1" > "drift_v2/logs/h2h/$1.log" 2>&1 || echo "FAILED $1"; }
export -f run
cat <<'L' | xargs -P 3 -L 1 bash -c 'run "$0" "$1" "$2"'
M_vs_U v2_M:6 v2_U:6
U_vs_S2 v2_U:6 v2_S2:6
M_vs_S2 v2_M:6 v2_S2:6
M_vs_S1 v2_M:6 v2_S1:3
U_vs_S1 v2_U:6 v2_S1:3
M_vs_Mcut v2_M:6 v2_Mcut:3
Mcut_vs_U v2_Mcut:3 v2_U:6
Md90_vs_U v2_Md90:3 v2_U:6
Md90_vs_M v2_Md90:3 v2_M:6
Mvol_vs_S2 v2_Mvol:3 v2_S2:6
S1_vs_S2 v2_S1:3 v2_S2:6
L
n=$(ls drift_v2/results/h2h/*.json 2>/dev/null | wc -l)
echo "h2h files: $n"
[ "$n" -eq 11 ] || { echo "missing head-to-heads"; exit 1; }
nice -n 19 python3 -u drift_rebuild/v2/run.py drift_rebuild/r8_score.py > drift_v2/logs/r8_v2.log 2>&1 || { echo "r8 failed"; exit 1; }
nice -n 19 python3 -u drift_rebuild/v2/run.py drift_rebuild/r15_net_struct.py > drift_v2/logs/r15_v2.log 2>&1 || { echo "r15 failed"; exit 1; }
echo "stage_h2h done"
