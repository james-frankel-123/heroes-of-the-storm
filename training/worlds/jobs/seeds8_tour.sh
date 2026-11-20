#!/bin/bash
# Paper 1, 8 seeds per arm: per-seed tournament drafts (paper1_revision/seed_tour.py, CPU) for every trained
# checkpoint as it appears, then the worlds rescoring and seed-level contrasts (worlds/jobs/p1_seeds8.py) under
# the log and the flat grade. Resumable: finished pairs and the rescoring cache are skipped.
# Polls every 20 min until all 16 checkpoints are trained and played (or until $D/tour_once exists: one pass).
set -uo pipefail
source ~/hots/env.sh
source ~/hots/td_env.sh
cd ~/hots/repo/training
D=~/hots/adhoc/seeds8; mkdir -p $D
P=${SEEDS8_PROCS:-6}
log() { echo "[$(date -Is)] $*"; }
T="env $P1ENV nice -n 19 taskset -c 16-35 python -u paper1_revision/seed_tour.py"
R=paper1_revision/site/results/seed_tour/tournament
nfiles() { ls $R 2>/dev/null | grep -c "\.json$"; }
while true; do
  n0=$(nfiles)
  $T queue --kinds policy --procs $P
  [ "$(nfiles)" != "$n0" ] && bash worlds/jobs/seeds8_analyze.sh   # interim results as checkpoints arrive
  $T queue --kinds search --procs $P
  st=$($T status); echo "$st"
  ntr=$(echo "$st" | grep "trained seeds" | python3 -c "import sys,ast; print(sum(len(ast.literal_eval(l.split(':',1)[1])) for l in sys.stdin))")
  left=$(echo "$st" | grep -o "[0-9]* pairs to play" | awk '{s+=$1} END {print s}')
  log "trained checkpoints $ntr/16, pairs left $left"
  if { [ "$ntr" = 16 ] && [ "$left" = 0 ]; } || [ -f $D/tour_once ]; then break; fi
  sleep 1200
done
bash worlds/jobs/seeds8_analyze.sh
log "SEEDS8 TOUR+ANALYSIS DONE"
