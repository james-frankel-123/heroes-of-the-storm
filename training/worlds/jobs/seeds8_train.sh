#!/bin/bash
# Paper 1, 8 seeds per arm: train F_floor s3-7 (train_mcts.py F <k> --floor <floor kernel>) and F_oof s5-7
# (train_mcts.py F <k>), same config/budget as the originals (400 sims, 300K episodes), two runs at a time on
# the 3090 GPU. Resumable: a finished run (log "Complete. ") is skipped, an interrupted one continues from its
# resume_state.pt (periodic checkpoint every MCTS_CKPT_EVERY_SEC; SIGTERM/pause file -> checkpoint, exit 75).
# Gate (once; marker $D/gate_passed): the party job's log has "# exit=", then no foreign GPU compute process
# and no running hotsjob named *greedy*/*matched* for 15 min, after either seeing such a job come and go or a
# 3 h grace period. touch $D/go to skip the gate.
set -uo pipefail
source ~/hots/env.sh
source ~/hots/td_env.sh
cd ~/hots/repo/training
D=~/hots/adhoc/seeds8; mkdir -p $D
SMI=/usr/lib/wsl/lib/nvidia-smi
KDIR=/home/max/hots/kernels/cuda_mcts_floor_20261111
LOGS=paper1_revision/site/logs
RUNS=paper1_revision/site/mcts_runs
GRACE=${SEEDS8_GRACE:-10800}
log() { echo "[$(date -Is)] $*"; }

foreign_busy() {
  [ -n "$($SMI --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -v '^\s*$')" ] && return 0
  for m in ~/hots/jobs/*.json; do
    python3 - "$m" <<'PY' && return 0
import json, re, sys
d = json.load(open(sys.argv[1]))
sys.exit(0 if d.get("status") == "running" and re.search("greedy|matched", d.get("name", ""), re.I) else 1)
PY
  done
  return 1
}

gate() {
  [ -f $D/gate_passed ] && { log "gate already passed"; return; }
  until grep -q "# exit=" ~/hots/logs/party_pcalf_1120b.log 2>/dev/null || [ -f $D/go ]; do sleep 120; done
  log "party job done (or go file)"
  local t0=$(date +%s) idle=0 seen=0
  while [ ! -f $D/go ]; do
    if foreign_busy; then seen=1; idle=0; else idle=$((idle + 60)); fi
    if [ $idle -ge 900 ] && { [ $seen = 1 ] || [ $(( $(date +%s) - t0 )) -ge $GRACE ]; }; then break; fi
    sleep 60
  done
  log "gate passed (seen_other=$seen)"; touch $D/gate_passed
}

done_run() { grep -q "^Complete\. " $LOGS/mcts_$1.log 2>/dev/null; }

train() {   # cfg seed
  local cfg=$1 k=$2 name extra=""
  if [ $cfg = floor ]; then name=F_floor_s$k; extra="--floor $KDIR"; else name=F_oof_s$k; fi
  done_run $name && { log "$name already complete"; return 0; }
  local res=""; [ -f $RUNS/$name/resume_state.pt ] && res="--resume"
  log "train $name $res"
  env $P1ENV MCTS_CKPT_EVERY_SEC=240 MCTS_PAUSE_FILE=$D/pause OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    nice -n 19 taskset -c 16-35 python -u paper1_revision/train_mcts.py F $k --gpu 0 $extra $res
  local rc=$?
  log "$name exit $rc"
  return $rc
}

gate
rm -f $D/pause
for pair in "floor 3 oof 5" "floor 4 oof 6" "floor 5 oof 7" "floor 6 floor 7"; do
  set -- $pair
  train $1 $2 & p1=$!
  sleep 60
  train $3 $4 & p2=$!
  wait $p1; r1=$?; wait $p2; r2=$?
  if [ $r1 != 0 ] || [ $r2 != 0 ]; then log "a run did not complete (rc $r1 $r2); stopping (rerun to resume)"; exit 1; fi
done
log "SEEDS8 TRAIN DONE"
