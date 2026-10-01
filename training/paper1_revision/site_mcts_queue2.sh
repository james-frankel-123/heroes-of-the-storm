#!/bin/bash
# Second wave of paper-1 MCTS retrains (site tiers, chance-node kernel), behind
# site_mcts_queue.sh (B/F/J_oof):
#   3090:     B_leak s0..4 (200 sims, in-sample-statistics control WP), then
#             K_oof s0..4 (400 sims, naive leaf WP: the no-features agent);
#             at most two p1site MCTS jobs at a time, pause-aware hotsjob jobs
#   this box: stream 1 F_leak s0..4 then E_oof s0..2; stream 2 J_leak s0..4 then
#             E_oof s3..4 (E = 200 sims, 1M episodes); each run starts only while
#             this lane holds fewer than 2 GPU processes, and only after the
#             first wave's local runs are done
# Usage (from training/): setsid nohup bash paper1_revision/site_mcts_queue2.sh &
set -u
cd "$(dirname "$0")/.."
SITE=paper1_revision/site
LOG=$SITE/logs/site_mcts_queue2.log
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

remote_running() {
  remote_workers/jobs_remote.sh max-windows-3090 2>/dev/null | grep -cE "p1site_[A-Z]_(oof|leak)_s[0-9] .*running"
}
remote_done() {   # $1 = run name; finished = hotsjob status done
  remote_workers/jobs_remote.sh max-windows-3090 2>/dev/null | grep -qE "p1site_$1 +(done|finished)"
}
launch_remote() {   # $1 = config, $2 = tag (oof|leak), $3 = seed
  local run=${1}_${2}_s$3 wp=""
  [ "$2" = leak ] && wp="--wp enriched_leak"
  RUN_NAME=p1site_$run \
  HOTSJOB_FLAGS="--save-dir $SITE/mcts_runs/$run --progress-log $SITE/logs/mcts_$run.log" \
    remote_workers/run_remote.sh max-windows-3090 \
    env P1_TIERS=site python paper1_revision/train_mcts.py $1 $3 --gpu 0 $wp | tail -1 | tee -a "$LOG"
}
(
  # wait for the first wave on the 3090 to be launched (it launches B_oof s4 last)
  until remote_workers/jobs_remote.sh max-windows-3090 2>/dev/null | grep -q "p1site_B_oof_s4"; do sleep 600; done
  for job in "B leak 0" "B leak 1" "B leak 2" "B leak 3" "B leak 4" \
             "K oof 0" "K oof 1" "K oof 2" "K oof 3" "K oof 4"; do
    set -- $job
    while [ "$(remote_running)" -ge 2 ]; do sleep 300; done
    launch_remote $1 $2 $3
    sleep 900
  done
) &

lane_gpu_procs() {
  n=0
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
    tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qE "^(P1_TIERS=site|RERUN_NS=p1site)$" && n=$((n+1))
  done
  echo $n
}
pick_gpu() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
    awk -F', ' '$1==1||$1==2 {print $2, $1}' | sort -rn | head -1 | awk '{print $2}'
}
wave1_local_done() {
  for c in F J; do for s in 0 1 2 3 4; do [ -f $SITE/mcts_runs/${c}_oof_s$s/.done ] || return 1; done; done
}
run_local() {   # $1 = config, $2 = tag, $3 = seed
  local run=${1}_${2}_s$3 wp=""
  [ "$2" = leak ] && wp="--wp enriched_leak"
  [ -f $SITE/mcts_runs/$run/.done ] && return
  until [ "$(lane_gpu_procs)" -lt 2 ]; do sleep 120; done
  local g; g=$(pick_gpu)
  log "local start $run on GPU $g"
  P1_TIERS=site nice -n 19 taskset -c 48-63 python3 paper1_revision/train_mcts.py $1 $3 --gpu $g $wp \
    && touch $SITE/mcts_runs/$run/.done && log "local done $run" || log "local FAILED $run"
}
until wave1_local_done; do sleep 600; done
( for s in 0 1 2 3 4; do run_local F leak $s; done; for s in 0 1 2; do run_local E oof $s; done ) &
sleep 600
( for s in 0 1 2 3 4; do run_local J leak $s; done; for s in 3 4; do run_local E oof $s; done ) &
wait
log "second MCTS wave finished"
