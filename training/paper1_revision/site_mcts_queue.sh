#!/bin/bash
# Paper-1 MCTS reruns on site tiers with the fixed kernel (MCTS_SEARCH_MODE=chance).
# Waits until the site GD pool (rerun2026/ns/p1site, fetched from the 3090 by
# site_fetch.sh) and the site leak-free enriched WP exist, then:
#   3090 (max-windows-3090): B_oof s0..s4 (200 sims; ~10 ep/s there while the
#        baseline queues share the GPU), two at a time, pause-aware hotsjob MCTS
#        jobs (checkpoint every 480 s, resume with --resume)
#   this box: two streams, F_oof s0..s4 and J_oof s0..s4 (about 3.7 h and 7.5 h
#        per run), each starting a run only while this lane holds fewer than 2
#        GPU processes; nice 19, cores 48-63, GPU with the most free memory
#        among 1,2
# Then bench_mcts for every finished run (site namespace).
# Usage (from training/): nohup paper1_revision/site_mcts_queue.sh &
set -u
cd "$(dirname "$0")/.."
NS=rerun2026/ns/p1site/models
SITE=paper1_revision/site
LOG=$SITE/logs/site_mcts_queue.log
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
ready() {
  for i in 0 1 2 3 4; do [ -f $NS/meta/gd_$i.json ] || return 1; done
  [ -f $SITE/models/enriched.json ]
}
until ready; do sleep 300; done
log "GD pool and site WP present"
P1_TIERS=site python3 -c "import sys; sys.path.insert(0, '.'); from paper1_revision import train_mcts; print(train_mcts.exclude_path())" >> "$LOG" 2>&1

# ── remote: push site WP/stats/GD pool, launch J_oof ──
L=$(mktemp)
{ find $SITE/models -name "enriched*" ; find $SITE/cache/stats -name "deploy*"; \
  echo $SITE/cache/paper_test_ids.json; echo $SITE/cache/mcts_pretrain_exclude.json; ls $NS/generic_draft_[0-4].pt; } | sed 's#^#training/#' > "$L"
(cd .. && nice -n 19 rsync -rlt --rsync-path="wsl rsync" -z --files-from="$L" ./ max-windows-3090:/home/max/hots/repo/) \
  && log "synced site WP, stats, GD pool to 3090"
rm -f "$L"
launch_remote() {   # $1 = seed
  RUN_NAME=p1site_B_oof_s$1 \
  HOTSJOB_FLAGS="--save-dir paper1_revision/site/mcts_runs/B_oof_s$1 --progress-log paper1_revision/site/logs/mcts_B_oof_s$1.log" \
    remote_workers/run_remote.sh max-windows-3090 \
    env P1_TIERS=site python paper1_revision/train_mcts.py B "$1" --gpu 0 | tail -1 | tee -a "$LOG"
}
remote_running() {
  remote_workers/jobs_remote.sh max-windows-3090 2>/dev/null | grep -c "p1site_B_oof_s[0-9].*running"
}
(
  for s in 0 1 2 3 4; do
    while [ "$(remote_running)" -ge 2 ]; do sleep 300; done
    launch_remote $s
    sleep 900
  done
) &

# ── local: two streams (F and J) ──
lane_gpu_procs() {   # GPU processes of this lane (site tiers or the p1site namespace)
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
stream() {   # $1 = config
  for s in 0 1 2 3 4; do
    d=$SITE/mcts_runs/${1}_oof_s$s
    [ -f $d/.done ] && continue
    until [ "$(lane_gpu_procs)" -lt 2 ]; do sleep 120; done
    g=$(pick_gpu)
    log "local start ${1}_oof_s$s on GPU $g"
    P1_TIERS=site nice -n 19 taskset -c 48-63 python3 paper1_revision/train_mcts.py $1 $s --gpu $g \
      && touch $d/.done && log "local done ${1}_oof_s$s" || log "local FAILED ${1}_oof_s$s"
  done
}
stream F &
sleep 600
stream J &
wait
log "MCTS queue finished"
