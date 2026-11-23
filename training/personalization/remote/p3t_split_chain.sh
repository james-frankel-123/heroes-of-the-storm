#!/bin/bash
# Split of the root-held control (p3t_split.py) on the 3090, paper-2 namespace (cache_def_def1027_hlsup /
# results/def_def1027_hlsup, headline lobbies), the env of p3t_promote_pcalf_chain.sh. Resumable: every step
# skips finished pair and score files; markers in p3t/support_chain/split_<step>.done.
# Launch (main box): P3_CORES=16-35 P3_MEM=60G P3_GPU=1 training/personalization/remote/p3_job.sh windows-3090-wsl \
#   p3t_split bash personalization/remote/p3t_split_chain.sh
# Smoke (tiny, no files): bash personalization/remote/p3t_split_chain.sh smoke
# Steps: run (PAR GPU processes,
# stable shards), score (only the control's pairs), summary (results/p3t_support/split_greedy*.json/md).
set -euo pipefail
cd /home/max/hots/repo/training
source ~/hots/env.sh
B=/home/max/hots/repo/training/personalization
CROOT=$B/cache_def_def1027_hlsup
RROOT=$B/results/def_def1027_hlsup
export P3_CACHE=$CROOT P3_RESULTS=$RROOT P3T_VINTAGES=$RROOT/p3t_vintages.json
export P3_RCF_ROWS=$CROOT/merged_rows_def1027_2695e2cac4d17884.npz P3_RCF_KERNELS=$CROOT/hs_kernels.npz
# at most ~8 worker processes: PAR tourney processes, each with a P3T_PROCS one-step pool (a1step-cur pairs)
PAR=${P3T_SPLIT_PAR:-2}
export P3T_PROCS=${P3T_PROCS:-3} OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2 PYTHONUNBUFFERED=1
T=/home/max/hots/repo/training
export P1_TIERS=site P1R_NPROC=3 WP_STATS_PATH=$T/paper1_revision/site/cache/stats/deploy.json P1R_COMP_PATH=$T/paper1_revision/site/cache/stats/deploy_compositions.json PYTHONPATH=$T/paper1_revision/oct2026_site
MK=$CROOT/p3t/support_chain
OUT=$RROOT/p3t_support
mkdir -p "$MK" "$OUT"
PM="python -u personalization/p3t_split.py"
if [ "${1:-}" = "smoke" ]; then
  $PM smoke --lobbies 4 --sims 16
  exit 0
fi
RT=""; [ "${P3T_SPLIT_ROUND:-1}" = 2 ] && RT="r2"
step() { local n=$1; shift
  if [ -e "$MK/split${RT}_$n.done" ]; then echo "$(date -Is) $n: done earlier"; return 0; fi
  echo "$(date -Is) $n: start"; "$@"; date -Is > "$MK/split${RT}_$n.done"; echo "$(date -Is) $n: done"; }
run() {
  local pids=() rc=0
  for i in $(seq 0 $((PAR - 1))); do
    $PM run --shard $i/$PAR > "$MK/split${RT}_run_$i.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  tail -n 3 "$MK"/split${RT}_run_*.log
  return $rc
}
score() { $PM score; }
summary() { $PM summary --out "$OUT"; }
# queued behind paper 1's 8-seed training (worlds/jobs/seeds8_train.sh); touch $MK/split_go to skip
until grep -q "SEEDS8 TRAIN DONE" ~/hots/logs/seeds8_train.log 2>/dev/null || [ -f "$MK/split${RT}_go" ]; do sleep 300; done
echo "$(date -Is) gate passed"
step run run
step score score
step summary summary
echo "P3T SPLIT CHAIN DONE $(date -Is)"
