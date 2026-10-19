#!/bin/bash
# P3 tournament: the whole pipeline in one command, for the full-data rerun.
#
# Run from the repo root on the main box (light orchestration only):
#   nohup nice -n 19 taskset -c 48-51 training/personalization/remote/p3t_pipeline.sh \
#     > training/personalization/logs/p3t_pipeline.log 2>&1 &
#
# Inputs (already on the workers; produced by the personalization rerun):
#   3090: personalization/cache_full/ (slot table x_slots_ext, merged rows, scoreboard cards,
#         rcf/parts_full_ms.npz, mmr_at_game.npz, x_post_games.json.gz) and results/rcf_full/
#         {rcf/params/full_ms.npz, p3_c_tests_main.json, rcf/p3c/full_ms_layer__merged.npz}
#   5090: personalization/cache_full/rcf/fit_merged.npz + fitm_merged_a_shrink.npz (judge refit)
#   Vintage artifacts listed in personalization/p3t_vintages.json (stale MCTS runs finished and
#   copied to the 5090; DraftRec model and histories if that system is in the schedule).
# Hosts: CPU stages on the 3090 (cores 20-35); GPU search stages on the 5090.
# Every stage writes outputs atomically and skips finished outputs, so a rerun after a pause
# continues. Delete $B/cache_full/p3t (both hosts) to start over. p3t_vintages.json is written
# only on the 3090 (combfit, vnet) and fetched back right after; do not edit it while this runs.
set -euo pipefail
cd "$(dirname "$0")/../../.."
source training/remote_workers/_remote.sh
CPU=max-windows-3090
GPU=windows-5090-wsl
B=/home/max/hots/repo/training/personalization
STAMP=$(date +%m%d%H%M)
RELAY=${P3T_RELAY:-/tmp/p3t_relay_$STAMP}
JOB=training/personalization/remote/p3_job.sh

wait_job() {   # host name: poll hotsjob state until the job ends; fail on anything but completed
  local h=$1 n=$2 st
  while true; do
    st=$(hotsjob "$h" state "$n" 2>/dev/null | tail -1)
    case "$st" in
      completed) echo "$(date -Is) $n completed"; return 0 ;;
      failed|died|paused) echo "$(date -Is) $n $st"; return 1 ;;
    esac
    sleep 60
  done
}
run_cpu() { local n="p3t_${1}_$STAMP"; shift
  P3_CORES=20-35 P3_MEM=100G P3_THREADS=8 $JOB $CPU "$n" env P3T_PROCS=12 "$@"
  wait_job $CPU "$n"; }
run_gpu() { local n="p3t_${1}_$STAMP"; shift
  P3_CORES=0-15 P3_MEM=11G P3_THREADS=2 $JOB $GPU "$n" env P3T_PROCS=6 "$@"
  wait_job $GPU "$n"; }
STEP="bash personalization/remote/p3t_run.sh"
relay() {      # from-host to-host: cache_full/p3t (scores stay on the 3090)
  mkdir -p "$RELAY"
  nice -n 19 rsync -rt --rsync-path="$(remote_rsync "$1")" --exclude 'scores*' --exclude 'premature' \
    --exclude '*_unmasked' "$1:$B/cache_full/p3t/" "$RELAY/"
  nice -n 19 rsync -rt --rsync-path="$(remote_rsync "$2")" "$RELAY/" "$2:$B/cache_full/p3t/"
}
fetch_vintages() {
  nice -n 19 rsync -t --rsync-path="$(remote_rsync $CPU)" "$CPU:$B/p3t_vintages.json" training/personalization/
  for h in $CPU $GPU; do training/personalization/remote/p3t_sync.sh $h; done
}

for h in $CPU $GPU; do training/personalization/remote/p3t_sync.sh $h; done

# 1. judge skill model: the main fit's settings on the other player half (5090 GPU, ~3 min)
run_gpu flipfit env P3_CACHE=$B/cache_full P3_RESULTS=$B/results/rcf_full python -u personalization/p3_rcf.py fit \
  --tag full_ms_flip --hist merged --kmax 2 --metrics --gshare --hero-s2 --sb a_shrink --f64 --rounds 3 --iters 500 --flip-half
mkdir -p "$RELAY"
nice -n 19 rsync -t --rsync-path="$(remote_rsync $GPU)" "$GPU:$B/results/rcf_full/rcf/params/full_ms_flip.*" "$RELAY/"
nice -n 19 rsync -t --rsync-path="$(remote_rsync $CPU)" "$RELAY"/full_ms_flip.* "$CPU:$B/results/rcf_full/rcf/params/"

# 2. data, states, combiners, judges, value net, rosters (3090 CPU)
run_cpu data $STEP p3t_data.py:lobbies p3t_data.py:states:--vintages:cur,s4m,s1y,judge:--judge-params:full_ms_flip \
  p3t_data.py:judgefit:--judge-params:full_ms_flip p3t_data.py:split_states p3t_data.py:split_judgefit \
  p3t_data.py:rosters p3t_value.py:combfit:--vintages:cur,s4m,s1y,split p3t_judge.py:fit \
  p3t_vnet.py:data p3t_vnet.py:train p3t_value.py:predeval p3t_tourney.py:select p3t_probe.py:extract \
  p3t_backtest.py:forced
run_cpu jsplit env P3T_SPLIT=1 $STEP p3t_judge.py:fit
fetch_vintages
relay $CPU $GPU

# 3. GPU: probe, player-feature prior targets, self-play prior, agreement backtest (5090)
run_gpu gpuprep $STEP p3t_probe.py:run p3t_pprior.py:targets:--lobbies:3000 \
  p3t_selfplay.py:--iters:4:--rosters:4096:--sims:800 p3t_backtest.py:agree:--games:20000:--vintage:cur_bc \
  p3t_backtest.py:agree:--games:20000:--vintage:cur
relay $GPU $CPU
run_cpu pprior $STEP p3t_pprior.py:train
relay $CPU $GPU

# 4. the tournament: every pair whose systems are ready (search pairs on the 5090, one-step pairs too)
run_gpu tourney $STEP p3t_tourney.py:run:--schedule:all:--only-ready

# 5. scoring and analysis (3090)
relay $GPU $CPU
run_cpu score $STEP p3t_judge.py:score p3t_diag.py p3t_analyze.py p3t_backtest.py:agree_analyze:--games:20000:--vintage:cur_bc \
  p3t_backtest.py:agree_analyze:--games:20000:--vintage:cur
run_cpu scoresplit env P3T_SPLIT=1 $STEP p3t_judge.py:score p3t_diag.py p3t_analyze.py

# 6. results home (small JSON only) and the tables
mkdir -p training/personalization/results/p3t
for h in $CPU $GPU; do
  nice -n 19 rsync -t --rsync-path="$(remote_rsync $h)" "$h:$B/results/rcf_full/p3t/*.json" training/personalization/results/p3t/ || true
done
python3 training/personalization/p3t_report.py
echo "P3T PIPELINE DONE $(date -Is)"
