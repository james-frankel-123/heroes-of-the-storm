#!/bin/bash
# P3 tournament: the whole pipeline in one command, for the full-data rerun.
#
# Run from the repo root on the main box (light orchestration only):
#   nohup nice -n 19 taskset -c 48-51 training/personalization/remote/p3t_pipeline.sh \
#     > training/personalization/logs/p3t_pipeline.log 2>&1 &
#
# Inputs (already on the workers; produced by the personalization rerun):
#   3090: personalization/cache_full/ (slot table x_slots_ext, merged rows, scoreboard cards,
#         rcf/parts_full_ms.npz, mmr_at_game.npz, x_post_games.json.gz, drafts_post.json.gz,
#         gd_trainonly/) and results/rcf_full/{rcf/params/full_ms.npz, p3_c_tests_main.json,
#         rcf/p3c/full_ms_layer__merged.npz}
#   5090: personalization/cache_full/rcf/fit_merged.npz + fitm_merged_a_shrink.npz (judge refit)
#   Stale vintages: paper1_revision/site/vint/<v>/ finished (vintage_launch.py; listed in
#   personalization/p3t_vintages.json) and copied to the 5090 (rsync of models, stats, mcts_runs).
# Hosts: CPU stages on the 3090 (cores 20-35); GPU stages on the 5090.
# Every stage writes outputs atomically and skips finished outputs, so a rerun after a pause
# continues. Delete $B/cache_full/p3t (both hosts) to start over. p3t_vintages.json is written
# only on the 3090 (combfit, vnet) and fetched back right after; do not edit it while this runs.
# The exploration design is the one the rule picked on the race-fixed tune rerun (x-combo1, with x-combo3
# reported next to it); the rerun re-reports the tune table but keeps that choice (P3_TOURNAMENT.md).
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
  P3_CORES=0-15 P3_MEM=11G P3_THREADS=2 $JOB $GPU "$n" env P3T_PROCS=4 "$@"
  wait_job $GPU "$n"; }
STEP="bash personalization/remote/p3t_run.sh"
relay() {      # from-host to-host: cache_full/p3t (scores stay on the 3090)
  mkdir -p "$RELAY"
  nice -n 19 rsync -rt --rsync-path="$(remote_rsync "$1")" --exclude 'scores*' --exclude 'premature' \
    --exclude '*_unmasked' "$1:$B/cache_full/p3t/" "$RELAY/"
  nice -n 19 rsync -rt --rsync-path="$(remote_rsync "$2")" "$RELAY/" "$2:$B/cache_full/p3t/"
}
push_results() {  # small result JSONs the GPU host reads (opponent model)
  nice -n 19 rsync -t --rsync-path="$(remote_rsync $CPU)" "$CPU:$B/results/rcf_full/p3t/opp_model.json" "$RELAY/"
  nice -n 19 rsync -t --rsync-path="$(remote_rsync $GPU)" "$RELAY/opp_model.json" "$GPU:$B/results/rcf_full/p3t/"
}
fetch_vintages() {
  nice -n 19 rsync -t --rsync-path="$(remote_rsync $CPU)" "$CPU:$B/p3t_vintages.json" training/personalization/
  for h in $CPU $GPU; do training/personalization/remote/p3t_sync.sh $h; done
}

for h in $CPU $GPU; do training/personalization/remote/p3t_sync.sh $h; done

# 0. rebuild every P3 kernel (race-fixed 2026-10-19) and gate on determinism and parity (5090)
run_gpu build bash personalization/remote/p3t_build_all.sh
run_gpu determ $STEP p3t_determinism.py p3t_verify.py p3t_pvnet_verify.py p3t_explore_verify.py

# 1. judge skill model: the main fit's settings on the other player half (5090 GPU, ~3 min)
run_gpu flipfit env P3_CACHE=$B/cache_full P3_RESULTS=$B/results/rcf_full python -u personalization/p3_rcf.py fit \
  --tag full_ms_flip --hist merged --kmax 2 --metrics --gshare --hero-s2 --sb a_shrink --f64 --rounds 3 --iters 500 --flip-half
mkdir -p "$RELAY"
nice -n 19 rsync -t --rsync-path="$(remote_rsync $GPU)" "$GPU:$B/results/rcf_full/rcf/params/full_ms_flip.*" "$RELAY/"
nice -n 19 rsync -t --rsync-path="$(remote_rsync $CPU)" "$RELAY"/full_ms_flip.* "$CPU:$B/results/rcf_full/rcf/params/"

# 2. data, states, combiners, judges, value study, rosters, lobby sets (3090 CPU)
run_cpu data $STEP p3t_data.py:lobbies p3t_data.py:states:--vintages:cur,s4m,s1y,judge:--judge-params:full_ms_flip \
  p3t_data.py:judgefit:--judge-params:full_ms_flip p3t_data.py:split_states p3t_data.py:split_judgefit \
  p3t_data.py:rosters p3t_data.py:prelobbies p3t_value.py:combfit:--vintages:cur,s4m,s1y,split p3t_judge.py:fit \
  p3t_vnet.py:data p3t_vnet.py:train p3t_vnet.py:train_pre p3t_vstudy.py p3t_value.py:predeval \
  p3t_tourney.py:select p3t_probe.py:extract p3t_backtest.py:forced
run_cpu vbest env P3T_VSTUDY_ASSIGN=best $STEP p3t_vstudy.py
run_cpu jsplit env P3T_SPLIT=1 $STEP p3t_judge.py:fit
run_cpu tunesel env P3T_LOBBYSET=tune $STEP p3t_tourney.py:select
fetch_vintages
relay $CPU $GPU

# 3. GPU: probe, player-feature prior targets, self-play prior, agreement backtests, DraftRec (5090)
run_gpu gpuprep $STEP p3t_probe.py:run p3t_pprior.py:targets:--lobbies:3000 \
  p3t_selfplay.py:--iters:4:--rosters:4096:--sims:800 p3t_backtest.py:agree:--games:20000:--vintage:cur_bc \
  p3t_backtest.py:agree:--games:20000:--vintage:cur
relay $GPU $CPU
run_cpu priors $STEP p3t_pprior.py:train p3t_oppmodel.py p3t_rprior.py:data_search p3t_rprior.py:train:--signal:search \
  p3t_rprior.py:data_bcow p3t_rprior.py:train:--signal:bcow p3t_rprior.py:data_onestep p3t_rprior.py:train:--signal:onestep \
  p3t_draftrec.py:data
relay $CPU $GPU
push_results
run_gpu draftrec $STEP p3t_draftrec.py:train p3t_draftrec.py:train:--final:--keep-epoch:auto p3t_draftrec.py:hist \
  p3t_probe.py:timing
# DraftRec opponent inside the search (p3t_oppdr.py): fit lobbies and histories (3090), the fit (5090)
run_cpu oppdrhist $STEP p3t_oppdr.py:lobbies p3t_oppdr.py:hist
relay $CPU $GPU
run_gpu oppdrfit $STEP p3t_oppdr.py:fit

# 4. the tournament (every ready pair), the tune set, the design ablation (5090)
run_gpu tourney $STEP p3t_tourney.py:run:--schedule:all:--only-ready
run_gpu tune env P3T_LOBBYSET=tune $STEP p3t_tourney.py:run:--schedule:tune
run_gpu ablation $STEP p3t_tourney.py:run:--schedule:ablation

# 5. scoring and analysis (3090)
relay $GPU $CPU
run_cpu score $STEP p3t_judge.py:score p3t_diag.py p3t_analyze.py p3t_backtest.py:agree_analyze:--games:20000:--vintage:cur_bc \
  p3t_backtest.py:agree_analyze:--games:20000:--vintage:cur
run_cpu scoresplit env P3T_SPLIT=1 $STEP p3t_judge.py:score p3t_diag.py p3t_analyze.py
run_cpu scoretune env P3T_LOBBYSET=tune $STEP p3t_judge.py:score p3t_analyze.py

# 6. results home (small JSON only) and the tables
mkdir -p training/personalization/results/p3t
for h in $CPU $GPU; do
  nice -n 19 rsync -t --rsync-path="$(remote_rsync $h)" "$h:$B/results/rcf_full/p3t/*.json" training/personalization/results/p3t/ || true
done
python3 training/personalization/p3t_report.py
echo "P3T PIPELINE DONE $(date -Is)"
