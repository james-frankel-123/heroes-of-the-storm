#!/bin/bash
# P3 tournament: the whole pipeline in one command, for the full-data rerun.
#
# Run from the repo root on the main box (light orchestration only):
#   nohup nice -n 19 taskset -c 48-51 training/personalization/remote/p3t_pipeline.sh \
#     > training/personalization/logs/p3t_pipeline.log 2>&1 &
#
# Inputs (already on the workers; produced by the personalization rerun):
#   CPU host (3080; was the 3090): personalization/cache_full/ or cache_def_<D>/ (slot table x_slots_ext, merged rows, scoreboard cards,
#         rcf/parts_full_ms.npz, mmr_at_game.npz, x_post_games.json.gz, drafts_post.json.gz,
#         gd_trainonly/) and results/rcf_full/{rcf/params/full_ms.npz, p3_c_tests_main.json,
#         rcf/p3c/full_ms_layer__merged.npz}
#   5090: personalization/cache_full/rcf/fit_merged.npz + fitm_merged_a_shrink.npz (judge refit)
#   Stale vintages: paper1_revision/site/vint/<v>/ finished (vintage_launch.py; listed in
#   personalization/p3t_vintages.json) and copied to the 5090 and the 3080 (rsync of models, stats, mcts_runs).
# Hosts: CPU stages on the 3080 (P3T_CPU_HOST); GPU stages on the 5090, tournament shards also on the 3080 GPU.
# Every stage writes outputs atomically and skips finished outputs, so a rerun after a pause
# continues. Delete $CROOT/p3t (both hosts) to start over. p3t_vintages.json is written
# only on the CPU host (combfit, vnet) and fetched back right after; do not edit it while this runs.
# The exploration design is the one the rule picked on the tune lobbies (x-combo1-dr-chance, with x-combo1 and x-combo3
# reported next to it); the rerun re-reports the tune table but keeps that choice (P3_TOURNAMENT.md).
set -euo pipefail
cd "$(dirname "$0")/../../.."
source training/remote_workers/_remote.sh
# Hosts (2026-10-21): the 3090 is out until its RAM is tested (single-bit flips seen), so the
# definitive rerun uses the 3080 for CPU stages and as a second GPU, and the 5090 as the main GPU.
CPU=${P3T_CPU_HOST:-3080-gaming-desktop}
GPU=${P3T_GPU_HOST:-windows-5090-wsl}
GPU2=${P3T_GPU2_HOST:-3080-gaming-desktop}
CPU_CORES=${P3T_CPU_CORES:-0-3,16-19}   # 3080: the nightly replay parse uses cores 4-15 from 22:40
CPU_MEM=${P3T_CPU_MEM:-20G}          # 3080: lean jobs, at most 20 GB per lane
B=/home/max/hots/repo/training/personalization
# Input roots (2026-10-22): the definitive build is versioned by tag D (personalization lane):
#   P3T_BUILD=<D>  ->  cache  $B/cache_def_<D>, results $B/results/def_<D>, merged rows from P3T_RCF_ROWS
#                      (the manifest's file name); the judge's other-half fit (full_ms_flip) comes
#                      with the build, so the refit step is skipped.
#   unset          ->  the old layout ($B/cache_full, $B/results/rcf_full) and the refit runs.
if [ -n "${P3T_BUILD:-}" ]; then
  CROOT=$B/cache_def_$P3T_BUILD; RROOT=$B/results/def_$P3T_BUILD
  : "${P3T_RCF_ROWS:?set P3T_RCF_ROWS to the merged-rows file of the build, see the manifest}"
else
  CROOT=$B/cache_full; RROOT=$B/results/rcf_full
  P3T_RCF_ROWS=${P3T_RCF_ROWS:-$CROOT/merged_rows_full_1017.npz}
fi
ROOTENV="P3T_CACHE_ROOT=$CROOT P3T_RESULTS_ROOT=$RROOT P3T_RCF_ROWS=$P3T_RCF_ROWS"
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
  P3_CORES=$CPU_CORES P3_MEM=$CPU_MEM P3_THREADS=6 $JOB $CPU "$n" env P3T_PROCS=6 $ROOTENV $WALKENV "$@"
  wait_job $CPU "$n"; }
run_gpu() { local n="p3t_${1}_$STAMP"; shift
  P3_CORES=0-15 P3_MEM=11G P3_THREADS=2 $JOB $GPU "$n" env P3T_PROCS=4 $ROOTENV "$@"
  wait_job $GPU "$n"; }
run_gpu_pair() {   # name step-args...: the same tournament step on both GPU hosts, shards 0/2 and 1/2
  local n="p3t_${1}_$STAMP"; shift
  P3_CORES=0-15 P3_MEM=11G P3_THREADS=2 $JOB $GPU "${n}_a" env P3T_PROCS=4 $ROOTENV "$@":--shard:0/2
  P3_CORES=$CPU_CORES P3_MEM=$CPU_MEM P3_THREADS=2 $JOB $GPU2 "${n}_b" env P3T_PROCS=4 $ROOTENV "$@":--shard:1/2
  wait_job $GPU "${n}_a"; wait_job $GPU2 "${n}_b"
  relay $GPU2 $GPU; }
STEP="bash personalization/remote/p3t_run.sh"

relay() {      # from-host to-host: cache_full/p3t (scores stay on the CPU host)
  mkdir -p "$RELAY"
  nice -n 19 rsync -rt --rsync-path="$(remote_rsync "$1")" --exclude 'scores*' --exclude 'premature' \
    --exclude '*_unmasked' "$1:$CROOT/p3t/" "$RELAY/"
  nice -n 19 rsync -rt --rsync-path="$(remote_rsync "$2")" "$RELAY/" "$2:$CROOT/p3t/"
}
push_results() {  # small result JSONs the GPU host reads (opponent model)
  nice -n 19 rsync -t --rsync-path="$(remote_rsync $CPU)" "$CPU:$RROOT/p3t/opp_model.json" "$RELAY/"
  nice -n 19 rsync -t --rsync-path="$(remote_rsync $GPU)" "$RELAY/opp_model.json" "$GPU:$RROOT/p3t/"
}
fetch_vintages() {
  nice -n 19 rsync -t --rsync-path="$(remote_rsync $CPU)" "$CPU:$B/p3t_vintages.json" training/personalization/
  for h in $CPU $GPU; do training/personalization/remote/p3t_sync.sh $h; done
}

for h in $CPU $GPU; do training/personalization/remote/p3t_sync.sh $h; done

# Inputs (2026-10-22): the pipeline refuses to start unless the definitive input manifest
# (personalization/p3_manifest.py, shared with the personalization lane) verifies on both hosts.
MANIFEST=${P3T_MANIFEST:?set P3T_MANIFEST to the definitive manifest (path relative to training/)}
for h in $CPU $GPU; do
  P3_CORES=0-3 P3_MEM=4G P3_THREADS=1 $JOB $h "p3t_manifest_$STAMP" python personalization/p3_manifest.py verify "$MANIFEST"
  wait_job $h "p3t_manifest_$STAMP" || { echo "manifest does not verify on $h: refusing to start"; exit 1; }
done
# Player states (coordinator, 2026-10-22): the definitive tournament uses the personalization
# lane's chosen model; the old full_ms states are the sensitivity. Two passes of this script:
#   P3T_STATES=record  full_ms states, every walk's query list written to $CROOT/p3t_walk_queries
#                      (archive $CROOT/p3t and $RROOT/p3t as *_oldmodel afterwards)
#   P3T_STATES=replay  P3T_WALK_ANSWERS=<dir>: the lane's answers to those queries replace the walk
case "${P3T_STATES:-record}" in
  record) WALKENV="P3T_WALK_RECORD=$CROOT/p3t_walk_queries" ;;
  replay) WALKENV="P3T_WALK_REPLAY=${P3T_WALK_ANSWERS:?set P3T_WALK_ANSWERS for the replay pass}" ;;
  *) echo "P3T_STATES must be record or replay"; exit 2 ;;
esac

# 0. rebuild every P3 kernel (race-fixed 2026-10-19) and gate on determinism and parity (5090)
# (the 5090's compile needs about 20 GB for the nvcc step: run its build with P3T_BUILD_MEM, default 20G)
P3_CORES=0-15 P3_MEM=${P3T_BUILD_MEM:-20G} P3_THREADS=2 $JOB $GPU "p3t_build_$STAMP" env MAX_JOBS=1 bash personalization/remote/p3t_build_all.sh
wait_job $GPU "p3t_build_$STAMP"
P3_CORES=$CPU_CORES P3_MEM=$CPU_MEM P3_THREADS=2 $JOB $GPU2 "p3t_build2_$STAMP" env MAX_JOBS=1 bash personalization/remote/p3t_build_all.sh
wait_job $GPU2 "p3t_build2_$STAMP"
run_gpu determ $STEP p3t_determinism.py p3t_verify.py p3t_pvnet_verify.py p3t_explore_verify.py
P3_CORES=$CPU_CORES P3_MEM=$CPU_MEM P3_THREADS=2 $JOB $GPU2 "p3t_determ2_$STAMP" bash personalization/remote/p3t_run.sh \
  p3t_determinism.py p3t_verify.py p3t_pvnet_verify.py p3t_explore_verify.py
wait_job $GPU2 "p3t_determ2_$STAMP"

# 1. judge skill model: the main fit's settings on the other player half (5090 GPU, ~3 min)
mkdir -p "$RELAY"
if [ -z "${P3T_BUILD:-}" ]; then
run_gpu flipfit env P3_CACHE=$CROOT P3_RESULTS=$RROOT P3_RCF_ROWS=$P3T_RCF_ROWS python -u personalization/p3_rcf.py fit \
  --tag full_ms_flip --hist merged --kmax 2 --metrics --gshare --hero-s2 --sb a_shrink --f64 --rounds 3 --iters 500 --flip-half
nice -n 19 rsync -t --rsync-path="$(remote_rsync $GPU)" "$GPU:$RROOT/rcf/params/full_ms_flip.*" "$RELAY/"
nice -n 19 rsync -t --rsync-path="$(remote_rsync $CPU)" "$RELAY"/full_ms_flip.* "$CPU:$RROOT/rcf/params/"
fi

# 2. data, states, combiners, judges, value study, rosters, lobby sets (CPU host)
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
# DraftRec opponent inside the search (p3t_oppdr.py): fit lobbies and histories (CPU host), the fit (5090)
run_cpu oppdrhist $STEP p3t_oppdr.py:lobbies p3t_oppdr.py:hist
relay $CPU $GPU
run_gpu oppdrfit $STEP p3t_oppdr.py:fit

# 4. the tournament (every ready pair), the tune set, the design ablation (5090)
relay $GPU $GPU2
run_gpu_pair tourney $STEP p3t_tourney.py:run:--schedule:all:--only-ready
run_gpu tune env P3T_LOBBYSET=tune $STEP p3t_tourney.py:run:--schedule:tune
run_gpu ablation $STEP p3t_tourney.py:run:--schedule:ablation   # leftovers only (all includes ablation)

# 5. scoring and analysis (CPU host)
relay $GPU $CPU
run_cpu score $STEP p3t_judge.py:score p3t_diag.py p3t_analyze.py p3t_backtest.py:agree_analyze:--games:20000:--vintage:cur_bc \
  p3t_backtest.py:agree_analyze:--games:20000:--vintage:cur
run_cpu scoresplit env P3T_SPLIT=1 $STEP p3t_judge.py:score p3t_diag.py p3t_analyze.py
run_cpu scoretune env P3T_LOBBYSET=tune $STEP p3t_judge.py:score p3t_analyze.py

# 5b. the party drafter (P3_PARTY_DRAFTER.md): party sizes 1-5 and pick positions (5090)
run_gpu party $STEP p3t_party.py:run:--plan:effect p3t_party.py:run:--plan:position p3t_party.py:score p3t_party_report.py

# 6. results home (small JSON only) and the tables
mkdir -p training/personalization/results/p3t
for h in $CPU $GPU; do
  nice -n 19 rsync -t --rsync-path="$(remote_rsync $h)" "$h:$RROOT/p3t/*.json" training/personalization/results/p3t/ || true
done
python3 training/personalization/p3t_report.py
echo "P3T PIPELINE DONE $(date -Is)"
