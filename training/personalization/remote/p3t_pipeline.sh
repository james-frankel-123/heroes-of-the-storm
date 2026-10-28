#!/bin/bash
# P3 tournament: the whole pipeline in one command, for the full-data rerun.
#
# Run from the repo root on the main box (light orchestration only):
#   nohup nice -n 19 taskset -c 44-47 training/personalization/remote/p3t_pipeline.sh \
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
GPU2_CORES=${P3T_GPU2_CORES:-16-19}  # 3080 tournament lane of run_gpu_pair (0-3 is the personalization agent's)
B=/home/max/hots/repo/training/personalization
# Input roots (2026-10-22): the definitive build is versioned by tag D (personalization lane):
#   P3T_BUILD=<D>  ->  cache  $B/cache_def_<D>, results $B/results/def_<D>, merged rows from P3T_RCF_ROWS
#                      (the manifest's file name); the judge's other-half fit (full_ms_flip) comes
#                      with the build, so the refit step is skipped.
#   unset          ->  the old layout ($B/cache_full, $B/results/rcf_full) and the refit runs.
if [ -n "${P3T_BUILD:-}" ]; then
  CROOT0=$B/cache_def_$P3T_BUILD; RROOT0=$B/results/def_$P3T_BUILD
  : "${P3T_RCF_ROWS:?set P3T_RCF_ROWS to the merged-rows file of the build, see the manifest}"
else
  CROOT0=$B/cache_full; RROOT0=$B/results/rcf_full
  P3T_RCF_ROWS=${P3T_RCF_ROWS:-$CROOT0/merged_rows_full_1017.npz}
fi
# Passes (review C2/C8/C16, 2026-10-24): each model pass runs in its own namespace on every host,
#   cache  $CROOT0_<pass>/   results  $RROOT0_<pass>/
# whose entries are symlinks to the build's inputs (everything but p3t/, p3t_walk_*, p3t_vintages.json),
# so drafts, scores, walk queries, the vintage table (P3T_VINTAGES) and every result JSON of one pass
# never meet the other's. P3T_PASS is required with P3T_BUILD (p3t_definitive.sh sets it).
if [ -n "${P3T_PASS:-}" ]; then
  CROOT=${CROOT0}_$P3T_PASS; RROOT=${RROOT0}_$P3T_PASS; TAG=${P3T_BUILD:-full}_$P3T_PASS
else
  [ -z "${P3T_BUILD:-}" ] || { echo "P3T_BUILD needs P3T_PASS (run p3t_definitive.sh)"; exit 2; }
  CROOT=$CROOT0; RROOT=$RROOT0; TAG=""
fi
SMOKE=${P3T_SMOKE:-0}
# the merged rows through the namespace (a link on the CPU host, a copy on the GPU hosts)
[ -z "$TAG" ] || P3T_RCF_ROWS=$CROOT/$(basename "$P3T_RCF_ROWS")
# P3_MMR_KEY_FIX=1 (2026-10-28): collision-free MMR stamp keys for the ban-holder ranks (p3t_data.ranks); def1027 ran without
ROOTENV="P3T_CACHE_ROOT=$CROOT P3T_RESULTS_ROOT=$RROOT P3T_RCF_ROWS=$P3T_RCF_ROWS P3T_VINTAGES=$RROOT/p3t_vintages.json P3_MMR_KEY_FIX=${P3_MMR_KEY_FIX:-1}"
# stage sizes (full run), and the smoke test's: every stage runs, at small sizes
PROBE_N=500; PROBE_RUN=""; PP_LOB=3000; SP_ARGS=":--iters:4:--rosters:4096:--sims:800"; AGREE=20000; DR_EP=""
TOURNEY="--schedule:all:--only-ready"; TUNE="--schedule:tune"; ABLATION="--schedule:ablation"
PARTY_EFF=""; PARTY_GRID=""; PARTY_LOCK=""
if [ "$SMOKE" = 1 ]; then
  ROOTENV="$ROOTENV P3T_LOBBIES=24 P3T_N_PARTY=40"
  PROBE_N=40; PROBE_RUN=":--systems:aMCTS-cur"; PP_LOB=200; SP_ARGS=":--iters:1:--rosters:256:--sims:100"
  AGREE=400; DR_EP=":--epochs:1"
  TOURNEY="--pairs:x-combo1-dr-chance__aMCTS-cur,aMCTS-cur__x-combo1-dr-chance,aMCTS-cur__GD,GD__aMCTS-cur"
  TUNE="--pairs:x-combo1-dr-chance__pMCTS-cur,pMCTS-cur__x-combo1-dr-chance"; ABLATION=""
  PARTY_EFF=":--only:eff_m1_t0,eff_m2_t1"
  PARTY_GRID=":--only:grid_m1_0_t0_ban,grid_m1_4_t0_ban,grid_m1_0_t1_noban,grid_m1_4_t1_noban"
  PARTY_LOCK=":--only:lock_m1_t0_ban,lock_m1_t1_noban"
fi
STAMP=$(date +%m%d%H%M)
RELAY=${P3T_RELAY:-/tmp/p3t_relay_$STAMP}
JOB=training/personalization/remote/p3_job.sh

wait_job() {   # host name: poll hotsjob state until the job ends; fail on anything but completed
  # (review C5/R06) a failed or empty status poll (ssh hiccup) is "unknown": keep polling; give up
  # only after P3T_POLL_UNKNOWN_MAX seconds of nothing but unknown answers
  local h=$1 n=$2 st since=""
  local umax=${P3T_POLL_UNKNOWN_MAX:-3600}
  while true; do
    st=$(hotsjob "$h" state "$n" 2>/dev/null | tail -1) || st=""
    case "$st" in
      completed) echo "$(date -Is) $n completed"; return 0 ;;
      failed|died|paused) echo "$(date -Is) $n $st"; return 1 ;;
      running|queued|starting) since="" ;;
      *) [ -n "$since" ] || since=$(date +%s)
         if [ $(( $(date +%s) - since )) -gt "$umax" ]; then
           echo "$(date -Is) $n: status unknown for over ${umax}s ('$st'); giving up"; return 1
         fi ;;
    esac
    sleep 60
  done
}
# per-stage markers on the main box: a rerun of the same pass skips the stages that finished
SMK=training/personalization/logs/p3t_pass_${TAG:-legacy}${P3T_SMOKE:+_smoke$P3T_SMOKE}
mkdir -p "$SMK"
done_before() { if [ -e "$SMK/$1.done" ]; then echo "$(date -Is) stage $1: done earlier, skipped"; return 0; fi; return 1; }
mark_done() { date -Is > "$SMK/$1.done"; }
run_cpu() { local s=$1 n="p3t_${1}_$STAMP"; shift; done_before $s && return 0
  P3_CORES=$CPU_CORES P3_MEM=$CPU_MEM P3_THREADS=6 $JOB $CPU "$n" env P3T_PROCS=6 $ROOTENV $WALKENV "$@"
  wait_job $CPU "$n"; mark_done $s; }
run_gpu() { local s=$1 n="p3t_${1}_$STAMP"; shift; done_before $s && return 0
  P3_CORES=0-15 P3_MEM=11G P3_THREADS=2 $JOB $GPU "$n" env P3T_PROCS=4 $ROOTENV "$@"
  wait_job $GPU "$n"; mark_done $s; }
run_gpu_pair() {   # name step-args...: the same tournament step on both GPU hosts, shards 0/2 and 1/2
  local s=$1 n="p3t_${1}_$STAMP"; shift; done_before $s && return 0
  P3_CORES=0-15 P3_MEM=11G P3_THREADS=2 $JOB $GPU "${n}_a" env P3T_PROCS=4 $ROOTENV "$@":--shard:0/2
  P3_CORES=$GPU2_CORES P3_MEM=$CPU_MEM P3_THREADS=2 $JOB $GPU2 "${n}_b" env P3T_PROCS=4 $ROOTENV "$@":--shard:1/2
  wait_job $GPU "${n}_a"; wait_job $GPU2 "${n}_b"
  relay $GPU2 $GPU; mark_done $s; }
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
push_gpu_results() {  # the 5090's opponent and ban-holder fits, read by the 3080 lane of run_gpu_pair
  for f in opp_dr.json opp_holder.json; do
    nice -n 19 rsync -t --rsync-path="$(remote_rsync $GPU)" "$GPU:$RROOT/p3t/$f" "$RELAY/"
  done
  nice -n 19 rsync -t --rsync-path="$(remote_rsync $GPU2)" "$RELAY/opp_dr.json" "$RELAY/opp_holder.json" "$GPU2:$RROOT/p3t/"
}
fetch_vintages() {   # this pass's vintage table (combfit, vnet write it on the CPU host) to the GPU hosts
  mkdir -p "$RELAY"
  nice -n 19 rsync -t --rsync-path="$(remote_rsync $CPU)" "$CPU:$RROOT/p3t_vintages.json" "$RELAY/"
  for h in $HOSTS; do
    [ "$h" = "$CPU" ] || nice -n 19 rsync -t --rsync-path="$(remote_rsync $h)" "$RELAY/p3t_vintages.json" "$h:$RROOT/"
  done
}
HOSTS=$(printf '%s\n' $CPU $GPU $GPU2 | awk '!seen[$0]++' | tr '\n' ' ')

for h in $HOSTS; do training/personalization/remote/p3t_sync.sh $h; done
# the pass namespace on every host (idempotent)
if [ -n "$TAG" ]; then
  for h in $HOSTS; do
    ssh "$h" "$(remote_shell "$h")" <<NS
set -e
for pair in "$CROOT0 $CROOT" "$RROOT0 $RROOT"; do
  set -- \$pair; src=\$1; dst=\$2
  mkdir -p "\$dst/p3t"
  [ -d "\$src" ] || continue
  for f in "\$src"/* "\$src"/.[!.]*; do
    [ -e "\$f" ] || continue
    b=\$(basename "\$f")
    case "\$b" in p3t|p3t_walk_*|p3t_vintages.json|manifest.json) continue ;; esac
    [ -e "\$dst/\$b" ] || ln -s "\$(readlink -f "\$f")" "\$dst/\$b"
  done
done
echo "namespace $TAG ready on \$(hostname)"
NS
  done
  # the GPU hosts hold only the tournament-role inputs of the build: copy the build root's input files
  # (top level, plus gd_trainonly) from the CPU host into their namespace (real files; about 12 GB)
  mkdir -p "$RELAY/inputs"
  nice -n 19 rsync -rtL --rsync-path="$(remote_rsync $CPU)" --exclude '/rcf' --exclude '/export' \
    --exclude '/full_history' --exclude '/p3t*' --exclude '/prefix_race_*' --exclude '/manifest.json' \
    "$CPU:$CROOT0/" "$RELAY/inputs/"
  for h in $HOSTS; do
    [ "$h" = "$CPU" ] && continue
    nice -n 19 rsync -rtL --rsync-path="$(remote_rsync $h)" "$RELAY/inputs/" "$h:$CROOT/"
  done
fi

# Inputs (2026-10-22): the pipeline refuses to start unless the definitive input manifest
# (personalization/p3_manifest.py, shared with the personalization lane) verifies on both hosts.
MANIFEST=${P3T_MANIFEST:?set P3T_MANIFEST to the definitive manifest (path relative to training/)}
# (review C2) the CPU host holds every input: the full manifest; the GPU hosts hold the tournament
# inputs only: --role tournament
[ -s "training/$MANIFEST" ] || { echo "no manifest at training/$MANIFEST on the main box"; exit 1; }
for h in $HOSTS; do
  # every host checks the same manifest: the main box's copy is shipped first
  ssh "$h" "$(remote_shell "$h")" <<<"mkdir -p $(dirname /home/max/hots/repo/training/$MANIFEST)"
  nice -n 19 rsync -t --rsync-path="$(remote_rsync $h)" "training/$MANIFEST" "$h:/home/max/hots/repo/training/$MANIFEST"
  role=""; [ "$h" = "$CPU" ] || role="--role tournament"
  P3_CORES=0-3 P3_MEM=4G P3_THREADS=1 $JOB $h "p3t_manifest_$STAMP" python personalization/p3_manifest.py verify "$MANIFEST" $role --complete
  wait_job $h "p3t_manifest_$STAMP" || { echo "manifest does not verify on $h: refusing to start"; exit 1; }
done
# Player states (coordinator, 2026-10-22): the definitive tournament uses the personalization
# lane's chosen model; the old full_ms states are the sensitivity. Two passes of this script, each in
# its own namespace (p3t_definitive.sh runs them, with the answer and check steps between):
#   P3T_STATES=record  full_ms states, every walk's query list written to $CROOT/p3t_walk_queries
#   P3T_STATES=replay  P3T_WALK_ANSWERS=<dir>: the lane's answers to those queries replace the walk
case "${P3T_STATES:-record}" in
  record) WALKENV="P3T_WALK_RECORD=$CROOT/p3t_walk_queries" ;;
  replay) WALKENV="P3T_WALK_REPLAY=${P3T_WALK_ANSWERS:?set P3T_WALK_ANSWERS for the replay pass}" ;;
  *) echo "P3T_STATES must be record or replay"; exit 2 ;;
esac

# 0. the P3 kernels: built once before the definitive run (and pinned by the personalization lane's
# collect: no rebuild between that pin and a pass, review C7); each pass only re-runs the determinism
# and parity gates on both GPU hosts. P3T_REBUILD=1 rebuilds first (the 5090's nvcc step needs about
# 20 GB: P3T_BUILD_MEM, default 20G); the old layout (no P3T_BUILD) rebuilds by default.
REBUILD=${P3T_REBUILD:-$([ -n "${P3T_BUILD:-}" ] && echo 0 || echo 1)}
if [ "$REBUILD" = 1 ]; then
  P3_CORES=0-15 P3_MEM=${P3T_BUILD_MEM:-20G} P3_THREADS=2 $JOB $GPU "p3t_build_$STAMP" env MAX_JOBS=1 bash personalization/remote/p3t_build_all.sh
  wait_job $GPU "p3t_build_$STAMP"
  P3_CORES=$GPU2_CORES P3_MEM=$CPU_MEM P3_THREADS=2 $JOB $GPU2 "p3t_build2_$STAMP" env MAX_JOBS=1 bash personalization/remote/p3t_build_all.sh
  wait_job $GPU2 "p3t_build2_$STAMP"
fi
# the gates read fixed fixture lobbies (probe lobbies, rosters) of an older complete cache, not the
# pass's (still empty) namespace; their reports go to the pass's results
GATES="p3t_determinism.py p3t_verify.py p3t_pvnet_verify.py p3t_explore_verify.py tests/test_p3t_kernel_sampling.py"
GATE_GPU=${P3T_GATE_CACHE_GPU:-$B/cache_full}
GATE_GPU2=${P3T_GATE_CACHE_GPU2:-/home/max/hots/p3t_dry/cache_full}
if ! done_before gates; then
P3_CORES=0-15 P3_MEM=11G P3_THREADS=2 $JOB $GPU "p3t_determ_$STAMP" env P3T_PROCS=4 P3T_CACHE_ROOT=$GATE_GPU \
  P3T_RESULTS_ROOT=$RROOT bash personalization/remote/p3t_run.sh $GATES
wait_job $GPU "p3t_determ_$STAMP"
P3_CORES=$GPU2_CORES P3_MEM=$CPU_MEM P3_THREADS=2 $JOB $GPU2 "p3t_determ2_$STAMP" env P3T_PROCS=4 P3T_CACHE_ROOT=$GATE_GPU2 \
  P3T_RESULTS_ROOT=$RROOT bash personalization/remote/p3t_run.sh $GATES
wait_job $GPU2 "p3t_determ2_$STAMP"
mark_done gates
fi
# the code this pass runs, per host (sources, scripts, built kernels): pinned into the pass's own code
# manifest now, re-verified at the end of the pass (review C7)
CODEMAN=personalization/results/$(basename "$RROOT")/code_manifest.json
code_pin() {   # verb: pin | verify
  for h in $HOSTS; do
    P3_CORES=0-3 P3_MEM=4G P3_THREADS=1 $JOB $h "p3t_code_${1}_$STAMP" bash personalization/remote/p3t_code_pin.sh \
      "$1" "$CODEMAN" "${TAG:-full}" "$h"
    wait_job $h "p3t_code_${1}_$STAMP" || { echo "code $1 failed on $h"; exit 1; }
  done
}
code_pin pin

# 1. judge skill model: the main fit's settings on the other player half (5090 GPU, ~3 min)
mkdir -p "$RELAY"
if [ -z "${P3T_BUILD:-}" ]; then
run_gpu flipfit env P3_CACHE=$CROOT P3_RESULTS=$RROOT P3_RCF_ROWS=$P3T_RCF_ROWS python -u personalization/p3_rcf.py fit \
  --tag full_ms_flip --hist merged --kmax 2 --metrics --gshare --hero-s2 --sb a_shrink --f64 --rounds 3 --iters 500 --flip-half
nice -n 19 rsync -t --rsync-path="$(remote_rsync $GPU)" "$GPU:$RROOT/rcf/params/full_ms_flip.*" "$RELAY/"
nice -n 19 rsync -t --rsync-path="$(remote_rsync $CPU)" "$RELAY"/full_ms_flip.* "$CPU:$RROOT/rcf/params/"
fi

# 2. data, states, combiners, judges, value study, rosters, lobby sets (CPU host)
run_cpu data $STEP p3t_data.py:lobbies p3t_data.py:ranks p3t_data.py:states:--vintages:cur,s4m,s1y,judge:--judge-params:full_ms_flip \
  p3t_data.py:judgefit:--judge-params:full_ms_flip p3t_data.py:split_states p3t_data.py:split_judgefit \
  p3t_data.py:rosters p3t_data.py:prelobbies p3t_value.py:combfit:--vintages:cur,s4m,s1y,split p3t_judge.py:fit \
  p3t_vnet.py:data p3t_vnet.py:train p3t_vnet.py:train_pre p3t_vstudy.py p3t_value.py:predeval \
  p3t_tourney.py:select
run_cpu vbest env P3T_VSTUDY_ASSIGN=best $STEP p3t_vstudy.py
run_cpu jsplit env P3T_SPLIT=1 $STEP p3t_judge.py:fit
run_cpu tunesel env P3T_LOBBYSET=tune $STEP p3t_tourney.py:select
# the party study's lobby set, before any lobby that fits a drafter component is drawn
run_cpu partysel env P3T_LOBBYSET=party $STEP p3t_party.py:select
run_cpu probesel $STEP p3t_probe.py:extract:--n:$PROBE_N p3t_backtest.py:forced
fetch_vintages
relay $CPU $GPU

# 3. GPU: probe, player-feature prior targets, self-play prior, agreement backtests, DraftRec (5090)
run_gpu gpuprep $STEP p3t_probe.py:run$PROBE_RUN p3t_pprior.py:targets:--lobbies:$PP_LOB \
  p3t_selfplay.py$SP_ARGS p3t_backtest.py:agree:--games:$AGREE:--vintage:cur_bc \
  p3t_backtest.py:agree:--games:$AGREE:--vintage:cur
relay $GPU $CPU
run_cpu priors $STEP p3t_pprior.py:train p3t_oppmodel.py p3t_rprior.py:data_search p3t_rprior.py:train:--signal:search \
  p3t_rprior.py:data_bcow p3t_rprior.py:train:--signal:bcow p3t_rprior.py:data_onestep p3t_rprior.py:train:--signal:onestep \
  p3t_draftrec.py:data
relay $CPU $GPU
push_results
run_gpu draftrec $STEP p3t_draftrec.py:train$DR_EP p3t_draftrec.py:train:--final:--keep-epoch:auto$DR_EP p3t_draftrec.py:hist \
  p3t_probe.py:timing
# DraftRec opponent inside the search (p3t_oppdr.py): fit lobbies and histories (CPU host), the fit (5090)
run_cpu oppdrhist $STEP p3t_oppdr.py:lobbies p3t_oppdr.py:hist
relay $CPU $GPU
# and the ban holder's model (the top-ranked player bans; party drafter)
run_gpu oppdrfit $STEP p3t_oppdr.py:fit p3t_oppdr.py:holderfit
push_gpu_results

# 4. the tournament (every ready pair), the tune set, the design ablation (5090)
relay $GPU $GPU2
run_gpu_pair tourney $STEP p3t_tourney.py:run:$TOURNEY
run_gpu tune env P3T_LOBBYSET=tune $STEP p3t_tourney.py:run:$TUNE
[ -z "$ABLATION" ] || run_gpu ablation $STEP p3t_tourney.py:run:$ABLATION   # leftovers only (all includes ablation)

# 5. scoring and analysis (CPU host)
relay $GPU $CPU
run_cpu score $STEP p3t_judge.py:score p3t_diag.py p3t_analyze.py p3t_backtest.py:agree_analyze:--games:$AGREE:--vintage:cur_bc \
  p3t_backtest.py:agree_analyze:--games:$AGREE:--vintage:cur
run_cpu scoresplit env P3T_SPLIT=1 $STEP p3t_judge.py:score p3t_diag.py p3t_analyze.py
run_cpu scoretune env P3T_LOBBYSET=tune $STEP p3t_judge.py:score p3t_analyze.py

# 5b. the party drafter (P3_PARTY_DRAFTER.md) on its own lobby set (up to 1,800 lobbies, 10-game rule):
# party sizes 1-5 with random members and lock turns (4 arms), the full fixed-timing grid (every set of
# team turns, both sides, with and without the top-ranked player in the party; aware and generic), and
# the solo lock-or-wait arm. Configurations sharded across both GPUs; scored on the CPU host (judge).
relay $GPU $GPU2
run_gpu_pair partyeff env P3T_LOBBYSET=party $STEP p3t_party.py:run:--plan:effect$PARTY_EFF
run_gpu_pair partygrid env P3T_LOBBYSET=party $STEP p3t_party.py:run:--plan:grid$PARTY_GRID
run_gpu_pair partylock env P3T_LOBBYSET=party $STEP p3t_party.py:run:--plan:lock$PARTY_LOCK
relay $GPU $CPU
run_cpu partyscore env P3T_LOBBYSET=party $STEP p3t_party.py:score p3t_party_report.py

# 6. results home (small JSON only) and the tables, per build and pass (review C16)
LRES=training/personalization/results/p3t${TAG:+_$TAG}
mkdir -p "$LRES"
for h in $HOSTS; do
  nice -n 19 rsync -t --rsync-path="$(remote_rsync $h)" "$h:$RROOT/p3t/*.json" "$LRES/"
done
nice -n 19 rsync -t --rsync-path="$(remote_rsync $CPU)" "$CPU:$RROOT/p3t_vintages.json" "$LRES/"
[ -s "$LRES/analysis.json" ] || { echo "no analysis.json collected from $RROOT: the pass is incomplete"; exit 1; }
code_pin verify
for h in $HOSTS; do
  nice -n 19 rsync -t --rsync-path="$(remote_rsync $h)" "$h:/home/max/hots/repo/training/$CODEMAN" "$LRES/code_manifest_$h.json"
done
if [ -n "$TAG" ]; then P3T_REPORT_DIR=$PWD/$LRES python3 training/personalization/p3t_report.py
else python3 training/personalization/p3t_report.py; fi
echo "P3T PIPELINE DONE $(date -Is)"
