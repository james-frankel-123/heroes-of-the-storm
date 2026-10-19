#!/bin/bash
# P3 tournament: the whole pipeline in one command, for the full-data rerun.
#
# Run from the repo root on the main box (light orchestration only):
#   nohup nice -n 19 taskset -c 48-51 training/personalization/remote/p3t_pipeline.sh > training/personalization/logs/p3t_pipeline.log 2>&1 &
#
# Inputs (already on the workers; produced by the personalization rerun):
#   3090: personalization/cache_full/ (slot table x_slots_ext, merged rows, scoreboard cards,
#         rcf/parts_full_ms.npz) and results/rcf_full/{rcf/params/full_ms.npz,p3_c_tests_main.json,
#         rcf/p3c/full_ms_layer__merged.npz}; set P3T_CACHE / P3T_RESULTS if the rerun uses other dirs.
#   5090: personalization/cache_full/rcf/fit_merged.npz + fitm_merged_a_shrink.npz (for the judge fit).
#   Vintage artifacts listed in personalization/p3t_vintages.json (stale MCTS runs finished).
# Hosts: CPU stages on the 3090 (cores 20-35), GPU search stages on the 5090, judge refit on the 5090.
# Every stage writes its outputs atomically and skips finished outputs, so a rerun after a pause
# continues; delete $P3_CACHE/p3t to start over.
set -euo pipefail
cd "$(dirname "$0")/../../.."
source training/remote_workers/_remote.sh
CPU=max-windows-3090
GPU=windows-5090-wsl
B=/home/max/hots/repo/training/personalization
STAMP=$(date +%m%d%H%M)
RELAY=${P3T_RELAY:-/tmp/p3t_relay}
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
  P3_CORES=20-35 P3_MEM=100G P3_THREADS=8 $JOB $CPU "$n" env P3T_PROCS=12 bash personalization/remote/p3t_run.sh "$@"
  wait_job $CPU "$n"; }
run_gpu() { local n="p3t_${1}_$STAMP"; shift
  P3_CORES=0-15 P3_MEM=11G P3_THREADS=2 $JOB $GPU "$n" bash personalization/remote/p3t_run.sh "$@"
  wait_job $GPU "$n"; }

relay() {      # from-host to-host (cache_full/p3t, small files only)
  mkdir -p "$RELAY"
  nice -n 19 rsync -rt --rsync-path="$(remote_rsync "$1")" --exclude 'scores*' --exclude '*_unmasked' \
    "$1:$B/cache_full/p3t/" "$RELAY/"
  nice -n 19 rsync -rt --rsync-path="$(remote_rsync "$2")" "$RELAY/" "$2:$B/cache_full/p3t/"
}

for h in $CPU $GPU; do training/personalization/remote/p3t_sync.sh $h; done

# 1. judge skill model: the main fit's settings on the other player half (5090 GPU, ~3 min)
P3_CORES=0-15 P3_MEM=11G P3_THREADS=4 $JOB $GPU "p3t_flipfit_$STAMP" env P3_CACHE=$B/cache_full \
  P3_RESULTS=$B/results/rcf_full python -u personalization/p3_rcf.py fit --tag full_ms_flip --hist merged \
  --kmax 2 --metrics --gshare --hero-s2 --sb a_shrink --f64 --rounds 3 --iters 500 --flip-half
wait_job $GPU "p3t_flipfit_$STAMP"
mkdir -p "$RELAY"
nice -n 19 rsync -t --rsync-path="$(remote_rsync $GPU)" "$GPU:$B/results/rcf_full/rcf/params/full_ms_flip.*" "$RELAY/"
nice -n 19 rsync -t --rsync-path="$(remote_rsync $CPU)" "$RELAY"/full_ms_flip.* "$CPU:$B/results/rcf_full/rcf/params/"

# 2. data, states, combiners, judge (3090 CPU)
run_cpu data p3t_data.py:lobbies p3t_data.py:states:--vintages:cur,s4m,s1y,judge:--judge-params:full_ms_flip \
  p3t_data.py:judgefit:--judge-params:full_ms_flip p3t_value.py:combfit:--vintages:cur,s4m,s1y \
  p3t_judge.py:fit p3t_tourney.py:select p3t_probe.py:extract
# the combiners land in the 3090's p3t_vintages.json: bring them home and push everywhere
nice -n 19 rsync -t --rsync-path="$(remote_rsync $CPU)" "$CPU:$B/p3t_vintages.json" training/personalization/
for h in $CPU $GPU; do training/personalization/remote/p3t_sync.sh $h; done
relay $CPU $GPU

# 3. searches (5090 GPU) and the one-step pairs (3090 CPU), in parallel
run_gpu search p3t_tourney.py:run:--schedule:all:--kind:no-onestep:--only-ready p3t_probe.py:run \
  p3t_pprior.py:targets:--lobbies:3000 p3t_backtest.py:agree:--games:20000 &
G=$!
run_cpu onestep p3t_tourney.py:run:--schedule:all:--kind:onestep:--only-ready p3t_backtest.py:forced &
C=$!
wait $G; wait $C

# 4. the player-feature prior: train on the 3090, then its pairs on the 5090
relay $GPU $CPU
run_cpu pprior p3t_pprior.py:train
relay $CPU $GPU
run_gpu pp p3t_tourney.py:run:--schedule:all:--kind:no-onestep:--only-ready

# 5. scoring and analysis (3090)
relay $GPU $CPU
run_cpu score p3t_judge.py:score p3t_analyze.py p3t_backtest.py:agree_analyze:--games:20000 p3t_value.py:predeval

# 6. results home (small JSON only)
mkdir -p training/personalization/results/p3t
for h in $CPU $GPU; do
  nice -n 19 rsync -t --rsync-path="$(remote_rsync $h)" "$h:$B/results/rcf_full/p3t/*.json" training/personalization/results/p3t/ || true
done
echo "P3T PIPELINE DONE $(date -Is)"
