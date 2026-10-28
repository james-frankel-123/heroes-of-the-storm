#!/bin/bash
# P3 tournament: rerun every kernel-dependent stage after the 2026-10-19 race fix.
# P3T_RF_FROM=3 P3T_RF_WAIT=<job> starts at the tournament after <job> ends.
# Picks up after p3t_fixA (probe + player-prior targets) and p3t_fixB (every pair that needs
# no new prior) on the 5090. Light orchestration on the main box:
#   nohup nice -n 19 taskset -c 48-51 training/personalization/remote/p3t_racefix_rerun.sh \
#     > training/personalization/logs/p3t_racefix_rerun.log 2>&1 &
# One GPU job at a time on the 5090; CPU stages on the 3090 (cores 20-35).
set -euo pipefail
cd "$(dirname "$0")/../../.."
source training/remote_workers/_remote.sh
CPU=max-windows-3090
GPU=windows-5090-wsl
B=/home/max/hots/repo/training/personalization
STAMP=$(date +%m%d%H%M)
RELAY=${P3T_RELAY:-/tmp/claude-1000/p3t_racefix_relay}
JOB=training/personalization/remote/p3_job.sh
VQ_PID=${VQ_PID:-1875739}     # paper-1 vintage queue (vintage_launch.py queue); explicit PID

wait_job() {
  local h=$1 n=$2 st
  while true; do
    st=$(hotsjob "$h" state "$n" 2>/dev/null | tail -1)
    case "$st" in
      completed) echo "$(date -Is) $n completed"; return 0 ;;
      failed|died|paused) echo "$(date -Is) $n $st"; return 1 ;;
    esac
    sleep 120
  done
}
run_cpu() { local n="p3t_rf_${1}_$STAMP"; shift
  P3_CORES=20-35 P3_MEM=100G P3_THREADS=8 $JOB $CPU "$n" env P3T_PROCS=12 "$@"
  wait_job $CPU "$n"; }
run_gpu() { local n="p3t_rf_${1}_$STAMP"; shift
  P3_CORES=0-15 P3_MEM=11G P3_THREADS=2 $JOB $GPU "$n" env P3T_PROCS=4 "$@"
  wait_job $GPU "$n"; }
STEP="bash personalization/remote/p3t_run.sh"
relay() {
  mkdir -p "$RELAY"
  nice -n 19 rsync -rt --rsync-path="$(remote_rsync "$1")" --exclude 'scores*' --exclude 'premature' \
    --exclude 'prefix_race' --exclude '*_unmasked' "$1:$B/cache_full/p3t/" "$RELAY/"
  nice -n 19 rsync -rt --rsync-path="$(remote_rsync "$2")" "$RELAY/" "$2:$B/cache_full/p3t/"
}

echo "$(date -Is) start ${P3T_RF_FROM:-1}"
if [ "${P3T_RF_FROM:-1}" -le 1 ]; then
# 1. player-feature prior and the search-signal rich prior from the post-fix targets
wait_job $GPU p3t_fixA
relay $GPU $CPU
run_cpu priors $STEP p3t_pprior.py:train p3t_rprior.py:data_search p3t_rprior.py:train:--signal:search
relay $CPU $GPU
fi

if [ "${P3T_RF_FROM:-1}" -le 2 ]; then
# 2. self-play prior and agreement backtests, after the no-new-prior pairs
wait_job $GPU p3t_fixB
run_gpu gpuprep $STEP p3t_selfplay.py:--iters:4:--rosters:4096:--sims:800 \
  p3t_backtest.py:agree:--games:20000:--vintage:cur_bc p3t_backtest.py:agree:--games:20000:--vintage:cur
fi
[ -n "${P3T_RF_WAIT:-}" ] && wait_job $GPU "$P3T_RF_WAIT"

if [ "${P3T_RF_FROM:-1}" -le 3 ]; then
# 3. every pair that is ready now, the tune set, the paper-1 bench and timing
run_gpu tourney $STEP p3t_tourney.py:run:--schedule:all:--only-ready
run_gpu tune env P3T_LOBBYSET=tune $STEP p3t_tourney.py:run:--schedule:tune
run_gpu bench $STEP p3t_sanity.py:bench p3t_probe.py:timing:--systems:x-combo3,x-combo1,x-pp-ban3,x-pp-opp2,x-pp-as1,x-pp,pMCTS-cur
fi

# 4. stale vintages: wait for every paper-1 vintage F run (the queue process exits once all are
# launched, not finished), fetch the fine-tune runs, copy the vintages to the 5090, run the stale pairs
while kill -0 "$VQ_PID" 2>/dev/null; do sleep 300; done
for hv in max-windows-3090:s1y_ft 3080-gaming-desktop:s4m_ft; do
  h=${hv%%:*}; v=${hv#*:}
  wait_job "$h" "p1vint_${v}_F_s0"
  (cd training && nice -n 19 python3 paper1_revision/vintage_launch.py fetch-run "$v" 0 "$h")
done
(cd training && nice -n 19 python3 paper1_revision/vintage_launch.py compute) || true
echo "$(date -Is) vintage runs fetched"
for v in s4m s1y s4m_ft s1y_ft; do
  nice -n 19 rsync -rt --rsync-path="$(remote_rsync $GPU)" training/paper1_revision/site/vint/$v/ \
    "$GPU:/home/max/hots/repo/training/paper1_revision/site/vint/$v/"
done
run_gpu stale $STEP p3t_tourney.py:run:--schedule:all:--only-ready

# 5. scoring and analysis on the 3090
relay $GPU $CPU
run_cpu score $STEP p3t_judge.py:score p3t_diag.py p3t_analyze.py \
  p3t_backtest.py:agree_analyze:--games:20000:--vintage:cur_bc p3t_backtest.py:agree_analyze:--games:20000:--vintage:cur
run_cpu scoresplit env P3T_SPLIT=1 $STEP p3t_judge.py:score p3t_diag.py p3t_analyze.py
run_cpu scoretune env P3T_LOBBYSET=tune $STEP p3t_judge.py:score p3t_analyze.py

mkdir -p training/personalization/results/p3t
for h in $CPU $GPU; do
  nice -n 19 rsync -t --rsync-path="$(remote_rsync $h)" "$h:$B/results/rcf_full/p3t/*.json" training/personalization/results/p3t/ || true
done
python3 training/personalization/p3t_report.py
echo "P3T RACEFIX RERUN DONE $(date -Is)"
