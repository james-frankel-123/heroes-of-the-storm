#!/bin/bash
# P3 tournament: measurement dry run of every CPU stage on one host, in an isolated directory
# (outputs never published). Per stage: peak memory (MAXRSS_KB), seconds, exit code, and the
# input files it read (p3t_trace.py) -> $D/dryrun.jsonl and $D/trace.jsonl.
# Usage (worker, from ~/hots/repo/training): bash personalization/remote/p3t_dryrun.sh <dir> [first-stage-index]
set -u
D=${1:?dir}
FIRST=${2:-0}
export P3_CACHE=$D/cache_full P3_RESULTS=$D/results
export P3_RCF_ROWS=$D/cache_full/merged_rows_full_1017.npz P3_RCF_KERNELS=$D/cache_full/hs_kernels.npz
export P3T_PROCS=${P3T_PROCS:-6}
mkdir -p $P3_CACHE/p3t $P3_RESULTS/p3t
# GPU-stage outputs the CPU stages read (copied in for the measurement only)
for f in $P3_CACHE/p3t_gpu/*; do [ -e "$f" ] && cp -n "$f" $P3_CACHE/p3t/; done
STAGES=(
  "p3t_data.py lobbies"
  "p3t_data.py states --vintages cur,s4m,s1y,judge --judge-params full_ms_flip"
  "p3t_data.py judgefit --judge-params full_ms_flip"
  "p3t_data.py split_states"
  "p3t_data.py split_judgefit"
  "p3t_data.py rosters"
  "p3t_data.py prelobbies"
  "p3t_value.py combfit --vintages cur,s4m,s1y,split"
  "p3t_judge.py fit"
  "p3t_vnet.py data"
  "p3t_vnet.py train"
  "p3t_vnet.py train_pre"
  "p3t_vstudy.py"
  "p3t_value.py predeval"
  "p3t_tourney.py select"
  "p3t_probe.py extract"
  "p3t_backtest.py forced"
  "ENV:P3T_VSTUDY_ASSIGN=best p3t_vstudy.py"
  "ENV:P3T_SPLIT=1 p3t_judge.py fit"
  "ENV:P3T_LOBBYSET=tune p3t_tourney.py select"
  "p3t_pprior.py train"
  "p3t_oppmodel.py"
  "p3t_rprior.py data_search"
  "p3t_rprior.py train --signal search"
  "p3t_rprior.py data_bcow"
  "p3t_rprior.py train --signal bcow"
  "p3t_rprior.py data_onestep"
  "p3t_rprior.py train --signal onestep"
  "p3t_draftrec.py data"
  "p3t_oppdr.py lobbies"
  "p3t_oppdr.py hist"
)
i=0
for st in "${STAGES[@]}"; do
  if [ $i -lt $FIRST ]; then i=$((i + 1)); continue; fi
  envs=""
  if [[ $st == ENV:* ]]; then envs=${st#ENV:}; envs=${envs%% *}; st=${st#ENV:$envs }; fi
  set -- $st
  script=$1; shift
  t0=$(date +%s)
  echo "=== [$i] $envs $script $* $(date -Is)"
  env $envs /usr/bin/time -o $D/.time -f "%M %e" python -u personalization/p3t_trace.py $D/trace.jsonl \
    personalization/$script "$@"
  rc=$?
  read -r kb secs < <(tail -1 $D/.time) 2>/dev/null || { kb=-1; secs=$(( $(date +%s) - t0 )); }
  echo "{\"i\": $i, \"stage\": \"$envs $script $*\", \"rc\": $rc, \"maxrss_gb\": $(awk "BEGIN{print $kb/1048576}"), \"secs\": $secs}" | tee -a $D/dryrun.jsonl
  i=$((i + 1))
done
echo "DRYRUN DONE $(date -Is)"
