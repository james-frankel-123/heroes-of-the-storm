#!/bin/bash
# Personal-term recalibration arms (coordinator 2026-11-14): s-pcal and s-pcal-floor25 (p3t_support_arms.json:
# the drafter's combiner b2, b3 times 0.88; the vintage table, its b2 = 4.2429 gate and the judges unchanged).
# Queued behind p3t_curve: waits for that hotsjob to complete (by its job state, never a process match).
# Launch on the 3090 from ~/hots/repo/training:
#   python remote_workers/hotsjob.py launch p3t_pcal "bash personalization/remote/p3t_pcal_chain.sh" --mem-max 60G
# Steps (markers p3t/support_chain/pcal_<step>.done):
#   wait      p3t_curve completed (fails if it failed or died)
#   gates     vintage table (unscaled headline combiner)
#   headline  the two arms vs the six support opponents, both seats; score; summary (all arms) -> summary.json/md
#   tune      the two arms vs the tune opponents, both seats; score; tune_pcal.json
set -euo pipefail
cd /home/max/hots/repo/training
source ~/hots/env.sh
B=/home/max/hots/repo/training/personalization
CROOT=$B/cache_def_def1027_hlsup
RROOT=$B/results/def_def1027_hlsup
export P3_CACHE=$CROOT P3_RESULTS=$RROOT P3T_VINTAGES=$RROOT/p3t_vintages.json
export P3_RCF_ROWS=$CROOT/merged_rows_def1027_2695e2cac4d17884.npz P3_RCF_KERNELS=$CROOT/hs_kernels.npz
export P3T_PROCS=12 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMBA_NUM_THREADS=4 PYTHONUNBUFFERED=1
T=/home/max/hots/repo/training
P1ENV="P1_TIERS=site P1R_NPROC=3 WP_STATS_PATH=$T/paper1_revision/site/cache/stats/deploy.json P1R_COMP_PATH=$T/paper1_revision/site/cache/stats/deploy_compositions.json PYTHONPATH=$T/paper1_revision/oct2026_site"
MK=$CROOT/p3t/support_chain
OUT=$RROOT/p3t_support
mkdir -p "$MK" "$OUT"
PAR=${P3T_SUP_PAR:-2}
step() { local n=$1; shift
  if [ -e "$MK/pcal_$n.done" ]; then echo "$(date -Is) $n: done earlier"; return 0; fi
  echo "$(date -Is) $n: start"; "$@"; date -Is > "$MK/pcal_$n.done"; echo "$(date -Is) $n: done"; }
PY=python
TT="nice -n 5 $PY -u personalization/p3t_tourney.py"
SP="nice -n 5 $PY -u personalization/p3t_support.py"
SC="nice -n 5 $PY -u personalization/p3t_judge.py score"

run_par() {   # schedule [env...]: PAR processes on the one GPU, stable shards
  local sch=$1; shift
  local pids=() rc=0
  for i in $(seq 0 $((PAR - 1))); do
    env "$@" $TT run --schedule "$sch" $SYSF --shard $i/$PAR > "$MK/pcal_${sch}_$i.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  tail -n 3 "$MK"/pcal_${sch}_*.log
  return $rc
}

waitcurve() {
  while true; do
    st=$($PY remote_workers/hotsjob.py state p3t_curve 2>/dev/null | tail -1 || true)
    case "$st" in
      completed) echo "p3t_curve completed"; return 0 ;;
      failed|died) echo "p3t_curve $st"; return 1 ;;
    esac
    sleep 60
  done
}
gates() {
  $PY -c "import sys; sys.path.insert(0, 'personalization'); import p3t_support as S; print('vintages ok:', S.check_vintages())"
}
headline() {
  SYSF="--systems s-pcal,s-pcal-floor25" run_par support
  $SC
  env $P1ENV $SP summary --out $OUT
}
tune() {
  SYSF="" run_par support_tune_pcal P3T_LOBBYSET=tune
  P3T_LOBBYSET=tune $SC
  env P3T_LOBBYSET=tune $P1ENV $SP tune_pcal --out $OUT
}

step wait waitcurve
step gates gates
step headline headline
step tune tune
echo "P3T PCAL CHAIN DONE $(date -Is)"
