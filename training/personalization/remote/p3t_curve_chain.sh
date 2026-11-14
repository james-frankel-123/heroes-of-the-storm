#!/bin/bash
# Graded-floor tradeoff curve (coordinator 2026-11-14): s-floor15 and s-floor20 (p3t_support_arms.json) next to
# s-floor10/25. A reported curve, not a promotion attempt. Same namespace, gates and conventions as
# p3t_support_chain.sh / p3t_promote_chain.sh. Launch on the 3090 from ~/hots/repo/training:
#   python remote_workers/hotsjob.py launch p3t_curve "bash personalization/remote/p3t_curve_chain.sh" --mem-max 60G
# Steps (markers p3t/support_chain/curve_<step>.done):
#   gates     vintage table (b2 = 4.2429); the kernel-equivalence / headline reproduction files of the first chain
#   headline  the two arms vs the six support opponents, both seats; score; summary (all arms) -> summary.json/md
#   tune      s-floor10/15/20 vs the tune opponents, both seats (s-floor25 done); score; tune_curve.json
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
  if [ -e "$MK/curve_$n.done" ]; then echo "$(date -Is) $n: done earlier"; return 0; fi
  echo "$(date -Is) $n: start"; "$@"; date -Is > "$MK/curve_$n.done"; echo "$(date -Is) $n: done"; }
PY=python
TT="nice -n 5 $PY -u personalization/p3t_tourney.py"
SP="nice -n 5 $PY -u personalization/p3t_support.py"
SC="nice -n 5 $PY -u personalization/p3t_judge.py score"

run_par() {   # schedule [env...]: PAR processes on the one GPU, stable shards
  local sch=$1; shift
  local pids=() rc=0
  for i in $(seq 0 $((PAR - 1))); do
    env "$@" $TT run --schedule "$sch" $SYSF --shard $i/$PAR > "$MK/curve_${sch}_$i.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  tail -n 3 "$MK"/curve_${sch}_*.log
  return $rc
}

gates() {
  $PY -c "import sys; sys.path.insert(0, 'personalization'); import p3t_support as S; print('vintages ok:', S.check_vintages())"
  cat "$MK/gates.txt" "$MK/promote_gates.txt"
}
headline() {
  SYSF="--systems s-floor15,s-floor20" run_par support
  $SC
  env $P1ENV $SP summary --out $OUT
}
tune() {
  SYSF="" run_par support_tune_curve P3T_LOBBYSET=tune
  P3T_LOBBYSET=tune $SC
  env P3T_LOBBYSET=tune $P1ENV $SP tune_curve --out $OUT
}

step gates gates
step headline headline
step tune tune
echo "P3T CURVE CHAIN DONE $(date -Is)"
