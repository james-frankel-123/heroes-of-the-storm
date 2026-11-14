#!/bin/bash
# Promotion of the floored drafter s-floor25 (p3t_support.PROMOTE) to paper 2's drafter, pending confirmation
# (coordinator 2026-11-14). One resumable chain on the 3090, after p3t_support_chain.sh, in the same namespace
# (cache_def_def1027_hlsup / results/def_def1027_hlsup; vintage table with the headline combiner, b2 = 4.2429).
# Launch (on the worker, from ~/hots/repo/training):
#   python remote_workers/hotsjob.py launch p3t_promote "bash personalization/remote/p3t_promote_chain.sh" --mem-max 60G
# Lobby sets (copies of the headline pass's, read from the 3080): tune = the frozen 300 tune lobbies
# (tourney_tune, hist_tune); loose = P3_TOURNAMENT_LOOSE's 300 (tourney_loose, hist_loose = its hist_tourney);
# rep<n> = the headline lobbies again (tourney_rep<n>, hist_rep<n> = hist_tourney) with P3T_SEED_OFFSET=<n>.
# The *_ref directories keep the original drafts and scores.
# Steps (markers in p3t/support_chain/promote_<step>.done; every step skips finished pairs and scores):
#   gates    vintage table; redraft the selected design's tune and loose pairs (drafted on the 5090, sm_120):
#            >= 97% identical (cross-architecture near-ties; the definitive run saw 0.9% differ)
#   confirm  s-floor25 vs the tune opponents, both seats; score the tune set; tune table (results p3t_support/tune.json)
#   matrix   s-floor25 vs every MAIN system, both seats, headline lobbies; score; matrix.json/md
#   rep      the REP set with s-floor25 at kernel-seed offsets 1-3; score; rep.json/md
#   loose    s-floor25 (and the selected design) vs pMCTS-cur and aMCTS-pp on the loose lobbies; score; loose.json/md
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
  if [ -e "$MK/promote_$n.done" ]; then echo "$(date -Is) $n: done earlier"; return 0; fi
  echo "$(date -Is) $n: start"; "$@"; date -Is > "$MK/promote_$n.done"; echo "$(date -Is) $n: done"; }
PY=python
TT="nice -n 5 $PY -u personalization/p3t_tourney.py"
SP="nice -n 5 $PY -u personalization/p3t_support.py"
SC="nice -n 5 $PY -u personalization/p3t_judge.py score"

run_par() {   # schedule [env...]: PAR processes on the one GPU, stable shards
  local sch=$1; shift
  local pids=() rc=0
  for i in $(seq 0 $((PAR - 1))); do
    env "$@" $TT run --schedule "$sch" --shard $i/$PAR > "$MK/promote_${sch}_$i.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  tail -n 3 "$MK"/promote_${sch}_*.log
  return $rc
}

gates() {
  $PY -c "import sys; sys.path.insert(0, 'personalization'); import p3t_support as S; print('vintages ok:', S.check_vintages())"
  for p in x-combo1-dr-chance__pMCTS-cur x-combo1-dr-chance__aMCTS-pp; do
    P3T_LOBBYSET=tune $SP verify --pair $p --ref $CROOT/p3t/tourney_tune_ref > "$MK/promote_verify_tune_$p.json"
  done
  for p in x-combo1-dr-chance__pMCTS-cur x-combo1-dr-chance__GD; do
    P3T_LOBBYSET=loose $SP verify --pair $p --ref $CROOT/p3t/tourney_loose_ref > "$MK/promote_verify_loose_$p.json"
  done
  $PY - "$MK" <<'PYC' | tee "$MK/promote_gates.txt"
import glob, json, sys
for f in sorted(glob.glob(sys.argv[1] + "/promote_verify_*.json")):
    t = open(f).read()
    d = json.loads(t[t.index("{"):])
    print(f.rsplit("/", 1)[1], d["identical_drafts"], "of", d["lobbies"], "identical; ref", d["ref_gpu"], "differing", d["first_diff_lobbies"])
    assert d["identical_drafts"] >= 0.97 * d["lobbies"], f
print("PROMOTE GATES OK")
PYC
}
confirm() {
  run_par support_confirm P3T_LOBBYSET=tune
  P3T_LOBBYSET=tune $SC
  env P3T_LOBBYSET=tune $P1ENV $SP tune --out $OUT
}
matrix() {
  run_par support_matrix
  $SC
  env $P1ENV $SP matrix --out $OUT
}
rep() {
  for n in 1 2 3; do
    run_par support_rep P3T_LOBBYSET=rep$n P3T_SEED_OFFSET=$n
    P3T_LOBBYSET=rep$n $SC
  done
  $SP rep --out $OUT
}
loose() {
  run_par support_loose P3T_LOBBYSET=loose
  P3T_LOBBYSET=loose $SC
  env P3T_LOBBYSET=loose $P1ENV $SP loose --out $OUT
}

step gates gates
step confirm confirm
step matrix matrix
step rep rep
step loose loose
echo "P3T PROMOTE CHAIN DONE $(date -Is)"
