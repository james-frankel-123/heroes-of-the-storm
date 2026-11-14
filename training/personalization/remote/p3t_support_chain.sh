#!/bin/bash
# Paper 2 support arms (p3t_support.py), end to end on one GPU worker (the 3090), resumable.
# Launch (on the worker, from ~/hots/repo/training) as a pause-aware hotsjob plain job:
#   python remote_workers/hotsjob.py launch p3t_support "bash personalization/remote/p3t_support_chain.sh" --mem-max 60G
# Pause/resume: hotsjob.py pause|resume p3t_support. Every step skips finished outputs (pair files,
# score files, step markers), so a resume continues where it stopped.
# Namespace: a copy of the headline pass (cache_def_def1027_headline/p3t: lobbies, states, priors,
# opponent fits, DraftRec; results/def_def1027_headline: judge, vintage table with the headline
# combiner) as cache_def_def1027_hlsup / results/def_def1027_hlsup; tourney/ holds the 28 headline
# pairs among the selected design and its opponents (rescored here: scoring check) plus the new pairs.
# Steps:
#   gates     the vintage table carries the headline combiner; redraft x-combo1-dr-chance__pMCTS-cur and
#             x-combo1-dr-chance__aMCTS-pp (drafted on the 3080, sm_86 like the 3090) with the explore
#             kernel and through the support kernel with every support term off: must be identical
#   smoke     each kernel-side arm against GD (one pair each, all 300 lobbies)
#   prior     pprior_ffloor (F_floor as p_F, the same visit targets) and a refit of pprior on F_oof
#             with the same code (pprior_refitcheck: the refit's quality next to the original)
#   drafts    every support pair (schedule support), P3T_SUP_PAR processes on the one GPU
#   score     p3t_judge.py score (the 28 headline pairs too), then p3t_support.py summary
set -euo pipefail
cd /home/max/hots/repo/training
source ~/hots/env.sh
B=/home/max/hots/repo/training/personalization
CROOT=$B/cache_def_def1027_hlsup
RROOT=$B/results/def_def1027_hlsup
export P3_CACHE=$CROOT P3_RESULTS=$RROOT P3T_VINTAGES=$RROOT/p3t_vintages.json
export P3_RCF_ROWS=$CROOT/merged_rows_def1027_2695e2cac4d17884.npz P3_RCF_KERNELS=$CROOT/hs_kernels.npz
export P3T_PROCS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMBA_NUM_THREADS=4 PYTHONUNBUFFERED=1
# paper 1 env (support corpus for the summary; never builds one: the corpus files are in place)
T=/home/max/hots/repo/training
P1ENV="P1_TIERS=site P1R_NPROC=3 WP_STATS_PATH=$T/paper1_revision/site/cache/stats/deploy.json P1R_COMP_PATH=$T/paper1_revision/site/cache/stats/deploy_compositions.json PYTHONPATH=$T/paper1_revision/oct2026_site"
MK=$CROOT/p3t/support_chain
mkdir -p "$MK"
PAR=${P3T_SUP_PAR:-3}
step() { local n=$1; shift
  if [ -e "$MK/$n.done" ]; then echo "$(date -Is) $n: done earlier"; return 0; fi
  echo "$(date -Is) $n: start"; "$@"; date -Is > "$MK/$n.done"; echo "$(date -Is) $n: done"; }
PY=python
TT="nice -n 5 $PY -u personalization/p3t_tourney.py"
SP="nice -n 5 $PY -u personalization/p3t_support.py"

gates() {
  $PY -c "import sys; sys.path.insert(0, 'personalization'); import p3t_support as S; print('vintages ok:', S.check_vintages())"
  for p in x-combo1-dr-chance__pMCTS-cur x-combo1-dr-chance__aMCTS-pp; do
    $SP verify --pair $p --ref $CROOT/p3t/tourney_headline | tee "$MK/verify_$p.json"
    $SP verify --pair $p --ref $CROOT/p3t/tourney_headline --force-support-kernel | tee "$MK/verify_sk_$p.json"
  done
  # every check must reproduce all 300 drafts
  $PY - "$MK" <<'PYC'
import glob, json, re, sys
bad = []
for f in glob.glob(sys.argv[1] + "/verify_*.json"):
    t = open(f).read()
    m = re.search(r'"identical_drafts": (\d+)', t)
    n = re.search(r'"lobbies": (\d+)', t)
    if not m or m.group(1) != n.group(1):
        bad.append(f)
if bad:
    sys.exit(f"reproduction failed: {bad}")
print("GATES OK")
PYC
}
smoke() {
  for a in s-floor10 s-floor25 s-leaf3 s-gd05; do
    $TT run --pairs ${a}__GD
  done
  $PY - <<'PYC'
import sys, numpy as np
sys.path.insert(0, "personalization")
import p3t_tourney as TT, p3t_support as SP
c = SP.c3_sym()
import itertools
for a in ("s-floor10", "s-floor25", "s-leaf3", "s-gd05"):
    acts = np.load(f"{TT.TDIR}/{a}__GD.npz")["acts"]
    t0 = acts[:, [k for k in range(16) if SP.IS_PICK[k] and SP.DRAFT_TEAM[k] == 0]]
    m3 = np.array([min(c[x, y, z] for x, y, z in itertools.combinations(r, 3)) for r in t0])
    print(a, "teams", len(m3), "<25", (m3 < 25).mean().round(3), "<10", (m3 < 10).mean().round(3), "median", np.median(m3))
PYC
}
prior() {
  [ -s $CROOT/p3t/pprior_ffloor.pt ] || nice -n 5 $PY -u personalization/p3t_pprior.py train --vintage cur_ffloor --out pprior_ffloor
  [ -s $CROOT/p3t/pprior_refitcheck.pt ] || nice -n 5 $PY -u personalization/p3t_pprior.py train --vintage cur --out pprior_refitcheck
}
drafts() {
  local pids=()
  for i in $(seq 0 $((PAR - 1))); do
    $TT run --schedule support --shard $i/$PAR > "$MK/drafts_$i.log" 2>&1 &
    pids+=($!)
  done
  local rc=0
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  tail -3 "$MK"/drafts_*.log
  return $rc
}
score() {
  nice -n 5 $PY -u personalization/p3t_judge.py score
  # scoring check: the 28 headline pairs rescored here against the headline score files
  $PY - "$CROOT/p3t" <<'PYC' | tee "$MK/score_check.txt"
import os, sys, numpy as np
d = sys.argv[1]
worst = 0.0
for f in sorted(os.listdir(d + "/scores_headline")):
    if not os.path.exists(f"{d}/scores/{f}") or "__" not in f:
        continue
    a, b = np.load(f"{d}/scores_headline/{f}"), np.load(f"{d}/scores/{f}")
    for k in ("judge_personal", "consensus", "rec_slot", "judge_personal_of"):
        x, y = a[k].astype(float), b[k].astype(float)
        dd = float(np.nanmax(np.abs(x - y))) if np.isfinite(x).any() else 0.0
        assert (np.isnan(x) == np.isnan(y)).all(), (f, k, "nan pattern")
        worst = max(worst, dd)
    print(f, "key same" if str(a["score_key"]) == str(b["score_key"]) else "key differs")
print("max abs diff over J, CONS, record, J_of:", worst)
PYC
  env $P1ENV nice -n 5 $PY -u personalization/p3t_support.py summary
}

step gates gates
step smoke smoke
step prior prior
step drafts drafts
step score score
echo "P3T SUPPORT CHAIN DONE $(date -Is)"
