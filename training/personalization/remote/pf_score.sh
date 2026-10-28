#!/bin/bash
# P3 post-freeze confirmation: score today's kernel, the previous main model and
# the chosen model on build <D> with every coefficient frozen from build <B>
# (P3_RCF_FROZEN), the offset table of <B>, window 4 = games from 2026-09-28.
# Usage (from training/): bash personalization/remote/pf_score.sh <D> <B>
set -euo pipefail
D=${1:?post-freeze tag}; B=${2:?frozen build tag}
cd "$(dirname "$0")/../.."
. personalization/remote/def_env.sh "$PWD" "$D"
BR=$PWD/personalization/results/def_$B/rcf/p3c
export P3_PF_WINDOW=1 P3_MMR_KEY_FIX=1 P3_RCF_FROZEN_TABLE=$PWD/personalization/cache_def_$B/rcf/aug_merged.npz
export P3_RCF_SHARDS=${P3_RCF_SHARDS:-3} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMBA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY="${PF_PY:-/usr/bin/python3} -u personalization/p3_rcf.py online --hist merged"
QW=$(python3 -c "print(0.02**2/365)"); QM=$(python3 -c "print(0.4**2/365)")
have() { [ -f $DEF_RESULTS/rcf/p3c/$1__merged.npz ] && echo "skip $1 (done)"; }
have today || P3_RCF_FROZEN=$BR/today__merged.npz $PY --tag today
have full_ms_layer || P3_RCF_FROZEN=$BR/full_ms_layer__merged.npz $PY --tag full_ms --layer \
  --parts $DEF_CACHE/rcf/parts_full_ms.npz --out full_ms_layer
have chosen || P3_RCF_FROZEN=$BR/chosen__merged.npz P3_RCF_POSTCAL=2 P3_RCF_MMR=combiner $PY --tag full_ms \
  --parts $DEF_CACHE/rcf/parts_chosen.npz --layer --q-win $QW --q-met $QM --tail shrinkfn2 --statlin \
  --statlin-extra share6 --postcal --out chosen
echo "PF SCORE DONE $(date -Is)"
