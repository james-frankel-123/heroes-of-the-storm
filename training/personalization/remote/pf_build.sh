#!/bin/bash
# P3 post-freeze confirmation build (main box, nice 19, 4 cores): the frozen
# build <B>'s inputs plus the 2.57 games dated 2026-09-28 .. before <CAP>,
# then the arms scored with every coefficient frozen from build <B>.
# Prerequisites (done by hand, see results/P3_POSTFREEZE.md):
#   p3_x_fetch_post.py and p3_x_wp_post.py with P3_PF_CAP / P3_PF_OUT -> cache_def_<D>/x_post_*, x_wp_post.npz
#   p3_export.py stamps|games|scoreboard --tag <D> --pf-cap <CAP> --base-tag <B>
#   P3_CAP_DATE=<CAP> p3_full_history.py build --tag <D>
# Usage (from training/): bash personalization/remote/pf_build.sh <D> <B>
set -euo pipefail
D=${1:?post-freeze tag}; B=${2:?frozen build tag}
cd "$(dirname "$0")/../.."
T=$PWD
DC=$T/personalization/cache_def_$D
BC=$T/personalization/cache_def_$B
for f in x_post_games.json.gz x_post_players.npz x_wp_post.npz; do [ -f $DC/$f ] || { echo "missing $DC/$f"; exit 2; }; done
# frozen inputs of build B: today's kernels (fit on E), MMR coefficients (fit on E)
mkdir -p $DC/rcf $DC/.done
cp -n $BC/hs_kernels.npz $DC/hs_kernels.npz
cp -n $BC/rcf/mmr_coef.npy $DC/rcf/mmr_coef.npy
touch $DC/.done/kernels $DC/.done/prep       # no refit of the kernels, no fit inputs
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMBA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
bash personalization/remote/def_build_a1.sh $D
# params of build B (no refit)
mkdir -p $T/personalization/results/def_$D/rcf/params
cp -n $T/personalization/results/def_$B/rcf/params/full_ms.* $T/personalization/results/def_$D/rcf/params/
echo "PF BUILD DONE $(date -Is)"
