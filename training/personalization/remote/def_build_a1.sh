#!/bin/bash
# P3 definitive build, worker stage (3080, <= 20 GB, from ~/hots/repo/training):
# A1 cache, merged rows, scoreboard card, model caches and fit inputs.
# Inputs already shipped: cache/export/<D>/{stamps,games,scoreboard}.csv.gz,
# cache/full_history/<D>/history.npz, the pinned upstream window-cache files.
# Usage: bash personalization/remote/def_build_a1.sh <D>
set -euo pipefail
D=${1:?build tag}
cd "$(dirname "$0")/../.."
T=$PWD
PY="python -u"
DONE=$T/personalization/cache_def_$D/.done
mkdir -p $DONE
mark() { echo "=== $1 $(date -Is)"; }
# step NAME CMD...: run once; a marker skips it on a rerun after a failure
step() { local n=$1; shift; if [ -e $DONE/$n ]; then echo "skip $n (done)"; return 0; fi; mark $n; "$@"; touch $DONE/$n; }
# 1. stamps parse (window cache, writes cache/export/<D>/stamps.npz)
step parse   $PY personalization/p3_hero_level_causal.py parse --tag "$D"
# 2. A1 cache from the export (players, hero MMR, game times, MMR stamps; links upstream inputs)
step history $PY personalization/p3_c_history.py --tag "$D" --out personalization/cache_def_$D
export P3_CACHE=$T/personalization/cache_def_$D P3_RESULTS=$T/personalization/results/def_$D
mkdir -p $P3_RESULTS
ln -sfn $T/personalization/cache/full_history $P3_CACHE/full_history
ln -sfn $T/personalization/cache/export $P3_CACHE/export
# 3. slot table, kernels (today's model, the baseline arm), MMR at game, extended slots
step slots   $PY personalization/p3_hs_fit.py slots
step kernels $PY personalization/p3_hs_fit.py fit
step mmr     $PY personalization/p3_mmr_at_game.py
step ext     $PY personalization/p3_x_common.py ext
# 4. merged rows, one versioned file: merged_rows_<D>_<slot fingerprint>.npz
if [ ! -e $DONE/rows ]; then
  mark rows; $PY personalization/p3_c_merged.py rows --store $D --slots $P3_CACHE/x_slots_ext.npz
  FP=$($PY -c "import numpy as np; print(str(np.load('$P3_CACHE/merged_rows_$D.npz')['fingerprint']))")
  mv $P3_CACHE/merged_rows_$D.npz $P3_CACHE/merged_rows_${D}_$FP.npz; touch $DONE/rows
fi
FP=$(ls $P3_CACHE/merged_rows_${D}_*.npz | head -1 | sed 's/.*_//; s/.npz$//')
export P3_RCF_ROWS=$P3_CACHE/merged_rows_${D}_$FP.npz P3_RCF_KERNELS=$P3_CACHE/hs_kernels.npz
# 5. scoreboard card a_shrink (full export)
step cards   $PY personalization/p3_rcf_cards.py --tag "$D" --cards a_shrink
step share6  $PY personalization/p3_rcf_cards.py --tag "$D" --cards share6
# 6. compact caches: aug (plain and MMR-adjusted) and the aug-indexed metrics, MMR coefficients
if [ ! -e $DONE/aug ]; then mark aug
P3_RCF_WRITE_AUG=1 $PY -c "
import sys; sys.path.insert(0, 'personalization')
import p3_rcf as R
d, aug = R.load_aug('merged'); R.aug_metrics(aug, 'a_shrink', d)"
P3_RCF_MMR=1 P3_RCF_WRITE_AUG=1 $PY -c "
import sys; sys.path.insert(0, 'personalization')
import p3_rcf as R
R.load_aug('merged')"
touch $DONE/aug; fi
if [ ! -e $DONE/zaug_share6 ]; then mark zaug_share6
P3_RCF_WRITE_AUG=1 $PY -c "
import sys; sys.path.insert(0, 'personalization')
import p3_rcf as R
d, aug = R.load_aug('merged'); R.aug_metrics(aug, 'share6', d)"
touch $DONE/zaug_share6; fi
# 7. fit inputs (predictive split at 2025-03-01)
step prep    $PY personalization/p3_rcf.py prep --hist merged --sb a_shrink
echo "A1 BUILD DONE $(date -Is)"
