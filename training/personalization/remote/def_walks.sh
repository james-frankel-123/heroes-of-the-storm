#!/bin/bash
# P3 definitive build, walks (3080, <= 20 GB): the baseline arms, the chosen
# shrinkage model (LK2_full2) for the drafter half and for the judge half
# (full_ms_flip), calibration curves, the display band.
# Usage: bash personalization/remote/def_walks.sh <D>
set -uo pipefail
D=${1:?build tag}
cd "$(dirname "$0")/../.."
. personalization/remote/def_env.sh "$PWD" "$D"
PY="python -u personalization/p3_rcf.py online --hist merged"
QW=$(python -c "print(0.02**2/365)"); QM=$(python -c "print(0.4**2/365)")
CHOSEN="--layer --q-win $QW --q-met $QM --tail shrinkfn2 --statlin --postcal"
mark() { echo "=== $1 $(date -Is)"; }
export P3_RCF_SHARDS=${P3_RCF_SHARDS:-3}
have() { [ -f $DEF_RESULTS/rcf/p3c/$1__merged.npz ] && echo "skip $1 (done)"; }
have today      || { mark today;      $PY --tag today; }
have full_ms    || { mark full_ms;    $PY --tag full_ms; }
have full_ms_layer || { mark ms_layer; $PY --tag full_ms --layer --parts $DEF_CACHE/rcf/parts_full_ms.npz --out full_ms_layer; }
have chosen     || { mark chosen;     P3_RCF_POSTCAL=2 P3_RCF_MMR=combiner $PY --tag full_ms --parts $DEF_CACHE/rcf/parts_chosen.npz $CHOSEN --out chosen; }
have chosen_flip || { mark chosen_flip; P3_RCF_POSTCAL=2 P3_RCF_MMR=combiner $PY --tag full_ms_flip --parts $DEF_CACHE/rcf/parts_chosen_flip.npz $CHOSEN --out chosen_flip; }
mark calib;      python -u personalization/p3_rcf_calib.py --arms today,full_ms_layer,chosen,chosen_flip --ref full_ms_layer --name def --plot full_ms_layer,chosen --labels "previous main model,chosen shrinkage model"
mark band;       python -u personalization/p3_rcf_band_def.py --arm chosen
mark states;     python -u personalization/p3_rcf_states.py selfcheck
mark report;     mkdir -p $DEF_RESULTS/p3c && for a in today full_ms_layer chosen; do ln -sf $DEF_RESULTS/rcf/p3c/${a}__merged.npz $DEF_RESULTS/p3c/$a.npz; done && \
                 python -u personalization/p3_c_tests.py report --arms today,full_ms_layer,chosen --ref today --name def
echo "WALKS DONE $(date -Is)"
