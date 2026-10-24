#!/bin/bash
# Team-share chosen model with metric x role and metric x hero direct-channel
# weights (full_1017 panel), reusing the LK2_share walk parts. Any host.
set -uo pipefail
cd "$(dirname "$0")/../.."
. personalization/remote/rcf_env.sh "$PWD" rcf_shrink
PY="${RCF_PY:-python} -u personalization/p3_rcf.py online --hist merged --tag full_ms"
QW=$(python -c "print(0.02**2/365)"); QM=$(python -c "print(0.4**2/365)")
P=$P3_CACHE/rcf/parts_kal2_share.npz
ARGS="--parts $P --layer --q-win $QW --q-met $QM --tail shrinkfn2 --statlin --statlin-extra share6 --postcal"
export P3_RCF_POSTCAL=2 P3_RCF_MMR=combiner P3_RCF_LAYER_SAVE=/tmp/rcf_share_group_layer.npz
$PY $ARGS --out LK2_share_rep
P3_RCF_STATLIN_GROUP=role $PY $ARGS --out LK2_share_role
P3_RCF_STATLIN_GROUP=hero $PY $ARGS --out LK2_share_hero
echo GROUP DONE
