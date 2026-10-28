#!/bin/bash
# Shrinkage study, baselines on the full_1017 panel (any host; local here).
set -uo pipefail
cd "$(dirname "$0")/../.."
. personalization/remote/rcf_env.sh "$PWD" rcf_shrink
PY="${RCF_PY:-python} -u personalization/p3_rcf.py"
P=$P3_CACHE/rcf/parts_full_ms.npz
$PY online --tag today --hist merged
$PY online --tag full_ms --hist merged
$PY online --tag full_ms --hist merged --layer --parts $P --out full_ms_layer
echo BASE DONE
