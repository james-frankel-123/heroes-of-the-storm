#!/bin/bash
# P3 definitive build, fits (5090 GPU, <= 11.5 GB host RAM): the chosen
# model's parameters on the drafter half (full_ms) and on the other half
# (full_ms_flip, the tournament judge). Same settings as P3_RICH_CF met2_hs2
# with the a_shrink card.
# Usage: bash personalization/remote/def_fit.sh <D>
set -euo pipefail
D=${1:?build tag}
cd "$(dirname "$0")/../.."
. personalization/remote/def_env.sh "$PWD" "$D"
F="--hist merged --kmax 2 --metrics --gshare --hero-s2 --sb a_shrink --f64 --rounds 3 --iters 500"
python -u personalization/p3_rcf.py fit --tag full_ms $F
python -u personalization/p3_rcf.py fit --tag full_ms_flip --flip-half $F
echo "FIT DONE $(date -Is)"
