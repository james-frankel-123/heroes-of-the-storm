#!/bin/bash
# Arms for the full-history tests on one cache (from ~/hots/repo/training):
# extra rows from the merged store, then the slot-only arm, the merged arm
# (A2) and the merged arm under rule "seen" (A2-seen).
# Usage: bash personalization/remote/run_p3c.sh <cache dir> <store history.npz> <slot arm> <merged arm> [refit]
set -euo pipefail
CACHE=${1:?cache}; STORE=${2:?store}; SA=${3:?slot arm}; MA=${4:?merged arm}; REFIT=${5:-}
export P3_CACHE=$CACHE
TAG=$(basename "$(dirname "$STORE")")
ROWS=$CACHE/merged_rows_$TAG.npz
[ -f "$ROWS" ] || python personalization/p3_c_merged.py rows --store "$STORE"
python personalization/p3_c_tests.py predict --arm "$SA" --rows "$ROWS"
python personalization/p3_c_tests.py predict --arm "$MA" --history merged --rows "$ROWS"
python personalization/p3_c_tests.py predict --arm "${MA}seen" --history merged --rule seen --rows "$ROWS"
if [ "$REFIT" = refit ]; then
  python personalization/p3_c_tests.py predict --arm "${MA}refit" --history merged --refit --rows "$ROWS"
fi
