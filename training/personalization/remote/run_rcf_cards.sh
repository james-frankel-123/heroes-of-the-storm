#!/bin/bash
# Scoreboard cards end to end: for each card, prep its metric arrays here,
# ship them to the 5090, fit met2_hs2's settings there (inner-validation fit
# and the final fit, one hotsjob at a time), then evaluate here.
# Usage (from training/): personalization/remote/run_rcf_cards.sh card ...
set -uo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMBA_NUM_THREADS=4
PY="nice -n 19 taskset -c 44-47 /usr/bin/python3 personalization/p3_rcf.py"
RP=hots/repo/training/personalization
source remote_workers/_remote.sh
rsync -t personalization/p3_rcf.py windows-5090-wsl:$RP/
for card in "$@"; do
  until [ -f personalization/cache/rcf/sb_$card.npz ]; do sleep 30; done
  echo "=== $card $(date -Is)"
  [ -f personalization/cache/rcf/fitm_merged_$card.npz ] || $PY prep --hist merged --sb $card
  rsync -t personalization/cache/rcf/fitm_merged_$card.npz windows-5090-wsl:$RP/cache/rcf/
  F="--kmax=2,--metrics,--gshare,--hero-s2,--sb=$card"
  RUN_NAME=rcf_card_$card HOTSJOB_FLAGS="--mem-max 11G" remote_workers/run_remote.sh windows-5090-wsl \
    "RCF_EXTRA='--f64 --rounds 3 --iters 500' OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 nice -n 19 bash personalization/remote/run_rcf_fits.sh val_sb_$card:merged:$F,--val sb_$card:merged:$F"
  until ssh windows-5090-wsl "grep -q 'exit=' ~/hots/logs/rcf_card_$card.log"; do sleep 30; done
  ssh windows-5090-wsl "grep -E '^\[fit .*effective|FAILED|Traceback' ~/hots/logs/rcf_card_$card.log"
  personalization/remote/run_rcf_eval.sh merged sb_$card
done
echo "CARDS DONE"
