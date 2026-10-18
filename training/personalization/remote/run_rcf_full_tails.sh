#!/bin/bash
# Step 2 of the tails round on the 3090: tail cards on the main model (walk
# parts cached by run_rcf_full_main.sh), then the p3_c_tests report with
# R1 calibration by record against A2S.
set -uo pipefail
cd ~/hots/repo/training
R="bash personalization/remote/run_rcf_full.sh"
P=/home/max/hots/repo/training/personalization/cache_full/rcf/parts_full_ms.npz
B=/home/max/hots/repo/training/personalization/results
for card in "$@"; do
  [ -f $B/rcf_full/rcf/p3c/full_ms_tail_${card}__merged.npz ] || \
    $R online --tag full_ms --hist merged --layer --parts $P --tail $card --out full_ms_tail_$card
done
mkdir -p $B/rcf_full/p3c
ln -sf $B/oct26_p3c/p3c/A2S.npz $B/rcf_full/p3c/A2S.npz
for f in $B/rcf_full/rcf/p3c/*__merged.npz; do n=$(basename $f __merged.npz); ln -sf $f $B/rcf_full/p3c/$n.npz; done
echo TAILS DONE
