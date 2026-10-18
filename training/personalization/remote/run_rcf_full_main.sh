#!/bin/bash
# Step 1 of the tails round on the 3090: the combined main model on full_1017.
set -uo pipefail
cd ~/hots/repo/training
R="bash personalization/remote/run_rcf_full.sh"
P=/home/max/hots/repo/training/personalization/cache_full/rcf/parts_full_ms.npz
$R online --tag full_ms --hist merged --layer --parts $P --out full_ms_layer
$R online --tag full_ms --hist merged
$R cover --tag full_ms --hist merged
$R cover --tag full_ms --hist merged --layer-arm full_ms_layer --parts $P
$R cassia --tag full_ms --hist merged
$R online --tag full_imp2 --hist merged
$R cover --tag full_imp2 --hist merged
$R cassia --tag full_imp2 --hist merged
echo MAIN DONE
