#!/bin/bash
# Personalized drafter with the merged (A2) skill state (from ~/hots/repo/training):
# build the linked cache with merged drafter tables, run the drafter and its analysis.
# Usage: bash personalization/remote/run_p3c_drafter.sh <base cache> <store history.npz> <out cache> <results dir> [procs]
set -euo pipefail
BASE=${1:?}; STORE=${2:?}; OUT=${3:?}; RES=${4:?}; PROCS=${5:-4}
mkdir -p "$RES"
cp -n personalization/results/oct26/p3_ph_role.json "$RES/"
P3_CACHE=$BASE python personalization/p3_c_merged.py drtables --store "$STORE" --out "$OUT"
P3_CACHE=$OUT P3_RESULTS=$RES python personalization/p3_dr_drafter.py --procs "$PROCS" --assign team
P3_CACHE=$OUT P3_RESULTS=$RES python personalization/p3_dr_analyze.py
