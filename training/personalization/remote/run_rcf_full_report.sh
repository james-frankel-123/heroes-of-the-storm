#!/bin/bash
# p3_c_tests report (subset lifts, R1 calibration by record) for rich-CF arms on full_1017.
# Usage: run_rcf_full_report.sh <name> <arm,arm,...>   (reference A2S)
set -uo pipefail
cd ~/hots/repo/training
B=/home/max/hots/repo/training/personalization
mkdir -p $B/results/rcf_full/p3c
ln -sf $B/results/oct26_p3c/p3c/A2S.npz $B/results/rcf_full/p3c/A2S.npz
for f in $B/results/rcf_full/rcf/p3c/*__merged.npz; do n=$(basename $f __merged.npz); ln -sf $f $B/results/rcf_full/p3c/$n.npz; done
P3_CACHE=$B/cache_full P3_RESULTS=$B/results/rcf_full python -u personalization/p3_c_tests.py report --arms "$2" --ref A2S --name "$1"
