#!/bin/bash
# Build the CUDA MCTS extensions in place (nvcc only; needs no GPU).
# Run inside WSL on a worker: bash ~/hots/repo/training/remote_workers/build_ext.sh
# Rebuild after every code sync that touches training/cuda_mcts or overfit2026/cuda_ofit.
# Paper-1 site-tier MCTS (paper1_revision/train_mcts.py, P1_TIERS=site) needs the
# composition fallback in the Python WP's order: HOTS_COMP_FALLBACK_MID_HIGH_LOW=1 bash build_ext.sh
# (train_mcts.py refuses a build without it; check: the module's COMP_FALLBACK_ORDER).
set -euo pipefail
source ~/hots/env.sh
# nvcc needs no GPU, but torch probes cuInit; CUDA_VISIBLE_DEVICES="" crashes the
# 3080 driver's WSL libcuda (double free), so keep the GPU visible.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
T=~/hots/repo/training
for d in "$T/cuda_mcts" "$T/overfit2026/cuda_ofit"; do
  echo "== building $d"
  cd "$d"
  rm -rf build ./*.so
  nice -n 10 python setup.py build_ext --inplace > build_remote.log 2>&1 || { tail -40 build_remote.log; exit 1; }
  ls -la ./*.so
done
echo BUILD OK
