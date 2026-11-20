#!/bin/bash
# Paper-1 robustness (2026-11-20): per-seed rescoring (log grade) then flat-grade rescoring, family a, CPU only.
set -eo pipefail
source ~/hots/env.sh
source ~/hots/td_env.sh
cd ~/hots/repo/training
export CUDA_VISIBLE_DEVICES=
R="env $P1ENV WORLDS_PROCS=8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 19 taskset -c 16-35 python -u worlds/rescore.py p1 --fams a"
WORLDS_P1_TAG=_seeds WORLDS_P1_SEEDS=1 $R
WORLDS_P1_TAG=_flat WORLDS_GRADE=flat WORLDS_P1_SEEDS=1 $R
echo P1_FLAT_SEEDS_DONE
