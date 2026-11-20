#!/bin/bash
# One command: per-seed worlds rescoring and seed-level contrasts (family a) from whatever seed_tour drafts
# exist, log and flat grade. Outputs worlds/site/results/p1_seeds8.json and p1_seeds8_flat.json.
# Usage on the 3090 (from anywhere): bash ~/hots/repo/training/worlds/jobs/seeds8_analyze.sh
set -eo pipefail
source ~/hots/env.sh
source ~/hots/td_env.sh
cd ~/hots/repo/training
for g in log flat; do
  env $P1ENV WORLDS_GRADE=$g WORLDS_PROCS=${WORLDS_PROCS:-8} OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    nice -n 19 taskset -c 16-35 python -u worlds/jobs/p1_seeds8.py --n ${SEEDS8_WORLDS:-1000}
done
