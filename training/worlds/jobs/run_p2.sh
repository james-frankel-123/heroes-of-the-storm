#!/bin/bash
# Paper-2 env (as personalization/remote/p3t_promote_pcalf_chain.sh) for worlds scripts on the 3090 GPU.
set -eo pipefail
cd /home/max/hots/repo/training
source ~/hots/env.sh
B=/home/max/hots/repo/training/personalization
CROOT=$B/cache_def_def1027_hlsup
RROOT=$B/results/def_def1027_hlsup
export P3_CACHE=$CROOT P3_RESULTS=$RROOT P3T_VINTAGES=$RROOT/p3t_vintages.json
export P3_RCF_ROWS=$CROOT/merged_rows_def1027_2695e2cac4d17884.npz P3_RCF_KERNELS=$CROOT/hs_kernels.npz
export P3T_PROCS=12 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMBA_NUM_THREADS=4 PYTHONUNBUFFERED=1
T=/home/max/hots/repo/training
export P1_TIERS=site P1R_NPROC=3 WP_STATS_PATH=$T/paper1_revision/site/cache/stats/deploy.json P1R_COMP_PATH=$T/paper1_revision/site/cache/stats/deploy_compositions.json PYTHONPATH=$T/paper1_revision/oct2026_site
export CUDA_VISIBLE_DEVICES=0
exec python -u "worlds/$@"
