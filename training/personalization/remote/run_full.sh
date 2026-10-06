#!/bin/bash
# Full-history rerun on a worker (from ~/hots/repo/training): parse the
# post-backfill export, build cache_full, run the "full" chain.
# Usage: bash personalization/remote/run_full.sh <export tag>
set -euo pipefail
T=${1:?export tag}
python personalization/p3_hero_level_causal.py parse --tag "$T"
python personalization/p3_c_history.py --tag "$T" --out personalization/cache_full
P3_CACHE=$HOME/hots/repo/training/personalization/cache_full \
P3_RESULTS=$HOME/hots/repo/training/personalization/results/oct26_full \
P3_EXPORT_TAG="$T" python personalization/p3_pipe.py full
