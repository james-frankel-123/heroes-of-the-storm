#!/bin/bash
# Rebuild every P3 kernel on a worker (from ~/hots/repo/training): the four base
# kernels, cuda_pvnet and cuda_explore. CPU only; does not touch cuda_mcts.
set -euo pipefail
for t in all pvnet explore; do bash personalization/build_p3_kernels.sh $t; done
echo "P3 BUILD ALL OK"
