#!/bin/bash
# Paper-1 MCQ runs on a worker: queue A or B (see mcq_real.py). From training/.
set -euo pipefail
T=$(cd "$(dirname "$0")/.." && pwd)
export P1_TIERS=site RERUN_NS=p1site RERUN_SPLIT=p1val RERUN_SLOW_GPU=-1 REPLAY_SNAPSHOT=1 PYTHON_CPU_COUNT=6
export REPLAY_SNAPSHOT_PATH=$T/snapshots/replay_snapshot_2026-05-22_1956753_p1site.json
export WP_STATS_PATH=$T/paper1_revision/site/cache/stats/deploy.json
export P1R_COMP_PATH=$T/paper1_revision/site/cache/stats/deploy_compositions.json
export PYTHONPATH=$T/paper1_revision/oct2026_site WANDB_MODE=disabled PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
M=$T/rerun2026/ns/p1site/models
case "${1:?A|B}" in
  A) for l in 0.7 0.9; do [ -f "$M/meta/mcq_real_l$l.json" ] || python paper1_revision/mcq_real.py train --lam $l; done ;;
  B) [ -f "$M/meta/mcq_real_l0.5.json" ] || python paper1_revision/mcq_real.py train --lam 0.5
     [ -f "$M/cql/_cql_temp_a0.0.pt" ] && [ -f "$M/meta/cql_naive_a0.0.json" ] || python rerun2026/train_jobs.py cql_naive --alpha 0.0 ;;
esac
