#!/bin/bash
# Print (repo-root-relative) the data files an MCTS training run and the
# paper1_revision benchmark/tournament scripts read. Feed to rsync --files-from.
# Usage: training/remote_workers/data_manifest.sh [--no-snapshot]
set -euo pipefail
cd "$(dirname "$0")/../.."
T=training
{
  # value pretraining (shared.load_replay_data, REPLAY_SNAPSHOT=1): pinned snapshot, 2.9 GB
  [ "${1:-}" = "--no-snapshot" ] || echo $T/snapshots/replay_snapshot_2026-05-22_1956753.json
  # default WP/GD checkpoints the worker can fall back to, frozen stats, HP compositions
  ls $T/*.pt
  echo $T/frozen_stats_2026-05-19.json
  echo src/lib/data/compositions.json
  # paper1_revision: leak-free leaf WPs + own deploy/oof stats + exclude lists
  find $T/paper1_revision/models -type f
  find $T/paper1_revision/cache/stats $T/paper1_revision/cache/stats_site $T/paper1_revision/cache/stats_stated -type f
  find $T/paper1_revision/cache -maxdepth 1 -type f \( -name "*.json" -o -name "*.npz" \)
  find $T/paper1_revision/mcts_runs -name draft_policy.pt -o -name "run_meta*.json"
  # rerun2026: GD opponents, submission WPs, exclude ids, reused tournament records,
  # and the submission MCTS policies the bench/tournament compare against
  find $T/rerun2026/models -type f -size -100M
  find $T/rerun2026 -maxdepth 1 -type f -name "*.json"
  find $T/rerun2026/results -type f -size -20M
  ls $T/rerun2026/mcts_runs/{K_truebase_s*,F_400sim_s*,J_800sim_s0}/draft_policy.pt
  # overfit2026: judges (gN), slim corpora, realized indices; qm2026 judges
  find $T/overfit2026/models -type f -size -100M
  find $T/overfit2026/cache -maxdepth 1 -type f -size -100M
  find $T/overfit2026/results -type f -size -20M
  find $T/qm2026/results -type f -size -20M
  # drift paper v2 rebuild (opt-in: DRIFT_V2_DATA=1): site-tiered stats and
  # feature caches, patch index/sidecar
  if [ "${DRIFT_V2_DATA:-}" = 1 ]; then
    echo $T/drift2026/patch_index.json
    echo $T/drift2026/patch_sidecar.npz
    echo $T/drift_v2/w9_mmr_sidecar.npz
    echo $T/drift2026/w4_exclude_ids.json
    ls $T/drift_rebuild/models/exclude_after_*.json
    find $T/drift_v2/patch_stats -type f
    # value-function feature caches (63 GB) only with DRIFT_V2_FEATURES=1;
    # opponent pools and MCTS do not read them
    [ "${DRIFT_V2_FEATURES:-}" = 1 ] && find $T/drift_v2/feature_cache -maxdepth 1 -name "features_*.npz"
    find $T/drift_v2/models -name "*.pt" 2>/dev/null
  fi
} | grep -v "/__pycache__/" | sort -u
