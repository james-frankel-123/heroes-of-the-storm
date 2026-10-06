#!/bin/bash
# Repo-root-relative data files the P3 (personalization) jobs read on a remote
# worker. Feed to rsync --files-from (see p3_sync.sh). No DB credentials go
# to the remotes: everything here is a frozen file.
set -euo pipefail
HERE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE_DIR/../../.."
T=training
P=$T/personalization
{
  # P3 caches (inputs and intermediate tables); not the HP history shards,
  # the legacy fix/ tree, partial files or the remote-built outputs
  find $P/cache -maxdepth 1 -type f ! -name "*.tmp" ! -name "*.tmp.npz"
  find $P/cache/export -type f ! -name "*.tmp" 2>/dev/null || true
  # population WP (d2c_cumprev seeds), cumulative patch stats, patch index
  find $T/drift2026/models -maxdepth 1 -name "d2c_cumprev_s*.pt"
  find $T/drift2026/patch_stats -type f
  echo $T/drift2026/patch_index.json
  echo $T/drift2026/patch_sidecar.npz
  # paper-1 GD pool and the BC prior (legacy in-sample models, kept for the A/B)
  ls $T/rerun2026/models/generic_draft_*.pt
  echo $T/overfit2026/models/bc_prior.pt
} | grep -v "/__pycache__/" | grep -v -E -f "$HERE_DIR/derived_caches.txt" | sort -u
