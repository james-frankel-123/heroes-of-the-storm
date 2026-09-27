#!/bin/bash
# Wave-3 self-driving chain (launch detached):
#   nohup bash drift2026/w3_chain.sh > drift2026/logs/w3_chain.log 2>&1 &
# Stages: (a) recovery -> (f) heterogeneity -> (c) never-fielded ->
#         (b) changepoints -> hybrid features -> (d)(e)(g) GPU phase ->
#         summaries. Each stage's outputs are idempotent; rerunning the
# chain skips finished work. Status: tail drift2026/logs/w3_chain.log
set -uo pipefail
cd /home/max/heroes-of-the-storm
set -a && source .env && set +a
cd training

PY=python3
LOG=drift2026/logs
mkdir -p "$LOG"

stage() {  # stage <name> <output-that-marks-done> <cmd...>
  local name=$1 marker=$2; shift 2
  if [ -e "$marker" ]; then
    echo "[w3_chain] SKIP $name (exists: $marker)"
    return 0
  fi
  echo "[w3_chain] ==== $name: $* ($(date)) ===="
  "$@" > "$LOG/$name.log" 2>&1
  local rc=$?
  echo "[w3_chain] ==== $name done rc=$rc ($(date)) ===="
  return $rc
}

# (a) recovery + daily cache (CPU) — everything downstream needs this
stage w3_recovery drift2026/results/w3_recovery.json \
  $PY drift2026/w3_recovery.py || exit 1

# (f) heterogeneity (CPU)
stage w3_heterogeneity drift2026/results/w3_heterogeneity.json \
  $PY drift2026/w3_heterogeneity.py || echo "[w3_chain] WARN: (f) failed"

# (c) never-fielded (CPU)
stage w3_neverfielded drift2026/results/w3_neverfielded.json \
  $PY drift2026/w3_neverfielded.py || echo "[w3_chain] WARN: (c) failed"

# (b) changepoints (CPU; validation auto-skips if ground truth missing)
stage w3_changepoints drift2026/results/w3_changepoints.json \
  $PY drift2026/w3_changepoints.py || echo "[w3_chain] WARN: (b) failed"

# (d) hybrid stats + feature pass (CPU, ~10 min)
stage w3_build_hybrid drift2026/feature_cache/features_hybrid.npz \
  $PY drift2026/w3_build_hybrid.py || exit 1

# (d)(e)(g) GPU phase — polite: 2 concurrent jobs while the diversity
# diagnostic shares the pool (its remaining trains end within ~2 h)
echo "[w3_chain] ==== phase_w3 ($(date)) ===="
$PY drift2026/phase_w3.py --num-gpus 2 > "$LOG/phase_w3.log" 2>&1
rc=$?
echo "[w3_chain] ==== phase_w3 done rc=$rc ($(date)) ===="
[ $rc -ne 0 ] && echo "[w3_chain] WARN: phase_w3 had failures (summaries will use what exists)"

# summaries (uses GPU briefly for the W3e eval-only check)
echo "[w3_chain] ==== summarize_w3 ($(date)) ===="
CUDA_VISIBLE_DEVICES=0 $PY drift2026/summarize_w3.py > "$LOG/summarize_w3.log" 2>&1
echo "[w3_chain] ==== summarize_w3 done rc=$? ($(date)) ===="

# re-run changepoint validation if the ground-truth file arrived after (b)
if [ -e drift2026/patch_notes_ground_truth.json ] && \
   ! grep -q "Validation vs official patch notes" drift2026/results/W3_CHANGEPOINTS.md 2>/dev/null; then
  echo "[w3_chain] ground truth present but unvalidated -> re-running (b)"
  $PY drift2026/w3_changepoints.py > "$LOG/w3_changepoints_reval.log" 2>&1
fi

echo "[w3_chain] ALL DONE ($(date))"
ls -la drift2026/results/W3_*.md 2>/dev/null
