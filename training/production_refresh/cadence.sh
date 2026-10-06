#!/usr/bin/env bash
# Monthly production model refresh (drift-paper recommendation in production).
#
# Runs the full refresh pipeline; on success (all deploy gates pass inside
# refresh.py export) commits the new models + stats artifact and pushes,
# which deploys via Vercel. On any failure the run directory keeps its logs
# and nothing is committed.
#
# Install (1st of each month, 04:00 local — quota-quiet hours):
#   crontab -e
#   0 4 1 * * /home/max/heroes-of-the-storm/training/production_refresh/cadence.sh >> /home/max/heroes-of-the-storm/training/production_refresh/cron.log 2>&1
set -euo pipefail

REPO=/home/max/heroes-of-the-storm
cd "$REPO"
set -a && source .env && set +a

DATE=$(date +%F)
echo "=== production refresh $DATE ==="

# Shared box: HotS work gets ~25% (one GPU, 16 cores, lowest priority).
REFRESH_GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t, -k2 -nr | head -1 | cut -d, -f1)
export REFRESH_GPU CUDA_VISIBLE_DEVICES=$REFRESH_GPU
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
echo "refresh pinned to GPU $REFRESH_GPU, cores 48-63, nice 19"
PY=/home/linuxbrew/.linuxbrew/bin/python3
fail() { echo "=== REFRESH FAILED $DATE: $1; NOTHING DEPLOYED (see training/production_refresh/$DATE/) ==="; exit 1; }

# Production MCTS kernel (91 heroes, 15 maps): rebuilt only when its sources
# changed, CPU only (GPU hidden; sm_120 = this box's GPUs), nice 19, same 16
# cores. refresh.py's ensure_kernel then finds it up to date, and phase
# kparity gates it against the Python WP (exits nonzero on mismatch).
CUDA_VISIBLE_DEVICES= TORCH_CUDA_ARCH_LIST=${H91_ARCH:-12.0} nice -n 19 taskset -c 48-63 \
    $PY training/cuda_mcts/build_h91.py --if-stale || fail "h91 kernel build"

# Production refits the gated WP and partial-WP on all data after the gates
# pass and deploys the refits (refresh.py phase refit, sanity-gated).
export REFRESH_REFIT=1

# Any failed phase or gate (kernel parity included) exits nonzero: no deploy.
nice -n 19 taskset -c 48-63 $PY training/production_refresh/refresh.py all \
    || fail "refresh.py all exited nonzero (a phase or a deploy gate failed)"

# Gates passed (refresh.py export exits nonzero otherwise) — deploy.
# Commit from a clean temporary worktree of origin/main: this working tree is
# shared with other jobs (uncommitted research edits, other sessions pushing
# to main), so never rebase or commit here.
# model-compositions.json: the models' comp_wr table (this run's export).
# compositions.json: the display panel's Heroes Profile table, as the nightly
# sync last wrote it (sync/sync-compositions.ts); deployed to keep it fresh.
DEPLOY_FILES="public/models/draft_policy.onnx public/models/generic_draft_0.onnx
public/models/win_probability.onnx public/models/partial_wp.onnx
src/lib/data/draft-stats-decayed.json src/lib/data/model-compositions.json
src/lib/data/compositions.json"
DEPLOY_DIR=$(mktemp -d /tmp/hots-deploy-XXXXXX)
trap 'git -C "$REPO" worktree remove --force "$DEPLOY_DIR" 2>/dev/null || true' EXIT
git fetch origin main
git worktree add --detach "$DEPLOY_DIR" origin/main
for f in $DEPLOY_FILES; do cp "$REPO/$f" "$DEPLOY_DIR/$f"; done
cd "$DEPLOY_DIR"
git add $DEPLOY_FILES
git commit -m "Production refresh $DATE: decayed-aggregate retrain (standing cadence)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
for attempt in 1 2 3; do
    git push origin HEAD:main && break
    echo "push rejected (attempt $attempt); rebasing onto the new origin/main"
    git fetch origin main && git rebase origin/main
done
git fetch origin main
git merge-base --is-ancestor HEAD origin/main \
    || { echo "DEPLOY FAILED: push never landed on origin/main"; exit 1; }
cd "$REPO"
echo "=== deployed $DATE ==="
