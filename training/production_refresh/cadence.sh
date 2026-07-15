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

python3 training/production_refresh/refresh.py all

# Gates passed (refresh.py export exits nonzero otherwise) — deploy.
git add public/models/draft_policy.onnx \
        public/models/generic_draft_0.onnx \
        public/models/win_probability.onnx \
        src/lib/data/draft-stats-decayed.json
git commit -m "Production refresh $DATE: decayed-aggregate retrain (standing cadence)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git push origin main
echo "=== deployed $DATE ==="
