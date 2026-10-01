#!/bin/bash
# Fetch finished p1site baseline artifacts from a remote worker every 15 min
# (models, meta, results; never feature caches or GD resume states).
# Usage: nohup paper1_revision/site_fetch.sh max-windows-3090 &   (from training/)
HOST=${1:-max-windows-3090}
cd "$(dirname "$0")/.."
SRC=/home/max/hots/repo/training/rerun2026/ns/p1site
DST=rerun2026/ns/p1site
while true; do
  for d in models results; do
    # FETCH_EXCLUDE: space-separated patterns trained on another host (e.g. "generic_draft_0* gd_0.json")
    EX=(); for e in ${FETCH_EXCLUDE:-}; do EX+=(--exclude "$e"); done
    nice -n 19 rsync -rlt --rsync-path="wsl rsync" --exclude "*.resume.pt*" --exclude "feature_cache" "${EX[@]}" \
      "$HOST:$SRC/$d/" "$DST/$d/" 2>/dev/null
  done
  nice -n 19 rsync -rlt --rsync-path="wsl rsync" "$HOST:/home/max/hots/logs/p1site_*.log" paper1_revision/site/logs/remote/ 2>/dev/null
  # site-tier MCTS runs (policies, best checkpoints, meta; not the resume state) and their logs
  nice -n 19 rsync -rlt --rsync-path="wsl rsync" --exclude "resume_state.*" \
    "$HOST:/home/max/hots/repo/training/paper1_revision/site/mcts_runs/" paper1_revision/site/mcts_runs/ 2>/dev/null
  nice -n 19 rsync -rlt --rsync-path="wsl rsync" --include "mcts_*" --exclude "*" \
    "$HOST:/home/max/hots/repo/training/paper1_revision/site/logs/" paper1_revision/site/logs/remote/ 2>/dev/null
  sleep 900
done
