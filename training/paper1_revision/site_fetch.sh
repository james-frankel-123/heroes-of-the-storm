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
    nice -n 19 rsync -rlt --rsync-path="wsl rsync" --exclude "*.resume.pt*" --exclude "feature_cache" \
      "$HOST:$SRC/$d/" "$DST/$d/" 2>/dev/null
  done
  nice -n 19 rsync -rlt --rsync-path="wsl rsync" "$HOST:/home/max/hots/logs/p1site_*.log" paper1_revision/site/logs/remote/ 2>/dev/null
  sleep 900
done
