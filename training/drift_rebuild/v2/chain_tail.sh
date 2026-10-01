#!/usr/bin/env bash
cd /home/max/heroes-of-the-storm/training
set -a; source /home/max/heroes-of-the-storm/.env; set +a
while ps -eo args | grep -q "^python3 -u drift_rebuild/run_capped.py drift_v2/jobs_vf_local.txt"; do sleep 60; done
python3 -u drift_rebuild/run_capped.py drift_v2/jobs_tail.txt 1
