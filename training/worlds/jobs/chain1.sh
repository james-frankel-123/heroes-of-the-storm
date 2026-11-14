#!/bin/bash
# Milestone-1 chain on the 3090 (resumable: every step skips finished outputs).
set -eo pipefail
R="bash worlds/jobs/run.sh"
cd ~/hots/repo/training
WORLDS_THREADS=24 $R fit.py base --procs 8
WORLDS_THREADS=12 $R fit.py layers
[ -f worlds/site/results/drift.json ] || WORLDS_THREADS=12 $R drift.py
[ -f worlds/dials.json ] || WORLDS_THREADS=4 $R families.py freeze
echo CHAIN1_DONE
