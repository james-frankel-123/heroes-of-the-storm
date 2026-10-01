#!/bin/bash
# Launch a detached, pause-aware job inside a worker's WSL.
# Usage: training/remote_workers/run_remote.sh <host> <command...>
#   RUN_NAME=<name> (default run_<timestamp>) names the job: manifest
#   ~/hots/jobs/<name>.json, log ~/hots/logs/<name>.log. cwd is
#   ~/hots/repo/training, ~/hots/env.sh is sourced, CUDA_VISIBLE_DEVICES=0.
#   Extra hotsjob.py launch flags via HOTSJOB_FLAGS (e.g. "--save-dir X --gpu 0").
# Example:
#   RUN_NAME=F_oof_s0 run_remote.sh max-windows-3090 python paper1_revision/train_mcts.py F 0 --gpu 0
set -euo pipefail
source "$(dirname "$0")/_remote.sh"
HOST=${1:?usage: run_remote.sh <host> <command...>}; shift
[ $# -gt 0 ] || { echo "no command given" >&2; exit 1; }
NAME=${RUN_NAME:-run_$(date +%Y%m%d_%H%M%S)}
# shellcheck disable=SC2086
hotsjob "$HOST" launch "$NAME" "$*" ${HOTSJOB_FLAGS:-}
