#!/usr/bin/env bash
# Launch one v2 MCTS agent on a remote worker through the pause-aware job
# manager. Usage: launch_mcts_remote.sh <host> <agent name, e.g. v2_M_s0>
set -euo pipefail
HOST=$1; NAME=$2
cd "$(dirname "$0")/../../.."
CMD=$(python3 - "$NAME" <<'PY'
import json, sys
j = {x["name"]: x for x in json.load(open("training/drift_v2/mcts_jobs_remote.json"))}[sys.argv[1]]
env = {k: v for k, v in j["env"].items() if k != "MCTS_CKPT_EVERY_SEC"}
print(" ".join(f"{k}={v}" for k, v in env.items()) + " " + j["cmd"].replace("python3 ", "python ")
      + f" && touch {j['env']['MCTS_SAVE_DIR']}/DONE")
PY
)
RUN_NAME=${RUN_NAME:-$NAME} training/remote_workers/run_remote.sh "$HOST" "$CMD"
