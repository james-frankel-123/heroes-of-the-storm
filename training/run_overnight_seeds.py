#!/usr/bin/env python3
"""
Overnight run: 10 extra seeds per variant for learning curves + confidence intervals.

E (enriched, 256→128): seeds 5-14  = 10 runs
G (augmented, 512→256→128): seeds 5-14 = 10 runs

Total: 20 runs. 4 GPUs, 5s stagger → 5 rounds × ~18 min = ~1.5 hours.
"""
import os
import sys
import subprocess
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKER_SCRIPT = os.path.join(SCRIPT_DIR, "train_mcts_worker.py")

ALL_RUNS = []

# 10 more G seeds (5-14)
for s in range(5, 15):
    ALL_RUNS.append({"name": f"G_seed{s}", "wp": "augmented", "done": False})

# 10 more E seeds (5-14)
for s in range(5, 15):
    ALL_RUNS.append({"name": f"E_seed{s}", "wp": "enriched", "done": False})

# 2 large model seeds (enriched WP, large policy net)
ALL_RUNS.append({"name": "EL_seed0", "wp": "enriched", "done": False, "net_size": "large"})
ALL_RUNS.append({"name": "EL_seed1", "wp": "enriched", "done": False, "net_size": "large"})

NUM_GPUS = 4
EPISODES = 300_000
NUM_SIMS = 200


def launch_run(run, gpu_id, log_dir="/tmp"):
    save_dir = os.path.join(SCRIPT_DIR, f"mcts_runs/{run['name']}")
    os.makedirs(save_dir, exist_ok=True)
    sims = run.get("sims", NUM_SIMS)
    log_path = os.path.join(log_dir, f"mcts_{run['name']}.log")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["MCTS_SAVE_DIR"] = save_dir
    env["MCTS_WP_MODEL"] = run["wp"]
    env["MCTS_NUM_EPISODES"] = str(EPISODES)
    env["MCTS_NUM_SIMS"] = str(sims)
    env["MCTS_BATCH_EPISODES"] = "128"
    env["MCTS_FRESH"] = "1"
    env["WANDB_RUN_NAME"] = run["name"]
    if "net_size" in run:
        env["MCTS_NET_SIZE"] = run["net_size"]

    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        [sys.executable, "-u", WORKER_SCRIPT],
        env=env, stdout=log_file, stderr=subprocess.STDOUT
    )
    return proc, log_file, log_path


def main():
    remaining = [r for r in ALL_RUNS if not r["done"]]
    print(f"Total: {len(remaining)} runs")

    rounds = []
    for i in range(0, len(remaining), NUM_GPUS):
        rounds.append(remaining[i:i + NUM_GPUS])

    print(f"Rounds: {len(rounds)}")
    for i, batch in enumerate(rounds):
        print(f"  Round {i+1}: {', '.join(r['name'] for r in batch)}")
    print()

    for round_idx, batch in enumerate(rounds):
        print(f"\n{'='*60}")
        print(f"ROUND {round_idx+1}/{len(rounds)}")
        print(f"{'='*60}")

        procs = []
        for gpu_id, run in enumerate(batch):
            if gpu_id > 0:
                time.sleep(5)
            proc, log_file, log_path = launch_run(run, gpu_id)
            procs.append((run["name"], proc, log_file, log_path))
            print(f"  {run['name']} on GPU {gpu_id} (PID {proc.pid}, log: {log_path})")

        print(f"\nWaiting for round {round_idx+1}...")
        for name, proc, log_file, log_path in procs:
            proc.wait()
            log_file.close()
            with open(log_path) as f:
                lines = f.readlines()
                last = lines[-1].strip() if lines else "empty"
            status = "OK" if proc.returncode == 0 else f"FAIL (code {proc.returncode})"
            print(f"  {name}: {status} — {last}")

        print(f"Round {round_idx+1} complete.")

    print(f"\nAll {len(remaining)} runs complete.")


if __name__ == "__main__":
    main()
