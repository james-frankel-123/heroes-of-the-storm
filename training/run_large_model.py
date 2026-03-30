#!/usr/bin/env python3
"""
Large model experiment: 2 seeds with enriched WP + large policy net (30M params).
Runs after the overnight base seeds complete.
~55 min per run at ~90 ep/s. 2 runs on 2 GPUs = ~55 min total.
"""
import os
import sys
import subprocess
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKER_SCRIPT = os.path.join(SCRIPT_DIR, "train_mcts_worker.py")

ALL_RUNS = [
    {"name": "EL_seed0", "wp": "enriched", "net_size": "large"},
    {"name": "EL_seed1", "wp": "enriched", "net_size": "large"},
]

EPISODES = 300_000
NUM_SIMS = 200


def launch_run(run, gpu_id, log_dir="/tmp"):
    save_dir = os.path.join(SCRIPT_DIR, f"mcts_runs/{run['name']}")
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"mcts_{run['name']}.log")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["MCTS_SAVE_DIR"] = save_dir
    env["MCTS_WP_MODEL"] = run["wp"]
    env["MCTS_NUM_EPISODES"] = str(EPISODES)
    env["MCTS_NUM_SIMS"] = str(NUM_SIMS)
    env["MCTS_BATCH_EPISODES"] = "128"
    env["MCTS_FRESH"] = "1"
    env["WANDB_RUN_NAME"] = run["name"]
    env["MCTS_NET_SIZE"] = run.get("net_size", "base")

    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        [sys.executable, "-u", WORKER_SCRIPT],
        env=env, stdout=log_file, stderr=subprocess.STDOUT
    )
    return proc, log_file, log_path


def main():
    print(f"Large model experiment: {len(ALL_RUNS)} runs")
    procs = []
    for gpu_id, run in enumerate(ALL_RUNS):
        if gpu_id > 0:
            time.sleep(5)
        proc, log_file, log_path = launch_run(run, gpu_id)
        procs.append((run["name"], proc, log_file, log_path))
        print(f"  {run['name']} on GPU {gpu_id} (PID {proc.pid}, log: {log_path})")

    print("\nWaiting...")
    for name, proc, log_file, log_path in procs:
        proc.wait()
        log_file.close()
        with open(log_path) as f:
            lines = f.readlines()
            last = lines[-1].strip() if lines else "empty"
        status = "OK" if proc.returncode == 0 else f"FAIL (code {proc.returncode})"
        print(f"  {name}: {status} — {last}")

    print("Done.")


if __name__ == "__main__":
    main()
