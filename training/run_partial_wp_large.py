#!/usr/bin/env python3
"""
Large backbone MCTS with partial WP model.
30M param policy (1536-dim, 6 res blocks).
4 seeds on 4 GPUs.
"""
import os, sys, subprocess, time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(SCRIPT_DIR, "train_mcts_worker.py")

ALL_RUNS = [{"name": f"PL_seed{s}"} for s in range(4)]
EPISODES = 300_000

def main():
    print(f"Large backbone + partial WP: {len(ALL_RUNS)} runs")
    procs = []
    for gi, run in enumerate(ALL_RUNS):
        save_dir = os.path.join(SCRIPT_DIR, f"mcts_runs/{run['name']}")
        os.makedirs(save_dir, exist_ok=True)
        log = f"/tmp/mcts_{run['name']}.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gi)
        env["MCTS_SAVE_DIR"] = save_dir
        env["MCTS_WP_MODEL"] = "enriched"
        env["MCTS_NET_SIZE"] = "large"
        env["MCTS_NUM_EPISODES"] = str(EPISODES)
        env["MCTS_NUM_SIMS"] = "200"
        env["MCTS_BATCH_EPISODES"] = "128"
        env["MCTS_FRESH"] = "1"
        env["WANDB_RUN_NAME"] = run["name"]
        lf = open(log, "w")
        p = subprocess.Popen([sys.executable, "-u", WORKER], env=env, stdout=lf, stderr=subprocess.STDOUT)
        procs.append((run["name"], p, lf, log))
        print(f"  {run['name']} on GPU {gi} (PID {p.pid})")
        if gi < len(ALL_RUNS) - 1: time.sleep(5)

    print("\nWaiting...")
    for name, p, lf, log in procs:
        p.wait(); lf.close()
        with open(log) as f: lines = f.readlines(); last = lines[-1].strip() if lines else "empty"
        print(f"  {name}: {'OK' if p.returncode==0 else 'FAIL'} — {last}")
    print("Done.")

if __name__ == "__main__":
    main()
