#!/usr/bin/env python3
"""
15 seeds with partial WP model, enriched features, 300K episodes.
4 GPUs, staggered launches.
"""
import os, sys, subprocess, time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(SCRIPT_DIR, "train_mcts_worker.py")

ALL_RUNS = [{"name": f"P_seed{s}"} for s in range(15)]
NUM_GPUS = 4
EPISODES = 300_000

def launch(run, gpu_id):
    save_dir = os.path.join(SCRIPT_DIR, f"mcts_runs/{run['name']}")
    os.makedirs(save_dir, exist_ok=True)
    log = f"/tmp/mcts_{run['name']}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["MCTS_SAVE_DIR"] = save_dir
    env["MCTS_WP_MODEL"] = "enriched"  # partial WP loaded by default now
    env["MCTS_NUM_EPISODES"] = str(EPISODES)
    env["MCTS_NUM_SIMS"] = "200"
    env["MCTS_BATCH_EPISODES"] = "128"
    env["MCTS_FRESH"] = "1"
    env["WANDB_RUN_NAME"] = run["name"]
    lf = open(log, "w")
    p = subprocess.Popen([sys.executable, "-u", WORKER], env=env, stdout=lf, stderr=subprocess.STDOUT)
    return p, lf, log

def main():
    rounds = [ALL_RUNS[i:i+NUM_GPUS] for i in range(0, len(ALL_RUNS), NUM_GPUS)]
    print(f"Partial WP MCTS: {len(ALL_RUNS)} runs, {len(rounds)} rounds")
    for ri, batch in enumerate(rounds):
        print(f"\n{'='*60}\nROUND {ri+1}/{len(rounds)}\n{'='*60}")
        procs = []
        for gi, run in enumerate(batch):
            if gi > 0: time.sleep(5)
            p, lf, log = launch(run, gi)
            procs.append((run["name"], p, lf, log))
            print(f"  {run['name']} on GPU {gi} (PID {p.pid})")
        print(f"\nWaiting...")
        for name, p, lf, log in procs:
            p.wait(); lf.close()
            with open(log) as f: lines = f.readlines(); last = lines[-1].strip() if lines else "empty"
            print(f"  {name}: {'OK' if p.returncode==0 else 'FAIL'} — {last}")
    print(f"\nAll {len(ALL_RUNS)} runs complete.")

if __name__ == "__main__":
    main()
