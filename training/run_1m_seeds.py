#!/usr/bin/env python3
"""
15 seeds × 1M episodes with enriched WP, 200 sims.
Tests whether longer training improves draft quality metrics.
4 GPUs, ~55 min per run at ~300 ep/s → 4 rounds × 55 min = ~3.7 hours.
"""
import os, sys, subprocess, time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKER_SCRIPT = os.path.join(SCRIPT_DIR, "train_mcts_worker.py")

ALL_RUNS = [{"name": f"E1M_seed{s}", "wp": "enriched"} for s in range(15)]

NUM_GPUS = 4
EPISODES = 1_000_000
NUM_SIMS = 200


def launch_run(run, gpu_id):
    save_dir = os.path.join(SCRIPT_DIR, f"mcts_runs/{run['name']}")
    os.makedirs(save_dir, exist_ok=True)
    log_path = f"/tmp/mcts_{run['name']}.log"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["MCTS_SAVE_DIR"] = save_dir
    env["MCTS_WP_MODEL"] = run["wp"]
    env["MCTS_NUM_EPISODES"] = str(EPISODES)
    env["MCTS_NUM_SIMS"] = str(NUM_SIMS)
    env["MCTS_BATCH_EPISODES"] = "128"
    env["MCTS_FRESH"] = "1"
    env["WANDB_RUN_NAME"] = run["name"]

    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        [sys.executable, "-u", WORKER_SCRIPT],
        env=env, stdout=log_file, stderr=subprocess.STDOUT
    )
    return proc, log_file, log_path


def main():
    remaining = ALL_RUNS
    rounds = [remaining[i:i + NUM_GPUS] for i in range(0, len(remaining), NUM_GPUS)]

    print(f"1M episode training: {len(remaining)} runs, {len(rounds)} rounds")
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
