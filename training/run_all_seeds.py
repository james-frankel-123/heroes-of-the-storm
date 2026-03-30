#!/usr/bin/env python3
"""
Run multiple MCTS training seeds across 4 GPUs.

5 seeds per variant × 3 variants = 15 runs total.
4 GPUs → 4 rounds → ~7 hours.

All runs use the new WP-in-kernel CUDA MCTS (enriched WP model with
lookup tables computed on-device). Variant A (base WP, 197d) is dropped
because the kernel requires enriched features.

Priority: G (augmented) first, then E (enriched), then G400 (more sims).
"""
import os
import sys
import subprocess
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKER_SCRIPT = os.path.join(SCRIPT_DIR, "train_mcts_worker.py")

# All runs to execute — ordered by priority (G augmented first).
ALL_RUNS = [
    # ── Variant G: augmented WP (283d, 512→256→128, synthetic) ──
    {"name": "G_seed0", "wp": "augmented", "gpu": None, "done": True},
    {"name": "G_seed1", "wp": "augmented", "gpu": None, "done": False},  # round 1 crash
    {"name": "G_seed2", "wp": "augmented", "gpu": None, "done": False},  # round 1 crash
    {"name": "G_seed3", "wp": "augmented", "gpu": None, "done": True},
    {"name": "G_seed4", "wp": "augmented", "gpu": None, "done": True},

    # ── Variant E: enriched WP (283d, 256→128, no augmentation) ──
    {"name": "E_seed0", "wp": "enriched",  "gpu": None, "done": True},
    {"name": "E_seed1", "wp": "enriched",  "gpu": None, "done": True},
    {"name": "E_seed2", "wp": "enriched",  "gpu": None, "done": True},
    {"name": "E_seed3", "wp": "enriched",  "gpu": None, "done": True},
    {"name": "E_seed4", "wp": "enriched",  "gpu": None, "done": True},

    # ── Variant G400: augmented WP, 400 MCTS sims ──
    {"name": "G400_seed0", "wp": "augmented", "gpu": None, "done": False, "sims": 400},
    {"name": "G400_seed1", "wp": "augmented", "gpu": None, "done": False, "sims": 400},
    {"name": "G400_seed2", "wp": "augmented", "gpu": None, "done": False, "sims": 400},
    {"name": "G400_seed3", "wp": "augmented", "gpu": None, "done": False, "sims": 400},
    {"name": "G400_seed4", "wp": "augmented", "gpu": None, "done": False, "sims": 400},
]

NUM_GPUS = 4
EPISODES = 300_000
DEFAULT_SIMS = 200


def launch_run(run, gpu_id, log_dir="/tmp"):
    """Launch one training run as a subprocess."""
    save_dir = os.path.join(SCRIPT_DIR, f"mcts_runs/{run['name']}")
    os.makedirs(save_dir, exist_ok=True)

    sims = run.get("sims", DEFAULT_SIMS)
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

    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        [sys.executable, "-u", WORKER_SCRIPT],
        env=env, stdout=log_file, stderr=subprocess.STDOUT
    )
    return proc, log_file, log_path


def main():
    # Filter to remaining runs
    remaining = [r for r in ALL_RUNS if not r["done"]]
    print(f"Total runs: {len(ALL_RUNS)}")
    print(f"Already done: {sum(1 for r in ALL_RUNS if r['done'])}")
    print(f"Remaining: {len(remaining)}")
    print()

    # Group into rounds of NUM_GPUS
    rounds = []
    for i in range(0, len(remaining), NUM_GPUS):
        rounds.append(remaining[i:i + NUM_GPUS])

    print(f"Rounds: {len(rounds)}")
    for i, batch in enumerate(rounds):
        names = [r["name"] for r in batch]
        print(f"  Round {i+1}: {', '.join(names)}")
    print()

    # Execute rounds sequentially
    for round_idx, batch in enumerate(rounds):
        print(f"\n{'='*60}")
        print(f"ROUND {round_idx+1}/{len(rounds)}")
        print(f"{'='*60}")

        procs = []
        for gpu_id, run in enumerate(batch):
            if gpu_id > 0:
                time.sleep(5)  # stagger launches to avoid CUDA driver contention
            proc, log_file, log_path = launch_run(run, gpu_id)
            procs.append((run["name"], proc, log_file, log_path))
            print(f"  {run['name']} on GPU {gpu_id} (PID {proc.pid}, log: {log_path})")

        # Wait for all in this round to finish
        print(f"\nWaiting for round {round_idx+1}...")
        for name, proc, log_file, log_path in procs:
            proc.wait()
            log_file.close()
            # Print final line of log
            with open(log_path) as f:
                lines = f.readlines()
                last = lines[-1].strip() if lines else "empty"
            status = "OK" if proc.returncode == 0 else f"FAIL (code {proc.returncode})"
            print(f"  {name}: {status} — {last}")

        print(f"Round {round_idx+1} complete.")

    print(f"\nAll {len(remaining)} runs complete.")


if __name__ == "__main__":
    main()
