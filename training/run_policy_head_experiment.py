#!/usr/bin/env python3
"""
Policy Head Capacity Experiment.

Run A: deep head (256→512→256→90), no step embedding
Run B: step head (272→256→128→90), with step embedding
Run C: deep_step head (272→512→256→90), with step embedding

15 seeds each, enriched WP, 200 sims, 300K episodes.
4 GPUs, staggered launches.
"""
import os, sys, subprocess, time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKER_SCRIPT = os.path.join(SCRIPT_DIR, "train_mcts_worker.py")

ALL_RUNS = []

# Run A: deep head, 15 seeds
for s in range(15):
    ALL_RUNS.append({"name": f"A_deep_s{s}", "wp": "enriched", "head": "deep"})

# Run B: step head, 15 seeds
for s in range(15):
    ALL_RUNS.append({"name": f"B_step_s{s}", "wp": "enriched", "head": "step"})

# Run C: deep_step head, 15 seeds
for s in range(15):
    ALL_RUNS.append({"name": f"C_dstep_s{s}", "wp": "enriched", "head": "deep_step"})

NUM_GPUS = 4
EPISODES = 300_000
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
    env["MCTS_POLICY_HEAD"] = run["head"]
    env["WANDB_RUN_NAME"] = run["name"]

    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        [sys.executable, "-u", WORKER_SCRIPT],
        env=env, stdout=log_file, stderr=subprocess.STDOUT
    )
    return proc, log_file, log_path


def main():
    rounds = [ALL_RUNS[i:i + NUM_GPUS] for i in range(0, len(ALL_RUNS), NUM_GPUS)]

    print(f"Policy Head Experiment: {len(ALL_RUNS)} runs, {len(rounds)} rounds")
    print(f"  A (deep):      15 seeds, head=deep")
    print(f"  B (step):      15 seeds, head=step")
    print(f"  C (deep_step): 15 seeds, head=deep_step")
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
            print(f"  {run['name']} on GPU {gpu_id} (PID {proc.pid})")

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

    print(f"\nAll {len(ALL_RUNS)} runs complete.")


if __name__ == "__main__":
    main()
