#!/usr/bin/env python3
"""
Post-bugfix MCTS experiments. 4 seeds each, 6 experiments.

A: Partial WP, base backbone, 300K episodes (primary)
B: Full (enriched) WP, base backbone, 300K episodes (comparison)
C: Partial WP, large backbone (1536d, 6 res), 300K episodes
D: Partial WP, deep policy head (256→512→256→90), 300K episodes
E: Partial WP, base backbone, 1M episodes (longer training)
F: Partial WP, base backbone, 300K episodes, 400 sims (more search)

All use the fixed CUDA kernel with correct feature computation.
"""
import os, sys, subprocess, time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(SCRIPT_DIR, "train_mcts_worker.py")
NUM_GPUS = 4

ALL_RUNS = [
    # A: Partial WP, base backbone, 300K
    *[{"name": f"A_partial_s{s}", "wp": "enriched", "episodes": 300000,
       "head": "linear", "size": "base"} for s in range(4)],

    # B: Full enriched WP (not partial), base backbone, 300K
    *[{"name": f"B_fullwp_s{s}", "wp": "enriched_full", "episodes": 300000,
       "head": "linear", "size": "base"} for s in range(4)],

    # C: Partial WP, large backbone, 300K
    *[{"name": f"C_large_s{s}", "wp": "enriched", "episodes": 300000,
       "head": "linear", "size": "large"} for s in range(4)],

    # D: Partial WP, deep policy head, 300K
    *[{"name": f"D_deep_s{s}", "wp": "enriched", "episodes": 300000,
       "head": "deep", "size": "base"} for s in range(4)],

    # E: Partial WP, base backbone, 1M episodes
    *[{"name": f"E_1M_s{s}", "wp": "enriched", "episodes": 1000000,
       "head": "linear", "size": "base"} for s in range(4)],

    # F: Partial WP, base backbone, 300K episodes, 400 sims (more search)
    *[{"name": f"F_400sim_s{s}", "wp": "enriched", "episodes": 300000,
       "head": "linear", "size": "base", "sims": 400} for s in range(4)],
]


def launch(run, gpu_id):
    save_dir = os.path.join(SCRIPT_DIR, f"mcts_runs/{run['name']}")
    os.makedirs(save_dir, exist_ok=True)
    log = f"/tmp/mcts_{run['name']}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["MCTS_SAVE_DIR"] = save_dir
    env["MCTS_WP_MODEL"] = run["wp"]
    env["MCTS_NUM_EPISODES"] = str(run["episodes"])
    env["MCTS_NUM_SIMS"] = str(run.get("sims", 200))
    env["MCTS_BATCH_EPISODES"] = "128"
    env["MCTS_FRESH"] = "1"
    env["MCTS_POLICY_HEAD"] = run.get("head", "linear")
    env["MCTS_NET_SIZE"] = run.get("size", "base")
    env["WANDB_RUN_NAME"] = run["name"]
    lf = open(log, "w")
    p = subprocess.Popen([sys.executable, "-u", WORKER], env=env, stdout=lf, stderr=subprocess.STDOUT)
    return p, lf, log


def main():
    rounds = [ALL_RUNS[i:i + NUM_GPUS] for i in range(0, len(ALL_RUNS), NUM_GPUS)]
    print(f"Post-bugfix experiments: {len(ALL_RUNS)} runs, {len(rounds)} rounds")
    print(f"  A: Partial WP, base (4 seeds)")
    print(f"  B: Full enriched WP, base (4 seeds)")
    print(f"  C: Partial WP, large backbone (4 seeds)")
    print(f"  D: Partial WP, deep head (4 seeds)")
    print(f"  E: Partial WP, 1M episodes (4 seeds)")
    print(f"  F: Partial WP, 400 sims (4 seeds)")
    print()

    for ri, batch in enumerate(rounds):
        print(f"\n{'=' * 60}")
        print(f"ROUND {ri + 1}/{len(rounds)}")
        print(f"{'=' * 60}")
        procs = []
        for gi, run in enumerate(batch):
            if gi > 0:
                time.sleep(5)
            p, lf, log = launch(run, gi)
            procs.append((run["name"], p, lf, log))
            print(f"  {run['name']} on GPU {gi} (PID {p.pid})")
        print(f"\nWaiting...")
        for name, p, lf, log in procs:
            p.wait()
            lf.close()
            with open(log) as f:
                lines = f.readlines()
                last = lines[-1].strip() if lines else "empty"
            status = "OK" if p.returncode == 0 else f"FAIL ({p.returncode})"
            print(f"  {name}: {status} — {last}")

    print(f"\nAll {len(ALL_RUNS)} runs complete.")


if __name__ == "__main__":
    main()
