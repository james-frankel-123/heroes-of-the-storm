"""
Pool runner for tournament.py pairs (CPU processes). Runs every pending pair
whose dependencies exist (MCTS checkpoints for mcts/constrained_mcts, the
augmented model for enriched_aug). Re-run to pick up the rest.

Usage: python3 paper1_revision/run_tournament.py [--procs 12]
"""
import os
import sys
import json
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)
from paper1_revision import core
from paper1_revision.tournament import todo, MCTS_CONFIG, MCTS_SEEDS, K_CONFIG, K_SEEDS


def ready(name):
    """A pooled MCTS agent is ready when every seed's final policy is present
    (runs train on the remote workers; only the checkpoints come back)."""
    runs = {"mcts": (MCTS_CONFIG, MCTS_SEEDS), "constrained_mcts": (MCTS_CONFIG, MCTS_SEEDS),
            "k_oof": (K_CONFIG, K_SEEDS)}
    if name in runs:
        cfg, seeds = runs[name]
        return all(os.path.exists(os.path.join(core.MCTS_RUNS, f"{cfg}_s{s}", "draft_policy.pt"))
                   for s in seeds)
    if name == "enriched_aug":
        return os.path.exists(os.path.join(core.MODEL_DIR, "aug_wr10_512.json"))
    return True


def run(pair):
    log = open(os.path.join(core.LOGS, f"tour_{pair}.log"), "w")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    r = subprocess.run([sys.executable, "-u", os.path.join(HERE, "tournament.py"), "--pair", pair],
                       cwd=TRAINING_DIR, stdout=log, stderr=subprocess.STDOUT, env=env)
    return pair, r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=12)
    a = ap.parse_args()
    pend = [p for p in todo() if all(ready(x) for x in p.split("__"))]
    # greedy-vs-greedy pairs are the slowest: start them first
    greedy = {"enriched", "enriched_aug", "constrained_greedy", "gourdeau"}
    pend.sort(key=lambda p: -sum(x in greedy for x in p.split("__")))
    print(f"{len(pend)} runnable pairs", flush=True)
    with ThreadPoolExecutor(a.procs) as ex:
        for pair, rc in ex.map(run, pend):
            print(f"{pair}: rc={rc}", flush=True)


if __name__ == "__main__":
    main()
