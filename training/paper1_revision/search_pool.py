"""
Process pool over search_eval.py tasks (CPU only; one thread per process).
Finished outputs are skipped by search_eval.py, so a rerun resumes.

Usage (from training/): python3 paper1_revision/search_pool.py --procs 30 <group>[,<group>...]
  groups: bench        F_oof_s0, enriched leaf, T0 and T1, 10 shards each
          bench_lcb    F_oof_s0, lcb:1 leaf, T0, 10 shards
          tour         pending pairs of mcts_search and mcts_lcb
          tour_search / rich_search   only the mcts_search items (no ensemble needed)
          tour_lcb / rich_lcb         only the mcts_lcb items
          tour_lcb2    pending pairs of mcts_lcb2
          rich         rich eval of mcts_search, mcts_lcb (5 eval seeds)
          rich_lcb2    rich eval of mcts_lcb2
"""
import os
import sys
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)
from paper1_revision import core
from paper1_revision import search_eval as se

LOG = os.path.join(core.LOGS, "search_agents")


def tasks(group):
    if group == "bench":
        return [["bench", "enriched", "0", t, str(s)] for t in ("0", "1") for s in range(10)
                if not os.path.exists(se.bench_path("enriched", 0, float(t), s))]
    if group == "bench_lcb":
        return [["bench", "lcb:1", "0", "0", str(s)] for s in range(10)
                if not os.path.exists(se.bench_path("lcb:1", 0, 0.0, s))]
    agents = {"tour": ("mcts_search", "mcts_lcb"), "tour_search": ("mcts_search",),
              "rich_search": ("mcts_search",), "rich": ("mcts_search", "mcts_lcb"),
              "tour_lcb": ("mcts_search", "mcts_lcb"), "rich_lcb": ("mcts_lcb",),
              "tour_lcb2": ("mcts_search", "mcts_lcb", "mcts_lcb2"),
              "rich_lcb2": ("mcts_lcb2",)}[group]
    out = []
    for line in se.todo(agents):
        kind, *rest = line.split()
        if kind == "pair" and group.startswith("tour"):
            out.append(["pair"] + rest)
        if kind == "rich" and group.startswith("rich"):
            out.append(["rich"] + rest)
    if group in ("tour_lcb", "tour_lcb2"):
        ag = group[5:].replace("lcb", "mcts_lcb")
        out = [t for t in out if ag in t[1].split("__")]
    # pairs where both sides search are the slowest: start them first
    out.sort(key=lambda t: -sum(x in se.NEW for x in t[1].split("__")) if t[0] == "pair" else 0)
    return out


def run(t):
    name = "_".join(x.replace(":", "") for x in t)
    log = open(os.path.join(LOG, f"{name}.log"), "w")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    r = subprocess.run([sys.executable, "-u", os.path.join(HERE, "search_eval.py")] + t,
                       cwd=TRAINING_DIR, stdout=log, stderr=subprocess.STDOUT, env=env)
    return name, r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("groups")
    ap.add_argument("--procs", type=int, default=30)
    ap.add_argument("--reverse", action="store_true", help="take tasks from the end (a second pool)")
    a = ap.parse_args()
    os.makedirs(LOG, exist_ok=True)
    T = [t for g in a.groups.split(",") for t in tasks(g)]
    if a.reverse:
        T = T[::-1]
    print(f"{len(T)} tasks", flush=True)
    with ThreadPoolExecutor(a.procs) as ex:
        for name, rc in ex.map(run, T):
            print(f"{name}: rc={rc}", flush=True)


if __name__ == "__main__":
    main()
