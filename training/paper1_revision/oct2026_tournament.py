"""
oct2026 tournament runner (expert study v6). Runs every ordered pair of the
phase3b round-robin and every constrained-search pair by IMPORTING the
unchanged rerun2026 modules (so oct2026_site/sitecustomize.py patches apply:
own composition table, F_400sim_s0 for both MCTS actors, and the v6 roster
without synthetic augmentation) and calling their task_pair with
--mcts-run F_400sim_s0. CPU only, 5 workers. Writes the usual pair files to
rerun2026/ns/oct2026/results/{roundrobin,constrained/roundrobin}/.

Usage: python3 paper1_revision/oct2026_tournament.py roundrobin|constrained
"""
import os
import sys
import json
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
MCTS_RUN = "F_400sim_s0"

RUNNER = """
import sys, types
sys.path.insert(0, %r)
mod = __import__(%r, fromlist=["x"])
args = types.SimpleNamespace(pair=%r, drafts=200, mcts_run=%r, task="pair", seeds=5,
                             strategy=None, dry_run=False, force=False, only=None)
mod.task_pair(args)
"""


def pairs(kind):
    code = ("import sys,json,itertools; sys.path.insert(0, %r); "
            % TRAINING_DIR) + (
        "import rerun2026.phase3b_roundrobin as p; print(json.dumps(list(itertools.permutations(p.STRATEGIES, 2))))"
        if kind == "roundrobin" else
        "import rerun2026.constrained_search as c; print(json.dumps(c.pair_list())); "
        "import rerun2026.phase3b_roundrobin as p; assert 'enriched_aug' not in c.ALL_STRATEGIES, c.ALL_STRATEGIES")
    r = subprocess.run([sys.executable, "-c", code], cwd=TRAINING_DIR, capture_output=True, text=True)
    if r.returncode:
        sys.exit(r.stderr[-2000:])
    return [tuple(x) for x in json.loads(r.stdout.strip().splitlines()[-1])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["roundrobin", "constrained"])
    a = ap.parse_args()
    from rerun2026 import common
    mod = "rerun2026.phase3b_roundrobin" if a.kind == "roundrobin" else "rerun2026.constrained_search"
    res_dir = os.path.join(common.RESULTS_DIR, "roundrobin" if a.kind == "roundrobin"
                           else os.path.join("constrained", "roundrobin"))
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(common.LOGS_DIR, exist_ok=True)
    todo = [(x, y) for x, y in pairs(a.kind) if not os.path.exists(os.path.join(res_dir, f"{x}__{y}.json"))]
    print(f"{a.kind}: {len(todo)} pairs to run", flush=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

    def one(pr):
        name = f"{pr[0]}__{pr[1]}"
        lf = open(os.path.join(common.LOGS_DIR, f"{'p3b' if a.kind == 'roundrobin' else 'p3c'}_{name}.log"), "w")
        rc = subprocess.call([sys.executable, "-u", "-c", RUNNER % (TRAINING_DIR, mod, name, MCTS_RUN)],
                             cwd=TRAINING_DIR, env=env, stdout=lf, stderr=subprocess.STDOUT)
        print(f"{name}: rc={rc}", flush=True)
        return rc
    with ThreadPoolExecutor(5) as ex:
        rcs = list(ex.map(one, todo))
    if any(rcs):
        sys.exit(1)
    marker = os.path.join(common.RESULTS_DIR, "roundrobin.done" if a.kind == "roundrobin"
                          else os.path.join("constrained", "pairs.done"))
    open(marker, "w").write(str(len(pairs(a.kind))))


if __name__ == "__main__":
    sys.path.insert(0, TRAINING_DIR)
    main()
