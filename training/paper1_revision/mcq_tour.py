"""
Tournament row for MCQ as published (mcq_real.py, lam chosen by lowest held-out
BCE, fixed before any tournament score): plays every ordered pair against the
11 tournament strategies, 200 drafts each, exactly as tournament.run_pair
(maps/tiers stratified, seed 1000 + pair index with mcq_real appended as the
12th strategy so existing indices are unchanged). Records go to
site/results/mcq_real/tournament/ and are scored with score_tournament.

Usage (from training/, p1site env):
  python3 paper1_revision/mcq_tour.py --list-todo
  python3 paper1_revision/mcq_tour.py --pair mcq_real__gd
  python3 paper1_revision/mcq_tour.py --pool 22
"""
import os
import sys
import json
import time
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)
from paper1_revision import core
from paper1_revision import tournament as T

NAME = "mcq_real"
LAM = os.environ.get("MCQ_LAM", "0.9")
OUT = os.path.join(core.RESULTS, "mcq_real", "tournament")
T.STRATEGIES = list(T.STRATEGIES) + [NAME]
_build = T.build


def build(name, device, gd_models):
    if name == NAME:
        from rerun2026 import common
        from experiment_rich_evaluation import make_cql_strategy
        inner = make_cql_strategy(os.path.join(common.MODELS_DIR, "mcq_real", f"mcq_real_l{LAM}.pt"), device)

        def act(state, team, step_type, game_map, tier):
            return inner(state, team, step_type, game_map, tier, gd_models, device)
        return act
    return _build(name, device, gd_models)


T.build = build


def pairs():
    others = T.STRATEGIES[:-1]
    return [f"{NAME}__{o}" for o in others] + [f"{o}__{NAME}" for o in others]


def path(p):
    return os.path.join(OUT, f"{p}.json")


def todo():
    return [p for p in pairs() if not os.path.exists(path(p))]


def run_one(p, n):
    if os.path.exists(path(p)):
        return
    t0 = time.time()
    a, b = p.split("__")
    out = T.run_pair(a, b, n)
    out["mcq_lam"] = float(LAM)
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(path(p) + ".tmp", "w"))
    os.replace(path(p) + ".tmp", path(p))
    print(f"{p}: {n} drafts in {time.time() - t0:.0f}s", flush=True)


def pool(procs, n):
    logs = os.path.join(core.LOGS, "mcq_tour")
    os.makedirs(logs, exist_ok=True)

    def run(p):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        with open(os.path.join(logs, f"{p}.log"), "w") as log:
            r = subprocess.run([sys.executable, "-u", __file__, "--pair", p, "--drafts", str(n)],
                               cwd=TRAINING_DIR, stdout=log, stderr=subprocess.STDOUT, env=env)
        print(p, "rc", r.returncode, flush=True)
    with ThreadPoolExecutor(procs) as ex:
        list(ex.map(run, todo()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair")
    ap.add_argument("--drafts", type=int, default=200)
    ap.add_argument("--list-todo", action="store_true")
    ap.add_argument("--pool", type=int)
    a = ap.parse_args()
    if a.list_todo:
        print("\n".join(todo()))
    elif a.pair:
        import torch
        torch.set_num_threads(1)
        run_one(a.pair, a.drafts)
    elif a.pool:
        pool(a.pool, a.drafts)


if __name__ == "__main__":
    main()
