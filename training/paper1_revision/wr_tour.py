"""
Tournament row for a masked highest-win-rate drafter (no interactions): every
pick is the available hero with the highest shrunken win rate on this map and
tier, under the same role mask as constrained greedy / constrained MCTS; every
ban removes the highest-ranked available hero. Win rates come from the deploy
statistics the greedy agents use; a hero's map win rate is shrunk toward its
tier-wide win rate with SHRINK pseudo-games. Rule fixed before any score.

Plays every ordered pair against the 11 tournament strategies, 200 drafts each,
exactly as tournament.run_pair (seed 1000 + pair index with constrained_wr
appended as the 12th strategy, so existing indices are unchanged). Records go
to site/results/wr_baseline/tournament/ and are scored by wr_score.py.

Usage (from training/, p1site env):
  python3 paper1_revision/wr_tour.py --list-todo
  python3 paper1_revision/wr_tour.py --pair constrained_wr__gd
  python3 paper1_revision/wr_tour.py --pool 4
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

NAME = "constrained_wr"
SHRINK = 500.0
OUT = os.path.join(core.RESULTS, "wr_baseline", "tournament")
T.STRATEGIES = list(T.STRATEGIES) + [NAME]
_build = T.build


def scores(st, game_map, tier):
    from shared import HEROES
    base = st.hero_wr[tier]
    hm = st.hero_map_wr[tier].get(game_map, {})
    out = []
    for h in HEROES:
        p = base.get(h, 50.0)
        if h in hm:
            wr, n = hm[h]
            p = (wr * n + p * SHRINK) / (n + SHRINK)
        out.append(p)
    return out


def make_wr_strategy():
    import numpy as np
    from rerun2026.constrained_search import constrain
    st = core.load_stats("deploy")
    cache = {}

    def inner(state, team, step_type, game_map, tier, gd_models, device):
        key = (game_map, tier)
        if key not in cache:
            cache[key] = np.array(scores(st, game_map, tier))
        s = np.where(state.valid_mask_np() > 0.5, cache[key], -np.inf)
        return int(np.argmax(s))
    return constrain(inner)


def build(name, device, gd_models):
    if name == NAME:
        inner = make_wr_strategy()

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
    out["wr_shrink"] = SHRINK
    out["secs"] = time.time() - t0
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(path(p) + ".tmp", "w"))
    os.replace(path(p) + ".tmp", path(p))
    print(f"{p}: {n} drafts in {out['secs']:.0f}s", flush=True)


def pool(procs, n):
    logs = os.path.join(core.LOGS, "wr_tour")
    os.makedirs(logs, exist_ok=True)

    def run(p):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        with open(os.path.join(logs, f"{p}.log"), "w") as log:
            r = subprocess.run([sys.executable, "-u", __file__, "--pair", p, "--drafts", str(n)],
                               cwd=TRAINING_DIR, stdout=log, stderr=subprocess.STDOUT, env=env)
        print(p, "rc", r.returncode, time.strftime("%H:%M"), flush=True)
    with ThreadPoolExecutor(procs) as ex:
        list(ex.map(run, todo()))
    print("WR TOUR DONE", flush=True)


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
