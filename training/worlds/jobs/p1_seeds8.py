"""
Per-seed true values of the floor-trained and ordinary MCTS policies from their own tournament drafts
(paper1_revision/seed_tour.py), family a, every dial cell of the frozen grid, and the seed-level contrasts.

  value(entrant, world, cell) = rescore.strat_values (mean over the 7 common opponents of the mean true win
  probability, both seats), entrants "<base>@s<k>".
Contrasts, per cell, with seeds as the unit (the question is whether arms differ by more than seed-to-seed):
  arm means over seeds, X - Y of the arm means (between-world 95% interval of that difference), the seed SDs,
  and Welch's t on the per-seed world-mean values (df by Welch-Satterthwaite, two-sided p).
Pairs: constrained_mcts_ft, constrained_mcts_ft_sf, search_ft_sf vs constrained_mcts; constrained_mcts_sf vs
constrained_mcts (the floor's cost, seed-paired); search_sf vs constrained_mcts.

Usage (from training/, p1site env, CPU; WORLDS_GRADE=flat for the flat grade, output tag _flat):
  python -u worlds/jobs/p1_seeds8.py [--n 1000] [--procs 8]
Outputs: worlds/site/cache/p1_seeds8<tag>_a.npz (worlds x cells x entrants; rebuilt when the entrant set
changes), worlds/site/results/p1_seeds8<tag>.json
"""
import os
import sys
import json
import glob
import time
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORLDS = os.path.dirname(HERE)
TRAINING = os.path.dirname(WORLDS)
sys.path.insert(0, WORLDS)
sys.path.insert(0, TRAINING)
import data as WD  # noqa: E402
import world as W  # noqa: E402
import rescore as R  # noqa: E402

TAG = "" if W.GRADE == "log" else f"_{W.GRADE}"
PAIRS = [("constrained_mcts_ft", "constrained_mcts"), ("constrained_mcts_ft_sf", "constrained_mcts"),
         ("search_ft_sf", "constrained_mcts"), ("constrained_mcts_sf", "constrained_mcts"),
         ("search_sf", "constrained_mcts"), ("constrained_mcts_ft_sf", "constrained_mcts_ft")]


def load_drafts():
    from paper1_revision import seed_tour as st
    recs = []
    for f in sorted(glob.glob(os.path.join(st.OUT, "*.json"))):
        J = json.load(open(f))
        recs += [(J["team0_strategy"], J["team1_strategy"], r) for r in J["records"]]
    D = WD.from_rows([(r["team0"]["picks"], r["team1"]["picks"], r["game_map"], r["tier"]) for _, _, r in recs])
    D["S0"] = np.array([a for a, _, _ in recs])
    D["S1"] = np.array([b for _, b, _ in recs])
    return D


def index(D):
    """entrant -> [(opponent, idx team 0, idx team 1)], entrants with all 7 opponents in both seats."""
    from paper1_revision import seed_tour as st
    out = {}
    for s in sorted(set(D["S0"]) | set(D["S1"])):
        if "@s" not in s:
            continue
        L = []
        for o in st.OPP7:
            f = np.flatnonzero((D["S0"] == s) & (D["S1"] == o))
            g = np.flatnonzero((D["S0"] == o) & (D["S1"] == s))
            if len(f) and len(g):
                L.append((o, f, g))
        if len(L) == len(st.OPP7):
            out[s] = L
    return out


def welch(x, y):
    from scipy import stats
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or len(y) < 2:
        return None
    t, p = stats.ttest_ind(x, y, equal_var=False)
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    df = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1)) if vx + vy > 0 else float("nan")
    return {"t": float(t), "df": float(df), "p": float(p), "se": float(np.sqrt(vx + vy))}


def summarize(V, cells, names):
    named = list(W.dials()["named"])
    ix = {s: i for i, s in enumerate(names)}
    seeds = {}
    for s in names:
        b, k = s.split("@s")
        seeds.setdefault(b, []).append(int(k))
    out = {"seeds": {b: sorted(v) for b, v in seeds.items()}, "cells": {}}
    for ci, cell in enumerate(cells):
        tag = named[ci] if ci < len(named) else json.dumps(cell, sort_keys=True)
        C = {"cell": cell, "per_seed": {}, "arms": {}, "contrasts": {}}
        arm_w = {}
        for b, ks in seeds.items():
            M = np.stack([V[:, ci, ix[f"{b}@s{k}"]] for k in sorted(ks)], 1)   # worlds x seeds
            C["per_seed"][b] = {str(k): float(100 * M[:, j].mean()) for j, k in enumerate(sorted(ks))}
            vals = 100 * M.mean(0)
            C["arms"][b] = {"mean": float(vals.mean()), "seed_sd": float(vals.std(ddof=1)) if len(vals) > 1 else None,
                            "n_seeds": len(vals)}
            arm_w[b] = M.mean(1)
        for x, y in PAIRS:
            if x not in arm_w or y not in arm_w:
                continue
            d = 100 * (arm_w[x] - arm_w[y])
            row = {"mean": float(d.mean()), "world_lo95": float(np.percentile(d, 2.5)),
                   "world_hi95": float(np.percentile(d, 97.5)),
                   "welch_seeds": welch(list(C["per_seed"][x].values()), list(C["per_seed"][y].values()))}
            common = sorted(set(seeds[x]) & set(seeds[y]))
            if cfg_same(x, y) and len(common) > 1:   # same checkpoints (e.g. the floor's cost): seed-paired
                pd = np.array([C["per_seed"][x][str(k)] - C["per_seed"][y][str(k)] for k in common])
                row["paired_seeds"] = {"mean": float(pd.mean()), "sd": float(pd.std(ddof=1)),
                                       "se": float(pd.std(ddof=1) / np.sqrt(len(pd))), "n": len(pd)}
            C["contrasts"][f"{x} - {y}"] = row
        out["cells"][tag] = C
    return out


def cfg_same(x, y):
    return ("_ft" in x) == ("_ft" in y)


def main():
    from multiprocessing import get_context
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--procs", type=int, default=int(os.environ.get("WORLDS_PROCS", "8")))
    a = ap.parse_args()
    D = load_drafts()
    idx = index(D)
    names = sorted(idx)
    print(f"{len(D['S0']):,} drafts, {len(names)} entrants: {names}", flush=True)
    cells, _ = R.cells_for("a")
    cache = os.path.join(WD.CACHE, f"p1_seeds8{TAG}_a.npz")
    V = None
    if os.path.exists(cache):
        z = np.load(cache)
        if list(z["names"]) == names and int(z["V"].shape[0]) == a.n:
            V = z["V"]
    if V is None:
        t0 = time.time()
        R._G.update(D=D, idx=idx, names=names)
        W.Worlds().base_draws([0])
        with get_context("fork").Pool(a.procs) as pool:
            V = np.stack(pool.map(R._p1_world, [("a", w, cells) for w in range(a.n)], chunksize=4))
        np.savez(cache, V=V, cells=json.dumps(cells), names=np.array(names))
        print(f"rescored {V.shape} in {time.time() - t0:.0f}s", flush=True)
    res = {"family": "a", "grade": W.GRADE, "worlds": int(V.shape[0]), "drafts": int(len(D["S0"])),
           **summarize(V, cells, names)}
    p = os.path.join(WD.RESULTS, f"p1_seeds8{TAG}.json")
    json.dump(res, open(p, "w"), indent=1)
    d = res["cells"]["default"]
    for b, v in d["arms"].items():
        print(f"{b:24s} mean {v['mean']:.2f} seed_sd {v['seed_sd']} n={v['n_seeds']}  seeds {d['per_seed'][b]}")
    for k, v in d["contrasts"].items():
        print(f"{k:48s} {v['mean']:+.2f} [{v['world_lo95']:+.2f},{v['world_hi95']:+.2f}] welch {v['welch_seeds']}"
              + (f" paired {v['paired_seeds']}" if "paired_seeds" in v else ""))
    print("wrote", p)


if __name__ == "__main__":
    main()
