"""
Are the judges still calibrated on real teams that look like the agents' teams?

The agents' final five-hero teams are far from the training data in a specific way
(team_support.py): almost never seen whole, one swap from a seen team, built from
rare triples and from hero pairs that humans field together less than their
popularity predicts. If the judges' interpolation is wrong there, every judge could
overrate those teams together. This script scores real games, which no judge trained
on, binned by the same support measures of each team, and compares each judge's
predicted win probability with the actual result.

Games: a random sample of training-split snapshot games, whose support measures are
computed leave-one-out (the game's own team removed from the counts), plus every
validation and paper-test game (held out from the counts entirely). No judge trained
on any of them: the judges are fit on post-snapshot ranked games and Quick Match.
Judges: v2, composition-corrected, exactly as the tournament (search_score.v1_v2).

Per team (each game gives two): predicted p for that team, outcome y.
Outputs (site/results/judge_ood_calibration.json):
  by_bin[judge][measure] = per-bin n, mean predicted, actual, gap = actual - predicted, SE
  joint[judge] = gap on a rarest-triple x pair-affinity grid
  reweighted[judge][strategy] = mean gap with real teams reweighted to the strategy's
      distribution over that grid: the judge's expected bias on teams like the strategy's

Usage (from training/, p1site env): python3 paper1_revision/judge_ood_calibration.py [--sample 300000]
"""
import os
import sys
import json
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
T = os.path.dirname(HERE)
sys.path.insert(0, T)
from paper1_revision import core
from paper1_revision import team_support as ts
from shared import HERO_TO_IDX

JUDGES = ["consensus", "gN", "gN_naive", "RN", "QM2026", "QM2021"]
OUT = os.path.join(ts.RES, "judge_ood_calibration.json")
if os.environ.get("JOC_OUT"):
    OUT = os.environ["JOC_OUT"]


def idx_teams(games, side):
    return np.array([sorted(HERO_TO_IDX[h] for h in g[side]) for g in games], np.int64)


def loo_metrics(C, A):
    """Exact leave-one-out version of Corpus.metrics for teams that are in the corpus."""
    from itertools import combinations
    key = ts.key
    n5 = C._lookup(C.k5, key(range(5), A)) - 1
    s4 = np.stack([C.c4[key(c, A)] for c in combinations(range(5), 4)], 1) - 1
    s3 = np.stack([C.c3[key(c, A)] for c in combinations(range(5), 3)], 1) - 1
    s2 = np.stack([C.c2[key(c, A)] for c in combinations(range(5), 2)], 1) - 1
    dist = np.where(n5 > 0, 0, np.where(s4.max(1) > 0, 1, np.where(s3.max(1) > 0, 2, 3)))
    p = C.c1 / C.n
    expect = np.stack([C.n * p[A[:, a]] * p[A[:, b]] for a, b in combinations(range(5), 2)], 1)
    pmi = np.log((s2 + 0.5) / expect).mean(1)
    return dict(n5=n5, dist=dist, min3=s3.min(1), min2=s2.min(1), pmi=pmi)


def judge_scores(games):
    from paper1_revision.score_tournament import score_rows
    from paper1_revision.search_score import v1_v2, rebuilt_gn
    rows = [(list(g[3]), list(g[4]), g[2], g[1]) for g in games]
    base = score_rows(rows)
    base["gN"] = rebuilt_gn(rows)
    return v1_v2(base, rows)["v2"]


def bin_stats(p, y, b, nb):
    out = []
    for i in range(nb):
        s = b == i
        n = int(s.sum())
        if n == 0:
            out.append({"n": 0})
            continue
        gap = y[s] - p[s]
        out.append({"n": n, "pred": float(p[s].mean()), "actual": float(y[s].mean()),
                    "gap": float(gap.mean()), "se": float(gap.std(ddof=1) / np.sqrt(n)) if n > 1 else None})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=300000)
    ap.add_argument("--held", type=int, default=0, help="cap held-out games (smoke tests)")
    a = ap.parse_args()
    G = core.split_games()
    C = ts.Corpus(np.concatenate([idx_teams(G["train"], 3), idx_teams(G["train"], 4)]))
    print(f"corpus teams {C.n:,}", flush=True)
    rng = np.random.default_rng(0)
    tr = [G["train"][i] for i in rng.choice(len(G["train"]), min(a.sample, len(G["train"])), replace=False)]
    held = G["val"] + G["test"]
    if a.held:
        held = held[:a.held]
    parts = []
    for name, games, in_corpus in (("train_loo", tr, True), ("heldout", held, False)):
        sc = judge_scores(games)
        y0 = np.array([1.0 if g[6] == 0 else 0.0 for g in games])
        for side in (3, 4):
            A = idx_teams(games, side)
            m = loo_metrics(C, A) if in_corpus else C.metrics(A)
            sgn = side == 3
            parts.append({"set": name, "y": y0 if sgn else 1 - y0, "rid": np.array([g[0] for g in games]), "team": np.full(len(games), side - 3),
                          "p": {j: (sc[j] if sgn else 1 - sc[j]) for j in JUDGES},
                          **{k: m[k] for k in ("n5", "dist", "min3", "min2", "pmi")}})
        print(f"scored {name}: {len(games):,} games", flush=True)
    cat = lambda k: np.concatenate([q[k] for q in parts])
    y = cat("y")
    P = {j: np.concatenate([q["p"][j] for q in parts]) for j in JUDGES}
    M = {k: cat(k) for k in ("n5", "dist", "min3", "min2", "pmi")}
    setname = np.concatenate([[q["set"]] * len(q["y"]) for q in parts])
    np.savez_compressed(OUT.replace(".json", "_teams.npz"), y=y, set=setname, **{f"p_{j}": P[j] for j in JUDGES},
                        **{f"m_{k}": M[k] for k in M}, rid=cat("rid"), team=cat("team"))

    # bins: human-team quantiles of each measure, with the tail split finer
    qs = [0, 1, 2.5, 5, 10, 25, 50, 75, 100]
    edges = {k: np.unique(np.percentile(M[k], qs)) for k in ("min3", "min2", "pmi")}
    B = {k: np.clip(np.searchsorted(edges[k], M[k], side="right") - 1, 0, len(edges[k]) - 2) for k in edges}
    B["dist"] = np.minimum(M["dist"], 2)
    B["seen"] = (M["n5"] > 0).astype(int)
    nb = {k: (len(edges[k]) - 1 if k in edges else (3 if k == "dist" else 2)) for k in B}
    res = {"n_teams": int(len(y)), "sets": {s: int((setname == s).sum()) for s in np.unique(setname)},
           "edges": {k: v.tolist() for k, v in edges.items()}, "by_bin": {}, "by_bin_heldout": {}, "joint": {},
           "reweighted": {}}
    ho = setname == "heldout"
    for j in JUDGES:
        res["by_bin"][j] = {k: bin_stats(P[j], y, B[k], nb[k]) for k in B}
        res["by_bin_heldout"][j] = {k: bin_stats(P[j][ho], y[ho], B[k][ho], nb[k]) for k in B}

    # joint grid: rarest triple x pair affinity, 4 x 4 on human quantiles
    jq = [0, 5, 25, 50, 100]
    je = {k: np.unique(np.percentile(M[k], jq)) for k in ("min3", "pmi")}
    jb = lambda k, x: np.clip(np.searchsorted(je[k], x, side="right") - 1, 0, len(je[k]) - 2)
    cell = jb("min3", M["min3"]) * 4 + jb("pmi", M["pmi"])
    for j in JUDGES:
        res["joint"][j] = bin_stats(P[j], y, cell, 16)
    res["joint_edges"] = {k: v.tolist() for k, v in je.items()}

    # reweight real teams to each strategy's distribution over the joint grid
    strat = ts.strategy_teams()
    gap_cell = {j: np.array([c.get("gap", np.nan) if c["n"] >= 200 else np.nan for c in res["joint"][j]]) for j in JUDGES}
    se_cell = {j: np.array([c.get("se") or np.nan if c["n"] >= 200 else np.nan for c in res["joint"][j]]) for j in JUDGES}
    human_w = np.bincount(cell, minlength=16) / len(cell)
    for name, A in sorted(strat.items()):
        m = C.metrics(A)
        w = np.bincount(jb("min3", m["min3"]) * 4 + jb("pmi", m["pmi"]), minlength=16) / len(A)
        res["reweighted"][name] = {"cell_share": w.tolist(), "covered": float(w[~np.isnan(gap_cell["gN"])].sum())}
        for j in JUDGES:
            ok = ~np.isnan(gap_cell[j])
            ww = w * ok
            res["reweighted"][name][j] = {"gap": float(np.nansum(ww * gap_cell[j]) / ww.sum()),
                                          "se": float(np.sqrt(np.nansum((ww / ww.sum()) ** 2 * se_cell[j] ** 2)))}
    res["reweighted"]["human"] = {j: {"gap": float(np.nansum(human_w * gap_cell[j]) / (human_w * ~np.isnan(gap_cell[j])).sum())}
                                  for j in JUDGES}
    json.dump(res, open(OUT, "w"), indent=1)
    print("wrote", OUT)
    for name in ("human", "mcts", "constrained_mcts", "k_oof", "constrained_wr", "enriched", "gourdeau", "gd"):
        if name in res["reweighted"]:
            r = res["reweighted"][name]
            print(name.ljust(18), {j: round(r[j]["gap"] * 100, 2) for j in JUDGES}, "covered", round(r.get("covered", 1), 3))


if __name__ == "__main__":
    main()
