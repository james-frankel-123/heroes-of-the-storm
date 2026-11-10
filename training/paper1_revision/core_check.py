"""
Four-hero-core check of the judges' support bias, with no judge in the comparison
of real outcomes.

Real teams that share a four-hero core and differ only in the fifth hero are
compared with each other. Within a core, any lasting difference in who fields the
core, and the core's own strength, cancel; what varies is the fifth hero and how
well it is supported next to the other four. For each core we demean, across its
real completions, the actual result y, each judge's predicted p, and the team's
support (leave-one-out pair affinity, PMI; and log rarest triple). If the judges
extrapolate badly into unsupported teams, low-support completions lose more often
than the core's other completions, by more than the judges predict.

  raw         within-core slope of the actual result on support: model-free
  judged      the same slope of the judge's prediction
  gap         actual minus judged: the bias the judge carries within a core
  agent_cores the same, restricted to cores that the search agents' final teams use

Games: the 1M leave-one-out training-split games of judge_ood_calibration (per-team
y, judge p, support), matched to their heroes by replay id. Each team gives five
cores. Standard errors are clustered by core and by game; the larger is reported.

Usage (from training/, p1site env): python3 paper1_revision/core_check.py
Output: paper1_revision/site/results/core_check.json
"""
import os
import sys
import json
from itertools import combinations

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
T = os.path.dirname(HERE)
sys.path.insert(0, T)
from paper1_revision import core
from paper1_revision import team_support as ts
from shared import HERO_TO_IDX

RES = os.path.join(HERE, "site", "results")
JUDGES = ["consensus", "gN", "gN_naive", "RN", "QM2026"]
AGENTS = ["mcts", "constrained_mcts", "constrained_wr", "gd", "constrained_greedy", "enriched"]
MIN_CORE = 20  # team-observations per core


def load():
    z = np.load(os.path.join(RES, "judge_ood_calibration_teams.npz"), allow_pickle=True)
    sel = z["set"] == "train_loo"
    d = {k: z[k][sel] for k in ("y", "rid", "team", "m_pmi", "m_min3")}
    d["p"] = {j: z["p_" + j][sel] for j in JUDGES}
    G = core.split_games()
    by = {g[0]: g for g in G["train"]}
    A = np.array([sorted(HERO_TO_IDX[h] for h in by[r][3 + t]) for r, t in zip(d["rid"], d["team"])], np.int64)
    return d, A


def long_cores(A):
    """Every (team row, core) pair: core key, fifth hero, row index."""
    ck, fifth, row = [], [], []
    for j in range(5):
        cols = [c for c in range(5) if c != j]
        ck.append(ts.key(cols, A))
        fifth.append(A[:, j])
        row.append(np.arange(len(A)))
    return np.concatenate(ck), np.concatenate(fifth), np.concatenate(row)


def demean(x, g, ng):
    m = np.bincount(g, x, ng) / np.bincount(g, minlength=ng)
    return x - m[g]


def slope(v, s, clusters):
    """OLS slope of demeaned v on demeaned s (no intercept) with the larger clustered SE."""
    ss = (s * s).sum()
    b = (v * s).sum() / ss
    u = s * (v - b * s)
    ses = [np.sqrt((np.bincount(c, u) ** 2).sum()) / ss for c in clusters]
    return float(b), float(max(ses))


def analyse(d, A, ck, fifth, row, keep_core=None):
    u, g = np.unique(ck, return_inverse=True)
    ng = len(u)
    cnt = np.bincount(g, minlength=ng)
    nf = np.bincount(np.unique(g.astype(np.int64) * ts.H + fifth) // ts.H, minlength=ng)
    ok = (cnt >= MIN_CORE) & (nf >= 2)
    if keep_core is not None:
        ok &= np.isin(u, keep_core)
    m = ok[g]
    g2 = np.unique(g[m], return_inverse=True)[1]
    ng2 = g2.max() + 1
    r = row[m]
    _, rid_c = np.unique(d["rid"][r], return_inverse=True)
    out = {"cores": int(ng2), "team_core_obs": int(m.sum()), "teams": int(len(np.unique(r)))}
    y = demean(d["y"][r], g2, ng2)
    for sname, sraw in (("pmi", d["m_pmi"]), ("log_min3", np.log1p(d["m_min3"]))):
        s = demean(sraw[r], g2, ng2)
        sd = float(s.std())
        q = np.percentile(s, [10, 90])
        lo, hi = s <= q[0], s >= q[1]
        res = {"within_core_sd": sd, "raw_slope_per_sd": [x * sd * 100 for x in slope(y, s, (g2, rid_c))],
               "bottom10_vs_top10": {"support_diff": float(s[lo].mean() - s[hi].mean()),
                                     "actual_pts": float(100 * (y[lo].mean() - y[hi].mean()))}}
        for j in JUDGES:
            p = demean(d["p"][j][r], g2, ng2)
            res[j] = {"judged_slope_per_sd": [x * sd * 100 for x in slope(p, s, (g2, rid_c))],
                      "gap_slope_per_sd": [x * sd * 100 for x in slope(y - p, s, (g2, rid_c))],
                      "bottom10_vs_top10_judged_pts": float(100 * (p[lo].mean() - p[hi].mean())),
                      "bottom10_vs_top10_gap_pts": float(100 * ((y - p)[lo].mean() - (y - p)[hi].mean()))}
        out[sname] = res
    return out


def agent_cores():
    st = ts.strategy_teams()
    keys = {}
    for a in AGENTS:
        if a in st:
            B = np.unique(st[a], axis=0)
            keys[a] = np.unique(np.concatenate([ts.key([c for c in range(5) if c != j], B) for j in range(5)]))
    return keys


def main():
    d, A = load()
    print(f"teams {len(A):,}", flush=True)
    ck, fifth, row = long_cores(A)
    res = {"min_core_obs": MIN_CORE, "all": analyse(d, A, ck, fifth, row)}
    print("all", json.dumps(res["all"]["pmi"]["consensus"]), res["all"]["pmi"]["raw_slope_per_sd"], flush=True)
    ak = agent_cores()
    res["agent_cores"] = {}
    for a, keys in ak.items():
        res["agent_cores"][a] = analyse(d, A, ck, fifth, row, keep_core=keys)
        r = res["agent_cores"][a]
        print(a, r["cores"], r["team_core_obs"], r["pmi"]["raw_slope_per_sd"], r["pmi"]["consensus"]["gap_slope_per_sd"], flush=True)
    json.dump(res, open(os.path.join(RES, "core_check.json"), "w"), indent=1)
    print("wrote core_check.json")


if __name__ == "__main__":
    main()
