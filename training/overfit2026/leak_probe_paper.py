"""
Leak-mechanism probe for the PAPER proxy (wp_enriched_256 + frozen stats) on
unseen real games (post-snapshot no-drift set BF + T97).

If training on outcome-encoding statistics taught the proxy to over-trust
statistics, and most of all sparse ones, then on games it never saw the
realized outcome should pull AGAINST the statistics the proxy relies on:
in   y ~ logit(proxy) + d_syn_sparse + d_syn_dense + d_ctr_sparse + d_ctr_dense
the sparse-cell coefficients should be negative (the proxy already
over-counted them), the dense ones near zero.

Cells are split by their game count in the frozen statistics (the counts the
proxy saw). d_* are team0-minus-team1 mean additive residuals (pp).
Output: results/leak_probe_paper.json
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)

import numpy as np

from overfit2026 import data, feats
from overfit2026.gold import fit_logistic

BINS = [(0, 300), (300, 2000), (2000, 10 ** 9)]


def team_terms(st, own, opp, tier):
    hw = lambda h: st.hero_wr.get(tier, {}).get(h, 50.0)
    syn = [[] for _ in BINS]
    ctr = [[] for _ in BINS]
    pw = st.pairwise.get(tier, {})
    for i, a in enumerate(own):
        for b in own[i + 1:]:
            e = pw.get("with", {}).get(a, {}).get(b)
            if e and e[1] >= 30:
                r = e[0] - (50 + hw(a) - 50 + hw(b) - 50)
                for k, (lo, hi) in enumerate(BINS):
                    if lo <= e[1] < hi:
                        syn[k].append(r)
        for b in opp:
            e = pw.get("against", {}).get(a, {}).get(b)
            if e and e[1] >= 30:
                r = e[0] - (hw(a) + (100 - hw(b)) - 50)
                for k, (lo, hi) in enumerate(BINS):
                    if lo <= e[1] < hi:
                        ctr[k].append(r)
    return ([np.sum(x) / 10.0 for x in syn], [np.sum(x) / 25.0 for x in ctr],
            [len(x) for x in syn], [len(x) for x in ctr])


def main():
    z = np.load(os.path.join(HERE, "cache", "gold_games_proxy.npz"))
    fut, bf = data.load_future(), data.load_backfill()
    sets = {"BF": [g for v in bf.values() for g in v], "T97": fut["2.55.16.97039"]}
    st = feats.paper_stats()
    out = {}
    for s in ("BF", "T97"):
        games = sets[s]
        p = np.clip(z[f"{s}_proxy"], 1e-6, 1 - 1e-6)
        y = z[f"{s}_y"]
        X, cnt = [], []
        for g in games:
            s0, c0, n0, m0 = team_terms(st, g[3], g[4], g[1])
            s1, c1, n1, m1 = team_terms(st, g[4], g[3], g[1])
            X.append([a - b for a, b in zip(s0, s1)] + [a - b for a, b in zip(c0, c1)])
            cnt.append(n0 + m0)
        X = np.array(X)
        zlog = np.log(p / (1 - p))
        names = ["logit_proxy"] + [f"syn_{lo}-{hi}" for lo, hi in BINS] + \
                [f"ctr_{lo}-{hi}" for lo, hi in BINS]
        coef, se = fit_logistic(np.column_stack([zlog, X]), y)
        base, bse = fit_logistic(zlog[:, None], y)
        out[s] = {"n": len(y), "names": ["intercept"] + names,
                  "coef": coef.tolist(), "se": se.tolist(),
                  "calib_only": {"coef": base.tolist(), "se": bse.tolist()},
                  "mean_cells_per_team_by_bin": np.mean(cnt, 0).tolist()}
        print(s, {n: f"{c:+.4f}±{e:.4f}" for n, c, e in zip(["b0"] + names, coef, se)},
              "calib slope", round(base[1], 3), flush=True)
    json.dump(out, open(os.path.join(HERE, "results", "leak_probe_paper.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
