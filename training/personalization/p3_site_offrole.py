"""
Refit the draft value's off-role weight b3 with the site's models
(docs/personalization-site-design.md, "Draft page"; P3_DRAFTER.md).

V = sigmoid(b0 + b1 logit WP + b2 D + b3 (O_0 - O_1)), where D is the team
difference of the model's per-slot predictions (the arm's p3c file) and O_t
counts team t's off-role players: the hero's fine role is under 10% of the
player's 50+ earlier games (lag-1 day slot counts, as p3_dr_core). Fit on V1
games, tested on V2 and OOT as every combiner in p3_c_tests.

Run on the 3090 with the full_1017 env:
  RCF_SCRIPT=p3_site_offrole.py bash personalization/remote/run_rcf_full.sh
Writes $P3_RESULTS/site/offrole_refit.json.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import p3_c_merged as M
import p3_hs_core as C
import p3_x_common as X


def main():
    d = X.load_ext()
    meta = C.hero_meta(d["hero_names"])
    fine = meta["fine"]
    nf = int(fine.max()) + 1
    pid = d["pid"].astype(np.int64)
    n_p = M.lag1_key_sums(d["day"], pid)
    n_f = M.lag1_key_sums(d["day"], pid * nf + fine[d["hero"]])
    off = (n_p >= 50) & (n_f < 0.10 * np.maximum(n_p, 1))
    n_games, y, wp0, cnt, gday = X.game_arrays(d)
    Od = X.team_diff(off.astype(np.float64), d["g"], d["team"], n_games)
    grid = np.zeros(n_games, np.int64)
    grid[d["g"]] = d["replay_id"]
    o = np.argsort(grid)
    res = os.environ["P3_RESULTS"]
    out = {"off_role_share_of_slots": float(off.mean())}
    for arm in ("full_ms", "full_imp2"):
        z = np.load(os.path.join(res, "rcf", "p3c", f"{arm}__merged.npz"))
        rid, win, yy, lo, D = (z[k] for k in ("g_replay_id", "g_win", "g_y", "g_lo", "g_D"))
        j = o[np.searchsorted(grid[o], rid)]
        assert (grid[j] == rid).all()
        od = Od[j]
        fit = win == 1
        w0 = C.fit_logistic(np.column_stack([lo, D])[fit], yy[fit])
        w1 = C.fit_logistic(np.column_stack([lo, D, od])[fit], yy[fit])
        p0 = C.predict(w0, np.column_stack([lo, D]))
        p1 = C.predict(w1, np.column_stack([lo, D, od]))
        r = {"coef_without": w0.tolist(), "coef_with": w1.tolist(), "b3": float(w1[3])}
        for k, nm in ((1, "V1"), (2, "V2"), (3, "OOT")):
            m = win == k
            dl = X.logloss(p0[m], yy[m]) - X.logloss(p1[m], yy[m])
            r[nm] = {"games": int(m.sum()), "gain_x1000": 1000 * float(dl.mean()),
                     "se": 1000 * float(dl.std(ddof=1) / np.sqrt(m.sum())),
                     "mean_abs_Odiff": float(np.abs(od[m]).mean())}
        out[arm] = r
        print(arm, json.dumps(r), flush=True)
    os.makedirs(os.path.join(res, "site"), exist_ok=True)
    with open(os.path.join(res, "site", "offrole_refit.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
