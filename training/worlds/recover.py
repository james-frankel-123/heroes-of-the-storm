"""
Recovery check (family a): take world w* as the truth, draw outcomes on the training games, refit the
whole world-model pipeline on them (base MAP at the frozen hyperparameters, Gauss-Newton Laplace,
triple layer at the fitted prior SD), and ask whether the refit posterior covers the truth on probe
games. Diagnostics are on predictions (the CP block is rotation-ambiguous), by support:
  probes  = NODRIFT human games (in support, mostly) + paper-1 tournament drafts (constrained_mcts and
            mcts teams, 63-78% below the floor)
  for each probe: truth logit z*, refit posterior mean and SD (300 refit-posterior draws), coverage of
  the central 50% / 90% intervals and the standardized error (z* - mean) / sd, by the game's lower min3.
-> site/results/check_recovery.json
"""
import os
import sys
import json
import time
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data as WD  # noqa: E402
import model as M  # noqa: E402
import world as W  # noqa: E402


def probes(n_human=20000, seed=3):
    import rescore as RS
    rng = np.random.default_rng(seed)
    Dn = WD.load("NODRIFT")
    i = rng.choice(len(Dn["y"]), n_human, replace=False)
    Dh = {k: Dn[k][i] for k in ("T0", "T1", "m", "t")}
    Dp = RS.p1_drafts()
    sel = np.flatnonzero(np.isin(Dp["S0"], ["constrained_mcts", "mcts"]) | np.isin(Dp["S1"], ["constrained_mcts", "mcts"]))
    sel = rng.choice(sel, min(20000, len(sel)), replace=False)
    Dd = {k: Dp[k][sel] for k in ("T0", "T1", "m", "t")}
    D = {k: np.concatenate([Dh[k], Dd[k]]) for k in Dh}
    src = np.r_[np.zeros(len(i)), np.ones(len(sel))]
    return D, src


def refit_one(wid, n_draws=300):
    import scipy.linalg as sl
    Ws = W.Worlds()
    D = WD.load("train")
    zt = Ws.logit("a", D, wid)
    y = (np.random.default_rng([int(wid), 21]).random(len(zt)) < M.sig(zt)).astype(float)
    Dy = dict(D, y=y)
    cfg = Ws.cfg
    t0 = time.time()
    B, th = M.fit_base(Dy, cfg["nm"], cfg["nt"], cfg["r"], cfg["sS"], cfg["sK"], cfg["sE"], init=Ws.theta, iters=1000)
    # GGN on the simulated labels: ggn_precision reads the stored training split, whose games are the same
    # (the labels do not enter J' diag(p(1-p)) J), so it is exact here
    Lam = M.ggn_precision(B, th, len(y), workers=8)
    Lam[np.diag_indices_from(Lam)] += B.prior_prec(cfg["sS"], cfg["sK"], cfg["sE"])
    L = sl.cholesky(Lam, lower=True)
    tri = M.load_layer("triple")
    lay = M.fit_layer("triple", Dy, B.logit(th, Dy), cfg["nm"], cfg["nt"], [tri["s"]])
    print(f"refit world {wid}: {time.time() - t0:.0f}s", flush=True)
    P, src = probes()
    z_true = Ws.logit("a", P, wid)
    m0, m1 = Ws.team_support("a", P)
    mn = np.minimum(m0, m1)
    Z = sl.solve_triangular(L, np.random.default_rng([wid, 77]).standard_normal((B.P, n_draws)), lower=True, trans="T")
    keys0, keys1 = M.team_keys("triple", P["T0"]), M.team_keys("triple", P["T1"])

    def look(K):
        i = np.clip(np.searchsorted(lay["keys"], K), 0, len(lay["keys"]) - 1)
        hit = lay["keys"][i] == K
        return np.where(hit, lay["mu"][i], 0.0), np.where(hit, lay["sd"][i], lay["s"])
    mu0, sd0 = look(keys0)
    mu1, sd1 = look(keys1)
    rng = np.random.default_rng([wid, 78])
    S = np.empty((n_draws, len(z_true)))
    for d in range(n_draws):
        zb = B.logit(th + Z[:, d], P)
        S[d] = zb + (mu0 + sd0 * rng.standard_normal(sd0.shape)).sum(1) - (mu1 + sd1 * rng.standard_normal(sd1.shape)).sum(1)
    mean, sd = S.mean(0), S.std(0)
    lo50, hi50 = np.percentile(S, [25, 75], axis=0)
    lo90, hi90 = np.percentile(S, [5, 95], axis=0)
    out = {"world": int(wid), "bins": []}
    for lo, hi in ((0, 1), (1, 5), (5, 25), (25, 100), (100, 10 ** 12)):
        for s_ in (0, 1):
            sel = (mn >= lo) & (mn < hi) & (src == s_)
            if sel.sum() < 50:
                continue
            e = (z_true[sel] - mean[sel]) / sd[sel]
            out["bins"].append({"min3_lo": lo, "min3_hi": hi if hi < 10 ** 12 else None, "probes": "human" if s_ == 0 else "search",
                                "n": int(sel.sum()), "cover50": float(((z_true >= lo50) & (z_true <= hi50))[sel].mean()),
                                "cover90": float(((z_true >= lo90) & (z_true <= hi90))[sel].mean()),
                                "z_mean": float(e.mean()), "z_sd": float(e.std()),
                                "post_sd_pts": float(25 * sd[sel].mean()),
                                "abs_err_pts": float(25 * np.abs(z_true[sel] - mean[sel]).mean())})
    return out


def main(wids):
    p = os.path.join(WD.RESULTS, "check_recovery.json")
    res = json.load(open(p)) if os.path.exists(p) else {"runs": []}
    done = {r["world"] for r in res["runs"]}
    for w in wids:
        if w in done:
            continue
        res["runs"].append(refit_one(w))
        json.dump(res, open(p, "w"), indent=1)
        print(json.dumps(res["runs"][-1], indent=1), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", default="900,901,902")
    a = ap.parse_args()
    main([int(x) for x in a.worlds.split(",")])
