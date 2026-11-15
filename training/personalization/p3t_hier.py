"""
Paper 2: a hierarchical s(p, h) for the drafter's personal term (hidden-worlds diagnosis follow-up, 2026-11-15).

Estimator (worlds/player.py v2 structure, fit on real games only):
  u(p, h) = gamma[bin(n_ph)] + A_p + AP_{p, half-year} + R_{p, role} + E_{p, h} - ubar_h       (logit)
  logistic outcome model with logit WP_pop as offset, Gaussian priors, MAP by LBFGS, prior SDs by evidence.
The candidate personal term stacks it with the current drafter state:
  adj = w_s dS + w_u dU + w_o dO      (team differences; dU on the best assignment as the kernel does)

Stages (main box, CPU; from training/):
  calib     held-out check on the judge window (2026-07-01..09-11), v2 fit through 06-30 (worlds
            player_fit2.npz) against the drafter state as of 06-30 (worlds player_cov.npz j_s):
            slope of r = y - WP_pop on the personal term P by player group, and held-out log loss
            (weights fit on half A of the window, evaluated on half B)  -> results/p3t_hier/calib.json
"""
import os
import sys
import json
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING = os.path.dirname(HERE)
WORLDS = os.path.join(TRAINING, "worlds")
sys.path.insert(0, HERE)
sys.path.insert(0, TRAINING)
sys.path.insert(0, WORLDS)
RES = os.path.join(HERE, "results", "p3t_hier")
H = 90
HEADLINE = [0.0, 1.2158510866657697, 4.242860855417209, -0.031338977228257625]
PSCALE = 0.864


def sig(x):
    return 1 / (1 + np.exp(-x))


def asof_counts(pid, hero, nprior, jw_rows):
    """n_ph as of the window start: the game-time count minus the player's earlier window games on the hero."""
    p, h, n = pid[jw_rows].ravel(), hero[jw_rows].ravel(), nprior[jw_rows].ravel()
    key = p * H + h
    o = np.lexsort((n, key))
    k = key[o]
    first = np.r_[True, k[1:] != k[:-1]]
    start = np.maximum.accumulate(np.where(first, np.arange(len(k)), 0))
    rank = np.empty(len(k), np.int64)
    rank[o] = np.arange(len(k)) - start
    return (n - rank).reshape(-1, 10)


def calib(n_boot=200):
    import player as PL
    src = np.load(PL.SRC)
    cov = np.load(PL.COV)
    jw = np.flatnonzero(src["jw"])
    pid, hero = src["pid"][jw], src["hero"][jw]
    n0 = asof_counts(src["pid"], src["hero"], src["nprior"], jw)
    y, wp = src["y0"][jw], np.clip(src["wp0"][jw], 1e-6, 1 - 1e-6)
    lw = np.log(wp / (1 - wp))
    G = len(jw)
    # v2 mean u' (fit through 06-30; period = 06-30's half-year), as-of counts
    U = PL.PlayerWorlds2().u(pid, hero, n0, 0, mean_only=True)
    S, O = cov["j_s"].astype(float), cov["j_off"].astype(float)
    known = cov["j_known"]
    # player total games before the window (as of 06-30): sum over heroes is not in src; use the
    # slot-level experience proxy = the player's games on all heroes before J0
    allp = src["pid"][src["fit"]].ravel()
    up, cnt = np.unique(allp, return_counts=True)
    j = np.clip(np.searchsorted(up, pid), 0, len(up) - 1)
    ntot = np.where(up[j] == pid, cnt[j], 0)
    sg = np.r_[np.ones(5), -np.ones(5)]
    dS, dO, dU = S @ sg, O @ sg, U @ sg
    half = np.arange(G) % 2                                   # half A = even games, B = odd
    b0, b1, b2, b3 = HEADLINE
    terms = {
        "cur_headline": b2 * dS + b3 * dO,
        "cur_pcal": PSCALE * (b2 * dS + b3 * dO),
        "v2_natural": dU,
    }
    # weights on half A: logistic with offset b1 * lw
    fits = {}
    for name, X in (("cur_refit", np.column_stack([dS, dO])), ("v2_refit", np.column_stack([dU])),
                    ("stack", np.column_stack([dS, dO, dU]))):
        w = _logit_fit(X[half == 0], y[half == 0], b1 * lw[half == 0])
        fits[name] = w.tolist()
        terms[name] = X @ w
    out = {"games": int(G), "half_B_games": int((half == 1).sum()), "weights_halfA": fits,
           "note": "P = sig(b1 lw + adj) - sig(b1 lw) per team; slope of r = y - WP_pop on P (1 = calibrated). "
                   "Held-out: half B log loss and its gain over WP_pop alone (b1 lw offset), x1e4 per game."}
    base_ll = _ll(y[half == 1], b1 * lw[half == 1])
    out["holdout_logloss"] = {"wp_only": base_ll}
    for k, a in terms.items():
        out["holdout_logloss"][k] = {"logloss": _ll(y[half == 1], b1 * lw[half == 1] + a[half == 1]),
                                     "gain_1e4": 1e4 * (base_ll - _ll(y[half == 1], b1 * lw[half == 1] + a[half == 1]))}
    # slopes on half B, team rows, by group
    team_mean_logn = np.stack([np.log1p(ntot[:, :5]).mean(1), np.log1p(ntot[:, 5:]).mean(1)], 1)
    q = np.quantile(team_mean_logn, [0.25, 0.5, 0.75])
    team_known = np.stack([known[:, :5].all(1), known[:, 5:].all(1)], 1)
    min3 = src["j_min3"]
    groups = {"all": np.ones((G, 2), bool), "all_known_team": team_known}
    for i, (lo, hi) in enumerate(zip(np.r_[-np.inf, q], np.r_[q, np.inf])):
        groups[f"exp_q{i + 1}"] = (team_mean_logn > lo) & (team_mean_logn <= hi)
    for lo, hi, lab in ((0, 24, "min3_0-24"), (25, 99, "min3_25-99"), (100, 499, "min3_100-499"), (500, 10 ** 12, "min3_500+")):
        groups[lab] = (min3 >= lo) & (min3 <= hi)
    rng = np.random.default_rng(0)
    gB = np.flatnonzero(half == 1)
    boots = [rng.choice(gB, len(gB)) for _ in range(n_boot)]
    out["slopes_halfB"] = {}
    for k, a in terms.items():
        P0 = sig(b1 * lw + a) - sig(b1 * lw)
        P1 = sig(-b1 * lw - a) - sig(-b1 * lw)
        P = np.stack([P0, P1], 1)
        r = np.stack([y - wp, (1 - y) - (1 - wp)], 1)
        o = {}
        for gname, gm in groups.items():
            est = _slope(P[gB], r[gB], gm[gB])
            bs = [_slope(P[b], r[b], gm[b]) for b in boots]
            o[gname] = {"slope": est, "se": float(np.std(bs)), "n_team_rows": int(gm[gB].sum()),
                        "sd_P_pts": float(100 * P[gB][gm[gB]].std())}
        out["slopes_halfB"][k] = o
    os.makedirs(RES, exist_ok=True)
    json.dump(out, open(os.path.join(RES, "calib.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


def _slope(P, r, m):
    x, z = P[m], r[m]
    x = x - x.mean()
    return float((x * (z - z.mean())).sum() / max((x * x).sum(), 1e-12))


def _ll(y, z):
    p = np.clip(sig(z), 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _logit_fit(X, y, off, iters=50):
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = sig(off + X @ w)
        g = X.T @ (y - p)
        Hm = (X * (p * (1 - p))[:, None]).T @ X
        step = np.linalg.solve(Hm + 1e-9 * np.eye(len(w)), g)
        w += step
        if np.abs(step).max() < 1e-10:
            break
    return w


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["calib"])
    a = ap.parse_args()
    calib()
