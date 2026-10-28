"""
P3 display band for the chosen shrinkage model (results/P3_SHRINKAGE.md,
section "Display band").

The band shown next to a player's per-hero number is m +- 1.28 sqrt(D), with
D the display variance of the estimate. The model's own posterior variance
s2 (the state-space walk's Gaussian posterior at the query hero) does not
account for the shrinkage function, the direct metric channel, the layer or
the final calibration, so D is fit to forward errors:

    D = s2 * exp(g0 + g1 l + g2 sh + g3 g) + k0^2

l = log1p(games on the hero) / log1p(300), sh = hero share of the player's
games, g = log1p(days since the last game on the hero) / log1p(1000), all
as of the cell's first game in the window.

Cells: player x hero with 10+ slot games in V1 (fit), V2 or OOT
(test). Per cell, e = mean over its games of (r - prediction), where each
prediction is the model's as-of online prediction for that game; noise =
s_w^2 / n from the spread of the cell's own errors (BAND_NOISE=model uses sum wp (1 - wp) / n^2 with a fitted scale); claimed = mean of D over the cell's games.
Fit: maximum likelihood of e ~ N(0, claimed + noise) on V1 cells. Test: 80%
coverage of |e| < 1.28 sqrt(claimed + noise) on V2 and OOT cells, overall and
by games on the hero, hero share and recency; and the realized / claimed
variance ratio (mean e^2 - mean noise) / mean claimed.

Usage (from training/, study env remote/rcf_env.sh):
  python3 personalization/p3_rcf_band.py --arms LK2_full2,full_ms_layer
Output: $P3_RESULTS/rcf/band_<arm>.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.optimize import minimize

import p3_rcf as R

Z80 = 1.2815516


def feats(nh, npl, gap):
    l = np.log1p(nh) / np.log1p(300.0)
    sh = nh / np.maximum(npl, 1)
    g = np.log1p(np.maximum(gap, 0)) / np.log1p(1000.0)
    return np.column_stack([np.ones(len(l)), l, sh, g])


def cells(A, win, min_games):
    s = np.flatnonzero(A["s_win"] == win)
    key = A["s_pid"][s].astype(np.int64) * 128 + A["s_hero"][s]
    o = np.lexsort((A["s_day"][s], key))
    s, key = s[o], key[o]
    u, start, cnt = np.unique(key, return_index=True, return_counts=True)
    keep = cnt >= min_games
    inv = np.repeat(np.arange(len(u)), cnt)
    e_g = (A["s_r"][s] - A["s_pred"][s]).astype(np.float64)
    v_g = (A["s_wp"][s] * (1 - A["s_wp"][s])).astype(np.float64)
    n = cnt.astype(np.float64)
    e = np.bincount(inv, weights=e_g) / n
    if os.environ.get("BAND_NOISE", "empirical") == "empirical":
        # game noise of the cell mean from the cell's own spread: s_w^2 / n
        # (no assumption on the WP noise model; slightly conservative because
        # the online prediction moves a little within the cell)
        ss = np.bincount(inv, weights=e_g ** 2)
        sw2 = (ss - n * e ** 2) / np.maximum(n - 1, 1)
        noise = sw2 / n
    else:
        noise = np.bincount(inv, weights=v_g) / n ** 2
    first = s[start]
    return {"e": e[keep], "noise": noise[keep], "n": n[keep], "inv": inv, "keep": keep, "rows": s,
            "F": feats(A["s_nh"][first].astype(float), A["s_np"][first].astype(float), A["s_gap"][first].astype(float))[keep],
            "nh": A["s_nh"][first][keep], "sh": (A["s_nh"][first] / np.maximum(A["s_np"][first], 1))[keep],
            "gap": A["s_gap"][first][keep], "pid": A["s_pid"][first][keep]}


def claimed(A, c, gam, k0):
    """Mean over each cell's games of D (D uses each game's own features)."""
    rows = c["rows"]
    Fg = feats(A["s_nh"][rows].astype(float), A["s_np"][rows].astype(float), A["s_gap"][rows].astype(float))
    D = A["s_var"][rows].astype(np.float64) * np.exp(Fg @ gam) + k0
    n = np.bincount(c["inv"]).astype(np.float64)
    return (np.bincount(c["inv"], weights=D) / n)[c["keep"]]


def coverage(A, c, gam, k0, kap=1.0):
    cl = claimed(A, c, gam, k0)
    z = np.abs(c["e"]) / np.sqrt(cl + kap * c["noise"])
    out = {"cells": int(len(z)), "cover80": float((z < Z80).mean()),
           "var_ratio": float((np.mean(c["e"] ** 2) - kap * np.mean(c["noise"])) / np.mean(cl)),
           "median_display_sd_pp": float(100 * np.median(np.sqrt(cl)))}
    bins = {"games": (c["nh"], [(0, 0), (1, 9), (10, 29), (30, 99), (100, 299), (300, 10 ** 9)]),
            "share": (c["sh"], [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)]),
            "days_since": (c["gap"], [(-1, -1), (0, 6), (7, 29), (30, 179), (180, 364), (365, 10 ** 9)])}
    for nm, (x, bb) in bins.items():
        out[nm] = {}
        for lo, hi in bb:
            m = (x >= lo) & (x <= hi) if nm != "share" else (x >= lo) & (x < hi)
            if m.sum() < 100:
                continue
            out[nm][f"{lo}-{hi}"] = {"cells": int(m.sum()), "cover80": float((z[m] < Z80).mean()),
                                    "var_ratio": float((np.mean(c["e"][m] ** 2) - kap * np.mean(c["noise"][m])) / np.mean(cl[m])),
                                    "display_sd_pp": float(100 * np.median(np.sqrt(cl[m])))}
    return out


def fit(A, c, smooth=True, kap=None):
    """ML on V1 cells; kap (game-noise scale on wp (1 - wp)) is fit too when
    None (the WP noise model is known to be slightly off, P3-10)."""
    def unpack(x):
        gam = np.r_[x[:4]] if smooth else np.r_[x[0], 0, 0, 0]
        k = 4 if smooth else 1
        return gam, np.exp(x[k]), (np.exp(x[k + 1]) if kap is None else kap)

    def nll(x):
        gam, k0, kp = unpack(x)
        tv = claimed(A, c, gam, k0) + kp * c["noise"]
        return 0.5 * np.sum(np.log(tv) + c["e"] ** 2 / tv)
    x0 = (np.r_[0.0, 0, 0, 0] if smooth else np.r_[0.0])
    x0 = np.r_[x0, np.log(1e-5)] if kap is not None else np.r_[x0, np.log(1e-5), 0.0]
    r = minimize(nll, x0, method="Nelder-Mead", options={"maxiter": 6000, "xatol": 1e-4, "fatol": 1e-3})
    gam, k0, kp = unpack(r.x)
    return gam, float(k0), float(kp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True)
    a = ap.parse_args()
    for arm in a.arms.split(","):
        z = np.load(os.path.join(R.RCF_RES, "p3c", f"{arm}__merged.npz"))
        A = {k: z[k] for k in z.files if k.startswith("s_")}
        c1 = cells(A, 1, 10)
        out = {"arm": arm, "v1_cells": int(len(c1["e"]))}
        kfix = 1.0 if os.environ.get("BAND_NOISE", "empirical") == "empirical" else None
        g_c, k_c, kp_c = fit(A, c1, smooth=False, kap=kfix)
        g_s, k_s, kp_s = fit(A, c1, smooth=True, kap=kfix)
        raw = (np.zeros(4), 0.0, kp_s)
        out["noise_scale_kappa"] = kp_s
        out["fit_scalar"] = {"c": float(np.exp(g_c[0])), "k0_sd_pp": 100 * np.sqrt(k_c), "kappa": kp_c}
        out["fit_smooth"] = {"gamma": g_s.tolist(), "k0_sd_pp": 100 * np.sqrt(k_s), "kappa": kp_s,
                             "formula": "D = s2 * exp(g0 + g1 l + g2 sh + g3 g) + k0^2; l = log1p(games)/log1p(300), "
                                        "sh = hero share, g = log1p(days since last game)/log1p(1000)"}
        for wk, wn in ((2, "V2"), (3, "OOT")):
            c = cells(A, wk, 10)
            out[wn] = {"raw": coverage(A, c, *raw), "scalar": coverage(A, c, g_c, k_c, kp_c),
                       "smooth": coverage(A, c, g_s, k_s, kp_s)}
        p = os.path.join(R.RCF_RES, f"band_{arm}.json")
        with open(p, "w") as f:
            json.dump(out, f, indent=1)
        print(f"[{arm}] noise scale {kp_s:.3f}; scalar c {out['fit_scalar']['c']:.2f} k0 {out['fit_scalar']['k0_sd_pp']:.2f}pp; smooth gamma "
              f"{np.round(g_s, 3).tolist()} k0 {out['fit_smooth']['k0_sd_pp']:.2f}pp", flush=True)
        for wn in ("V2", "OOT"):
            for kind in ("raw", "scalar", "smooth"):
                r = out[wn][kind]
                print(f"  {wn} {kind:6s} cells {r['cells']:,} cover80 {r['cover80']:.3f} var_ratio {r['var_ratio']:.2f} "
                      f"sd {r['median_display_sd_pp']:.2f}pp | var ratio by games " + ", ".join(
                          f"{k} {v['var_ratio']:.2f}" for k, v in r["games"].items()), flush=True)


if __name__ == "__main__":
    main()
