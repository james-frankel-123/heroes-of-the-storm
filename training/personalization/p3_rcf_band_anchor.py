"""
P3 anchored display-band check (review item C17): does the band shown for a
player's hero describe the games that follow it?

For every (player, hero) cell with at least K scored games inside a window
(V1, V2, OOT), the displayed estimate s and the display variance
D = s2 (1 + exp(-games on the hero / 10)) are frozen at the cell's first scored
game (the state is as of the day before that game). They are compared with
the mean realized residual r = y - WP of the cell's first K games (the anchor
game and the next K - 1):

  e = mean_K(r) - s,   noise = sum_K wp (1 - wp) / K^2   (binomial game noise)
  realized / claimed = sum(e^2 - noise) / sum(D)          (1 = calibrated band)
  coverage of the 80% band: |e| < 1.2816 sqrt(D + noise)

Player-bootstrap 95% CIs (cells of one player resampled together). By window
and by games on the hero at the anchor.

If the ratio is far from 1, a corrected band D' = s2 exp(b0 + b1 l + b2 l^2)
(l = log1p(games on the hero) / log1p(300)) is fit by maximum likelihood of
e ~ N(0, D' + noise) on V1 cells and checked on V2 and OOT.

Usage: python3 personalization/p3_rcf_band_anchor.py --arm chosen [--k 10]
BAND_NOISE=empirical replaces the binomial noise with the cell's own spread
over its K games (which also carries the teammates' and opponents' effects).
Output: $P3_RESULTS/rcf/band_anchor_<arm>_k<K>_<noise>.json
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
BINS = ((0, 0), (1, 9), (10, 29), (30, 99), (100, 299), (300, 10 ** 9))
WINS = ((1, "V1"), (2, "V2"), (3, "OOT"))


def anchored(A, win, k):
    """One row per cell with >= k scored games in the window."""
    s = np.flatnonzero(A["s_win"] == win)
    key = A["s_pid"][s].astype(np.int64) * 128 + A["s_hero"][s]
    o = np.lexsort((A["s_replay_id"][s], A["s_day"][s], key))
    s, key = s[o], key[o]
    u, start, cnt = np.unique(key, return_index=True, return_counts=True)
    keep = cnt >= k
    start = start[keep]
    first = s[start]
    take = s[start[:, None] + np.arange(k)[None, :]]           # (cells, k) the first k games
    r = A["s_r"][take].astype(np.float64)
    wp = A["s_wp"][take].astype(np.float64)
    nh = A["s_nh"][first].astype(np.float64)
    s2 = A["s_var"][first].astype(np.float64)
    noise = (wp * (1 - wp)).sum(1) / k ** 2
    if os.environ.get("BAND_NOISE") == "empirical":
        # the cell's own spread over its K games: binomial noise plus the
        # teammates' and opponents' effects, which also move a player's residual
        noise = r.var(1, ddof=1) / k
    return {"e": r.mean(1) - A["s_pred"][first].astype(np.float64),
            "noise": noise,
            "D": s2 * (1 + np.exp(-nh / 10.0)), "s2": s2, "nh": nh,
            "pid": A["s_pid"][first], "day": A["s_day"][first]}


def summarize(c, D, rng, nboot=300):
    ex = c["e"] ** 2 - c["noise"]
    z = np.abs(c["e"]) / np.sqrt(D + c["noise"])
    up, inv = np.unique(c["pid"], return_inverse=True)
    bs, bc = [], []
    for _ in range(nboot):
        w = np.bincount(rng.integers(0, len(up), len(up)), minlength=len(up))[inv].astype(float)
        bs.append(np.sum(w * ex) / np.sum(w * D))
        bc.append(np.sum(w * (z < Z80)) / np.sum(w))
    row = {"cells": int(len(ex)), "ratio": float(ex.sum() / D.sum()),
           "ratio_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
           "cover80": float((z < Z80).mean()),
           "cover80_ci": [float(np.percentile(bc, 2.5)), float(np.percentile(bc, 97.5))],
           "claimed_sd_pp_median": float(100 * np.median(np.sqrt(D))),
           "noise_sd_pp_median": float(100 * np.median(np.sqrt(c["noise"]))), "by_games": {}}
    for lo, hi in BINS:
        m = (c["nh"] >= lo) & (c["nh"] <= hi)
        if m.sum() < 30:
            continue
        bsm = []
        for _ in range(nboot):
            w = np.bincount(rng.integers(0, len(up), len(up)), minlength=len(up))[inv].astype(float)
            bsm.append(np.sum((w * ex)[m]) / np.sum((w * D)[m]))
        row["by_games"][f"{lo}-{hi}"] = {"cells": int(m.sum()), "ratio": float(ex[m].sum() / D[m].sum()),
                                         "ratio_ci": [float(np.percentile(bsm, 2.5)), float(np.percentile(bsm, 97.5))],
                                         "cover80": float((z[m] < Z80).mean()),
                                         "claimed_sd_pp": float(100 * np.median(np.sqrt(D[m])))}
    return row


def basis(nh):
    l = np.log1p(nh) / np.log1p(300.0)
    return np.column_stack([np.ones(len(l)), l, l * l])


FORMS = {
    # name: (number of parameters, D'(params, s2, nh))
    "scale": (1, lambda b, s2, nh: np.exp(b[0]) * s2 * (1 + np.exp(-nh / 10.0))),
    "scale_plus_floor": (2, lambda b, s2, nh: np.exp(b[0]) * s2 + np.exp(b[1])),
    "smooth": (3, lambda b, s2, nh: s2 * np.exp(basis(nh) @ b)),
    "smooth_plus_floor": (4, lambda b, s2, nh: s2 * np.exp(basis(nh) @ b[:3]) + np.exp(b[3])),
}


def fit_form(c, name):
    k, f = FORMS[name]

    def nll(b):
        v = f(b, c["s2"], c["nh"]) + c["noise"]
        return 0.5 * np.sum(np.log(v) + c["e"] ** 2 / v)
    x0 = np.zeros(k)
    if name in ("scale_plus_floor",):
        x0 = np.array([0.0, np.log(1e-4)])
    if name == "smooth_plus_floor":
        x0 = np.r_[np.zeros(3), np.log(1e-4)]
    r = minimize(nll, x0, method="Nelder-Mead", options={"maxiter": 20000, "xatol": 1e-6, "fatol": 1e-6})
    return r.x, float(r.fun)


def fit_band(c):
    """ML fit of D' = s2 exp(B b) on e ~ N(0, D' + noise)."""
    Bm = basis(c["nh"])

    def nll(b):
        v = c["s2"] * np.exp(np.clip(Bm @ b, -20, 20)) + c["noise"]
        return 0.5 * np.sum(np.log(v) + c["e"] ** 2 / v)
    r = minimize(nll, np.zeros(Bm.shape[1]), method="BFGS")
    return r.x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()
    z = np.load(os.path.join(R.RCF_RES, "p3c", f"{a.arm}__merged.npz"))
    A = {k: z[k] for k in z.files if k.startswith("s_")}
    rng = np.random.default_rng(0)
    out = {"arm": a.arm, "k": a.k, "noise": os.environ.get("BAND_NOISE", "binomial"), "target": f"mean residual of the cell's first {a.k} scored games in the window",
           "anchor": "the cell's first scored game in the window (state as of the day before)",
           "formula": "D = s2 (1 + exp(-games_on_hero / 10)); noise = sum wp (1 - wp) / K^2", "windows": {}}
    C = {wn: anchored(A, wk, a.k) for wk, wn in WINS}
    for wn, c in C.items():
        out["windows"][wn] = summarize(c, c["D"], rng)
        r = out["windows"][wn]
        print(f"{wn}: {r['cells']:,} cells, realized/claimed {r['ratio']:.3f} "
              f"[{r['ratio_ci'][0]:.3f}, {r['ratio_ci'][1]:.3f}], cover80 {r['cover80']:.3f}; by games "
              + ", ".join(f"{k} {v['ratio']:.2f}/{v['cover80']:.2f}" for k, v in r["by_games"].items()), flush=True)
    b = fit_band(C["V1"])
    out["corrected"] = {"form": "D' = s2 exp(b0 + b1 l + b2 l^2), l = log1p(games_on_hero) / log1p(300)",
                        "fit": "maximum likelihood on V1 cells", "b": b.tolist(),
                        "multiplier_by_games": {str(n): float(np.exp(basis(np.array([float(n)])) @ b)[0])
                                                for n in (0, 1, 3, 10, 30, 100, 300, 1000)},
                        "windows": {}}
    for wn, c in C.items():
        Dc = c["s2"] * np.exp(basis(c["nh"]) @ b)
        out["corrected"]["windows"][wn] = summarize(c, Dc, rng)
        r = out["corrected"]["windows"][wn]
        print(f"  corrected {wn}: realized/claimed {r['ratio']:.3f} [{r['ratio_ci'][0]:.3f}, {r['ratio_ci'][1]:.3f}], "
              f"cover80 {r['cover80']:.3f}; by games "
              + ", ".join(f"{k} {v['ratio']:.2f}/{v['cover80']:.2f}" for k, v in r["by_games"].items()), flush=True)
    print("corrected multiplier of s2 by games:", {k: round(v, 2) for k, v in out["corrected"]["multiplier_by_games"].items()})
    # alternative forms, fit on V1, judged on V2 and OOT by held-out Gaussian log-likelihood per cell
    out["forms"] = {}
    for name in FORMS:
        b_, f1 = fit_form(C["V1"], name)
        row = {"b": b_.tolist(), "v1_nll": f1, "held_out": {}}
        for wn in ("V2", "OOT"):
            c = C[wn]
            Dn = FORMS[name][1](b_, c["s2"], c["nh"])
            v = Dn + c["noise"]
            row["held_out"][wn] = {"nll_per_cell": float(0.5 * np.mean(np.log(v) + c["e"] ** 2 / v)),
                                   "summary": summarize(c, Dn, rng, nboot=100)}
        sd_by = {str(n): float(100 * np.sqrt(FORMS[name][1](b_, np.array([np.median(C["OOT"]["s2"][(C["OOT"]["nh"] >= lo) & (C["OOT"]["nh"] <= hi)])]),
                                                             np.array([float(n)]))[0]))
                 for n, (lo, hi) in zip((0, 5, 20, 60, 200, 500), BINS)}
        row["display_sd_pp_at_median_s2"] = sd_by
        out["forms"][name] = row
        print(f"  form {name}: V1 nll {f1:.1f}; held-out nll/cell V2 {row['held_out']['V2']['nll_per_cell']:.5f} "
              f"OOT {row['held_out']['OOT']['nll_per_cell']:.5f}; OOT ratio {row['held_out']['OOT']['summary']['ratio']:.2f} by games "
              + ", ".join(f"{k} {v['ratio']:.2f}" for k, v in row["held_out"]["OOT"]["summary"]["by_games"].items())
              + f"; display sd pp {dict((k, round(v, 1)) for k, v in sd_by.items())}", flush=True)
    base_v = {wn: float(0.5 * np.mean(np.log(C[wn]["D"] + C[wn]["noise"]) + C[wn]["e"] ** 2 / (C[wn]["D"] + C[wn]["noise"])))
              for wn in ("V2", "OOT")}
    out["current_band_nll_per_cell"] = base_v
    print(f"  current band: held-out nll/cell V2 {base_v['V2']:.5f} OOT {base_v['OOT']:.5f}", flush=True)
    with open(os.path.join(R.RCF_RES, f"band_anchor_{a.arm}_k{a.k}_{out['noise']}.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
