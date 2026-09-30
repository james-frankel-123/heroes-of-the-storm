"""
P3 hero strength, stage 3: the Cassia test (masked cold start).

Players: the held-out half from p3_hs_fit (never used to fit kernels or the
co-strength matrix), with >= 300 E games. Target cells: heroes the player
rarely plays, 10 <= n <= 60 E games on the hero and <= 5% of their games.
For each target cell all of that cell's games are hidden; the player's
other heroes are kept. Each kernel predicts the hidden cell (posterior at a
hero with no data) and is scored against the cell's observed mean
(detrended) residual:

  latent R^2   1 - (MSE - noise) / (E[rbar^2] - noise), noise = sum v / n^2
               (share of the true cell-effect variance explained)
  cover80      80% predictive interval m +- 1.28 sqrt(s^2 + noise)
  var_ratio    realized latent error variance / claimed posterior variance

A second pass hides only the cell's first k games in time order and scores
the prediction of the remaining games after adding back the first k
(k = 0, 1, 3, 5, 10): how fast a new hero's estimate becomes useful.

Usage (from training/): python3 personalization/p3_hs_cassia.py
Output: results/p3_hs_cassia.json
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import p3_hs_core as C
from p3_hs_fit import KERNELS, KOUT, main_kernels, prepare

Z80 = 1.2815516


def score(rbar, noise, m, s2):
    err2 = (rbar - m) ** 2
    tot = (rbar ** 2).mean() - noise.mean()
    z = np.abs(rbar - m) / np.sqrt(s2 + noise)
    return {"cells": int(len(rbar)),
            "latent_R2": float(1 - (err2.mean() - noise.mean()) / tot),
            "rmse_latent_pp": float(100 * np.sqrt(max(err2.mean() - noise.mean(), 0))),
            "claimed_sd_pp": float(100 * np.sqrt(s2.mean())),
            "cover80": float((z < Z80).mean()),
            "var_ratio": float((err2.mean() - noise.mean()) / s2.mean())}


def main():
    d = C.load_slots()
    MAIN, BEST = main_kernels()
    H = len(d["hero_names"])
    meta = C.hero_meta(d["hero_names"])
    kz = np.load(KOUT)
    bases = {k[6:]: kz[k] for k in kz.files if k.startswith("basis_")}
    Ks = {n: C.kernel_from([bases[b] for b in bl], kz["theta_" + n]) for n, bl in KERNELS.items()}
    e_mask, n_p, n_ph, table, r_adj = prepare(d)
    S, P, N, _, _ = C.static_state(d, e_mask, d["n_players"], H, r_adj)
    hold = ~kz["fit_players"]
    tot = N.sum(1)
    deep = hold & (tot >= 300)
    pi, hi = np.nonzero((N >= 10) & (N <= 60) & (N <= 0.05 * tot[:, None]) & deep[:, None])
    print(f"deep held-out players {deep.sum():,}; target cells {len(pi):,}")
    # cell means from E rows
    key = d["pid"] * H + d["hero"]
    tk = pi * H + hi
    sel = e_mask & np.isin(key, tk)
    order = np.argsort(tk)
    tk_s = tk[order]
    ci = np.searchsorted(tk_s, key[sel])
    rs = np.bincount(ci, weights=r_adj[sel], minlength=len(tk))
    vs = np.bincount(ci, weights=d["v"][sel], minlength=len(tk))
    ns = np.bincount(ci, minlength=len(tk))
    rbar = np.empty(len(tk))
    noise = np.empty(len(tk))
    rbar[order] = rs / ns
    noise[order] = vs / ns ** 2
    Sm = S[pi].copy()
    Pm = P[pi].copy()
    Sm[np.arange(len(pi)), hi] = 0
    Pm[np.arange(len(pi)), hi] = 0
    out = {"cells": int(len(pi)), "players": int(deep.sum()), "full_mask": {}, "by_role": {},
           "first_k": {}}
    roles = meta["blizz"][hi]
    for name, K in Ks.items():
        m, s2 = C.gp_query(Sm, Pm, hi, K)
        out["full_mask"][name] = score(rbar, noise, m, s2)
        out["by_role"][name] = {C.BLIZZ_ROLES[r]: score(rbar[roles == r], noise[roles == r],
                                                        m[roles == r], s2[roles == r])
                                for r in range(6) if (roles == r).sum() > 200}
        a = out["full_mask"][name]
        print(f"  {name:26s} R2 {a['latent_R2']:+.3f} rmse {a['rmse_latent_pp']:.2f}pp "
              f"claimed sd {a['claimed_sd_pp']:.2f} cover80 {a['cover80']:.3f} "
              f"var_ratio {a['var_ratio']:.2f}", flush=True)
    zero = np.zeros(len(rbar))
    out["full_mask"]["zero (population)"] = score(rbar, noise, zero, np.full(len(rbar), 1e-12))
    # raw player mean residual (unshrunk, across the player's other heroes)
    pm = Sm.sum(1) / Pm.sum(1)
    out["full_mask"]["raw player mean"] = score(rbar, noise, pm, np.full(len(rbar), 1e-12))

    # first-k pass: reveal the first k games of the cell, predict the rest
    rows = np.flatnonzero(sel)
    rows = rows[np.lexsort((d["replay_id"][rows], d["day"][rows], key[rows]))]
    kr = key[rows]
    brk = np.flatnonzero(np.r_[True, kr[1:] != kr[:-1]])
    rank = np.arange(len(rows)) - np.repeat(brk, np.diff(np.r_[brk, len(rows)]))
    cell_of_row = np.empty(len(rows), np.int64)
    cell_of_row = np.searchsorted(tk_s, kr)
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))  # sorted position -> original target index
    tgt_of_row = order[cell_of_row]
    for k in (0, 1, 3, 5, 10):
        early = rank < k
        late = ~early
        Sk, Pk = Sm.copy(), Pm.copy()
        rr = rows[early]
        np.add.at(Sk, (tgt_of_row[early], hi[tgt_of_row[early]]), r_adj[rr] / d["v"][rr])
        np.add.at(Pk, (tgt_of_row[early], hi[tgt_of_row[early]]), 1.0 / d["v"][rr])
        rl = rows[late]
        tl = tgt_of_row[late]
        nl = np.bincount(tl, minlength=len(tk))
        rb = np.bincount(tl, weights=r_adj[rl], minlength=len(tk)) / np.maximum(nl, 1)
        nz = np.bincount(tl, weights=d["v"][rl], minlength=len(tk)) / np.maximum(nl, 1) ** 2
        ok = nl >= 5
        out["first_k"][k] = {}
        for name in MAIN:
            m, s2 = C.gp_query(Sk[ok], Pk[ok], hi[ok], Ks[name])
            out["first_k"][k][name] = score(rb[ok], nz[ok], m, s2)
        # raw mean of the revealed games (the naive "your WR so far")
        if k > 0:
            rawm = np.where(Pk[ok, hi[ok]] > 0, Sk[ok, hi[ok]] / np.maximum(Pk[ok, hi[ok]], 1e-9), 0)
            out["first_k"][k]["raw revealed mean"] = score(rb[ok], nz[ok], rawm,
                                                           np.full(ok.sum(), 1e-12))
        print(f"  first {k}: " + ", ".join(
            f"{n} R2 {v['latent_R2']:+.3f}" for n, v in out["first_k"][k].items()), flush=True)
    with open(os.path.join(C.RESULTS, "p3_hs_cassia.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
