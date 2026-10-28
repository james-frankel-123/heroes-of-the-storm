"""
Part A.3 of P3_HERO_RECS_AND_REPLAYS: heroes a player is currently bad at.
Do the models predict which ones improve with practice and which stay bad?

Cells: at a cutoff t, player x hero cells with 20-299 games before t and a
raw record (mean residual r, pp) of -5pp or worse, that the player keeps
playing (20+ games in the 183 days after t). Outcome: mean r over games
11-50 after t (pp); games 1-10 are reported but are selected on (players keep
heroes that start well). Cutoffs and players as the pooled backtest
(held-out half before 2026-02-10, all pulled players at 2026-02-10).

Predictors, all as of t:
  record   raw mean r on the hero;
  main     full_ms posterior mean (pp) plus the experience offset;
  trend    the cell's last 10 games minus its earlier games (pp, raw);
  speed    the player's learning speed: over their OTHER heroes with 30+
           games before t, mean r in games 6-30 minus games 1-5,
           shrunk (k = 3 heroes) and centred on the population;
  early    the player's level on heroes new to them: mean r over first-5
           games on other heroes, shrunk (k = 10 games), centred.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import p3_hr_common as HC  # noqa: E402
import p3_hs_core as C  # noqa: E402
from p3_hr_backtest import POOL_CUTS, post_outcomes  # noqa: E402

H = 90


def per_cell_history(a, t, players):
    pos = np.full(int(a["n_players"]), -1, np.int64)
    pos[players] = np.arange(len(players))
    rows = np.flatnonzero((a["day"] < t) & (pos[a["pid"]] >= 0))
    n = len(players)
    cell = pos[a["pid"][rows]] * H + a["hero"][rows].astype(np.int64)
    o = np.lexsort((a["day"][rows], cell))
    rows, cell = rows[o], cell[o]
    brk = np.flatnonzero(np.r_[True, cell[1:] != cell[:-1]])
    ln = np.diff(np.r_[brk, len(rows)])
    rank = np.arange(len(rows)) - np.repeat(brk, ln)
    rev = np.repeat(ln, ln) - 1 - rank                   # 0 = most recent
    r = a["r"][rows].astype(np.float64) * 100
    bc = lambda m, w=None: np.bincount(cell[m], None if w is None else w[m], n * H).reshape(n, H)
    allm = np.ones(len(rows), bool)
    out = {"N": bc(allm), "Rsum": bc(allm, r),
           "e5s": bc(rank < 5, r), "e5c": bc(rank < 5), "m30s": bc((rank >= 5) & (rank < 30), r),
           "m30c": bc((rank >= 5) & (rank < 30)), "l10s": bc(rev < 10, r), "l10c": bc(rev < 10)}
    return out


def build(a, t, players, follow=183):
    hist = per_cell_history(a, t, players)
    st = HC.state_at(a, t, players)
    m, _ = HC.post_all(HC.params("full_ms"), st)
    # experience offset at slot-row counts (the model's own offset)
    pos = np.full(int(a["n_players"]), -1, np.int64)
    pos[players] = np.arange(len(players))
    sl = (a["slot"] >= 0) & (a["day"] < t) & (pos[a["pid"]] >= 0)
    cs = pos[a["pid"][sl]] * H + a["hero"][sl].astype(np.int64)
    nph = np.bincount(cs, None, len(players) * H).reshape(-1, H)
    npl = np.repeat(nph.sum(1, keepdims=True), H, 1)
    off = a["table"][C.exp_bins(npl.ravel().astype(np.int64), nph.ravel().astype(np.int64))].reshape(-1, H)
    pred = 100 * (m + off)
    out = post_outcomes(a, t, players, t + follow)
    N = hist["N"]
    rec = hist["Rsum"] / np.maximum(N, 1)
    # learning speed and early level from OTHER heroes (leave-own-hero-out)
    q = N >= 30
    imp = np.where(q, hist["m30s"] / np.maximum(hist["m30c"], 1) - hist["e5s"] / np.maximum(hist["e5c"], 1), 0.0)
    pop_imp = imp[q].mean()
    Simp = (np.where(q, imp - pop_imp, 0)).sum(1, keepdims=True) - np.where(q, imp - pop_imp, 0)
    Cimp = q.sum(1, keepdims=True) - q
    speed = Simp / (Cimp + 3.0)
    e5 = hist["e5s"].sum(1, keepdims=True) - hist["e5s"]
    e5c = hist["e5c"].sum(1, keepdims=True) - hist["e5c"]
    pop_e5 = hist["e5s"].sum() / hist["e5c"].sum()
    early = (e5 - pop_e5 * e5c) / (e5c + 10.0)
    earlier = (hist["Rsum"] - hist["l10s"]) / np.maximum(N - hist["l10c"], 1)
    trend = hist["l10s"] / np.maximum(hist["l10c"], 1) - earlier
    bad = (N >= 20) & (N < 300) & (rec <= -5)
    keep = bad & (out["n"] >= 20)
    pi, hi = np.nonzero(keep)
    R = {"pid": players[pi], "t": np.full(len(pi), t), "N": N[pi, hi], "record": rec[pi, hi], "main": pred[pi, hi],
         "trend": trend[pi, hi], "speed": speed[pi, hi], "early": early[pi, hi],
         "y30": out["b30_s"][pi, hi] / np.maximum(out["b30_c"][pi, hi], 1), "n_post": out["n"][pi, hi],
         "y_late": out["e2_s"][pi, hi] / np.maximum(out["e2_c"][pi, hi], 1),
         "y_early": out["e1_s"][pi, hi] / np.maximum(out["e1_c"][pi, hi], 1)}
    print(f"  cut {t}: bad cells {bad.sum():,}, kept playing (20+) {keep.sum():,}, "
          f"stopped (<5 after) {(bad & (out['n'] < 5)).sum():,}", flush=True)
    stop = {"bad": int(bad.sum()), "kept20": int(keep.sum()), "stopped_lt5": int((bad & (out["n"] < 5)).sum())}
    return R, stop


def ols(X, y, w):
    Xw = X * w[:, None]
    return np.linalg.solve(Xw.T @ X + 1e-9 * np.eye(X.shape[1]), Xw.T @ y)


def main():
    t0 = time.time()
    a = HC.load()
    import p3_rcf as R_
    fit = R_.fit_half(int(a["n_players"]))
    allp = np.sort(a["pulled_pid"]).astype(np.int64)
    parts, stops = [], {}
    for ds, who in POOL_CUTS:
        t = C.day_of(ds)
        players = allp if who == "all" else allp[~fit[allp]]
        r, s = build(a, t, players)
        parts.append(r)
        stops[ds] = s
    R = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    y = R["y_late"]          # games 11-50 after t: less selected than games 1-10
    up, pinv = np.unique(R["pid"], return_inverse=True)
    reps = 200
    boot = np.random.RandomState(4).poisson(1.0, (reps, len(up))).astype(np.float64)
    out = {"cells": int(len(y)), "players": int(len(up)), "stops": stops,
           "mean_record": float(R["record"].mean()), "mean_main": float(R["main"].mean()), "mean_realized": float(y.mean()), "mean_early_1_10": float(R["y_early"].mean()),
           "share_realized_ge0": float((y >= 0).mean())}
    # terciles of the model prediction
    qs = np.quantile(R["main"], [1 / 3, 2 / 3])
    terc = np.digitize(R["main"], qs)
    out["by_main_tercile"] = [{"pred": float(R["main"][terc == k].mean()), "record": float(R["record"][terc == k].mean()),
                               "realized": float(y[terc == k].mean()), "share_ge0": float((y[terc == k] >= 0).mean()),
                               "early_1_10": float(R["y_early"][terc == k].mean()), "cells": int((terc == k).sum())}
                              for k in range(3)]
    qr = np.quantile(R["record"], [1 / 3, 2 / 3])
    tr = np.digitize(R["record"], qr)
    out["by_record_tercile"] = [{"record": float(R["record"][tr == k].mean()), "pred": float(R["main"][tr == k].mean()),
                                 "realized": float(y[tr == k].mean()), "cells": int((tr == k).sum())} for k in range(3)]
    nb = np.digitize(R["N"], [40, 100])
    out["by_games"] = [{"games": g, "pred": float(R["main"][nb == k].mean()), "record": float(R["record"][nb == k].mean()),
                        "realized": float(y[nb == k].mean()), "cells": int((nb == k).sum())}
                       for k, g in enumerate(["20-39", "40-99", "100-299"])]

    def fitset(cols):
        X = np.column_stack([np.ones(len(y))] + [R[c] for c in cols])
        b = ols(X, y, np.ones(len(y)))
        bs = np.array([ols(X, y, boot[i][pinv]) for i in range(reps)])
        res = y - X @ b
        r2 = 1 - (res ** 2).mean() / ((y - y.mean()) ** 2).mean()
        return {"coef": dict(zip(["const"] + cols, b.tolist())), "se": dict(zip(["const"] + cols, bs.std(0, ddof=1).tolist())),
                "r2": float(r2)}
    noise = float(np.mean(2400.0 / (np.minimum(R["n_post"], 50) - 10)))
    out["outcome_var"] = float(y.var())
    out["noise_var_approx"] = noise
    out["models"] = {"record": fitset(["record"]), "main": fitset(["main"]), "main+record": fitset(["main", "record"]),
                     "main+speed+early+trend+logN": None}
    R["logN"] = np.log(R["N"])
    out["models"]["main+speed+early+trend+logN"] = fitset(["main", "speed", "early", "trend", "logN"])
    # does speed sort improvers? realized minus model prediction by speed tercile
    qsp = np.quantile(R["speed"], [1 / 3, 2 / 3])
    ts = np.digitize(R["speed"], qsp)
    out["by_speed_tercile"] = [{"speed": float(R["speed"][ts == k].mean()), "pred": float(R["main"][ts == k].mean()),
                                "realized": float(y[ts == k].mean()),
                                "realized_minus_pred": float((y - R["main"])[ts == k].mean()), "cells": int((ts == k).sum())}
                               for k in range(3)]
    HC.save_json("trajectories.json", out)
    np.savez(os.path.join(HC.HR_CACHE, "trajectories.npz"), **R)
    import json
    print(json.dumps({k: v for k, v in out.items() if k != "stops"}, indent=1, default=float)[:6000])
    print(f"[traj] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
