"""
P3 phase 2: does same-day information help? (session / form effects)

The drift fit finds a fast player component (half-life of days or less).
With predictions frozen at the previous day (the phase-1 and p3_sd_eval
protocol) it cannot help. A live product also knows the player's earlier
games today. Here predictions use every earlier game including same-day
ones (sequential Kalman predictive; the order within a day is replay_id
order). Compared: static kernel (no drift) vs drift model (random-sample
fit), both at lag 0 and lag 1 day, same combiner protocol (fit V1, test
V2), plus the slot-level gain by position in the day's session.

Usage (from training/): python3 personalization/p3_sd_session.py
Output: results/p3_sd_session.json
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import p3_hs_core as C
import p3_sd_kalman as K
from p3_hs_fit import prepare
from p3_hs_eval import team_diff, slot_stats


def main():
    d = C.load_slots()
    meta = C.hero_meta(d["hero_names"])
    kz = np.load(K.KOUT)
    e_mask, n_p, n_ph, table, r_adj = prepare(d)
    V = kz["V_cf2"]
    with open(K.PARAMS) as f:
        prm = json.load(f)
    mu = table[C.exp_bins(n_p, n_ph)]
    days = d["day"]
    post = ~d["in_sample"]
    _, first_row = np.unique(d["g"][post], return_index=True)
    med = np.median(days[post][first_row])
    v1r, v2r = post & (days < med), post & (days >= med)
    filt = K.Filter(d, meta, V, r_adj)
    day_real = filt.day.copy()
    preds = {}
    for lagname in ("lag 1 day", "lag 0 (same-day games used)"):
        if lagname.startswith("lag 0"):
            filt.day = np.arange(len(filt.rows), dtype=np.int64)  # every row its own block
        else:
            filt.day = day_real
        for mname, key in (("static", "static (phase-1 kernel)"), ("drift", "drift days")):
            sd, lam, br = K.split(prm[key])
            m, s2, _ = filt.run(sd, filt.lam_state(lam, br))
            pm = np.empty(len(d["pid"]))
            pm[filt.rows] = m
            preds[f"{mname}, {lagname}"] = mu + pm
            print(mname, lagname, flush=True)
    # position within the player's day (0 = first game of the day)
    o = np.lexsort((d["replay_id"], d["day"], d["pid"]))
    key = d["pid"][o] * 100000 + d["day"][o]
    brk = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    pos_s = np.arange(len(o)) - np.repeat(brk, np.diff(np.r_[brk, len(o)]))
    pos = np.empty(len(o), np.int64)
    pos[o] = pos_s
    out = {"slot": {}, "game": {}}
    for nm, pr in preds.items():
        res = {}
        for lab, sel in (("all", v2r), ("first game of day", v2r & (pos == 0)),
                         ("games 2-3 of day", v2r & (pos >= 1) & (pos <= 2)),
                         ("game 4+ of day", v2r & (pos >= 3))):
            res[lab] = slot_stats(d["r"][sel], d["wp"][sel], pr[sel])
        out["slot"][nm] = res
        print(nm, {k: round(v["ll_gain_x1000"], 3) for k, v in res.items()}, flush=True)
    out["share_by_position"] = {"first": float((pos[v2r] == 0).mean()),
                                "2-3": float(((pos[v2r] >= 1) & (pos[v2r] <= 2)).mean()),
                                "4+": float((pos[v2r] >= 3).mean())}
    n_games = int(d["g"].max()) + 1
    y = np.zeros(n_games)
    wg = np.full(n_games, 0.5)
    t0 = d["team"] == 0
    y[d["g"][t0]] = d["y"][t0]
    wg[d["g"][t0]] = d["wp"][t0]
    lo = np.log(wg / (1 - wg))
    full = np.bincount(d["g"][post], minlength=n_games) == 10
    gv1 = np.zeros(n_games, bool)
    gv1[d["g"][v1r]] = True
    gv2 = np.zeros(n_games, bool)
    gv2[d["g"][v2r]] = True
    fit_m, test_m = gv1 & full, gv2 & full
    rng = np.random.RandomState(0)
    idx = np.flatnonzero(test_m)
    boots = [rng.choice(idx, len(idx)) for _ in range(200)]
    lls = {}
    X0 = lo[:, None]
    w0 = C.fit_logistic(X0[fit_m], y[fit_m])
    p0 = C.predict(w0, X0)
    ll0 = -(y * np.log(p0) + (1 - y) * np.log(1 - p0))
    for nm, pr in preds.items():
        X = np.column_stack([lo, team_diff(pr, d["g"], d["team"], n_games)])
        w = C.fit_logistic(X[fit_m], y[fit_m])
        p = C.predict(w, X)
        lls[nm] = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        m = C.game_metrics(p[test_m], y[test_m])
        dd = ll0 - lls[nm]
        m["gain_vs_M0"] = [float(dd[idx].mean())] + [float(np.percentile([dd[b].mean() for b in boots], q))
                                                     for q in (2.5, 97.5)]
        out["game"][nm] = m
    ref = "static, lag 1 day"
    for nm in preds:
        dd = lls[ref] - lls[nm]
        out["game"][nm]["gain_vs_static_lag1"] = [float(dd[idx].mean())] + [
            float(np.percentile([dd[b].mean() for b in boots], q)) for q in (2.5, 97.5)]
        g = out["game"][nm]
        print(f"{nm:40s} acc {g['acc']:.4f} ll {g['logloss']:.5f} gain {g['gain_vs_M0'][0]:+.5f} "
              f"vs static lag1 {g['gain_vs_static_lag1'][0]:+.5f} "
              f"[{g['gain_vs_static_lag1'][1]:+.5f},{g['gain_vs_static_lag1'][2]:+.5f}]", flush=True)
    dd = lls["static, lag 0 (same-day games used)"] - lls["drift, lag 0 (same-day games used)"]
    out["drift_vs_static_lag0"] = [float(dd[idx].mean())] + [
        float(np.percentile([dd[b].mean() for b in boots], q)) for q in (2.5, 97.5)]
    print("drift vs static, both lag 0:", out["drift_vs_static_lag0"], flush=True)
    with open(os.path.join(C.RESULTS, "p3_sd_session.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
