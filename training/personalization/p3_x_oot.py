"""
P3 extensions, task 1: out-of-time validation of the phase-1 skill model.

Frozen as of the snapshot (nothing refit on post-snapshot data):
  - experience table (fit on E), "+CF rank 2" kernel and the nested
    baselines (fit on E), and the game-level combiner weights (fit on V1,
    exactly as in p3_hs_eval: logistic on logit WP + team difference of the
    summed slot estimates).
Scored on every post-snapshot game (day > 2026-05-22) with all ten player
rows: the rest of build 2.55.16.97039 and all of 2.55.17 (97605, 97650,
97771, 98025), causal WP from p3_x_wp_post.py.

State variants for the post games:
  online      per-player state from every game (snapshot + post) with
              day <= game day - 1: the nightly-updated product with frozen
              hyperparameters
  frozen@May  state and counts frozen at 2026-05-22 (snapshot games only):
              the model exactly as fit in May, never updated
  frozen@May+counts  frozen residual state, experience counts kept current
              (counts carry no outcomes)
  experience only  offset from current counts, no residual skill
Reproduction check: the same code on V2 with snapshot-only state must give
the phase-1 +0.0142.

Usage (from training/): python3 personalization/p3_x_oot.py
Outputs: results/p3_x_oot.json, results/p3_x_oot.txt
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import p3_hs_core as C
import p3_x_common as X

N_EDGES = [0, 1, 3, 6, 11, 21, 51, 101, 201, 10 ** 9]
N_LABELS = ["0", "1-2", "3-5", "6-10", "11-20", "21-50", "51-100", "101-200", "201+"]
KNAMES = ["player", "player+hero", "+CF rank 2"]
BEST = "+CF rank 2"


def slot_gain(r, wp, pred):
    p1 = np.clip(wp + pred, 1e-3, 1 - 1e-3)
    y = r + wp
    ll0 = -(y * np.log(wp) + (1 - y) * np.log(1 - wp))
    ll1 = -(y * np.log(p1) + (1 - y) * np.log(1 - p1))
    out = {"n": int(len(r)), "ll_gain_x1000": float(1000 * (ll0 - ll1).mean())}
    if pred.std() > 1e-9:
        out["slope"] = float(np.cov(r, pred)[0, 1] / pred.var())
    return out


def main():
    t0 = time.time()
    d = X.load_ext()
    Ks, table = X.kernels()
    H = len(d["hero_names"])
    n_p, n_ph = C.experience_counts(d)
    r_adj = d["r"] - table[C.exp_bins(n_p, n_ph)]
    day = d["day"]
    post = d["post"]
    snap_post = (~post) & (~d["in_sample"])
    _, first_row = np.unique(d["g"][snap_post], return_index=True)
    med = np.median(day[snap_post][first_row])
    v1, v2 = snap_post & (day < med), snap_post & (day >= med)
    oot = post & (day > X.SNAP_LAST_DAY)
    n_games, y, wp0, cnt, gday = X.game_arrays(d)
    lo = np.log(wp0 / (1 - wp0))
    gv = {}
    for nm, m in (("V1", v1), ("V2", v2), ("OOT", oot)):
        gm = np.zeros(n_games, bool)
        gm[d["g"][m]] = True
        gv[nm] = gm & (cnt == 10)
    out = {"windows": {k: int(v.sum()) for k, v in gv.items()}, "split_day": float(med)}
    print(out["windows"], flush=True)

    extra = [Ks[k] for k in KNAMES if k != BEST]
    preds = {}   # name -> per-row prediction (NaN outside queries)

    def put(name, qidx, val):
        a = preds.setdefault(name, np.full(len(day), np.nan))
        a[qidx] = val

    # A. snapshot protocol (state from snapshot rows only) for V1, V2
    q = v1 | v2
    qi, m, v, Nh, Np, ex = X.online_predict(d, ~post, q, Ks[BEST], r_adj, extra=extra)
    mu = table[C.exp_bins(Np.astype(np.int64), Nh.astype(np.int64))]
    put("experience only", qi, mu)
    put(BEST, qi, mu + m)
    for k, e in zip([k for k in KNAMES if k != BEST], ex):
        put(k, qi, mu + e)
    nh_all = np.full(len(day), -1.0)
    nh_all[qi] = Nh
    print(f"snapshot V1/V2 states: {time.time() - t0:.0f}s", flush=True)

    # B. post games, online (all rows, lag 1 day)
    qi, m, v, Nh, Np, ex = X.online_predict(d, np.ones(len(day), bool), oot, Ks[BEST], r_adj,
                                            extra=extra)
    mu_on = table[C.exp_bins(Np.astype(np.int64), Nh.astype(np.int64))]
    put("experience only", qi, mu_on)
    put(BEST, qi, mu_on + m)
    for k, e in zip([k for k in KNAMES if k != BEST], ex):
        put(k, qi, mu_on + e)
    nh_all[qi] = Nh
    sd_on = np.sqrt(v)
    print(f"OOT online: {time.time() - t0:.0f}s", flush=True)
    # C. frozen at May (snapshot rows only, state and counts at the snapshot end)
    qi2, m2, v2_, Nh2, Np2, ex2 = X.online_predict(d, ~post, oot, Ks[BEST], r_adj,
                                                   freeze_day=X.SNAP_LAST_DAY, extra=extra)
    assert np.array_equal(qi, qi2)
    mu_fr = table[C.exp_bins(Np2.astype(np.int64), Nh2.astype(np.int64))]
    fz = {"experience only": mu_fr, BEST: mu_fr + m2}
    for k, e in zip([k for k in KNAMES if k != BEST], ex2):
        fz[k] = mu_fr + e
    fz_counts = {BEST: mu_on + m2}
    print(f"OOT frozen: {time.time() - t0:.0f}s", flush=True)

    # ------------------------------------------------ game level
    g_all, team_all = d["g"], d["team"]

    def D_of(pr_rows):
        ok = ~np.isnan(pr_rows)
        return X.team_diff(np.where(ok, pr_rows, 0.0), g_all, team_all, n_games)

    base = {"M0 population WP": None}
    coefs = {}
    ll = {}
    w0 = C.fit_logistic(lo[gv["V1"]][:, None], y[gv["V1"]])
    p0 = C.predict(w0, lo[:, None])
    ll0 = X.logloss(p0, y)
    out["game"] = {}
    specs = {}
    for name in preds:
        Dv = D_of(preds[name])
        w = C.fit_logistic(np.column_stack([lo, Dv])[gv["V1"]], y[gv["V1"]])
        coefs[name] = w
        specs[name] = Dv
    # frozen variants: same V1 combiner, post-period D from frozen predictions
    for name, arr in list(fz.items()) + [(f"{BEST} (frozen@May, counts current)", fz_counts[BEST])]:
        rows = np.full(len(day), np.nan)
        rows[qi] = arr
        key = f"{name} [frozen@May]" if "counts" not in name else name
        base_name = name.split(" (")[0]
        specs[key] = D_of(rows)
        coefs[key] = coefs[base_name]

    periods = {"V2 (reproduction)": gv["V2"], "OOT all": gv["OOT"]}
    # by month and by build
    months = [("2026-05-23", "2026-06-30"), ("2026-07-01", "2026-07-31"),
              ("2026-08-01", "2026-08-31"), ("2026-09-01", "2026-09-27")]
    for a, b in months:
        periods[f"OOT {a[:7]}"] = gv["OOT"] & (gday >= C.day_of(a)) & (gday <= C.day_of(b))
    gver = np.full(n_games, -1)
    gver[d["g"]] = d["version"]
    vers = [str(x) for x in d["versions"]]
    for vi, vname in enumerate(vers):
        m_ = gv["OOT"] & (gver == vi)
        if m_.sum() > 1000:
            periods[f"OOT build {vname}"] = m_
    periods["OOT 2.55.16.97039"] = gv["OOT"] & (gver == vers.index("2.55.16.97039"))
    periods["OOT 2.55.17.*"] = gv["OOT"] & np.isin(gver, [i for i, v_ in enumerate(vers)
                                                          if v_.startswith("2.55.17")])
    out["periods"] = {k: int(v_.sum()) for k, v_ in periods.items()}
    for name, Dv in specs.items():
        w = coefs[name]
        p = C.predict(w, np.column_stack([lo, Dv]))
        l1 = X.logloss(p, y)
        res = {"coef": w.tolist()}
        for pn, pm in periods.items():
            if "frozen" in name and not pn.startswith("OOT"):
                continue
            idx = np.flatnonzero(pm)
            gain, ci = X.boot_gain(ll0, l1, idx, n=200 if pn in ("OOT all", "V2 (reproduction)") else 100)
            mt = C.game_metrics(p[pm], y[pm])
            res[pn] = {"gain": gain, "ci": ci, "acc": mt["acc"], "logloss": mt["logloss"],
                       "ece": mt["ece"], "cal_slope": mt["cal_slope"]}
        out["game"][name] = res
        a = res.get("OOT all", {})
        rv = res.get("V2 (reproduction)", {})
        print(f"  {name:48s} V2 {rv.get('gain', float('nan')):+.5f}  OOT {a['gain']:+.5f} "
              f"[{a['ci'][0]:+.5f},{a['ci'][1]:+.5f}] acc {a['acc']:.4f} slope {a['cal_slope']:.3f}",
              flush=True)
    m0 = {}
    for pn, pm in periods.items():
        mt = C.game_metrics(p0[pm], y[pm])
        m0[pn] = {"acc": mt["acc"], "logloss": mt["logloss"], "ece": mt["ece"],
                  "cal_slope": mt["cal_slope"], "games": int(pm.sum())}
    out["M0"] = m0

    # refit check: combiner refit on OOT (is the V1 weight still right?)
    Dbest = specs[BEST]
    w_oot = C.fit_logistic(np.column_stack([lo, Dbest])[gv["OOT"]], y[gv["OOT"]])
    out["combiner_refit_on_OOT"] = {"V1": coefs[BEST].tolist(), "OOT": w_oot.tolist()}
    Dfz = specs[f"{BEST} [frozen@May]"]
    out["combiner_refit_on_OOT_frozen"] = C.fit_logistic(
        np.column_stack([lo, Dfz])[gv["OOT"]], y[gv["OOT"]]).tolist()

    # ------------------------------------------------ slot level (as-is), OOT
    r_q, wp_q = d["r"][qi], d["wp"][qi]
    nb = np.searchsorted(N_EDGES, nh_all[qi], side="right") - 1
    out["slot"] = {}
    cand = {"online " + k: preds[k][qi] for k in ["experience only"] + KNAMES}
    cand.update({"frozen@May " + k: fz[k] for k in ["experience only"] + KNAMES})
    for name, pr in cand.items():
        res = {"all": slot_gain(r_q, wp_q, pr)}
        for i, lab in enumerate(N_LABELS):
            mm = nb == i
            if mm.sum() > 200:
                res[lab] = slot_gain(r_q[mm], wp_q[mm], pr[mm])
        out["slot"][name] = res
        print(f"  slot {name:34s} {res['all']['ll_gain_x1000']:+.3f} slope "
              f"{res['all'].get('slope', float('nan')):.3f}", flush=True)
    out["slot_share_by_n"] = {lab: float((nb == i).mean()) for i, lab in enumerate(N_LABELS)}
    # who the OOT players are
    pid_q = d["pid"][qi]
    first_seen = np.full(int(d["n_players"]), 10 ** 9)
    np.minimum.at(first_seen, d["pid"], d["day"])
    new_acct = first_seen[pid_q] > X.SNAP_LAST_DAY
    out["oot_slots_new_players_share"] = float(new_acct.mean())
    out["slot_new_vs_known"] = {
        "known players": slot_gain(r_q[~new_acct], wp_q[~new_acct], preds[BEST][qi][~new_acct]),
        "new players": slot_gain(r_q[new_acct], wp_q[new_acct], preds[BEST][qi][new_acct])}
    # per-slot gain by days since freeze, frozen vs online
    dd = d["day"][qi] - X.SNAP_LAST_DAY
    out["slot_by_month_since_freeze"] = {}
    for a, b in ((1, 30), (31, 60), (61, 90), (91, 128)):
        mm = (dd >= a) & (dd <= b) & ~new_acct
        out["slot_by_month_since_freeze"][f"{a}-{b}d"] = {
            "online": slot_gain(r_q[mm], wp_q[mm], preds[BEST][qi][mm]),
            "frozen@May": slot_gain(r_q[mm], wp_q[mm], fz[BEST][mm]),
            "frozen@May counts current": slot_gain(r_q[mm], wp_q[mm], fz_counts[BEST][mm]),
            "experience only (current counts)": slot_gain(r_q[mm], wp_q[mm], mu_on[mm])}
    # interval coverage four months out: cells frozen at May vs their OOT mean
    key = d["pid"][oot] * H + d["hero"][oot]
    u, inv, nf = np.unique(key, return_inverse=True, return_counts=True)
    rbar = np.bincount(inv, weights=r_adj[oot]) / nf
    noise = np.bincount(inv, weights=d["v"][oot]) / nf ** 2
    Sst, Pst, Nst, _, _ = C.static_state(d, ~post, d["n_players"], H, r_adj)
    pid_u, h_u = u // H, u % H
    mm_, s2 = C.gp_query(np.ascontiguousarray(Sst[pid_u]), np.ascontiguousarray(Pst[pid_u]),
                         h_u, Ks[BEST])
    nh_u = Nst[pid_u, h_u]
    z = (rbar - mm_) / np.sqrt(s2 + noise)
    cal = 1.20 * s2 + (0.07 ** 2) * (nh_u == 0)
    zc = (rbar - mm_) / np.sqrt(cal + noise)
    cov = {}
    for lab, sel in (("all", nf >= 10), ("never played", (nf >= 10) & (nh_u == 0)),
                     ("played before", (nf >= 10) & (nh_u > 0)),
                     ("30+ future games", nf >= 30)):
        err2 = (rbar[sel] - mm_[sel]) ** 2
        cov[lab] = {"cells": int(sel.sum()), "cover80_raw": float((np.abs(z[sel]) < 1.2816).mean()),
                    "cover80_calibrated": float((np.abs(zc[sel]) < 1.2816).mean()),
                    "var_ratio_raw": float((err2.mean() - noise[sel].mean()) / s2[sel].mean())}
    out["coverage_frozen_may_to_oot"] = cov
    print(json.dumps(cov), flush=True)
    with open(os.path.join(X.RESULTS, "p3_x_oot.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
