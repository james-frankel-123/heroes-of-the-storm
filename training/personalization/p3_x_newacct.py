"""
P3 extensions, task 4 follow-up: which part of the new-account prior carries
the gain, and does it hold out of time?

Account status at each game (causal):
  0  first seen in the window before 2024-07-01
  1  first seen on or after 2024-07-01, first day (no earlier-day hero levels)
  2  first seen on or after 2024-07-01, every hero level on earlier days <= 5
     (a genuinely new account)
  3  first seen on or after 2024-07-01, some earlier hero level > 5 (an old
     account new to our corpus)
The experience table (games seen x games on hero) is split by status and fit
on E; the skill state is rebuilt on the matching detrended residuals. Variants:
  4 statuses; status 2 split out only (1 and 3 pooled with 0); status 3 split
  out only; both 2 and 3 (1 pooled with 0).
Scored on V2 (combiner fit on V1) and on the post-snapshot OOT games of task 1
(frozen tables and combiner; online state with everything before the game day).

Usage (from training/): python3 personalization/p3_x_newacct.py
Output: results/p3_x_newacct.json
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import p3_hs_core as C
import p3_x_common as X


def status_of(d):
    n = len(d["pid"])
    npl = int(d["n_players"])
    day = d["day"]
    first_day = np.full(npl, 10 ** 9)
    np.minimum.at(first_day, d["pid"], day)
    hl = d["hero_level"].astype(float)
    hl[hl < 0] = 99
    o = np.lexsort((d["replay_id"], day, d["pid"]))
    pid, dy = d["pid"][o], day[o].astype(np.int64)
    hv = hl[o]
    starts = np.flatnonzero(np.r_[True, pid[1:] != pid[:-1]])
    ends = np.r_[starts[1:], n]
    cm = np.empty(n)
    for a, b in zip(starts, ends):
        cm[a:b] = np.maximum.accumulate(hv[a:b])
    comp = pid * 100000 + dy
    fod = np.searchsorted(comp, comp, side="left")
    fop = np.searchsorted(pid, pid, side="left")
    prev_max = np.where(fod > fop, cm[np.maximum(fod - 1, 0)], -1)
    late = first_day[pid] >= C.day_of("2024-07-01")
    st = np.where(~late, 0, np.where(prev_max < 0, 1, np.where(prev_max <= 5, 2, 3)))
    out = np.empty(n, np.int64)
    out[o] = st
    return out


VARIANTS = {"phase-1 table": lambda s: np.zeros_like(s),
            "4 statuses": lambda s: s,
            "new low-level accounts split out": lambda s: (s == 2).astype(np.int64),
            "old accounts new to the corpus split out": lambda s: (s == 3).astype(np.int64),
            "both split out": lambda s: np.where(s == 2, 1, np.where(s == 3, 2, 0))}


def run(d, status, e_mask, K, fit_rows, q_rows, add_rows, freeze=None):
    """Tables fit on e_mask rows; predictions for q_rows (state from add_rows)."""
    n_p, n_ph = C.experience_counts(d)
    b = C.exp_bins(n_p, n_ph)
    nb = len(C.NP_EDGES) * len(C.NPH_EDGES)
    res = {}
    for nm, f in VARIANTS.items():
        g = f(status)
        ng = int(g.max()) + 1
        key = g * nb + b
        ssum = np.bincount(key[e_mask], weights=d["r"][e_mask], minlength=ng * nb)
        scnt = np.bincount(key[e_mask], minlength=ng * nb)
        tab = ssum / (scnt + 200.0)
        r_adj = d["r"] - tab[key]
        qi, m, _, Nh, Np, _ = X.online_predict(d, add_rows, q_rows, K, r_adj)
        bl = C.exp_bins(Np.astype(np.int64), Nh.astype(np.int64))
        s = np.zeros(len(d["pid"]))
        s[qi] = tab[g[qi] * nb + bl] + m
        res[nm] = (s, tab.reshape(ng, len(C.NP_EDGES), len(C.NPH_EDGES)))
    return res


def score(d, preds, fit_g, test_sets):
    n_games, y, wp0, cnt, gday = X.game_arrays(d)
    lo = np.log(wp0 / (1 - wp0))
    base = None
    out = {}
    ll = {}
    for nm, (s, _) in preds.items():
        D = X.team_diff(s, d["g"], d["team"], n_games)
        w = C.fit_logistic(np.column_stack([lo, D])[fit_g], y[fit_g])
        ll[nm] = X.logloss(C.predict(w, np.column_stack([lo, D])), y)
    w0 = C.fit_logistic(lo[fit_g][:, None], y[fit_g])
    ll0 = X.logloss(C.predict(w0, lo[:, None]), y)
    for tn, tm in test_sets.items():
        idx = np.flatnonzero(tm)
        out[tn] = {}
        for nm in preds:
            g, ci = X.boot_gain(ll0, ll[nm], idx, n=200)
            vs, vci = X.boot_gain(ll["phase-1 table"], ll[nm], idx, n=200)
            out[tn][nm] = {"gain": g, "ci": ci, "vs_phase1_table": vs, "vs_ci": vci}
    return out


def main():
    t0 = time.time()
    Ks, _ = X.kernels()
    K = Ks["+CF rank 2"]
    # ---------------- snapshot: V2
    d = C.load_slots()
    st = status_of(d)
    day = d["day"]
    e_mask = d["in_sample"] & (day >= C.day_of(C.E_START))
    post = ~d["in_sample"]
    _, fr = np.unique(d["g"][post], return_index=True)
    med = np.median(day[post][fr])
    preds = run(d, st, e_mask, K, None, post, np.ones(len(day), bool))
    n_games, y, wp0, cnt, gday = X.game_arrays(d)
    cntp = np.bincount(d["g"][post], minlength=n_games)
    fit_g = (cntp == 10) & (gday < med)
    v2 = (cntp == 10) & (gday >= med)
    has2 = np.zeros(n_games, bool)
    has2[d["g"][st == 2]] = True
    has3 = np.zeros(n_games, bool)
    has3[d["g"][st == 3]] = True
    out = {"status_share": {"V2 slots": {str(k): float(np.mean(st[post & (day >= med)] == k)) for k in range(4)}},
           "V2": score(d, preds, fit_g, {"V2 all": v2, "V2 games with a new low-level account": v2 & has2,
                                          "V2 games without one": v2 & ~has2})}
    nb = len(C.NPH_EDGES)
    tab4 = preds["4 statuses"][1]
    out["offset_pp_4_statuses"] = {f"status {s_}": (100 * tab4[s_]).round(2).tolist() for s_ in range(4)}
    for tn, v in out["V2"].items():
        print(tn, {k: (round(x["gain"], 5), round(x["vs_phase1_table"], 5)) for k, x in v.items()}, flush=True)
    del d, preds
    # ---------------- OOT (tables from E, combiner from V1, online state)
    de = X.load_ext()
    ste = status_of(de)
    daye = de["day"]
    e_mask = de["in_sample"] & (daye >= C.day_of(C.E_START))
    snap = ~de["post"]
    spost = snap & ~de["in_sample"]
    v1r = spost & (daye < med)
    oot = de["post"] & (daye > X.SNAP_LAST_DAY)
    # V1 rows with snapshot-only state, OOT rows with all state (as in p3_x_oot)
    p1 = run(de, ste, e_mask, K, None, v1r, snap)
    p2 = run(de, ste, e_mask, K, None, oot, np.ones(len(daye), bool))
    preds = {k: (p1[k][0] + p2[k][0], p1[k][1]) for k in p1}
    ng, ye, wpe, cnte, gde = X.game_arrays(de)
    fg = np.zeros(ng, bool)
    fg[de["g"][v1r]] = True
    fg &= cnte == 10
    go = np.zeros(ng, bool)
    go[de["g"][oot]] = True
    go &= cnte == 10
    h2 = np.zeros(ng, bool)
    h2[de["g"][ste == 2]] = True
    out["status_share"]["OOT slots"] = {str(k): float(np.mean(ste[oot] == k)) for k in range(4)}
    out["OOT"] = score(de, preds, fg, {"OOT all": go, "OOT games with a new low-level account": go & h2,
                                        "OOT games without one": go & ~h2})
    for tn, v in out["OOT"].items():
        print(tn, {k: (round(x["gain"], 5), round(x["vs_phase1_table"], 5)) for k, x in v.items()}, flush=True)
    with open(os.path.join(C.RESULTS, "p3_x_newacct.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
