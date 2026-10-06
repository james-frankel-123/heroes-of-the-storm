"""
P3 review item #2: a lifetime-experience axis for the experience offset.

The headline offset is the mean residual by (games seen, games on hero) in
the window since 2024-04-01, so "games seen" partly measures how long the
corpus has watched a player. This script adds what the player brings from
before the window, using only information available at the game:
  * hero level as of the game's start (p3_hero_level_causal: latest stamp
    parsed before the game started; unknown kept as its own bin)
  * account status 0-3 (p3_hero_level_causal.account_status)
Offsets are tables over (group, games-seen bin, games-on-hero bin), fit on
E with the same shrinkage (200 pseudo-games) as the headline; the skill
state is rebuilt on each variant's detrended residuals (lag 1 day).
Variants
  window counts                 the headline table
  status split                  status 2 and 3 split out (P3_EXTENSIONS 4)
  hero level                    x hero-level bin (unknown, 1-5, 6-10, 11-20, 21-50, 51+)
  hero level + status split     both
Scored on V2 (combiner fit on V1) and OOT (frozen tables and combiner,
online state), overall and by status and by hero-level bin. Run once on the
window caches and once on the full-history caches (P3_CACHE pointing at the
backfilled tables; same target games) to separate better estimation from
more coverage. The sealed build 2.55.17.98025 is scored only with --final.

Usage (from training/): python3 personalization/p3_c_lifetime.py [--final]
Output: results/p3_c_lifetime.json
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import p3_hs_core as C
import p3_x_common as X
import p3_hero_level_causal as HL

LEVEL_EDGES = np.array([1, 6, 11, 21, 51])
SEALED_BUILD = "2.55.17.98025"


def level_bin(lo):
    """0 = unknown (no stamp parsed before the game), 1..5 = 1-5, 6-10,
    11-20, 21-50, 51+ (lower end of a band)."""
    b = np.searchsorted(LEVEL_EDGES, np.nan_to_num(lo, nan=0.0), side="right")
    return np.where(np.isnan(lo), 0, b).astype(np.int64)


def groups(st, lv):
    split = np.where(st == 2, 1, np.where(st == 3, 2, 0))
    return {"window counts": np.zeros_like(st),
            "status split": split,
            "hero level": lv,
            "hero level + status split": split * 6 + lv}


def run(d, G, e_mask, K, q_rows, add_rows):
    n_p, n_ph = C.experience_counts(d)
    b = C.exp_bins(n_p, n_ph)
    nb = len(C.NP_EDGES) * len(C.NPH_EDGES)
    res = {}
    for nm, g in G.items():
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
        res[nm] = (s, None)
        print(f"  {nm}: {ng} groups", flush=True)
    return res


def score(d, preds, fit_g, test_sets, base="window counts"):
    """Game-level log-loss gains over WP alone and over the headline table;
    combiners fit on fit_g (V1) games only."""
    n_games, y, wp0, cnt, gday = X.game_arrays(d)
    lo = np.log(wp0 / (1 - wp0))
    ll = {}
    coef = {}
    for nm, (s_, _) in preds.items():
        D = X.team_diff(s_, d["g"], d["team"], n_games)
        w = C.fit_logistic(np.column_stack([lo, D])[fit_g], y[fit_g])
        ll[nm] = X.logloss(C.predict(w, np.column_stack([lo, D])), y)
        coef[nm] = w.tolist()
    w0 = C.fit_logistic(lo[fit_g][:, None], y[fit_g])
    ll0 = X.logloss(C.predict(w0, lo[:, None]), y)
    out = {}
    for tn, tm in test_sets.items():
        idx = np.flatnonzero(tm)
        if len(idx) < 500:
            continue
        out[tn] = {"games": int(len(idx))}
        for nm in preds:
            g, ci = X.boot_gain(ll0, ll[nm], idx, n=200)
            vs, vci = X.boot_gain(ll[base], ll[nm], idx, n=200)
            out[tn][nm] = {"gain": g, "ci": ci, f"vs_{base}": vs, "vs_ci": vci, "coef": coef[nm]}
    return out


def strata(d, rows, st, lv, n_games):
    out = {}
    for name, msk in (("with a status-0 player", st == 0), ("with a status-3 player", st == 3),
                      ("with a status-2 player", st == 2), ("with an unknown-level slot", lv == 0),
                      ("with a level 51+ slot", lv == 5)):
        has = np.zeros(n_games, bool)
        has[d["g"][rows & msk]] = True
        out[name] = has
    return out


def main():
    final = "--final" in sys.argv
    t0 = time.time()
    Ks, _ = X.kernels()
    K = Ks["+CF rank 2"]
    out = {"cache": C.CACHE, "final": final}
    # ---------------- V2
    d = C.load_slots()
    hl = HL.load(d)
    st = HL.account_status(d, hl)
    lv = level_bin(hl["hero_lo"])
    day = d["day"]
    e_mask = d["in_sample"] & (day >= C.day_of(C.E_START))
    post = ~d["in_sample"]
    _, fr = np.unique(d["g"][post], return_index=True)
    med = np.median(day[post][fr])
    preds = run(d, groups(st, lv), e_mask, K, post, np.ones(len(day), bool))
    n_games, y, wp0, cnt, gday = X.game_arrays(d)
    cntp = np.bincount(d["g"][post], minlength=n_games)
    fit_g = (cntp == 10) & (gday < med)
    v2 = (cntp == 10) & (gday >= med)
    sets = {"V2 all": v2}
    for k, h in strata(d, post & (day >= med), st, lv, n_games).items():
        sets[f"V2 games {k}"] = v2 & h
    out["V2"] = score(d, {k: (v[0], None) for k, v in preds.items()}, fit_g, sets)
    out["V2 shares"] = {"status": {str(k): float(np.mean(st[post & (day >= med)] == k)) for k in range(4)},
                        "level bin": {str(k): float(np.mean(lv[post & (day >= med)] == k)) for k in range(6)}}
    for tn, v in out["V2"].items():
        print(tn, {k: round(x["gain"], 5) for k, x in v.items() if k != "games"}, flush=True)
    del d, preds
    # ---------------- OOT
    de = X.load_ext()
    hle = HL.load(de)
    ste = HL.account_status(de, hle)
    lve = level_bin(hle["hero_lo"])
    daye = de["day"]
    e_mask = de["in_sample"] & (daye >= C.day_of(C.E_START))
    snap = ~de["post"]
    spost = snap & ~de["in_sample"]
    v1r = spost & (daye < med)
    oot = de["post"] & (daye > X.SNAP_LAST_DAY)
    vers = [str(v) for v in de["versions"]]
    if SEALED_BUILD in vers and not final:
        oot &= de["version"] != vers.index(SEALED_BUILD)
    p1 = run(de, groups(ste, lve), e_mask, K, v1r, snap)
    p2 = run(de, groups(ste, lve), e_mask, K, oot, np.ones(len(daye), bool))
    preds = {k: (p1[k][0] + p2[k][0], None) for k in p1}
    ng, ye, wpe, cnte, gde = X.game_arrays(de)
    fg = np.zeros(ng, bool)
    fg[de["g"][v1r]] = True
    fg &= cnte == 10
    go = np.zeros(ng, bool)
    go[de["g"][oot]] = True
    go &= cnte == 10
    sets = {"OOT all": go}
    for k, h in strata(de, oot, ste, lve, ng).items():
        sets[f"OOT games {k}"] = go & h
    out["OOT"] = score(de, preds, fg, sets)
    for tn, v in out["OOT"].items():
        print(tn, {k: round(x["gain"], 5) for k, x in v.items() if k != "games"}, flush=True)
    with open(os.path.join(C.RESULTS, "p3_c_lifetime.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
