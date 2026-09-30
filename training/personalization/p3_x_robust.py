"""
P3 extensions, task 7: robustness of the lift and key effects by skill tier
and by region.

Lift: the phase-1 model ("+CF rank 2", online, lag 1 day, from
cache/x_predall.npz), combiner fit on all V1 games, scored within each tier
and region of V2 games (and of the post-snapshot OOT games from task 1, via
cache/x_slots_ext.npz, with the combiner and hyperparameters frozen). Also a
per-subset refit of the skill coefficient to see whether the effect size
differs.
Key effects per subset (V1 + V2 games, established players with 300+ earlier
games unless noted):
  never-played-hero penalty: mean residual r on heroes with no earlier games
  comfort bonus: mean r on heroes with 50+ earlier games
  residual after the skill model for off-pool and comfort picks
  3+ stack party effect after the skill model
Region codes follow Heroes Profile (1 Americas, 2 Europe, 3 Asia).

Usage (from training/): python3 personalization/p3_x_robust.py
Output: results/p3_x_robust.json
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import p3_hs_core as C
import p3_x_common as X
import p3_x_side as XS

PRED = os.path.join(C.CACHE, "x_predall.npz")
REGION = {1: "Americas", 2: "Europe", 3: "Asia"}


def ci_clusters(v, clus, rng, n=200):
    u, inv = np.unique(clus, return_inverse=True)
    s = np.bincount(inv, weights=v)
    c = np.bincount(inv)
    bs = []
    for _ in range(n):
        k = rng.randint(0, len(u), len(u))
        bs.append(s[k].sum() / c[k].sum())
    return [float(100 * v.mean()), float(100 * np.percentile(bs, 2.5)), float(100 * np.percentile(bs, 97.5))]


def lift_by(d, pred_rows, fit_m, test_sets, n_games, y, lo):
    D = X.team_diff(pred_rows, d["g"], d["team"], n_games)
    w0 = C.fit_logistic(lo[fit_m][:, None], y[fit_m])
    ll0 = X.logloss(C.predict(w0, lo[:, None]), y)
    w = C.fit_logistic(np.column_stack([lo, D])[fit_m], y[fit_m])
    ll1 = X.logloss(C.predict(w, np.column_stack([lo, D])), y)
    out = {}
    for nm, m in test_sets.items():
        idx = np.flatnonzero(m)
        if len(idx) < 2000:
            continue
        g, ci = X.boot_gain(ll0, ll1, idx, n=100)
        wr = C.fit_logistic(np.column_stack([lo, D])[m], y[m])
        mt0 = C.game_metrics(C.predict(w0, lo[:, None])[m], y[m])
        mt1 = C.game_metrics(C.predict(w, np.column_stack([lo, D]))[m], y[m])
        out[nm] = {"games": int(m.sum()), "gain": g, "ci": ci, "acc_M0": mt0["acc"], "acc_skill": mt1["acc"],
                   "skill_coef_V1": float(w[2]), "skill_coef_refit_here": float(wr[2]),
                   "sd_of_team_skill_diff_pp": float(100 * D[m].std())}
    return out


def main():
    t0 = time.time()
    rng = np.random.RandomState(0)
    d = C.load_slots()
    P = np.load(PRED)
    s = (P["mu"] + P["m_cf2"]).astype(np.float64)
    sg = np.load(os.path.join(C.CACHE, "x_side_games.npz"))
    gi = XS.game_lookup(d, sg["replay_ids"])
    tier_names = [str(t) for t in sg["tier_names"]]
    n_games, y, wp0, cnt, gday = X.game_arrays(d)
    gtier = np.full(n_games, -1)
    greg = np.full(n_games, -1)
    gtier[d["g"]] = np.where(gi >= 0, sg["tier"][np.maximum(gi, 0)], -1)
    greg[d["g"]] = np.where(gi >= 0, sg["region"][np.maximum(gi, 0)], -1)
    post = ~d["in_sample"]
    _, fr = np.unique(d["g"][post], return_index=True)
    med = np.median(d["day"][post][fr])
    cntp = np.bincount(d["g"][post], minlength=n_games)
    lo = np.log(wp0 / (1 - wp0))
    fit_m = (cntp == 10) & (gday < med)
    v2 = (cntp == 10) & (gday >= med)
    sets = {"all": v2}
    for ti, tn in enumerate(tier_names):
        sets[f"tier {tn}"] = v2 & (gtier == ti)
    for rv, rn in REGION.items():
        sets[f"region {rn}"] = v2 & (greg == rv)
    for ti, tn in enumerate(tier_names):
        for rv, rn in REGION.items():
            sets[f"{tn} x {rn}"] = v2 & (gtier == ti) & (greg == rv)
    out = {"tiers": tier_names, "V2_lift": lift_by(d, s, fit_m, sets, n_games, y, lo)}
    print(json.dumps({k: round(v["gain"], 5) for k, v in out["V2_lift"].items()}), flush=True)

    # key effects by subset (V1+V2 slots)
    stier = gtier[d["g"]]
    sreg = greg[d["g"]]
    n_ph = P["nh_lag"]
    n_p = P["np_lag"]
    r = d["r"]
    e = r - s
    est = n_p >= 300
    party = d["party"]
    t = d["team"].astype(np.int64)
    gid = d["g"] * 2 + t
    pk = gid * (1 << 20) + (party % (1 << 20))
    u2, inv2, c2 = np.unique(np.where(party != 0, pk, -1 - np.arange(len(pk))), return_inverse=True,
                             return_counts=True)
    size = c2[inv2]
    eff = {}
    subsets = {"all": np.ones(len(r), bool)}
    for ti, tn in enumerate(tier_names):
        subsets[f"tier {tn}"] = stier == ti
    for rv, rn in REGION.items():
        subsets[f"region {rn}"] = sreg == rv
    for nm, m in subsets.items():
        m = m & post
        row = {}
        for lab, mm, val in (("never-played hero, 300+ games: raw r", m & est & (n_ph == 0), r),
                             ("50+ games on hero, 300+ games: raw r", m & est & (n_ph >= 50), r),
                             ("never-played hero, 300+ games: after skill model", m & est & (n_ph == 0), e),
                             ("3+ stack party member: after skill model", m & (party != 0) & (size >= 3), e),
                             ("solo: after skill model", m & (party == 0), e)):
            if mm.sum() > 500:
                row[lab] = {"slots": int(mm.sum()), "pp": ci_clusters(val[mm], d["pid"][mm], rng)}
        eff[nm] = row
    out["key_effects_V1V2"] = eff
    print(json.dumps({k: {kk: round(vv["pp"][0], 2) for kk, vv in v.items()} for k, v in eff.items()}),
          flush=True)
    # smurf flags by tier/region from task 4, if present
    fz = os.path.join(C.CACHE, "x_smurf_flags.npz")
    if os.path.exists(fz):
        out["smurf_note"] = "see p3_x_smurf.json census (by tier of first game and by region)"

    # OOT (task 1) by tier and region: frozen hyperparameters, online state
    if os.path.exists(X.EXT):
        import gzip
        de = X.load_ext()
        Ks, table = X.kernels()
        n_pe, n_phe = C.experience_counts(de)
        r_adj = de["r"] - table[C.exp_bins(n_pe, n_phe)]
        oot = de["post"] & (de["day"] > X.SNAP_LAST_DAY)
        sp = de["post"] == False  # noqa: E712
        spost = sp & ~de["in_sample"]
        q = oot | (spost & (de["day"] < med))
        # V1 rows use snapshot-only state (as in p3_x_oot); OOT rows use all rows
        qi1, m1, _, Nh1, Np1, _ = X.online_predict(de, sp, spost & (de["day"] < med), Ks["+CF rank 2"], r_adj)
        qi2, m2, _, Nh2, Np2, _ = X.online_predict(de, np.ones(len(de["day"]), bool), oot,
                                                   Ks["+CF rank 2"], r_adj)
        pr = np.zeros(len(de["day"]))
        pr[qi1] = table[C.exp_bins(Np1.astype(np.int64), Nh1.astype(np.int64))] + m1
        pr[qi2] = table[C.exp_bins(Np2.astype(np.int64), Nh2.astype(np.int64))] + m2
        ng, ye, wpe, cnte, gde = X.game_arrays(de)
        loe = np.log(wpe / (1 - wpe))
        g_fit = np.zeros(ng, bool)
        g_fit[de["g"][spost & (de["day"] < med)]] = True
        g_fit &= cnte == 10
        g_oot = np.zeros(ng, bool)
        g_oot[de["g"][oot]] = True
        g_oot &= cnte == 10
        games = __import__("json").load(gzip.open(os.path.join(C.CACHE, "x_post_games.json.gz"), "rt"))
        tmap = {g["replay_id"]: (g["skill_tier"], g["region"]) for g in games}
        grid = np.zeros(ng, np.int64)
        grid[de["g"]] = de["replay_id"]
        gt_e = np.full(ng, -1)
        gr_e = np.full(ng, -1)
        for i in np.flatnonzero(g_oot):
            tr_ = tmap.get(int(grid[i]))
            if tr_:
                gt_e[i] = tier_names.index(tr_[0]) if tr_[0] in tier_names else -1
                gr_e[i] = tr_[1] if tr_[1] is not None else -1
        sets_e = {"OOT all": g_oot}
        for ti, tn in enumerate(tier_names):
            sets_e[f"OOT tier {tn}"] = g_oot & (gt_e == ti)
        for rv, rn in REGION.items():
            sets_e[f"OOT region {rn}"] = g_oot & (gr_e == rv)
        out["OOT_lift"] = lift_by(de, pr, g_fit, sets_e, ng, ye, loe)
        print(json.dumps({k: round(v["gain"], 5) for k, v in out["OOT_lift"].items()}), flush=True)
    with open(os.path.join(C.RESULTS, "p3_x_robust.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
