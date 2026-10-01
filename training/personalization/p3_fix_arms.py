"""
Audit fixes P3-09, P3-12, P3-11, P3-19, P3-13 (small, CPU).

  arms     P3-09: the static@cutoff and lag-7/30 arms of p3_hs_eval reused the
           lag-1 experience offset. Here every arm uses counts at the same
           horizon as its GP state (and, for reference, live lag-1 counts):
             online lag 1                state lag 1, counts lag 1
             static@cutoff, true static  state and counts frozen at the cutoff
             static@cutoff, live counts  state frozen, counts lag 1
             lag G (7, 30), consistent   state and counts lagged G days
             lag G, live counts          state lagged G, counts lag 1
           Lag-1 count contract and lag-1 experience table (P3-03). Game level:
           logit(WP) + team difference, fit V1, test V2, replay bootstrap and
           a player-cluster bootstrap for the headline (P3-16).
  role     P3-12: off-role penalty after the skill model with player- and
           game-cluster bootstrap SEs, by window (2025-01+, post-cutoff, V2).
  misc     P3-11: smurf prevalence over represented games; P3-19: BH-FDR on
           the per-hero transfer tests of P3_HERO_SIMILARITY.
  cf       P3-13: profile likelihood of the CF-factor drift rate.

Run (from training/): OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4 nice -n 19 taskset -c 48-63 \
  python3 personalization/p3_fix_arms.py arms|role|misc|cf
Output: results/fix/p3_fix_<what>.json
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import p3_hs_core as C
import p3_x_common as X
import p3_fix_counts as F
from p3_hs_eval import team_diff


def save(name, out):
    os.makedirs(F.FIX_RESULTS, exist_ok=True)
    with open(os.path.join(F.FIX_RESULTS, f"p3_fix_{name}.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)


def game_setup(d):
    n_games = int(d["g"].max()) + 1
    y = np.zeros(n_games)
    wg = np.full(n_games, 0.5)
    t0 = d["team"] == 0
    y[d["g"][t0]] = d["y"][t0]
    wg[d["g"][t0]] = d["wp"][t0]
    days = d["day"]
    post = ~d["in_sample"]
    _, first_row = np.unique(d["g"][post], return_index=True)
    med = np.median(days[post][first_row])
    full = np.bincount(d["g"][post], minlength=n_games) == 10
    gv1 = np.zeros(n_games, bool)
    gv1[d["g"][post & (days < med)]] = True
    gv2 = np.zeros(n_games, bool)
    gv2[d["g"][post & (days >= med)]] = True
    return n_games, y, np.log(wg / (1 - wg)), gv1 & full, gv2 & full, med


def arms():
    t0 = time.time()
    rng = np.random.RandomState(0)
    d = C.load_slots()
    e_mask, n_p, n_ph, table, r_adj = F.prepare_l1(d)
    Ks, _ = X.kernels()
    K = Ks["+CF rank 2"]
    n_games, y, lo, fit_m, test_m, med = game_setup(d)
    post = ~d["in_sample"]
    allrows = np.ones(len(d["day"]), bool)
    cut = int(d["day"][post].min()) - 1
    preds = {}

    def mu_of(Np, Nh):
        return table[C.exp_bins(Np.astype(np.int64), Nh.astype(np.int64))]
    qi, m, v, Nh, Np, _ = X.online_predict(d, allrows, post, K, r_adj, lag=1)
    mu1 = mu_of(Np, Nh)
    preds["online lag 1 (state and counts lag 1)"] = (qi, mu1 + m)
    preds["experience offset only (lag 1)"] = (qi, mu1)
    q2, m2, _, Nh2, Np2, _ = X.online_predict(d, allrows, post, K, r_adj, lag=1, freeze_day=cut)
    assert np.array_equal(qi, q2)
    preds["static@cutoff, true static (state and counts frozen)"] = (qi, mu_of(Np2, Nh2) + m2)
    preds["static@cutoff, live counts (state frozen, counts lag 1)"] = (qi, mu1 + m2)
    for G in (7, 30):
        q3, m3, _, Nh3, Np3, _ = X.online_predict(d, allrows, post, K, r_adj, lag=G)
        preds[f"lag {G}, consistent (state and counts lag {G})"] = (qi, mu_of(Np3, Nh3) + m3)
        preds[f"lag {G}, live counts (state lag {G}, counts lag 1)"] = (qi, mu1 + m3)
        print(f"lag {G}: {time.time() - t0:.0f}s", flush=True)
    base_X = lo[:, None]
    w0 = C.fit_logistic(base_X[fit_m], y[fit_m])
    p0 = C.predict(w0, base_X)
    ll0 = -(y * np.log(p0) + (1 - y) * np.log(1 - p0))
    idx = np.flatnonzero(test_m)
    boots = [rng.choice(idx, len(idx)) for _ in range(200)]
    out = {"split_day": float(med)}
    lls = {}
    for nm, (q, val) in preds.items():
        full = np.zeros(len(d["day"]))
        full[q] = val
        Xg = np.column_stack([lo, team_diff(full, d["g"], d["team"], n_games)])
        w = C.fit_logistic(Xg[fit_m], y[fit_m])
        p = C.predict(w, Xg)
        ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        lls[nm] = ll
        dd = ll0 - ll
        bs = [dd[b].mean() for b in boots]
        out[nm] = {"acc": float(((p > 0.5) == (y == 1))[test_m].mean()), "gain_vs_M0": float(dd[idx].mean()),
                   "ci95_replay": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
        print(nm, json.dumps(out[nm]), flush=True)
    # player-cluster bootstrap for the headline: each test game is resampled
    # through the player in its first team-0 slot (one-way approximation of
    # the ten-way crossed design; the audit measured a design effect of 1.7)
    hd = "online lag 1 (state and counts lag 1)"
    dd = (ll0 - lls[hd])
    first_p = np.full(n_games, -1)
    t0m = d["team"] == 0
    first_p[d["g"][t0m][::-1]] = d["pid"][t0m][::-1]
    gp = first_p[idx]
    up, inv = np.unique(gp, return_inverse=True)
    s_ = np.bincount(inv, weights=dd[idx])
    c_ = np.bincount(inv).astype(float)
    bs = []
    for _ in range(300):
        w = np.bincount(rng.randint(0, len(up), len(up)), minlength=len(up))
        bs.append((w * s_).sum() / (w * c_).sum())
    out[hd]["ci95_player_cluster_one_way"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    for a, b in (("online lag 1 (state and counts lag 1)", "static@cutoff, true static (state and counts frozen)"),
                 ("online lag 1 (state and counts lag 1)", "static@cutoff, live counts (state frozen, counts lag 1)"),
                 ("static@cutoff, live counts (state frozen, counts lag 1)", "static@cutoff, true static (state and counts frozen)")):
        dd2 = lls[b] - lls[a]
        bs = [dd2[x].mean() for x in boots]
        out[f"{a} minus {b}"] = [float(dd2[idx].mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    save("arms", out)


def role():
    import p3_ph_role as R
    rng = np.random.RandomState(0)
    d = C.load_slots()
    meta = C.hero_meta(d["hero_names"])
    e_mask, n_p, n_ph, table, r_adj = F.prepare_l1(d)
    blizz = meta["blizz"][d["hero"]]
    fine = meta["fine"][d["hero"]]
    n_pb = F.lag1_key_counts(d, d["pid"] * 8 + blizz)
    n_pf = F.lag1_key_counts(d, d["pid"] * 16 + fine)
    share_b = n_pb / np.maximum(n_p, 1)
    share_f = n_pf / np.maximum(n_p, 1)
    est = n_p >= 50
    pm = np.load(os.path.join(C.CACHE, "fix_sd_pred_static_lag1.npz"))["m"]
    e = d["r"] - (table[C.exp_bins(n_p, n_ph)] + pm)
    post = ~d["in_sample"]
    days = d["day"]
    _, first_row = np.unique(d["g"][post], return_index=True)
    med = np.median(days[post][first_row])
    wins = {"2025-01+ (in-sample for the WP)": est & (days >= C.day_of("2025-01-01")),
            "post-cutoff": est & post, "V2 only": est & post & (days >= med)}
    hb = np.searchsorted(R.NPH_EDGES, n_ph, side="right") - 1

    def pen(val, share, sel, w=None):
        num, den = 0.0, 0.0
        for i in range(len(R.NPH_LABELS)):
            off = sel & (hb == i) & (share < 0.10)
            on = sel & (hb == i) & (share >= 0.35)
            if off.sum() < 500 or on.sum() < 500:
                continue
            if w is None:
                a, b = val[off].mean(), val[on].mean()
                wt = off.sum()
            else:
                a = np.average(val[off], weights=w[off])
                b = np.average(val[on], weights=w[on])
                wt = w[off].sum()
            num += wt * (a - b)
            den += wt
        return 100 * num / den
    out = {}
    for wn, sel in wins.items():
        for rn, sh in (("Blizzard role", share_b), ("fine role", share_f)):
            est_ = pen(e, sh, sel)
            res = {"penalty_pp": est_}
            for cname, key in (("player", d["pid"]), ("game", d["g"])):
                ss = np.flatnonzero(sel)
                uk, inv = np.unique(key[ss], return_inverse=True)
                bs = []
                for _ in range(200):
                    w = np.zeros(len(d["day"]))
                    w[ss] = np.bincount(rng.randint(0, len(uk), len(uk)), minlength=len(uk))[inv]
                    bs.append(pen(e, sh, sel & (w > 0), w))
                res[f"se_{cname}_cluster"] = float(np.std(bs))
            out[f"{wn} | {rn}"] = res
            print(wn, rn, res, flush=True)
    save("role", out)


def misc():
    out = {}
    d = C.load_slots()
    v = json.load(open(os.path.join(F.FIX_RESULTS, "p3_val_validity.json")))["A"]
    rep = len(np.unique(d["g"]))
    out["represented_games"] = rep
    out["games_with_smurf_like_first20"] = v["games_with_smurf_like_first20"]
    out["share_of_represented_games"] = v["games_with_smurf_like_first20"] / rep
    out["old_denominator_wp_index_space"] = int(d["g"].max()) + 1
    # per-player exposure: share of a typical player's games with a smurf-like account
    hs = json.load(open(os.path.join(C.RESULTS, "p3_hs_similarity_pool.json")))
    from scipy.stats import norm
    fdr = {}
    for ver in ("total", "specific"):
        ph = [p for p in hs["pooled"][ver]["per_hero_transfer"] if np.isfinite(p["diff"])]
        se = np.array([(p["diff_ci"][1] - p["diff_ci"][0]) / 3.92 for p in ph])
        z = np.array([p["diff"] for p in ph]) / np.maximum(se, 1e-9)
        pv = 2 * norm.sf(np.abs(z))
        o = np.argsort(pv)
        m = len(pv)
        q = np.empty(m)
        q[o] = np.minimum.accumulate((pv[o] * m / np.arange(1, m + 1))[::-1])[::-1]
        fdr[ver] = {"tests": m, "raw_p_below_0.05": int((pv < 0.05).sum()),
                    "BH_q_below_0.05": [ph[i]["hero"] for i in np.flatnonzero(q < 0.05)],
                    "BH_q_below_0.10": [ph[i]["hero"] for i in np.flatnonzero(q < 0.10)],
                    "q_values": {ph[i]["hero"]: float(q[i]) for i in np.argsort(q)[:12]}}
    out["fdr_per_hero_transfer"] = fdr
    print(json.dumps(out, indent=1))
    save("misc", out)


def cf():
    import p3_sd_kalman as Kk
    d = C.load_slots()
    meta = C.hero_meta(d["hero_names"])
    kz = np.load(Kk.KOUT)
    from p3_hs_fit import prepare
    e_mask, n_p, n_ph, table, r_adj = prepare(d)
    fits = json.load(open(os.path.join(C.RESULTS, "p3_sd_fit.json")))
    out = {}
    rng = np.random.RandomState(11)
    fitp = kz["fit_players"]
    nE = np.bincount(d["pid"][e_mask], minlength=int(d["n_players"]))
    samp = np.zeros(int(d["n_players"]), bool)
    samp[rng.choice(np.flatnonzero(fitp & (nE > 0)), 40000, replace=False)] = True
    filt = Kk.Filter(d, meta, kz["V_cf2"], r_adj, e_mask & samp[d["pid"]])
    p = dict(fits["random 40k"]["fits"]["drift days"]["params"])
    for comp in ("cf", "hero_s"):
        prof = {}
        for hl_years in (1, 2, 3, 5, 8, 15, 30, 1000):
            q = dict(p)
            q[f"lam_{comp}"] = np.log(2) / (365.0 * hl_years)
            sd, lam, br = Kk.split(q)
            prof[hl_years] = float(filt.run(sd, filt.lam_state(lam, br))[2].sum())
        best = max(prof.values())
        out[comp] = {"correlation_half_life_years": {str(k): v - best for k, v in prof.items()},
                     "within_1.92_of_best": [k for k, v in prof.items() if v >= best - 1.92]}
        print(comp, out[comp], flush=True)
    save("cf", out)


if __name__ == "__main__":
    {"arms": arms, "role": role, "misc": misc, "cf": cf}[sys.argv[1]]()
