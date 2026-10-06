"""
P3 extensions, task 4: smurfs. Proposal #7's EM-style purification loop, a
new-account prior, and a causal smurf detector with honest validation.

New account (as in P3_VALIDITY): first seen on or after 2024-07-01 with median
hero level <= 5 over its first 10 games; accounts with 20+ games are
candidates.

A. Purification loop (retroactive, for labels and the census)
   1. Every slot has a skill estimate s = offset + CF posterior mean (lag 1 day).
   2. For each candidate, x = mean over its first 20 games of the residual net
      of the other nine players' estimates (r - teammates' s + opponents' s).
      A two-component Gaussian mixture (known per-account noise) is fit to x
      by EM; accounts with posterior P(smurf) > 0.5 are flagged.
   3. A flagged account's excess theta is estimated per block of its games
      (1-20, 21-50, 51-100), shrunk toward 0 (k = 30 games). Every other
      slot in those games gets its residual corrected: r' = r - (flagged
      teammates' theta) + (flagged opponents' theta).
   4. Skill estimates are recomputed from r' for everyone; back to 2.
   Iterated until the flag set is stable (at most 4 rounds).
B. Effect on the headline and on estimates. The V2 game-level gain (combiner
   fit on V1) with the state built from (a) raw residuals, (b) retroactive
   purification (uses smurfs' later games: labels view, not causal), and
   (c) boundary purification: flags and theta estimated only from games
   before the V1/V2 split, applied to games before it (causal for V2).
C. New-account prior: the experience table split by a causal new-account
   status (first seen on or after 2024-07-01, and every hero level seen on
   the account's earlier days <= 5), fit on E, scored on V2.
D. Smurf detector. At game k (10 or 20) of a new account, predict whether its
   NEXT 80 games run hot (mean net residual > +8pp) from its first k games
   only: residual z, hero levels, scoreboard style, talent conformity,
   party share, hero variety, pace. Logistic regression trained on accounts
   that started before 2025-07-01, tested on later accounts (time split).

Usage (from training/): python3 personalization/p3_x_smurf.py
Output: results/p3_x_smurf.json
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from p3_heroes import HKEY
import p3_hs_core as C
import p3_x_common as X
import p3_x_side as XS

PRED = os.path.join(C.CACHE, "x_predall.npz")
BLOCKS = [(0, 20), (20, 50), (50, 100)]


def mixture_em(x, s2, iters=300):
    """x_i ~ pi N(m1, t1 + s2_i) + (1 - pi) N(m0, t0 + s2_i)."""
    pi, m0, m1 = 0.1, float(np.median(x)), float(np.percentile(x, 95))
    t0, t1 = max(x.var() - s2.mean(), 1e-4) / 2, 1e-3
    for _ in range(iters):
        v0, v1 = t0 + s2, t1 + s2
        l0 = np.log(1 - pi) - 0.5 * (np.log(v0) + (x - m0) ** 2 / v0)
        l1 = np.log(pi) - 0.5 * (np.log(v1) + (x - m1) ** 2 / v1)
        mx = np.maximum(l0, l1)
        q = np.exp(l1 - mx) / (np.exp(l0 - mx) + np.exp(l1 - mx))
        pi = q.mean()
        w0, w1 = (1 - q) / v0, q / v1
        m0, m1 = (w0 * x).sum() / w0.sum(), (w1 * x).sum() / w1.sum()
        # moment updates for the latent variances
        t0 = max(((1 - q) * ((x - m0) ** 2 - s2)).sum() / (1 - q).sum(), 1e-6)
        t1 = max((q * ((x - m1) ** 2 - s2)).sum() / q.sum(), 1e-6)
    return q, {"pi": float(pi), "mean_normal_pp": 100 * float(m0), "sd_normal_pp": 100 * float(np.sqrt(t0)),
               "mean_smurf_pp": 100 * float(m1), "sd_smurf_pp": 100 * float(np.sqrt(t1))}


def team_sums(d, vals, n_games):
    t = d["team"].astype(np.int64)
    s = np.zeros((n_games, 2))
    np.add.at(s, (d["g"], t), vals)
    return s


def net_of_others(d, r, s, n_games):
    ts = team_sums(d, s, n_games)
    t = d["team"].astype(np.int64)
    return r - (ts[d["g"], t] - s) + ts[d["g"], 1 - t]


def purify(d, r, theta, n_games):
    ts = team_sums(d, theta, n_games)
    t = d["team"].astype(np.int64)
    return r - (ts[d["g"], t] - theta) + ts[d["g"], 1 - t]


def main():
    t_start = time.time()
    d = C.load_slots()
    P = np.load(PRED)
    P = {k: P[k] for k in P.files}
    Ks, table = X.kernels()
    K = Ks["+CF rank 2"]
    n = len(d["pid"])
    n_games = int(d["g"].max()) + 1
    npl = int(d["n_players"])
    n_p, n_ph = P["n_p"].astype(np.int64), P["n_ph"].astype(np.int64)
    mu_exact = table[C.exp_bins(n_p, n_ph)]
    day = d["day"]
    # ---------------- new accounts
    first_day = np.full(npl, 10 ** 9)
    np.minimum.at(first_day, d["pid"], day)
    ngames = np.bincount(d["pid"], minlength=npl)
    early = n_p < 10
    hl = d["hero_level"].astype(float)
    hl[hl < 0] = np.nan
    med_hl = np.full(npl, np.nan)
    pe = d["pid"][early]
    he = hl[early]
    srt = np.lexsort((he, pe))
    pe, he = pe[srt], he[srt]
    st = np.flatnonzero(np.r_[True, pe[1:] != pe[:-1]])
    en = np.r_[st[1:], len(pe)]
    for a, b in zip(st, en):
        v = he[a:b]
        v = v[~np.isnan(v)]
        if len(v):
            med_hl[pe[a]] = np.median(v)
    new_acct = (first_day >= C.day_of("2024-07-01")) & (med_hl <= 5)
    cand = new_acct & (ngames >= 20)
    slot_new = new_acct[d["pid"]]
    kidx = n_p  # game index of the account (window count = lifetime for new accounts)
    out = {"new_accounts": int(new_acct.sum()), "candidates_20plus": int(cand.sum())}
    print(out, flush=True)
    post = ~d["in_sample"]
    _, fr = np.unique(d["g"][post], return_index=True)
    med = np.median(day[post][fr])

    def rounds(s0, before=None, max_rounds=4):
        """Purification loop. before: only use / purify games with day < before."""
        s = s0.copy()
        flags_prev = None
        hist = []
        r = d["r"]
        use = np.ones(n, bool) if before is None else day < before
        for it in range(max_rounds):
            x_net = net_of_others(d, r, s, n_games)
            # candidate accounts whose first 20 games are all usable
            m20 = slot_new & (kidx < 20) & use
            c20 = np.bincount(d["pid"][m20], minlength=npl)
            ok = cand & (c20 == 20)
            xs = np.bincount(d["pid"][m20], weights=x_net[m20], minlength=npl)[ok] / 20
            s2 = np.bincount(d["pid"][m20], weights=d["v"][m20], minlength=npl)[ok] / 400
            q, pars = mixture_em(xs, s2)
            flags = np.zeros(npl, bool)
            flags[np.flatnonzero(ok)[q > 0.5]] = True
            pflag = np.zeros(npl)
            pflag[np.flatnonzero(ok)] = q
            theta = np.zeros(n)
            blk_means = {}
            for a, b in BLOCKS:
                mb = slot_new & (kidx >= a) & (kidx < b) & flags[d["pid"]] & use
                sb = np.bincount(d["pid"][mb], weights=x_net[mb], minlength=npl)
                cb = np.bincount(d["pid"][mb], minlength=npl)
                th = sb / (cb + 30.0)
                theta[mb] = th[d["pid"][mb]]
                blk_means[f"{a + 1}-{b}"] = float(100 * (sb[flags] / np.maximum(cb[flags], 1)).mean())
            r_p = purify(d, r, theta, n_games)
            r_adj_p = r_p - mu_exact
            qi, m, v, Nh, Np, _ = X.online_predict(d, np.ones(n, bool), np.ones(n, bool), K, r_adj_p)
            s = table[C.exp_bins(Np.astype(np.int64), Nh.astype(np.int64))] + m
            changed = None if flags_prev is None else int((flags != flags_prev).sum())
            hist.append({"round": it + 1, "flagged": int(flags.sum()), "mixture": pars,
                         "changed_flags": changed, "block_excess_pp": blk_means,
                         "games_touched": int((team_sums(d, theta != 0, n_games).sum(1) > 0).sum())})
            print(f"  round {it + 1}: {hist[-1]}", flush=True)
            if flags_prev is not None and changed == 0:
                break
            flags_prev = flags
        return s, r_p, flags, pflag, theta, hist

    s_raw = (P["mu"] + P["m_cf2"]).astype(np.float64)
    print("retroactive purification:", flush=True)
    s_ret, r_ret, flags, pflag, theta, hist = rounds(s_raw)
    out["retroactive"] = {"rounds": hist}
    # census
    side_g = np.load(os.path.join(C.CACHE, "x_side_games.npz"))
    gi = XS.game_lookup({"replay_id": d["replay_id"]}, side_g["replay_ids"])
    tier_names = [str(t) for t in side_g["tier_names"]]
    slot_tier = np.where(gi >= 0, side_g["tier"][np.maximum(gi, 0)], -1)
    region = (d["player_keys"][d["pid"]] >> 40).astype(int)
    touched = np.zeros(n_games, bool)
    touched[d["g"][theta != 0]] = True
    gq = {}
    gday = np.zeros(n_games, np.int64)
    gday[d["g"]] = day
    for y_ in (2024, 2025, 2026):
        for hq in (1, 2):
            a = C.day_of(f"{y_}-{'01' if hq == 1 else '07'}-01")
            b = C.day_of(f"{y_}-07-01") if hq == 1 else C.day_of(f"{y_ + 1}-01-01")
            m = (gday >= a) & (gday < b) & (np.bincount(d["g"], minlength=n_games) > 0)
            if m.sum() > 1000:
                gq[f"{y_}H{hq}"] = float(touched[m].mean())
    first_slot = np.zeros(npl, np.int64) - 1
    fs = np.flatnonzero(n_p == 0)
    first_slot[d["pid"][fs]] = fs
    census = {"flagged_accounts": int(flags.sum()),
              "share_of_candidates": float(flags.sum() / cand.sum()),
              "share_of_games_touched": float(touched[np.bincount(d["g"], minlength=n_games) > 0].mean()),
              "games_touched_by_half_year": gq, "by_tier_of_first_game": {}, "by_region": {}}
    ft = slot_tier[first_slot[cand]]
    for ti, tn in enumerate(tier_names):
        m = ft == ti
        if m.sum() > 100:
            census["by_tier_of_first_game"][tn] = {"candidates": int(m.sum()),
                                                   "flag_rate": float(flags[cand][m].mean())}
    rg = (d["player_keys"][np.flatnonzero(cand)] >> 40).astype(int)
    for rv in np.unique(rg):
        m = rg == rv
        if m.sum() > 100:
            census["by_region"][str(rv)] = {"candidates": int(m.sum()),
                                           "flag_rate": float(flags[cand][m].mean())}
    # flagged accounts over their game index (raw residual)
    traj = {}
    for a, b in ((0, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, 300), (300, 10 ** 6)):
        m = slot_new & flags[d["pid"]] & (kidx >= a) & (kidx < b)
        m2 = slot_new & cand[d["pid"]] & ~flags[d["pid"]] & (kidx >= a) & (kidx < b)
        traj[f"{a + 1}-{b if b < 10 ** 6 else 'up'}"] = {
            "flagged_raw_r_pp": float(100 * d["r"][m].mean()) if m.any() else None,
            "unflagged_new_raw_r_pp": float(100 * d["r"][m2].mean()) if m2.any() else None}
    census["residual_by_game_index"] = traj
    out["census"] = census
    print(json.dumps(census)[:800], flush=True)

    # ---------------- B. estimates and headline
    v2 = post & (day >= med)
    nf = ~flags[d["pid"]]
    mv = v2 & nf
    ds = s_ret[mv] - s_raw[mv]
    opp_flag = np.zeros(n)
    ts_f = team_sums(d, (flags[d["pid"]] & slot_new & (kidx < 100)).astype(float), n_games)
    tt = d["team"].astype(np.int64)
    opp_flag = ts_f[d["g"], 1 - tt]
    # exposure: share of a player's history games that had a flagged opponent (lag-free summary)
    expo = np.bincount(d["pid"], weights=(opp_flag > 0), minlength=npl) / np.maximum(
        np.bincount(d["pid"], minlength=npl), 1)
    ex_v2 = expo[d["pid"][mv]]
    qx = np.quantile(ex_v2, [0.5, 0.9, 0.99])
    out["estimate_change_V2_unflagged"] = {
        "corr": float(np.corrcoef(s_ret[mv], s_raw[mv])[0, 1]),
        "mean_change_pp": float(100 * ds.mean()), "mean_abs_change_pp": float(100 * np.abs(ds).mean()),
        "share_gt_0.5pp": float((np.abs(ds) > 0.005).mean()),
        "by_exposure": {lab: {"slots": int(m.sum()), "mean_change_pp": float(100 * ds[m].mean())}
                        for lab, m in (("below median", ex_v2 < qx[0]), ("top 10%", ex_v2 >= qx[1]),
                                       ("top 1%", ex_v2 >= qx[2]))},
        "exposure_quantiles_share_of_games": qx.tolist()}
    print(out["estimate_change_V2_unflagged"], flush=True)
    # boundary purification (causal for V2)
    print("boundary purification (games before the V1/V2 split only):", flush=True)
    s_bnd, _, flags_b, _, theta_b, hist_b = rounds(s_raw, before=med)
    out["boundary"] = {"rounds": hist_b}
    # game level: V1-fit combiner, V2 test, for each state
    y = np.zeros(n_games)
    wp0 = np.full(n_games, 0.5)
    t0m = d["team"] == 0
    y[d["g"][t0m]] = d["y"][t0m]
    wp0[d["g"][t0m]] = d["wp"][t0m]
    lo = np.log(wp0 / (1 - wp0))
    cnt = np.bincount(d["g"][post], minlength=n_games)
    fit_m = (cnt == 10) & (gday < med)
    test_m = (cnt == 10) & (gday >= med)
    rng = np.random.RandomState(0)
    idx = np.flatnonzero(test_m)
    boots = [rng.choice(idx, len(idx)) for _ in range(200)]
    w0 = C.fit_logistic(lo[fit_m][:, None], y[fit_m])
    ll0 = X.logloss(C.predict(w0, lo[:, None]), y)
    gl = {}
    lls = {}
    for nm, s in (("raw residuals", s_raw), ("retroactive purification (not causal)", s_ret),
                  ("boundary purification (causal)", s_bnd)):
        D = X.team_diff(s, d["g"], d["team"], n_games)
        w = C.fit_logistic(np.column_stack([lo, D])[fit_m], y[fit_m])
        l1 = X.logloss(C.predict(w, np.column_stack([lo, D])), y)
        lls[nm] = l1
        gain, ci = X.boot_gain(ll0, l1, idx)
        gl[nm] = {"gain": gain, "ci": ci, "coef": w.tolist()}
    for nm in list(lls)[1:]:
        dd = [float((lls["raw residuals"][b] - lls[nm][b]).mean()) for b in boots]
        gl[nm]["vs_raw"] = {"est": float((lls["raw residuals"][idx] - lls[nm][idx]).mean()),
                            "ci": [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]}
    # V2 games without any flagged new account in its first 100 games (retro flags)
    clean = test_m & ~touched
    for nm in lls:
        gl[nm]["gain_on_untouched_V2_games"] = float((ll0[clean] - lls[nm][clean]).mean())
    out["game_level_V2"] = gl
    print(json.dumps(gl), flush=True)

    # ---------------- C. new-account prior
    hl_f = np.nan_to_num(hl, nan=99)
    # causal status: max hero level on the account's earlier days
    o2 = np.lexsort((d["replay_id"], day, d["pid"]))
    pid2 = d["pid"][o2]
    day2 = day[o2].astype(np.int64)
    runmax = np.zeros(n)
    hv = hl_f[o2]
    comp = pid2 * 100000 + day2
    fod = np.searchsorted(comp, comp, side="left")
    fop = np.searchsorted(pid2, pid2, side="left")
    # running max by player via segment-wise maximum.accumulate
    cm = np.empty(n)
    starts = np.flatnonzero(np.r_[True, pid2[1:] != pid2[:-1]])
    ends = np.r_[starts[1:], n]
    for a, b in zip(starts, ends):
        cm[a:b] = np.maximum.accumulate(hv[a:b])
    prev_max = np.where(fod > fop, cm[np.maximum(fod - 1, 0)], -1)
    status = np.empty(n, np.int64)
    fd2 = first_day[pid2]
    late = fd2 >= C.day_of("2024-07-01")
    status_o = np.where(~late, 0, np.where(prev_max < 0, 1, np.where(prev_max <= 5, 2, 3)))
    status[o2] = status_o  # 0 old/early window, 1 first day unknown, 2 new low-level, 3 new window but high level
    e_mask = d["in_sample"] & (day >= C.day_of(C.E_START))
    b = C.exp_bins(n_p, n_ph)
    nb = len(C.NP_EDGES) * len(C.NPH_EDGES)
    key = status * nb + b
    ssum = np.bincount(key[e_mask], weights=d["r"][e_mask], minlength=4 * nb)
    scnt = np.bincount(key[e_mask], minlength=4 * nb)
    table4 = ssum / (scnt + 200.0)
    # predall mu uses lagged counts; use the same lag for the prior
    bl = C.exp_bins(P["np_lag"].astype(np.int64), P["nh_lag"].astype(np.int64))
    mu4 = table4[status * nb + bl]
    r_adj4 = d["r"] - table4[key]
    qi, m4, _, _, _, _ = X.online_predict(d, np.ones(n, bool), post, K, r_adj4)
    s4 = np.zeros(n)
    s4[qi] = mu4[qi] + m4
    D4 = X.team_diff(s4, d["g"], d["team"], n_games)
    D0 = X.team_diff(s_raw, d["g"], d["team"], n_games)
    w4 = C.fit_logistic(np.column_stack([lo, D4])[fit_m], y[fit_m])
    l4 = X.logloss(C.predict(w4, np.column_stack([lo, D4])), y)
    dd = [float((lls["raw residuals"][bb] - l4[bb]).mean()) for bb in boots]
    newg = np.zeros(n_games, bool)
    newg[d["g"][status >= 1]] = True
    out["new_account_prior"] = {
        "status_share_V2_slots": {s_: float((status[v2] == s_).mean()) for s_ in range(4)},
        "offset_pp_by_status_first_bins": {
            f"status {s_}": [float(100 * table4[s_ * nb + bb]) for bb in range(0, nb, len(C.NPH_EDGES))]
            for s_ in range(4)},
        "gain_V2": X.boot_gain(ll0, l4, idx)[0],
        "vs_skill_model": {"est": float((lls["raw residuals"][idx] - l4[idx]).mean()),
                           "ci": [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]},
        "vs_skill_model_on_games_with_new_accounts": float(
            (lls["raw residuals"][test_m & newg] - l4[test_m & newg]).mean()),
        "games_with_new_accounts_share_V2": float(newg[test_m].mean())}
    print(out["new_account_prior"], flush=True)

    # ---------------- D. smurf detector (causal, time split)
    side = XS.load()
    x_net0 = net_of_others(d, d["r"], s_raw, n_games)
    det = {}
    names_sb = [str(x) for x in side["sb_names"]]
    for k in (10, 20):
        acc = new_acct & (ngames >= k + 80)
        mk = slot_new & (kidx < k) & acc[d["pid"]]
        mf = slot_new & (kidx >= k) & (kidx < k + 80) & acc[d["pid"]]
        pid_k = d["pid"][mk]
        cntk = np.bincount(pid_k, minlength=npl)
        A = np.flatnonzero(acc & (cntk == k))
        bc = lambda w, m=mk: np.bincount(d["pid"][m], weights=w, minlength=npl)[A]
        rsum = bc(d["r"][mk])
        vsum = bc(d["v"][mk])
        z = rsum / np.sqrt(vsum)
        xnet = bc(x_net0[mk]) / k
        hlm = bc(np.nan_to_num(hl[mk], nan=5)) / k
        zsb = np.column_stack([bc(side["z"][mk][:, j].astype(float)) / k for j in range(side["z"].shape[1])])
        cf = np.nan_to_num(side["conf"][mk].astype(float), nan=np.nanmean(side["conf"]))
        conf = bc(cf) / k
        party = bc((d["party"][mk] != 0).astype(float)) / k
        uph = np.unique(pid_k * HKEY + d["hero"][mk])
        heroes = np.bincount(uph // HKEY, minlength=npl)[A].astype(float)
        span = np.zeros(len(A))
        dmin = np.full(npl, 10 ** 9)
        dmax = np.zeros(npl)
        np.minimum.at(dmin, pid_k, day[mk])
        np.maximum.at(dmax, pid_k, day[mk])
        span = (dmax - dmin)[A]
        fut = np.bincount(d["pid"][mf], weights=x_net0[mf], minlength=npl)[A] / 80
        fut_noise = np.bincount(d["pid"][mf], weights=d["v"][mf], minlength=npl)[A] / 6400
        label = fut > 0.08
        start = first_day[A]
        tr = start < C.day_of("2025-07-01")
        te = ~tr
        feats = {"z only": np.column_stack([z]),
                 "z + net residual": np.column_stack([z, xnet]),
                 "z + hero level": np.column_stack([z, xnet, hlm]),
                 "all (z, level, style, talents, party, variety, pace)": np.column_stack(
                     [z, xnet, hlm, zsb, conf, party, heroes,
                      np.log1p(span)]),
                 "side information only (no outcomes)": np.column_stack(
                     [hlm, conf, party, heroes, np.log1p(span)]),
                 "scoreboard style only": zsb}
        resk = {"accounts": int(len(A)), "train": int(tr.sum()), "test": int(te.sum()),
                "label_rate_train": float(label[tr].mean()), "label_rate_test": float(label[te].mean()),
                "future_latent_sd_pp": float(100 * np.sqrt(max(fut[te].var() - fut_noise[te].mean(), 0))),
                "models": {}}
        for nm, F in feats.items():
            mu_, sd_ = F[tr].mean(0), F[tr].std(0) + 1e-9
            Fz = (F - mu_) / sd_
            w = C.fit_logistic(Fz[tr], label[tr].astype(float), l2=1.0)
            p = C.predict(w, Fz)
            auc = auc_score(p[te], label[te])
            # regression of the future excess (R^2 latent)
            A1 = np.column_stack([np.ones(tr.sum()), Fz[tr]])
            beta = np.linalg.solve(A1.T @ A1 + np.eye(A1.shape[1]), A1.T @ fut[tr])
            pf = np.column_stack([np.ones(te.sum()), Fz[te]]) @ beta
            r2 = 1 - (np.mean((fut[te] - pf) ** 2) - fut_noise[te].mean()) / (fut[te].var() - fut_noise[te].mean())
            thr = np.quantile(p[tr], 1 - label[tr].mean())
            flag = p[te] >= thr
            resk["models"][nm] = {"auc": auc, "future_excess_latent_r2": float(r2),
                                  "precision_at_train_rate": float(label[te][flag].mean()) if flag.any() else None,
                                  "recall_at_train_rate": float(flag[label[te]].mean()),
                                  "mean_future_excess_flagged_pp": float(100 * fut[te][flag].mean()),
                                  "mean_future_excess_unflagged_pp": float(100 * fut[te][~flag].mean())}
        det[f"k={k}"] = resk
        print(f"detector k={k}: " + json.dumps({m: round(v["auc"], 3) for m, v in resk["models"].items()}),
              flush=True)
    out["detector"] = det
    out["detector_feature_names"] = {"style": names_sb}
    with open(os.path.join(C.RESULTS, "p3_x_smurf.json"), "w") as f:
        json.dump(out, f, indent=1)
    np.savez(os.path.join(C.CACHE, "x_smurf_flags.npz"), flags=flags, pflag=pflag,
             flags_boundary=flags_b)
    print(f"done in {time.time() - t_start:.0f}s")


def auc_score(p, y):
    y = y.astype(bool)
    if y.all() or (~y).all():
        return float("nan")
    r = np.argsort(np.argsort(p)) + 1
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


if __name__ == "__main__":
    main()
