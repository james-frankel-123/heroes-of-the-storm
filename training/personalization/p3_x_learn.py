"""
P3 extensions, task 3: learning curves on newly adopted heroes, and rust.

A. Learning curves. Adoption events from p3_x_adopt (cache/x_adopt_events.npz)
   split by the hero level on the first game (new: <= 5; returning: >= 10).
   Balanced panels (at least K games, K = 10, 20, 30, 50): every event
   contributes every game index, so the mix of players is fixed, but players
   keep a hero after good early games, so early games are selected upward.
   Unbalanced curve: every event still playing at game k contributes game k;
   game k's own outcome does not decide whether it is seen, but the mix
   shifts toward players who stayed. The two bracket the true curve.
   Gap at game k = residual r (y - WP) on the new hero minus the player's own
   baseline: mean r on his established heroes (20+ earlier games on that hero)
   over the same calendar span (first to K-th game on the new hero, widened
   to +-30 days when needed; 10+ baseline games required). Curves are pooled
   across players with player-clustered bootstrap CIs; an exponential
   gap(k) = c + A exp(-(k-1)/tau) is fit to the per-k means. "Games to
   baseline" = first k where the fitted gap is above -1pp. Heterogeneity:
   the across-event sd of the early gap (games 1-10) after removing noise,
   and its correlation with the late gap.
B. Rust. For established players (100+ earlier games), residual after the
   skill model e = r - (offset + CF posterior mean, lag 1 day), by days since
   the player's previous game (play-time order), for the first game back and
   the games after it; per hero, by days since the player last played that
   hero while active otherwise. A game-level check adds rust terms to the
   V1-fit combiner and scores V2.

Usage (from training/): python3 personalization/p3_x_learn.py
Output: results/p3_x_learn.json, results/fig_x_learning_curves.png
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from p3_heroes import HKEY
import p3_hs_core as C
from p3_sd_similarity import HARD

PRED = os.path.join(C.CACHE, "x_predall.npz")
EVENTS = os.path.join(C.CACHE, "x_adopt_events.npz")
GAMETIME = os.path.join(C.CACHE, "gametime_2024q2.npz")


def cluster_ci(vals, clus, n=300, seed=0):
    u, inv = np.unique(clus, return_inverse=True)
    s = np.bincount(inv, weights=vals, minlength=len(u))
    c = np.bincount(inv, minlength=len(u))
    rng = np.random.RandomState(seed)
    bs = []
    for _ in range(n):
        k = rng.randint(0, len(u), len(u))
        bs.append(s[k].sum() / c[k].sum())
    return float(vals.mean()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def fit_exp(k, m, w):
    from scipy.optimize import curve_fit
    f = lambda k, c, A, tau: c + A * np.exp(-(k - 1) / tau)
    try:
        p, _ = curve_fit(f, k, m, p0=[0, -0.05, 5], sigma=1 / np.sqrt(w), maxfev=20000,
                         bounds=([-0.2, -0.5, 0.2], [0.2, 0.5, 200]))
    except Exception:
        return None
    c, A, tau = p
    kk = np.arange(1, 201)
    fk = f(kk, *p)
    above = np.flatnonzero(fk >= -0.01)
    return {"c_pp": float(100 * c), "A_pp": float(100 * A), "tau_games": float(tau),
            "games_to_within_1pp": int(kk[above[0]]) if len(above) else None,
            "fitted_pp_at": {str(x): float(100 * f(x, *p)) for x in (1, 5, 10, 20, 50)}}


def learning(d, P, meta, names):
    E = np.load(EVENTS)
    o, es, el, ef, hl0 = E["order"], E["start"], E["length"], E["first_row"], E["hl0"]
    r = d["r"]
    day = d["day"].astype(np.int64)
    n_ph = P["n_ph"]
    est = n_ph >= 20
    # player's rows sorted by day for baseline windows
    po = np.lexsort((d["replay_id"], day, d["pid"]))
    ppid = d["pid"][po]
    pstart = np.searchsorted(ppid, np.arange(int(d["n_players"])), side="left")
    pend = np.searchsorted(ppid, np.arange(int(d["n_players"])), side="right")
    pday = day[po]
    res = {}
    kz = np.load(os.path.join(C.CACHE, "hs_kernels.npz"))
    ev_, U = np.linalg.eigh(kz["basis_cf2"])
    ax1 = U[:, -1] * np.sqrt(ev_[-1])
    # orient axis 1 so that high-execution heroes (Genji) are positive
    if ax1[names.index("Genji")] < 0:
        ax1 = -ax1
    for lev, K in (("new", 10), ("new", 20), ("new", 30), ("new", 50),
                   ("returning", 10), ("returning", 20), ("returning", 30)):
        lmask = (hl0 <= 5) if lev == "new" else (hl0 >= 10)
        sel = np.flatnonzero(lmask & (el >= K))
        rows = o[es[sel][:, None] + np.arange(K)[None, :]]  # (nev, K)
        base = np.full(len(sel), np.nan)
        base_n = np.zeros(len(sel))
        for i, e in enumerate(sel):
            p = d["pid"][ef[e]]
            hh = d["hero"][ef[e]]
            a, b = pstart[p], pend[p]
            d0, d1 = day[rows[i, 0]], day[rows[i, -1]]
            for pad in (0, 30, 90):
                lo_ = np.searchsorted(pday[a:b], d0 - pad, side="left") + a
                hi_ = np.searchsorted(pday[a:b], d1 + pad, side="right") + a
                rr = po[lo_:hi_]
                rr = rr[est[rr] & (d["hero"][rr] != hh)]
                if len(rr) >= 10:
                    base[i] = r[rr].mean()
                    base_n[i] = len(rr)
                    break
        okb = ~np.isnan(base)
        sel, rows, base = sel[okb], rows[okb], base[okb]
        gap = r[rows] - base[:, None]
        pid_ev = d["pid"][ef[sel]]
        h_ev = d["hero"][ef[sel]]
        role = meta["blizz"][h_ev]
        mP = P["m_player"][ef[sel]]
        vol = P["n_p"][ef[sel]]
        groups = {"all": np.ones(len(sel), bool)}
        for rn in range(6):
            groups[f"role {C.BLIZZ_ROLES[rn]}"] = role == rn
        hard = np.array([n in HARD for n in names])[h_ev]
        groups["high mechanics (hand label)"] = hard
        groups["other heroes"] = ~hard
        a1 = ax1[h_ev]
        q = np.quantile(ax1, [1 / 3, 2 / 3])
        groups["CF axis 1 top third (high-execution)"] = a1 >= q[1]
        groups["CF axis 1 bottom third (straightforward)"] = a1 < q[0]
        qm = np.quantile(mP, [1 / 3, 2 / 3])
        groups["player level top third"] = mP >= qm[1]
        groups["player level bottom third"] = mP < qm[0]
        qv = np.quantile(vol, [1 / 3, 2 / 3])
        groups["player volume top third"] = vol >= qv[1]
        groups["player volume bottom third"] = vol < qv[0]
        bins = [(1, 2), (3, 5), (6, 10), (11, 20), (21, 30), (31, 50)]
        if lev == "returning" or K == 10:
            groups = {k_: v_ for k_, v_ in groups.items() if k_ == "all" or k_.startswith("role")}
        rk = {"events": int(len(sel)), "players": int(len(np.unique(pid_ev))),
              "baseline_games_median": float(np.median(base_n[okb])), "groups": {}}
        for gn, gm in groups.items():
            if gm.sum() < 100:
                continue
            g = {"events": int(gm.sum()), "bins": {}}
            for a_, b_ in bins:
                if b_ > K:
                    continue
                vals = gap[gm][:, a_ - 1:b_].mean(1)
                est_, ci = cluster_ci(vals, pid_ev[gm])
                g["bins"][f"{a_}-{b_}"] = {"gap_pp": 100 * est_, "ci_pp": [100 * ci[0], 100 * ci[1]]}
            km = gap[gm].mean(0)
            wk = np.full(K, gm.sum())
            g["exp_fit"] = fit_exp(np.arange(1, K + 1), km, wk)
            g["per_k_mean_pp"] = (100 * km).round(2).tolist()
            # heterogeneity of the early gap
            early = gap[gm][:, :10].mean(1)
            noise = (d["v"][rows[gm][:, :10]].sum(1) / 100)
            g["early_gap_sd_latent_pp"] = float(100 * np.sqrt(max(early.var() - noise.mean(), 0)))
            if K >= 30:
                late = gap[gm][:, 20:K].mean(1)
                noise_l = d["v"][rows[gm][:, 20:K]].sum(1) / (K - 20) ** 2
                cv = np.cov(early, late)[0, 1]
                vl = max(late.var() - noise_l.mean(), 1e-12)
                ve = max(early.var() - noise.mean(), 1e-12)
                g["early_late_latent_corr"] = float(cv / np.sqrt(ve * vl))
            rk["groups"][gn] = g
        res[f"{lev} K={K}"] = rk
        a = rk["groups"]["all"]
        print(f"{lev} K={K}: {len(sel):,} events; bins {[(k, round(v['gap_pp'], 2)) for k, v in a['bins'].items()]}; "
              f"fit {a['exp_fit']}", flush=True)
    # Unbalanced curve: every event still playing at game k contributes its
    # game k. Game k's own outcome does not decide whether it is observed (no
    # noise selection at k), but the mix shifts toward players who stayed.
    for lev in ("new", "returning"):
        lmask = (hl0 <= 5) if lev == "new" else (hl0 >= 10)
        sel = np.flatnonzero(lmask)
        Kmax = 30
        base = np.full(len(sel), np.nan)
        for i, e in enumerate(sel):
            p = d["pid"][ef[e]]
            hh = d["hero"][ef[e]]
            a, b = pstart[p], pend[p]
            d0 = day[ef[e]]
            lo_ = np.searchsorted(pday[a:b], d0 - 60, side="left") + a
            hi_ = np.searchsorted(pday[a:b], d0 + 60, side="right") + a
            rr = po[lo_:hi_]
            rr = rr[est[rr] & (d["hero"][rr] != hh)]
            if len(rr) >= 10:
                base[i] = r[rr].mean()
        okb = ~np.isnan(base)
        sel, base = sel[okb], base[okb]
        L_ = np.minimum(el[sel], Kmax)
        curve = {}
        pid_ev = d["pid"][ef[sel]]
        for a_, b_ in ((1, 1), (2, 2), (3, 5), (6, 10), (11, 20), (21, 30)):
            vals, cl = [], []
            for k in range(a_, b_ + 1):
                m = L_ >= k
                rows_k = o[es[sel[m]] + (k - 1)]
                vals.append(r[rows_k] - base[m])
                cl.append(pid_ev[m])
            vals, cl = np.concatenate(vals), np.concatenate(cl)
            est_, ci = cluster_ci(vals, cl)
            curve[f"{a_}-{b_}"] = {"slots": int(len(vals)), "gap_pp": 100 * est_, "ci_pp": [100 * ci[0], 100 * ci[1]]}
        res[f"unbalanced {lev}"] = {"events": int(len(sel)), "bins": curve}
        print(f"unbalanced {lev}: {len(sel):,} events; " +
              str({k: round(v["gap_pp"], 2) for k, v in curve.items()}), flush=True)
    return res


def rust(d, P):
    gt = np.load(GAMETIME)
    o = np.argsort(gt["replay_ids"])
    j = np.searchsorted(gt["replay_ids"][o], d["replay_id"])
    ts = gt["ts"][o][np.minimum(j, len(o) - 1)].astype(np.int64)
    e = d["r"] - (P["mu"] + P["m_cf2"])
    n = len(e)
    # play-time order per player
    po = np.lexsort((d["replay_id"], ts, d["pid"]))
    pid = d["pid"][po]
    newp = np.r_[True, pid[1:] != pid[:-1]]
    gapd = np.full(n, np.nan)
    tmp = np.r_[np.nan, np.diff(ts[po]) / 86400.0]
    tmp[newp] = np.nan
    gapd[po] = tmp
    # games since the last break of >= 30 days (0 = first game back)
    brk = np.where(newp, False, np.r_[False, np.diff(ts[po]) >= 30 * 86400])
    idx = np.arange(n)
    last_brk = np.where(brk, idx, -1)
    last_brk = np.maximum.accumulate(last_brk)
    pstart = np.maximum.accumulate(np.where(newp, idx, 0))
    since = np.where(last_brk >= pstart, idx - last_brk, -1)
    since_break = np.full(n, -1)
    since_break[po] = since
    # hero-level gap: days since this player last played this hero
    ho = np.lexsort((d["replay_id"], ts, d["hero"], d["pid"]))
    key = d["pid"][ho] * HKEY + d["hero"][ho]
    newk = np.r_[True, key[1:] != key[:-1]]
    hg = np.r_[np.nan, np.diff(ts[ho]) / 86400.0]
    hg[newk] = np.nan
    hgap = np.full(n, np.nan)
    hgap[ho] = hg
    estab = P["np_lag"] >= 100
    out = {}
    bins = [(0, 0.5), (0.5, 1), (1, 3), (3, 7), (7, 14), (14, 30), (30, 90), (90, 180), (180, 2000)]
    tab = {}
    for a, b in bins:
        m = estab & (gapd >= a) & (gapd < b)
        if m.sum() < 500:
            continue
        est_, ci = cluster_ci(e[m], d["pid"][m], n=200)
        tab[f"{a}-{b}d"] = {"slots": int(m.sum()), "e_pp": 100 * est_, "ci_pp": [100 * ci[0], 100 * ci[1]],
                            "raw_r_pp": float(100 * d["r"][m].mean())}
    out["overall_first_game_after_gap"] = tab
    rec = {}
    for a, b in ((0, 0), (1, 2), (3, 9), (10, 29), (30, 99)):
        m = estab & (since_break >= a) & (since_break <= b)
        if m.sum() < 500:
            continue
        est_, ci = cluster_ci(e[m], d["pid"][m], n=200)
        rec[f"games {a}-{b} after a 30+ day break"] = {"slots": int(m.sum()), "e_pp": 100 * est_,
                                                         "ci_pp": [100 * ci[0], 100 * ci[1]]}
    out["recovery_after_30d_break"] = rec
    hb = [(0, 1), (1, 7), (7, 30), (30, 90), (90, 180), (180, 365), (365, 2000)]
    htab = {}
    active = gapd < 3
    for a, b in hb:
        m = estab & active & (P["nh_lag"] >= 20) & (hgap >= a) & (hgap < b)
        if m.sum() < 500:
            continue
        est_, ci = cluster_ci(e[m], d["pid"][m], n=200)
        htab[f"{a}-{b}d"] = {"slots": int(m.sum()), "e_pp": 100 * est_, "ci_pp": [100 * ci[0], 100 * ci[1]]}
    out["hero_gap_while_active"] = htab
    print("rust overall:", {k: round(v["e_pp"], 2) for k, v in tab.items()}, flush=True)
    print("recovery:", {k: round(v["e_pp"], 2) for k, v in rec.items()}, flush=True)
    print("hero gap:", {k: round(v["e_pp"], 2) for k, v in htab.items()}, flush=True)
    # game-level: do rust terms help the V1-fit combiner on V2?
    post = ~d["in_sample"]
    _, fr = np.unique(d["g"][post], return_index=True)
    med = np.median(d["day"][post][fr])
    n_games = int(d["g"].max()) + 1
    y = np.zeros(n_games)
    wp0 = np.full(n_games, 0.5)
    t0m = d["team"] == 0
    y[d["g"][t0m]] = d["y"][t0m]
    wp0[d["g"][t0m]] = d["wp"][t0m]
    lo = np.log(wp0 / (1 - wp0))
    cnt = np.bincount(d["g"][post], minlength=n_games)
    gday = np.zeros(n_games)
    gday[d["g"]] = d["day"]
    fit_m = (cnt == 10) & (gday < med)
    test_m = (cnt == 10) & (gday >= med)
    sign = np.where(d["team"] == 0, 1.0, -1.0)
    td = lambda x: np.bincount(d["g"], weights=sign * x, minlength=n_games)
    S = td(P["mu"] + P["m_cf2"])
    g = np.nan_to_num(gapd, nan=0.0)
    f1 = td(np.log1p(np.minimum(g, 365)))
    f2 = td(((g >= 30)).astype(float))
    hgc = np.nan_to_num(hgap, nan=0.0)
    f3 = td(np.log1p(np.minimum(hgc, 365)) * (P["nh_lag"] >= 1))
    eps = 1e-7
    res = {}
    base = None
    rng = np.random.RandomState(0)
    idx = np.flatnonzero(test_m)
    boots = [rng.choice(idx, len(idx)) for _ in range(200)]
    for nm, cols in (("skill model", [S]), ("+ log days since last game", [S, f1]),
                     ("+ 30+ day break count", [S, f2]), ("+ log days since hero last played", [S, f3]),
                     ("+ all rust terms", [S, f1, f2, f3])):
        Xm = np.column_stack([lo] + cols)
        w = C.fit_logistic(Xm[fit_m], y[fit_m])
        p = C.predict(w, Xm)
        ll = -(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        if base is None:
            base = ll
        diffs = [float((base[b] - ll[b]).mean()) for b in boots]
        res[nm] = {"gain_vs_skill_model": float((base[test_m] - ll[test_m]).mean()),
                   "ci": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
                   "coef": w.tolist()}
    out["game_level_V2"] = res
    print("game level:", {k: round(v["gain_vs_skill_model"], 5) for k, v in res.items()}, flush=True)
    return out


def figure(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    mids = {"1-1": 1, "2-2": 2, "3-5": 4, "6-10": 8, "11-20": 15.5, "21-30": 25.5, "1-2": 1.5, "31-50": 40.5}
    for key, col, lab in (("new K=30", "tab:blue", "balanced, 30+ games (931 events)"),
                          ("new K=50", "tab:green", "balanced, 50+ games (357 events)"),
                          ("unbalanced new", "k", "unbalanced, all players still on the hero")):
        bins = res["learning"][key]["groups"]["all"]["bins"] if "K=" in key else res["learning"][key]["bins"]
        xs = [mids[k] for k in bins]
        ys = [v["gap_pp"] for v in bins.values()]
        lo_ = [v["gap_pp"] - v["ci_pp"][0] for v in bins.values()]
        hi_ = [v["ci_pp"][1] - v["gap_pp"] for v in bins.values()]
        ax[0].errorbar(xs, ys, yerr=[lo_, hi_], fmt="o-", color=col, label=lab, capsize=2)
    ax[0].legend(fontsize=7)
    ax[0].axhline(0, color="grey", lw=0.5)
    ax[0].set_xlabel("game number on the new hero (hero level <= 5 at the first game)")
    ax[0].set_ylabel("residual minus own established-hero level (pp)")
    t = res["rust"]["overall_first_game_after_gap"]
    labs = list(t)
    ax[1].errorbar(range(len(labs)), [t[k]["e_pp"] for k in labs],
                   yerr=np.array([[t[k]["e_pp"] - t[k]["ci_pp"][0], t[k]["ci_pp"][1] - t[k]["e_pp"]]
                                  for k in labs]).T, fmt="o")
    ax[1].set_xticks(range(len(labs)))
    ax[1].set_xticklabels(labs, rotation=45, fontsize=7)
    ax[1].axhline(0, color="grey", lw=0.5)
    ax[1].set_ylabel("residual after skill model (pp)")
    ax[1].set_xlabel("days since the player's previous game")
    fig.tight_layout()
    fig.savefig(os.path.join(C.RESULTS, "fig_x_learning_curves.png"), dpi=110)


def main():
    t0 = time.time()
    d = C.load_slots()
    P = np.load(PRED)
    P = {k: P[k] for k in P.files}
    meta = C.hero_meta(d["hero_names"])
    names = [str(h) for h in d["hero_names"]]
    out = {"learning": learning(d, P, meta, names)}
    out["rust"] = rust(d, P)
    with open(os.path.join(C.RESULTS, "p3_x_learn.json"), "w") as f:
        json.dump(out, f, indent=1)
    figure(out)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
