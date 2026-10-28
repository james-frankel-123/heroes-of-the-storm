"""
Part A of P3_HERO_RECS_AND_REPLAYS: "heroes you'll become good at", a
backtest of hero recommenders on the full_1017 panel.

Set-up. At a cutoff day t, using only history rows with day < t, every
recommender scores each pulled player's 90 heroes. Candidates are the heroes
the player has played fewer than 5 times (variant: 10) in their full history
before t. After t we follow every row of the player's full history (corpus
slots and Heroes Profile rows) up to the end of the store (2026-09-28). A
candidate is "taken up" when it reaches 10 games after t (tiers 20 and 30).
Realized edge is the mean raw residual r in pp (y - draft WP on corpus
games, y - the hero's quarterly win rate on HP rows):
  early = games 1-10 after t (learning cost);
  later = games 11-50 after t (needs 20+ games after t).

Cutoffs:
  t0 = 2026-02-10 (end of E, the fit window of every model): all pulled
       players, follow-up 7.5 months;
  t1 = 2025-06-01: held-out half of players only (the params were fit on
       the other half, whose target rows reach into E after t1), follow-up
       16 months.

Recommenders (scores per player x hero, higher = recommend):
  main      full_ms posterior mean (win + scoreboard, the main model);
  main_nosb the same params with the scoreboard evidence removed;
  imp2      full_imp2 posterior mean (win + play preference);
  today     today's win-only CF kernel (A1 kernels, with role columns);
  pref      imp2's preference prior only (W z, no results at all);
  sbsim     scoreboard similarity: per-hero ridge of a player's cell
            residual on their hero-relative metric profile (fit on cells
            with 20+ games before t), applied to the player's profile;
  hero_wr   hero's raw win rate on corpus games in the 90 days before t;
  popular   hero's pick count in the 90 days before t;
  role_best same Blizzard role as the player's best hero (main model,
            20+ games), ties by popularity;
  random    uniform.

Run from training/: python personalization/p3_hr_backtest.py [--cut t0|t1]
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import p3_hr_common as HC  # noqa: E402
import p3_hs_core as C  # noqa: E402

H = 90
CUTS = {"t0": "2026-02-10", "t1": "2025-06-01"}
RECS = ["main", "main_nosb", "imp2", "impmet", "today", "pref", "sbsim", "hero_wr", "popular", "role_best", "random"]


# ------------------------------------------------------------------ outcomes

def post_outcomes(a, t, players, t_end=None):
    """Per (player, hero) after t: games, early edge (games 1-10), later edge
    (games 11-50), first-game r, all in pp; plus game-ordered r for
    trajectories (games 1-50 sums by block)."""
    pos = np.full(int(a["n_players"]), -1, np.int64)
    pos[players] = np.arange(len(players))
    sel = (a["day"] >= t) & (pos[a["pid"]] >= 0)
    if t_end is not None:
        sel &= a["day"] < t_end
    rows = np.flatnonzero(sel)
    n = len(players)
    cell = pos[a["pid"][rows]] * H + a["hero"][rows].astype(np.int64)
    o = np.lexsort((a["day"][rows], cell))       # aug rows are day-sorted; stable within day
    rows, cell = rows[o], cell[o]
    brk = np.flatnonzero(np.r_[True, cell[1:] != cell[:-1]])
    rank = np.arange(len(rows)) - np.repeat(brk, np.diff(np.r_[brk, len(rows)]))
    r = a["r"][rows].astype(np.float64) * 100
    out = {"n": np.bincount(cell, None, n * H).reshape(n, H)}
    for nm, lo, hi in (("e1", 0, 10), ("e2", 10, 50), ("f1", 0, 1), ("b30", 0, 30), ("b2_30", 20, 50)):
        m = (rank >= lo) & (rank < hi)
        out[nm + "_s"] = np.bincount(cell[m], r[m], n * H).reshape(n, H)
        out[nm + "_c"] = np.bincount(cell[m], None, n * H).reshape(n, H)
    out["first_day"] = np.full(n * H, -1, np.int64)
    out["first_day"][cell[brk]] = a["day"][rows[brk]]
    out["first_day"] = out["first_day"].reshape(n, H)
    return out


# ------------------------------------------------------------------ recommenders

def sbsim_scores(st, fitmask, k_prof=20.0, ridge=50.0, min_games=20):
    """Per-hero ridge of the cell mean residual (r_adj, pp) on the player's
    hero-relative metric profile from their OTHER heroes, fit on cells with
    min_games+ games before t of the fit players; score = beta_h . profile."""
    X, Cm = st["X"].astype(np.float64), st["Cm"].astype(np.float64)
    Xt, Ct = X.sum(1), Cm.sum(1)                                   # (n, M)
    M = X.shape[2]
    ybar = np.where(st["P"] > 0, st["S"] / np.maximum(st["P"], 1e-9), 0.0) * 100  # (n, H), r_adj units pp
    prof_all = Xt / (Ct + k_prof)
    beta = np.zeros((H, M + 1))
    info = {}
    for h in range(H):
        sel = fitmask & (st["N"][:, h] >= min_games)
        if sel.sum() < 200:
            continue
        F = (Xt[sel] - X[sel, h]) / (Ct[sel] - Cm[sel, h] + k_prof)
        F = np.c_[np.ones(len(F)), F]
        nh = st["N"][sel, h]
        wt = nh / (nh + 150.0)
        A = (F * wt[:, None]).T @ F + ridge * np.diag(np.r_[0.0, np.ones(M)])
        beta[h] = np.linalg.solve(A, (F * wt[:, None]).T @ ybar[sel, h])
        info[h] = int(sel.sum())
    score = prof_all @ beta[:, 1:].T + beta[:, 0][None, :]
    return score, beta, info


def recommenders(a, t, players, st, fitmask, rng):
    roles = HC.hero_roles(a["hero_names"])["blizz"]
    sc = {}
    t1 = time.time()
    for nm, tag, kw in (("main", "full_ms", {}), ("main_nosb", "full_ms", {"use_metrics": False}),
                        ("imp2", "full_imp2", {}), ("impmet", "impmet", {}), ("today", "today", {})):
        m, v = HC.post_all(HC.params(tag), st, **kw)
        sc[nm] = m.astype(np.float64) * 100
        if nm == "main":
            sc["_main_var"] = v.astype(np.float64) * 1e4
    st0 = dict(st, S=np.zeros_like(st["S"]), P=np.zeros_like(st["P"]))
    sc["pref"] = HC.post_all(HC.params("full_imp2"), st0)[0].astype(np.float64) * 100
    sc["sbsim"], beta, info = sbsim_scores(st, fitmask)
    print(f"  model scores {time.time() - t1:.0f}s; sbsim heroes fit {len(info)}", flush=True)
    win = (a["s_day"] >= t - 90) & (a["s_day"] < t)
    hh = a["s_hero"][win].astype(np.int64)
    cnt = np.bincount(hh, None, H)
    wr = np.bincount(hh, a["s_y"][win].astype(np.float64), H) / np.maximum(cnt, 1)
    n = len(players)
    sc["hero_wr"] = np.broadcast_to(100 * (wr - 0.5), (n, H)).copy()
    sc["popular"] = np.broadcast_to(np.log1p(cnt), (n, H)).copy()
    best_m = np.where(st["N"] >= 20, sc["main"], -np.inf)
    has_best = np.isfinite(best_m.max(1))
    best = best_m.argmax(1)
    same = (roles[None, :] == roles[best][:, None]) & has_best[:, None]
    pr = np.argsort(np.argsort(cnt)) / (H - 1)
    sc["role_best"] = same.astype(np.float64) + 0.5 * pr[None, :]
    sc["random"] = rng.rand(n, H)
    return sc, {"sbsim_beta": beta, "hero_wr_pp": 100 * (wr - 0.5), "pop": cnt, "roles": roles}


# ------------------------------------------------------------------ evaluation

def cluster_boot(stat_fn, pid_idx, n_players, reps=200, seed=1):
    """Poisson bootstrap over players: stat_fn(weights per row) -> float."""
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(reps):
        wp_ = rng.poisson(1.0, n_players).astype(np.float64)
        vals.append(stat_fn(wp_[pid_idx]))
    return float(np.std(vals, ddof=1))


def wmean(x, w):
    return float((x * w).sum() / max(w.sum(), 1e-12))


def evaluate(sc, out, st, cand, active, tiers=(10, 20, 30)):
    """Per recommender: within-player rank of each candidate, then realized
    edges of the taken-up candidates by rank."""
    n = cand.shape[0]
    res = {}
    took = cand & (out["n"] >= 10) & active[:, None]
    e1 = np.where(out["e1_c"] > 0, out["e1_s"] / np.maximum(out["e1_c"], 1), np.nan)
    e2 = np.where(out["e2_c"] > 0, out["e2_s"] / np.maximum(out["e2_c"], 1), np.nan)
    ncand = cand.sum(1)
    for nm in RECS:
        s = np.where(cand, sc[nm], -np.inf)
        # rank 0 = top recommendation among the player's candidates (ties random via tiny jitter)
        jit = np.random.RandomState(5).rand(*s.shape) * 1e-9
        order = np.argsort(-(s + jit), 1)
        rk = np.empty_like(order)
        np.put_along_axis(rk, order, np.arange(H)[None, :].repeat(n, 0), 1)
        pct = rk / np.maximum(ncand[:, None] - 1, 1)
        r_nm = {}
        for tier in tiers:
            T = took & (out["n"] >= tier)
            pi, hi = np.nonzero(T)
            if len(pi) < 50:
                continue
            p_ = pct[pi, hi]
            for en, E in (("early", e1), ("later", e2)):
                if en == "later" and tier < 20:
                    continue
                y_ = E[pi, hi]
                ok = np.isfinite(y_)
                pi2, p2, y2 = pi[ok], p_[ok], y_[ok]
                # within-player demeaned (players with 2+ taken-up heroes)
                cntp = np.bincount(pi2, None, n)
                mp = np.bincount(pi2, y2, n) / np.maximum(cntp, 1)
                multi = cntp[pi2] >= 2
                yd = y2 - mp[pi2]
                top = p2 <= 1 / 3
                bot = p2 > 2 / 3

                def tb(w, y=y2, m=np.ones(len(y2), bool)):
                    return wmean(y[top & m], w[top & m]) - wmean(y[bot & m], w[bot & m])

                def slope(w, y=yd, m=multi):
                    x = p2[m] - wmean(p2[m], w[m])
                    return float(-(w[m] * x * y[m]).sum() / max((w[m] * x * x).sum(), 1e-12))
                one = np.ones(len(y2))
                d_tb = tb(one)
                d_tbw = tb(one, yd, multi)
                sl = slope(one)
                r_nm[f"{en}_{tier}"] = {
                    "heroes": int(len(y2)), "players": int((cntp > 0).sum()),
                    "top_third_n": int(top.sum()), "bottom_third_n": int(bot.sum()),
                    "top_third_edge": wmean(y2[top], one[top]), "bottom_third_edge": wmean(y2[bot], one[bot]),
                    "top_minus_bottom": d_tb, "top_minus_bottom_se": cluster_boot(tb, pi2, n),
                    "within_top_minus_bottom": d_tbw,
                    "within_top_minus_bottom_se": cluster_boot(lambda w: tb(w, yd, multi), pi2, n),
                    "within_slope_top_vs_bottom": sl, "within_slope_se": cluster_boot(slope, pi2, n),
                    "within_players": int((cntp >= 2).sum())}
        # precision at 3: of each active player's top-3 candidates, the share
        # taken up (10+ games), and taken up with a positive later edge
        top3 = (rk < 3) & cand & active[:, None]
        bot3 = (rk >= ncand[:, None] - 3) & cand & active[:, None]
        good = took & (out["n"] >= 20) & np.nan_to_num(e2 > 0)
        r_nm["p_at_3"] = {"taken_up_top3": float(took[top3].mean()), "taken_up_bottom3": float(took[bot3].mean()),
                          "good_top3": float(good[top3].mean()), "good_bottom3": float(good[bot3].mean()),
                          "base_taken_up": float(took[cand & active[:, None]].mean()),
                          "base_good": float(good[cand & active[:, None]].mean())}
        # literal top-3 vs bottom-3 among a player's own taken-up heroes (players with 6+)
        pi, hi = np.nonzero(took & (out["n"] >= 20))
        y_ = e2[pi, hi]
        p_ = pct[pi, hi]
        cntp = np.bincount(pi, None, n)
        diffs = []
        for p in np.flatnonzero(cntp >= 6):
            m = pi == p
            o = np.argsort(p_[m])
            yy = y_[m][o]
            diffs.append(yy[:3].mean() - yy[-3:].mean())
        r_nm["own_top3_vs_bottom3_later"] = {"players": len(diffs),
                                            "mean_diff": float(np.mean(diffs)) if diffs else None,
                                            "se": float(np.std(diffs, ddof=1) / np.sqrt(len(diffs))) if len(diffs) > 2 else None}
        res[nm] = r_nm
        # rank percentile per candidate for downstream use
        sc["_pct_" + nm] = pct
    return res


def calibration(sc, out, cand, active):
    """Model scores are in pp: slope of realized edge on the score among
    taken-up heroes, within player, for the model recommenders."""
    took = cand & (out["n"] >= 20) & active[:, None]
    e2 = np.where(out["e2_c"] > 0, out["e2_s"] / np.maximum(out["e2_c"], 1), np.nan)
    e1 = np.where(out["e1_c"] > 0, out["e1_s"] / np.maximum(out["e1_c"], 1), np.nan)
    res = {}
    pi, hi = np.nonzero(took)
    n = cand.shape[0]
    for nm in ("main", "main_nosb", "imp2", "today", "pref", "sbsim"):
        x = sc[nm][pi, hi]
        r = {}
        for en, E in (("early", e1), ("later", e2)):
            y = E[pi, hi]
            ok = np.isfinite(y)
            p2, x2, y2 = pi[ok], x[ok], y[ok]
            cntp = np.bincount(p2, None, n)
            m = cntp[p2] >= 2
            xd = x2 - (np.bincount(p2, x2, n) / np.maximum(cntp, 1))[p2]
            yd = y2 - (np.bincount(p2, y2, n) / np.maximum(cntp, 1))[p2]

            def sl(w, xd=xd, yd=yd, m=m):
                return float((w[m] * xd[m] * yd[m]).sum() / max((w[m] * xd[m] ** 2).sum(), 1e-12))

            def slp(w, x2=x2, y2=y2):
                xx = x2 - wmean(x2, w)
                return float((w * xx * y2).sum() / max((w * xx * xx).sum(), 1e-12))
            r[en] = {"pooled_slope": slp(np.ones(len(x2))), "pooled_slope_se": cluster_boot(slp, p2, n),
                     "within_slope": sl(np.ones(len(x2))), "within_slope_se": cluster_boot(sl, p2, n),
                     "sd_score_within": float(np.sqrt((xd[m] ** 2).mean())), "n": int(m.sum())}
        res[nm] = r
    return res



POOL_CUTS = [("2024-08-10", "heldout"), ("2025-02-10", "heldout"), ("2025-08-10", "heldout"), ("2026-02-10", "all")]
SETS = [("tried_first_game", 1, "f1"), ("taken10_early", 10, "e1"), ("taken20_early", 20, "e1"),
        ("taken20_later", 20, "e2"), ("taken30_later", 30, "e2")]


def records(a, t, players, fitmask_players, follow, max_n, rng):
    """One cutoff: per-candidate records for every candidate the active
    player tried (1+ games in [t, t + follow))."""
    st = HC.state_at(a, t, players)
    out = post_outcomes(a, t, players, t + follow)
    active = (st["N"].sum(1) >= 50) & (out["n"].sum(1) >= 30)
    cand = (st["N"] < max_n) & active[:, None]
    sc, extra = recommenders(a, t, players, st, fitmask_players, rng)
    ncand = cand.sum(1)
    pcts = {}
    for nm in RECS:
        s = np.where(cand, sc[nm], -np.inf) + rng.rand(*cand.shape) * 1e-9
        order = np.argsort(-s, 1)
        rk = np.empty_like(order)
        np.put_along_axis(rk, order, np.arange(H)[None, :].repeat(len(players), 0), 1)
        pcts[nm] = rk / np.maximum(ncand[:, None] - 1, 1)
    # combination: mean of the main model's and the preference prior's ranks
    sc["combo"] = -(pcts["main"] + pcts["pref"])
    s = np.where(cand, sc["combo"], -np.inf) + rng.rand(*cand.shape) * 1e-9
    order = np.argsort(-s, 1)
    rk = np.empty_like(order)
    np.put_along_axis(rk, order, np.arange(H)[None, :].repeat(len(players), 0), 1)
    pcts["combo"] = rk / np.maximum(ncand[:, None] - 1, 1)
    pi, hi = np.nonzero(cand)
    rec = {"pid": players[pi], "hero": hi.astype(np.int16), "t": np.full(len(pi), t, np.int32),
           "n_pre": st["N"][pi, hi].astype(np.int16), "n_post": out["n"][pi, hi].astype(np.int32),
           "ncand": ncand[pi].astype(np.int16)}
    for k in ("e1", "e2", "f1", "b30", "b2_30"):
        c = out[k + "_c"][pi, hi]
        rec[k] = np.where(c > 0, out[k + "_s"][pi, hi] / np.maximum(c, 1), np.nan).astype(np.float32)
    for nm in RECS + ["combo"]:
        rec["pct_" + nm] = pcts[nm][pi, hi].astype(np.float32)
        rec["s_" + nm] = sc[nm][pi, hi].astype(np.float32)
    rec["s_main_var"] = sc["_main_var"][pi, hi].astype(np.float32)
    print(f"  cut {t}: players {len(players):,}, active {active.sum():,}, candidates {len(pi):,}, "
          f"tried {(rec['n_post'] >= 1).sum():,}, taken10 {(rec['n_post'] >= 10).sum():,}, "
          f"taken20 {(rec['n_post'] >= 20).sum():,}", flush=True)
    return rec, {"active": int(active.sum()), "cand_per_player": float(ncand[active].mean())}


def pooled_eval(R, recs, reps=200):
    gid = R["pid"].astype(np.int64) * 100000 + R["t"]             # player x cutoff group
    ug, ginv = np.unique(gid, return_inverse=True)
    up, pinv = np.unique(R["pid"], return_inverse=True)
    ng, npl = len(ug), len(up)
    boot = np.random.RandomState(3).poisson(1.0, (reps, npl)).astype(np.float64)
    res = {}
    for sname, tier, col in SETS:
        sel = (R["n_post"] >= tier) & np.isfinite(R[col])
        g, p, y = ginv[sel], pinv[sel], R[col][sel].astype(np.float64)
        cg = np.bincount(g, None, ng)
        multi = cg[g] >= 2
        yd = y - (np.bincount(g, y, ng) / np.maximum(cg, 1))[g]
        rs = {"heroes": int(sel.sum()), "players": int(len(np.unique(p))), "mean_edge": float(y.mean()),
              "within_heroes": int(multi.sum()), "within_groups": int((cg >= 2).sum())}
        for nm in recs:
            x = R["pct_" + nm][sel].astype(np.float64)
            top, bot = x <= 1 / 3, x > 2 / 3

            def stats(w):
                d_pool = wmean(y[top], w[top]) - wmean(y[bot], w[bot])
                m = multi
                d_with = wmean(yd[top & m], w[top & m]) - wmean(yd[bot & m], w[bot & m])
                xm = x[m] - wmean(x[m], w[m])
                xd_ = x[m] - (np.bincount(g[m], w[m] * x[m], ng) / np.maximum(np.bincount(g[m], w[m], ng), 1e-12))[g[m]]
                sl = -(w[m] * xd_ * yd[m]).sum() / max((w[m] * xd_ * xd_).sum(), 1e-12)
                return np.array([d_pool, d_with, sl, wmean(y[top], w[top]), wmean(y[bot], w[bot])])
            est = stats(np.ones(len(y)))
            bs = np.array([stats(boot[b][p]) for b in range(reps)])
            se = bs.std(0, ddof=1)
            rs[nm] = {"top_third_edge": est[3], "bottom_third_edge": est[4], "n_top": int(top.sum()), "n_bottom": int(bot.sum()),
                      "top_minus_bottom": est[0], "top_minus_bottom_se": se[0],
                      "within_top_minus_bottom": est[1], "within_top_minus_bottom_se": se[1],
                      "within_slope": est[2], "within_slope_se": se[2]}
        res[sname] = rs
    # take-up rates: share of top-3 / bottom-3 / all candidates taken up (10+), and
    # taken up with positive later edge
    act = np.ones(len(R["pid"]), bool)
    took = R["n_post"] >= 10
    good = (R["n_post"] >= 20) & (np.nan_to_num(R["e2"]) > 0)
    tried = R["n_post"] >= 1
    pa = {"base": {"tried": float(tried.mean()), "taken10": float(took.mean()), "good": float(good.mean())}}
    for nm in recs:
        k = R["pct_" + nm] * np.maximum(R["ncand"] - 1, 1)
        t3 = k < 2.5
        b3 = k > R["ncand"] - 3.5
        pa[nm] = {"tried_top3": float(tried[t3].mean()), "taken10_top3": float(took[t3].mean()),
                  "good_top3": float(good[t3].mean()), "taken10_bottom3": float(took[b3].mean()),
                  "good_bottom3": float(good[b3].mean())}
    res["take_up"] = pa
    return res


def pooled(max_n=5, follow=183):
    t0_ = time.time()
    a = HC.load()
    import p3_rcf as R_
    fit = R_.fit_half(int(a["n_players"]))
    allp = np.sort(a["pulled_pid"]).astype(np.int64)
    rng = np.random.RandomState(0)
    parts, meta = [], {}
    for day_s, who in POOL_CUTS:
        t = C.day_of(day_s)
        players = allp if who == "all" else allp[~fit[allp]]
        fm = fit[players] if who == "all" else np.zeros(len(players), bool) | True
        # sbsim betas: fit on fit-half players at t0; held-out cutoffs fit on
        # the held-out players' own pre-t cells (no post-t data either way)
        rec, m = records(a, t, players, fm, follow, max_n, rng)
        parts.append(rec)
        meta[day_s] = dict(m, players=int(len(players)), who=who)
    Rr = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    np.savez(os.path.join(HC.HR_CACHE, f"pooled_n{max_n}.npz"), **Rr)
    recs = RECS + ["combo"]
    out = {"cuts": meta, "max_n": max_n, "follow_days": follow, "by": {}}
    out["by"]["pooled"] = pooled_eval(Rr, recs)
    for day_s, _ in POOL_CUTS:
        m = Rr["t"] == C.day_of(day_s)
        out["by"][day_s] = pooled_eval({k: v[m] for k, v in Rr.items()}, recs, reps=100)
    HC.save_json(f"pooled_n{max_n}.json", out)
    P = out["by"]["pooled"]
    for sname, _, _ in SETS:
        print(f"  [{sname}] heroes {P[sname]['heroes']:,} within {P[sname]['within_heroes']:,} mean {P[sname]['mean_edge']:+.2f}")
        for nm in recs:
            r = P[sname][nm]
            print(f"    {nm:10s} top {r['top_third_edge']:+.2f} bot {r['bottom_third_edge']:+.2f} "
                  f"diff {r['top_minus_bottom']:+.2f} ({r['top_minus_bottom_se']:.2f}) within {r['within_top_minus_bottom']:+.2f} "
                  f"({r['within_top_minus_bottom_se']:.2f}) slope {r['within_slope']:+.2f} ({r['within_slope_se']:.2f})", flush=True)
    print(f"[pooled n{max_n}] {time.time() - t0_:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut", default="t0", choices=list(CUTS))
    ap.add_argument("--max-n", type=int, default=5, help="candidate: fewer than this many games before t")
    ap.add_argument("--pooled", action="store_true")
    ap.add_argument("--follow", type=int, default=183)
    a_ = ap.parse_args()
    if a_.pooled:
        return pooled(a_.max_n, a_.follow)
    t0_ = time.time()
    a = HC.load()
    import p3_rcf as R
    t = C.day_of(CUTS[a_.cut])
    fit = R.fit_half(int(a["n_players"]))
    players = np.sort(a["pulled_pid"]).astype(np.int64)
    if a_.cut == "t1":
        players = players[~fit[players]]
    st = HC.state_at(a, t, players)
    out = post_outcomes(a, t, players)
    tot_pre = st["N"].sum(1)
    tot_post = out["n"].sum(1)
    active = (tot_pre >= 50) & (tot_post >= 30)
    cand = st["N"] < a_.max_n
    print(f"[backtest {a_.cut} <{a_.max_n}] players {len(players):,}, active {active.sum():,}; "
          f"candidates per active player {cand[active].sum(1).mean():.1f}; taken up (10+) per active player "
          f"{(cand & (out['n'] >= 10))[active].sum(1).mean():.2f}", flush=True)
    rng = np.random.RandomState(0)
    sc, extra = recommenders(a, t, players, st, fit[players] if a_.cut == "t0" else np.ones(len(players), bool), rng)
    res = evaluate(sc, out, st, cand, active)
    cal = calibration(sc, out, cand, active)
    summ = {"cut": CUTS[a_.cut], "max_n": a_.max_n, "players": int(len(players)), "active": int(active.sum()),
            "cand_per_player": float(cand[active].sum(1).mean()),
            "taken10_per_player": float((cand & (out["n"] >= 10))[active].sum(1).mean()),
            "taken20_per_player": float((cand & (out["n"] >= 20))[active].sum(1).mean()),
            "taken30_per_player": float((cand & (out["n"] >= 30))[active].sum(1).mean()),
            "results": res, "calibration": cal, "seconds": round(time.time() - t0_)}
    took = cand & (out["n"] >= 10) & active[:, None]
    e1 = out["e1_s"] / np.maximum(out["e1_c"], 1)
    e2 = out["e2_s"] / np.maximum(out["e2_c"], 1)
    t20 = cand & (out["n"] >= 20) & active[:, None]
    summ["mean_edges"] = {"early_all_taken10": float(e1[took].mean()), "later_taken20": float(e2[t20].mean()),
                          "early_taken20": float(e1[t20].mean()), "first_game_taken10": float(
                              (out["f1_s"] / np.maximum(out["f1_c"], 1))[took].mean())}
    tag = f"{a_.cut}_n{a_.max_n}"
    HC.save_json(f"backtest_{tag}.json", summ)
    np.savez(os.path.join(HC.HR_CACHE, f"backtest_{tag}.npz"), players=players,
             **{k: v.astype(np.float32) for k, v in sc.items()}, N=st["N"].astype(np.float32),
             R=st["R"].astype(np.float32), **{f"o_{k}": v for k, v in out.items()}, active=active, cand=cand,
             sbsim_beta=extra["sbsim_beta"])
    for nm in RECS:
        r = res[nm]
        e = r.get("early_10", {})
        l = r.get("later_20", {})
        print(f"  {nm:10s} early(10+) top-bot {e.get('top_minus_bottom', np.nan):+.2f} ({e.get('top_minus_bottom_se', np.nan):.2f}) "
              f"within {e.get('within_top_minus_bottom', np.nan):+.2f} ({e.get('within_top_minus_bottom_se', np.nan):.2f}) | "
              f"later(20+) top-bot {l.get('top_minus_bottom', np.nan):+.2f} ({l.get('top_minus_bottom_se', np.nan):.2f}) "
              f"within {l.get('within_top_minus_bottom', np.nan):+.2f} ({l.get('within_top_minus_bottom_se', np.nan):.2f}) "
              f"slope {l.get('within_slope_top_vs_bottom', np.nan):+.2f} ({l.get('within_slope_se', np.nan):.2f}) | "
              f"p@3 taken {r['p_at_3']['taken_up_top3']:.3f} good {r['p_at_3']['good_top3']:.3f}", flush=True)
    print(f"[backtest {tag}] done {time.time() - t0_:.0f}s", flush=True)


if __name__ == "__main__":
    main()
