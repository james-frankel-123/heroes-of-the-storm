"""
P3 extensions, ban redo, step 3: opponent-aware ban model.

Strength of player p on hero h (pp of win probability beyond the population
WP), several arms, all causal (games before the draft):
  s        phase-1 skill model (experience offset + CF rank-2 GP mean)
  MAWP     Max's MAWP - 0.5, times its V1 slope on the residual
  EWMA     EWMA residual on the hero (10 games), times its V1 slope
  combo    linear model on V1 slots of r_self (r net of the other nine
           players' skill) on: s, MAWP - 0.5, EWMA residual, log1p(games on h),
           share of p's games on h, p's main share, [h is the main], main
           share x [h is not the main], [never played]; the concentration terms
           let one-tricks be worse off their main than the GP pools them
  combo without MAWP / without s (to compare MAWP with the residual skill)
Slot-level and game-level (V1 fit, V2 test) comparison of the arms.

Ban value at every real ban step of the 5,724 held-out V2 lobbies (all ten
players with 50+ earlier games), for every available hero c:
  for each player still to pick, pick probabilities from the imitation model
  (GD log-prob at the current draft state + personal history features),
  renormalized over available heroes; strength = population part (hero win
  rate in the build's causal cumulative stats, tier) + personal part (arm)
  value(c) = sum over opponents still to pick of [E sigma - E sigma | c removed]
             - the same over own players still to pick
  population value: GD pick probabilities (no identities), population part only
Personalized ban = argmax value(c); population ban = argmax population value.
Both are scored with the full personalized value. Pools are NOT used (pick
probabilities come from the imitation model over all available heroes), so
the game's own hero is never added (fixes the audit leak).
Validation: the predicted personal drop from banning a one-trick's main vs the
natural experiment (p3_x_ban_nat.json).

Usage (from training/): python3 personalization/p3_x_ban_model.py
Output: results/p3_x_ban_model.json
"""
import os
import sys
import json
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from p3_heroes import NUM_HEROES
import p3_hs_core as C
import p3_x_common as X
import p3_x_ban_feat as F

SLOTFEAT = os.path.join(C.CACHE, "x_ban_slotfeat.npz")


def slot_design(s, mawp, ewr, n_h, tot, top, is_main, forced=None):
    share = n_h / np.maximum(tot, 1)
    ms = top / np.maximum(tot, 1)
    f = np.zeros_like(share) if forced is None else forced.astype(float)
    return np.stack([s, mawp - 0.5, ewr, np.log1p(n_h), share, ms, is_main.astype(float),
                     ms * (1 - is_main), (n_h == 0).astype(float), ms * f], -1)


ARMS = {"s (skill model)": [0], "MAWP": [1], "EWMA residual": [2],
        "combo": list(range(9)), "combo without MAWP": [0, 2, 3, 4, 5, 6, 7, 8],
        "combo without s": [1, 2, 3, 4, 5, 6, 7, 8],
        "s + concentration terms": [0, 3, 4, 5, 6, 7, 8],
        "combo + forced off main": list(range(10))}


def fit_arms(d, Fz, P):
    """Fit each arm on V1 slots (target r_self), report V2 slot/game metrics."""
    n = len(d["pid"])
    g, team = d["g"], d["team"].astype(int)
    s = (P["mu"] + P["m_cf2"]).astype(np.float64)
    n_games = int(g.max()) + 1
    S = np.zeros((n_games, 2))
    np.add.at(S, (g, team), s)
    r = d["r"]
    r_self = r - (S[g, team] - s) + S[g, 1 - team]
    fl = np.load(os.path.join(C.CACHE, "x_ban_flags.npz"))
    Xs = slot_design(s, Fz["mawp_own"], Fz["ewr_own"], Fz["n_own"], Fz["tot"], Fz["top_cnt"],
                     d["hero"] == Fz["main"], fl["unavailable"])
    post = ~d["in_sample"]
    _, fr = np.unique(g[post], return_index=True)
    med = np.median(d["day"][post][fr])
    v1 = post & (d["day"] < med)
    v2 = post & (d["day"] >= med)
    coefs, res = {}, {}
    _, y, wp0, cnt, gday = X.game_arrays(d)
    lo = np.log(wp0 / (1 - wp0))
    cntp = np.bincount(g[post], minlength=n_games)
    fit_m = (cntp == 10) & (gday < med)
    test_m = (cntp == 10) & (gday >= med)
    w0 = C.fit_logistic(lo[fit_m][:, None], y[fit_m])
    ll0 = X.logloss(C.predict(w0, lo[:, None]), y)
    idx = np.flatnonzero(test_m)
    for nm, cols in ARMS.items():
        A = np.column_stack([np.ones(v1.sum()), Xs[v1][:, cols]])
        b = np.linalg.lstsq(A, r_self[v1], rcond=None)[0]
        coefs[nm] = (cols, b)
        pred = b[0] + Xs[:, cols] @ b[1:]
        wpv, yv = d["wp"][v2], d["y"][v2]
        p1 = np.clip(wpv + pred[v2], 1e-3, 1 - 1e-3)
        llg = 1000 * np.mean(-(yv * np.log(wpv) + (1 - yv) * np.log(1 - wpv))
                             + (yv * np.log(p1) + (1 - yv) * np.log(1 - p1)))
        D = X.team_diff(pred, g, d["team"], n_games)
        wg = C.fit_logistic(np.column_stack([lo, D])[fit_m], y[fit_m])
        l1 = X.logloss(C.predict(wg, np.column_stack([lo, D])), y)
        gain, ci = X.boot_gain(ll0, l1, idx, n=100)
        res[nm] = {"coef": b.tolist(), "slot_ll_gain_x1000_V2": float(llg), "game_gain_V2": gain, "ci": ci,
                   "sd_pp": float(100 * pred[v2].std())}
        print(f"  {nm:28s} slot {llg:+.3f} game {gain:+.5f} [{ci[0]:+.5f},{ci[1]:+.5f}]", flush=True)
    return coefs, res, med


def main():
    t0 = time.time()
    import torch
    torch.set_num_threads(4)
    import p3_dr_core as D
    import p3_dr_imitation as I
    from drift2026 import common
    rng = np.random.RandomState(0)
    d = C.load_slots()
    meta = C.hero_meta(d["hero_names"])
    names = [str(h) for h in d["hero_names"]]
    P = np.load(os.path.join(C.CACHE, "x_predall.npz"))
    Fz = dict(np.load(SLOTFEAT))
    out = {}
    print("strength arms (V1 fit, V2 test):", flush=True)
    coefs, out["strength_arms"], med = fit_arms(d, Fz, P)

    # ---------------- lobbies and per-slot 90-hero features
    L = D.load_lobbies()
    T = D.load_personal()
    picks = L["steps"][:, :, 3]
    prow = picks[picks >= 0].reshape(len(L["replay_id"]), 10)
    npos = np.vectorize(lambda r_: T["pos"][int(r_)])(prow)
    est = (T["n_p"][npos] >= 50).all(1)
    lob = np.flatnonzero((L["day"] >= med) & est)
    rows = np.unique(prow[lob].ravel())
    Q = F.compute(d, rows)
    qpos = {int(r_): i for i, r_ in enumerate(rows)}
    rec = I.recency_features(d, rows)
    iw = np.load(I.OUT)["w"]
    gd = D.GDPolicy(d["hero_names"])
    print(f"lobbies {len(lob):,}, slots {len(rows):,} ({time.time() - t0:.0f}s)", flush=True)

    def strength(row, arm):
        q = qpos[int(row)]
        p = T["pos"][int(row)]
        n_h = Q["q_cnt"][q].astype(float)
        tot = n_h.sum()
        top = n_h.max()
        ism = np.arange(NUM_HEROES) == np.argmax(n_h)
        Xh = slot_design(T["s"][p].astype(float), Q["q_mawp"][q].astype(float), Q["q_ewr10"][q].astype(float),
                         n_h, np.full(NUM_HEROES, tot), np.full(NUM_HEROES, top), ism)
        cols, b = coefs[arm]
        return b[0] + Xh[:, cols] @ b[1:]

    fcoef = float(coefs["combo + forced off main"][1][-1])
    out["forced_off_main_coef_pp_per_unit_share"] = 100 * fcoef
    print(f"forced off main: {100 * fcoef:+.2f}pp x main share", flush=True)
    # population hero strength per lobby: causal cumulative stats of the previous build
    w = np.load(C.WP)
    o = np.argsort(w["replay_ids"])
    bidx_all = w["build_idx"][o]
    present = np.unique(w["build_idx"])
    builds = common.load_patch_index()["builds"]
    stats_cache = {}

    def pop_strength(gi):
        bi = int(bidx_all[L["g"][gi]])
        key = builds[present[int(np.searchsorted(present, bi)) - 1]]
        if key not in stats_cache:
            stats_cache[key] = common.load_patch_stats("cumulative", key)
        st = stats_cache[key]
        tier = str(L["tier"][gi])
        hw = st.hero_wr.get(tier, {})
        return np.array([(hw.get(h, 50.0) - 50.0) / 100.0 for h in names])

    arms_eval = ["combo + forced off main", "combo", "s (skill model)", "MAWP", "combo without MAWP"]
    PRIMARY = "combo + forced off main"
    recs = []
    valid = []
    for li, gi in enumerate(lob):
        st = L["steps"][gi]
        mo, to = D.one_hots(str(L["map"][gi]), str(L["tier"][gi]))
        sp = pop_strength(gi)
        pick_step = {int(row): k for k, (h, ty, tm, row) in enumerate(st) if ty == 1}
        team_of = {int(row): tm for (h, ty, tm, row) in st if ty == 1}
        sig = {a: {r_: strength(r_, a) for r_ in pick_step} for a in arms_eval}
        t0_ = np.zeros(NUM_HEROES, np.float32)
        t1_ = np.zeros(NUM_HEROES, np.float32)
        bans = np.zeros(NUM_HEROES, np.float32)
        for k, (h, ty, tm, row) in enumerate(st):
            if ty == 1:
                (t0_ if tm == 0 else t1_)[h] = 1
                continue
            avail = (t0_ + t1_ + bans) == 0
            todo = [r_ for r_, kk in pick_step.items() if kk > k]
            # GD at the current state, at each remaining player's pick step
            B = len(todo)
            lp = gd.logprobs(np.repeat(t0_[None], B, 0), np.repeat(t1_[None], B, 0), np.repeat(bans[None], B, 0),
                             np.repeat(mo[None], B, 0), np.repeat(to[None], B, 0),
                             np.array([pick_step[r_] for r_ in todo], np.float32), np.ones(B, np.float32),
                             np.repeat(avail[None], B, 0))
            fake = {"row": np.array(todo), "recpos": np.array([qpos[r_] for r_ in todo]), "lp": lp.astype(np.float32)}
            Xf = I.feature_tensor(fake, T, rec, meta)
            u = (Xf * iw).sum(-1)
            u = np.where(avail[None], u, -1e9)
            Pi = np.exp(u - u.max(1, keepdims=True))
            Pi /= Pi.sum(1, keepdims=True)
            Pg = np.where(avail[None], np.exp(lp), 0)
            Pg /= Pg.sum(1, keepdims=True)
            sign = np.array([1.0 if team_of[r_] != tm else -1.0 for r_ in todo])

            mains = np.array([int(np.argmax(Q["q_cnt"][qpos[r_]])) for r_ in todo])
            mshare = np.array([Q["q_cnt"][qpos[r_]].max() / max(Q["q_cnt"][qpos[r_]].sum(), 1) for r_ in todo])

            def value(Pm, sig_rows, forced_coef=0.0):
                # E sigma with all available minus with c removed, for every c
                E = (Pm * sig_rows).sum(1)  # (B,)
                # removing c: (E - P_c sig_c) / (1 - P_c)
                Ec = (E[:, None] - Pm * sig_rows) / np.maximum(1 - Pm, 1e-9)
                # a player whose main is removed plays his replacement "forced"
                Ec[np.arange(B), mains] += forced_coef * mshare
                return (sign[:, None] * (E[:, None] - Ec)).sum(0)  # (90,)
            vals = {}
            for a in arms_eval:
                fc = fcoef if a == PRIMARY else 0.0
                sr = np.stack([sp + sig[a][r_] for r_ in todo])
                vals[a] = np.where(avail, value(Pi, sr, fc), -1e9)
                pers_only = np.stack([sig[a][r_] for r_ in todo])
                vals[a + "|personal part"] = np.where(avail, value(Pi, pers_only, fc), -1e9)
            vpop = np.where(avail, value(Pg, np.repeat(sp[None], B, 0)), -1e9)
            vpop_imit = np.where(avail, value(Pi, np.repeat(sp[None], B, 0)), -1e9)
            full = vals[PRIMARY]
            c_pers, c_pop, c_popi = int(np.argmax(full)), int(np.argmax(vpop)), int(np.argmax(vpop_imit))
            # one-trick opponents still to pick, with main available
            ot = []
            for jj, r_ in enumerate(todo):
                if sign[jj] < 0:
                    continue
                q = qpos[r_]
                n_h = Q["q_cnt"][q]
                m_ = int(np.argmax(n_h))
                if n_h.sum() >= 30 and avail[m_]:
                    drops = {}
                    for a in arms_eval:
                        sj = sig[a][r_]
                        Ej = (Pi[jj] * sj).sum()
                        Ec = (Ej - Pi[jj, m_] * sj[m_]) / max(1 - Pi[jj, m_], 1e-9)
                        if a == PRIMARY:
                            Ec += fcoef * mshare[jj]
                        drops[a] = float(Ej - Ec)
                    ot.append((r_, m_, float(Pi[jj, m_]), drops, float(n_h[m_] / n_h.sum())))
            ot_all = ot
            ot = [x for x in ot_all if x[4] >= 0.5]
            recs.append({"gi": int(gi), "step": k, "phase": 1 if k < 4 else 2, "n_todo_opp": int((sign > 0).sum()),
                         "ot_all": ot_all,
                         "v_pers_ban": float(full[c_pers]), "v_pop_ban": float(full[c_pop]),
                         "v_popimit_ban": float(full[c_popi]), "v_real_ban": float(full[h]) if avail[h] else np.nan,
                         "differ": c_pers != c_pop, "pers_part_of_pers_ban": float(vals["combo|personal part"][c_pers]),
                         "pers_ban_is_an_opp_main": any(x[1] == c_pers for x in ot),
                         "best_by_arm": {a: float(vals[a][int(np.argmax(vals[a]))]) for a in arms_eval},
                         "pers_ban_by_arm_scored_combo": {a: float(full[int(np.argmax(vals[a]))]) for a in arms_eval},
                         "ot": ot, "real_hero": int(h), "team": int(tm),
                         "real_ban_personal": {a: (float(vals[a + "|personal part"][h]) if avail[h] else 0.0)
                                               for a in arms_eval},
                         "real_ban_total_combo": float(full[h]) if avail[h] else 0.0})
            bans[h] = 1
        if (li + 1) % 500 == 0:
            print(f"  lobbies {li + 1}/{len(lob)} ({time.time() - t0:.0f}s)", flush=True)

    # ---------------- summaries
    def q(x):
        x = np.asarray(x, float)
        x = x[~np.isnan(x)]
        return {"mean_pp": float(100 * x.mean()), "p10": float(100 * np.percentile(x, 10)),
                "p50": float(100 * np.percentile(x, 50)), "p90": float(100 * np.percentile(x, 90)),
                "p99": float(100 * np.percentile(x, 99)), "n": int(len(x))}

    def cimean(x):
        x = np.asarray(x, float)
        x = x[~np.isnan(x)]
        bs = [x[rng.randint(0, len(x), len(x))].mean() for _ in range(500)]
        return [float(100 * x.mean()), float(100 * np.percentile(bs, 2.5)), float(100 * np.percentile(bs, 97.5))]
    R = recs
    has_ot = np.array([len(r_["ot"]) > 0 for r_ in R])
    gain = np.array([r_["v_pers_ban"] - r_["v_pop_ban"] for r_ in R])
    gain_real = np.array([r_["v_pers_ban"] - r_["v_real_ban"] for r_ in R])
    ph1 = np.array([r_["phase"] == 1 for r_ in R])
    summ = {"ban_decisions": len(R), "share_with_a_one_trick_opponent_to_pick_and_main_available": float(has_ot.mean()),
            "value_of_personalized_ban": q([r_["v_pers_ban"] for r_ in R]),
            "value_of_population_ban (scored with the personalized model)": q([r_["v_pop_ban"] for r_ in R]),
            "value_of_real_ban": q([r_["v_real_ban"] for r_ in R]),
            "gain_over_population_ban": q(gain), "gain_over_population_ban_ci": cimean(gain),
            "gain_over_real_ban_ci": cimean(gain_real),
            "share_personalized_differs_from_population": float(np.mean([r_["differ"] for r_ in R])),
            "personal_part_of_personalized_ban": q([r_["pers_part_of_pers_ban"] for r_ in R]),
            "share_personalized_ban_is_an_opponents_one_trick_main": float(np.mean([r_["pers_ban_is_an_opp_main"] for r_ in R])),
            "with a one-trick opponent: gain over population ban": cimean(gain[has_ot]),
            "with a one-trick opponent: share differ": float(np.mean([R[i]["differ"] for i in np.flatnonzero(has_ot)])),
            "without: gain over population ban": cimean(gain[~has_ot]),
            "first phase: gain over population ban": cimean(gain[ph1]),
            "second phase: gain over population ban": cimean(gain[~ph1])}
    # per draft (team's three bans summed), assuming the decisions add up
    per_team = {}
    for r_ in R:
        per_team.setdefault((r_["gi"], r_["team"]), []).append(r_["v_pers_ban"] - r_["v_pop_ban"])
    summ["gain_per_draft_sum_of_3_bans"] = cimean([sum(v) for v in per_team.values()])
    summ["arms: mean value of the best ban (each arm's own value)"] = {
        a: float(100 * np.mean([r_["best_by_arm"][a] for r_ in R])) for a in arms_eval}
    summ["arms: mean combo value of each arm's chosen ban"] = {
        a: float(100 * np.mean([r_["pers_ban_by_arm_scored_combo"][a] for r_ in R])) for a in arms_eval}
    # one-trick main: predicted personal drop (personal part, that player only)
    ot = [x for r_ in R for x in r_["ot"]]
    summ["one_trick_main_ban: predicted"] = {
        "cases": len(ot), "pick_probability_of_main (imitation)": float(np.mean([x[2] for x in ot])),
        **{f"expected personal drop, {a} (pp)": float(100 * np.mean([x[3][a] for x in ot])) for a in arms_eval},
        **{f"drop given he would have picked the main, {a} (pp)": float(
            100 * np.mean([x[3][a] / max(x[2], 1e-3) for x in ot])) for a in arms_eval}}
    nat = json.load(open(os.path.join(C.RESULTS, "p3_x_ban_nat.json")))
    # predicted vs realized main-ban drop by concentration stratum (first ban
    # decision of each lobby only, so each opponent counts once)
    first = {}
    for r_ in R:
        if r_["gi"] not in first:
            first[r_["gi"]] = r_
    allx = [x for r_ in first.values() for x in r_["ot_all"]]
    sh = np.array([x[4] for x in allx])
    val = {}
    for lab, m in (("one-trick (main share >= 50%)", sh >= 0.5), ("specialist (25-50%)", (sh >= 0.25) & (sh < 0.5)),
                   ("flexible (< 25%)", sh < 0.25)):
        xs = [allx[i] for i in np.flatnonzero(m)]
        val[lab] = {"opponents": len(xs), "pick_probability_of_main": float(np.mean([x[2] for x in xs])),
                    **{f"predicted drop, {a} (pp)": float(100 * np.mean([x[3][a] for x in xs])) for a in arms_eval},
                    "realized drop (natural experiment, r_self, pp)": -nat["effects"][lab][
                        "r_self (net of the other nine players' skill)"]["effect_pp"],
                    "realized CI": [-v for v in nat["effects"][lab]["r_self (net of the other nine players' skill)"]["ci_pp"]][::-1],
                    "realized: played main when available": nat["effects"][lab]["played_main_when_available"]}
    summ["main_ban_drop_predicted_vs_realized"] = val
    for k_ in ("one-trick (main share >= 50%)", "one-trick, V2 window (2026-04-01 on)"):
        if k_ in nat["effects"]:
            summ[f"one_trick_main_ban: realized, {k_} (r_self)"] = nat["effects"][k_][
                "r_self (net of the other nine players' skill)"]
            summ[f"one_trick_main_ban: realized, {k_}: played main when available"] = nat["effects"][k_][
                "played_main_when_available"]
    # realized check: team residual (y - WP) on the personal value of the bans the
    # two teams really made (own minus opponents'), lobby level
    n_games, yg, wpg, _, _ = X.game_arrays(d)
    acc = {}
    for r_ in R:
        a_ = acc.setdefault(r_["gi"], {a: np.zeros(2) for a in arms_eval})
        for a in arms_eval:
            a_[a][r_["team"]] += r_["real_ban_personal"][a]
    gis = np.array(list(acc))
    gg = L["g"][gis]
    res0 = yg[gg] - wpg[gg]  # team 0 residual
    rc = {}
    for a in arms_eval:
        x = np.array([acc[gi_][a][0] - acc[gi_][a][1] for gi_ in gis])
        A = np.column_stack([np.ones(len(x)), x])
        beta = np.linalg.lstsq(A, res0, rcond=None)[0]
        bs = []
        for _ in range(500):
            ii = rng.randint(0, len(x), len(x))
            bs.append(np.linalg.lstsq(A[ii], res0[ii], rcond=None)[0][1])
        qx = np.percentile(x, [10, 90])
        rc[a] = {"slope (1 = calibrated)": float(beta[1]), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                 "sd_of_ban_difference_pp": float(100 * x.std()),
                 "residual_pp_bottom_decile": float(100 * res0[x <= qx[0]].mean()),
                 "residual_pp_top_decile": float(100 * res0[x >= qx[1]].mean())}
    summ["realized_check_real_bans (team residual on own - opponent personal ban value)"] = rc
    summ["lobbies"] = int(len(gis))
    out["ban_model"] = summ
    print(json.dumps(summ, indent=1), flush=True)
    with open(os.path.join(C.RESULTS, "p3_x_ban_model.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
