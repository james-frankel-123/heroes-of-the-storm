"""
P3 extensions, ban redo, step 2: model-free natural experiment on real bans.

For every snapshot slot (2024-07-01 .. 2026-05-22; 2024-04..06 only builds
history) whose player has 30+ earlier games: the main hero (most games before
this game, play-time order) and concentration (main share).
  treated   the OPPOSING team banned the main: in the first ban phase, or in
            the second phase while this player had not picked yet (pick rank
            >= 5; second-phase bans come after the first five picks)
  control   the main was not banned by anyone and not picked by another player
  excluded  own team banned it; banned by opponents after the player picked;
            picked by someone else (reported separately as "taken by a pick")
Outcomes (pp, team perspective of the player's slot):
  y     win
  WP    population WP of the realized draft (what the ban did to the draft as
        the population model sees it)
  r     y - WP (beyond the population model)
  r_self  r net of the other nine players' skill estimates
        (r - (team S - own s) + opponents' S): the player's own contribution
  s_self  the skill model's estimate for the player on the hero he actually
        played (the model's predicted counterpart of r_self)
  mawp_self  Max's MAWP on the hero actually played, minus 0.5 (raw, and
        scaled by its V1 slot-level slope on r)
Estimates: OLS with player fixed effects (within-player), controlling for the
main's ban rate in that build (share of games where it was banned by either
team) and the opposing team's skill sum, with player-clustered bootstrap CIs.
Strata: one-trick (main share >= 50%), specialist (25-50%), flexible (< 25%);
games on main; MAWP of the main (momentum); real league tier.
Targeting: P(opponents ban the main) vs the expected rate from the hero's
per-team ban rate in that build and tier group.

Usage (from training/): python3 personalization/p3_x_ban_nat.py
Output: results/p3_x_ban_nat.json, cache/x_ban_slotfeat.npz
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import p3_hs_core as C
import p3_x_common as X
import p3_x_ban_feat as F

BANS = os.path.join(C.CACHE, "x_bans.npz")
PICKO = os.path.join(C.CACHE, "pickorder_2024q2.npz")
SLOTFEAT = os.path.join(C.CACHE, "x_ban_slotfeat.npz")


def fe_ols(yv, treat, ctrl, grp, rng, nboot=200):
    """Within-group OLS of yv on [treat, ctrl...]; player-cluster bootstrap."""
    u, inv = np.unique(grp, return_inverse=True)
    Xm = np.column_stack([treat.astype(float)] + list(ctrl))
    cnt = np.bincount(inv).astype(float)

    def demean(a):
        if a.ndim == 1:
            return a - (np.bincount(inv, weights=a) / cnt)[inv]
        return np.column_stack([demean(a[:, j]) for j in range(a.shape[1])])
    Xd, yd = demean(Xm), demean(yv)
    # cluster sums of cross products for fast bootstrap
    k = Xd.shape[1]
    XX = np.zeros((len(u), k, k))
    Xy = np.zeros((len(u), k))
    for a in range(k):
        Xy[:, a] = np.bincount(inv, weights=Xd[:, a] * yd, minlength=len(u))
        for b in range(a, k):
            XX[:, a, b] = XX[:, b, a] = np.bincount(inv, weights=Xd[:, a] * Xd[:, b], minlength=len(u))
    beta = np.linalg.solve(XX.sum(0), Xy.sum(0))
    bs = []
    for _ in range(nboot):
        w = np.bincount(rng.randint(0, len(u), len(u)), minlength=len(u)).astype(float)
        bs.append(np.linalg.solve(np.tensordot(w, XX, 1), w @ Xy)[0])
    return float(beta[0]), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main():
    t0 = time.time()
    rng = np.random.RandomState(0)
    d = C.load_slots()
    n = len(d["pid"])
    P = np.load(os.path.join(C.CACHE, "x_predall.npz"))
    s_self = (P["mu"] + P["m_cf2"]).astype(np.float64)
    if os.path.exists(SLOTFEAT):
        Fz = dict(np.load(SLOTFEAT))
    else:
        Fz = F.compute(d)
        np.savez(SLOTFEAT, **Fz)
    print(f"features ({time.time() - t0:.0f}s)", flush=True)
    n_games = int(d["g"].max()) + 1
    # ---- bans per game (aligned to d's game index)
    B = np.load(BANS)
    grid = np.zeros(n_games, np.int64)
    grid[d["g"]] = d["replay_id"]
    o = np.argsort(B["replay_ids"])
    j = np.searchsorted(B["replay_ids"][o], grid)
    okg = B["replay_ids"][o][np.minimum(j, len(o) - 1)] == grid
    bh = np.where(okg[:, None], B["hero"][o][np.minimum(j, len(o) - 1)], -1)
    bt = np.where(okg[:, None], B["team"][o][np.minimum(j, len(o) - 1)], -1)
    bk = np.where(okg[:, None], B["pick_number"][o][np.minimum(j, len(o) - 1)], -1)
    # ---- pick rank per slot
    po = np.load(PICKO)
    names = [str(x) for x in d["hero_names"]]
    hid = {h: i for i, h in enumerate(names)}
    pk = po["replay_ids"].astype(np.int64) * 128 + np.array([hid.get(h, 127) for h in po["hero"]])
    so = np.argsort(pk)
    key = d["replay_id"].astype(np.int64) * 128 + d["hero"]
    jj = np.searchsorted(pk[so], key)
    okp = pk[so][np.minimum(jj, len(so) - 1)] == key
    rank = np.where(okp, po["pick_rank"][so][np.minimum(jj, len(so) - 1)], -1)
    print(f"bans matched {okg.mean():.4f}, pick ranks matched {okp.mean():.4f}", flush=True)
    # ---- treatment
    main = Fz["main"]
    g, team = d["g"], d["team"].astype(int)
    opp_ban = np.zeros(n, bool)
    opp_ban_late = np.zeros(n, bool)
    own_ban = np.zeros(n, bool)
    phase2 = np.zeros(n, bool)
    for k in range(6):
        hit = bh[g, k] == main
        is_opp = bt[g, k] == 1 - team
        is_own = bt[g, k] == team
        second = bk[g, k] >= 5
        valid = hit & is_opp & (~second | (rank >= 5))
        opp_ban |= valid
        phase2 |= valid & second
        opp_ban_late |= hit & is_opp & second & (rank < 5) & (rank >= 0)
        own_ban |= hit & is_own
    # main picked by another slot of this game
    gk = g.astype(np.int64) * 128 + d["hero"]
    sgk = np.sort(gk)
    mk = g.astype(np.int64) * 128 + main
    jj2 = np.searchsorted(sgk, mk)
    taken = (sgk[np.minimum(jj2, len(sgk) - 1)] == mk) & (d["hero"] != main)
    elig = (Fz["tot"] >= 30) & (d["day"] >= C.day_of("2024-07-01")) & (rank >= 0)
    share = Fz["top_cnt"] / np.maximum(Fz["tot"], 1)
    treat = elig & opp_ban & ~own_ban
    ctrl = elig & ~opp_ban & ~own_ban & ~opp_ban_late & ~taken
    takenm = elig & taken & ~opp_ban & ~own_ban
    # ---- outcomes
    S = np.zeros((n_games, 2))
    np.add.at(S, (g, team), s_self)
    r = d["r"]
    r_self = r - (S[g, team] - s_self) + S[g, 1 - team]
    opp_S = S[g, 1 - team]
    # ban rate of the main per build (either team) and per-team rate per build x tier group
    w = np.load(C.WP)
    bidx = w["build_idx"][np.argsort(w["replay_ids"])]
    bg = bidx[np.searchsorted(np.sort(w["replay_ids"]), grid)]
    nb_build = bg.max() + 1
    banned_any = np.zeros((nb_build, 90))
    gcount = np.bincount(bg, minlength=nb_build).astype(float)
    for k in range(6):
        m = bh[:, k] >= 0
        np.add.at(banned_any, (bg[m], bh[m, k]), 1.0)
    # avoid double counting when both teams ban the same hero (impossible: unique) -> fine
    ban_rate = banned_any / np.maximum(gcount[:, None], 1)
    br_main = ban_rate[bg[g], main]
    # MAWP V1 slot-level calibration (slope of r on mawp - 0.5 at the played hero)
    post = ~d["in_sample"]
    _, fr = np.unique(g[post], return_index=True)
    med = np.median(d["day"][post][fr])  # V1/V2 split
    v1 = post & (d["day"] < med)
    mw = Fz["mawp_own"] - 0.5
    b_mawp = float(np.cov(r[v1], mw[v1])[0, 1] / mw[v1].var())
    b_s = float(np.cov(r[v1], s_self[v1])[0, 1] / s_self[v1].var())
    outcomes = {"y (win)": d["y"].astype(float), "WP of the realized draft": d["wp"], "r = y - WP": r,
                "r_self (net of the other nine players' skill)": r_self,
                "model: s on the hero played": s_self,
                "model: MAWP on the hero played x V1 slope": b_mawp * mw}
    tier = F.real_tier(grid)[g]
    out = {"slots_eligible": int(elig.sum()), "treated": int(treat.sum()), "control": int(ctrl.sum()),
           "taken_by_pick": int(takenm.sum()), "treated_second_phase": int((treat & phase2).sum()),
           "mawp_V1_slope_on_r": b_mawp, "s_V1_slope_on_r": b_s, "effects": {}}
    print(json.dumps({k: v for k, v in out.items() if k != "effects"}), flush=True)
    strata = {"all": np.ones(n, bool), "one-trick (main share >= 50%)": share >= 0.5,
              "specialist (25-50%)": (share >= 0.25) & (share < 0.5), "flexible (< 25%)": share < 0.25,
              "one-trick, 200+ games on main": (share >= 0.5) & (Fz["top_cnt"] >= 200),
              "one-trick, main MAWP >= 0.55": (share >= 0.5) & (Fz["mawp_main"] >= 0.55),
              "one-trick, main MAWP < 0.50": (share >= 0.5) & (Fz["mawp_main"] < 0.50),
              "specialist or one-trick, main MAWP >= 0.55": (share >= 0.25) & (Fz["mawp_main"] >= 0.55),
              "one-trick, main 90-day share >= 50%": (share >= 0.5) & (Fz["dshare_main"] >= 0.5)}
    for tn in (1, 2, 3, 4, 5, 6):
        strata[f"one-trick or specialist, {F.TIER_NAMES[tn]}"] = (share >= 0.25) & (tier == tn)
    strata["one-trick, V2 window (2026-04-01 on)"] = (share >= 0.5) & (d["day"] >= med)
    strata["specialist, V2 window"] = (share >= 0.25) & (share < 0.5) & (d["day"] >= med)
    strata["one-trick, before V1 (to 2026-02-10)"] = (share >= 0.5) & d["in_sample"]
    for sn, sm in strata.items():
        tm, cm = treat & sm, ctrl & sm
        if tm.sum() < 200:
            continue
        sel = tm | cm
        res = {"treated": int(tm.sum()), "control": int(cm.sum()),
               "played_main_when_available": float((d["hero"][cm] == main[cm]).mean()),
               "mean_share": float(share[sel].mean())}
        for on, ov in outcomes.items():
            est, ci = fe_ols(ov[sel], tm[sel], [br_main[sel], opp_S[sel]], d["pid"][sel], rng,
                             nboot=200 if sn in ("all", "one-trick (main share >= 50%)") else 100)
            res[on] = {"effect_pp": 100 * est, "ci_pp": [100 * ci[0], 100 * ci[1]],
                       "raw_diff_pp": float(100 * (ov[tm].mean() - ov[cm].mean()))}
        out["effects"][sn] = res
        print(sn, res["treated"], {k: round(v["effect_pp"], 2) for k, v in res.items() if isinstance(v, dict)},
              flush=True)
    # "taken by a pick" comparison (main unavailable without a targeted ban)
    tk = {}
    for sn in ("all", "one-trick (main share >= 50%)"):
        sm = strata[sn]
        sel = (takenm | ctrl) & sm
        tk[sn] = {}
        for on in ("r = y - WP", "r_self (net of the other nine players' skill)", "model: s on the hero played"):
            est, ci = fe_ols(outcomes[on][sel], (takenm & sm)[sel], [br_main[sel], opp_S[sel]],
                             d["pid"][sel], rng, nboot=100)
            tk[sn][on] = {"effect_pp": 100 * est, "ci_pp": [100 * ci[0], 100 * ci[1]]}
    out["main_taken_by_another_pick"] = tk
    # placebo: opponents ban the main in the second phase after this player already
    # picked another hero (the ban cannot change his pick); vs players who also
    # picked off-main in the first five picks and whose main was not banned
    plc = {}
    base_pl = elig & (rank >= 0) & (rank < 5) & (d["hero"] != main) & ~own_ban & ~taken
    first_ph = np.zeros(n, bool)
    for k in range(6):
        first_ph |= (bh[g, k] == main) & (bt[g, k] == 1 - team) & (bk[g, k] < 5)
    base_pl &= ~first_ph
    for sn in ("all", "one-trick (main share >= 50%)", "specialist (25-50%)"):
        sm = strata[sn] & base_pl
        tmk = sm & opp_ban_late
        if tmk.sum() < 100:
            continue
        plc[sn] = {"placebo_treated": int(tmk.sum())}
        for on in ("y (win)", "r = y - WP", "r_self (net of the other nine players' skill)"):
            est, ci = fe_ols(outcomes[on][sm], tmk[sm], [br_main[sm], opp_S[sm]], d["pid"][sm], rng, nboot=100)
            plc[sn][on] = {"effect_pp": 100 * est, "ci_pp": [100 * ci[0], 100 * ci[1]]}
    out["placebo_ban_after_player_picked"] = plc
    print("placebo", json.dumps(plc), flush=True)
    # by ban phase
    ph = {}
    for sn in ("all", "one-trick (main share >= 50%)"):
        for lab, tmk in (("first phase", treat & ~phase2), ("second phase, before his pick", treat & phase2)):
            sm = strata[sn] & (tmk | ctrl)
            est, ci = fe_ols(r_self[sm], tmk[sm], [br_main[sm], opp_S[sm]], d["pid"][sm], rng, nboot=100)
            ph[f"{sn} | {lab}"] = {"treated": int((tmk & strata[sn]).sum()), "r_self_effect_pp": 100 * est,
                                   "ci_pp": [100 * ci[0], 100 * ci[1]]}
    out["by_ban_phase"] = ph
    print("phase", json.dumps(ph), flush=True)
    # ---- targeting: do opponents ban the main more than the hero's base rate?
    # per-team ban rate of hero h in build b: banned_by_team / (2 * games)
    base = ban_rate[bg[g], main] / 2.0
    tgt = {}
    for sn in ("all", "one-trick (main share >= 50%)", "specialist (25-50%)", "flexible (< 25%)",
               "one-trick, 200+ games on main"):
        sm = strata[sn] & elig & ~own_ban
        any_opp = np.zeros(n, bool)
        for k in range(6):
            any_opp |= (bh[g, k] == main) & (bt[g, k] == 1 - team)
        tgt[sn] = {"slots": int(sm.sum()), "opponents_ban_main": float(any_opp[sm].mean()),
                   "expected_from_hero_ban_rate": float(base[sm].mean()),
                   "ratio": float(any_opp[sm].mean() / max(base[sm].mean(), 1e-9))}
    for tn in (1, 2, 3, 4, 5, 6):
        sm = (share >= 0.5) & elig & ~own_ban & (tier == tn)
        any_opp = np.zeros(n, bool)
        for k in range(6):
            any_opp |= (bh[g, k] == main) & (bt[g, k] == 1 - team)
        if sm.sum() > 500:
            tgt[f"one-trick, {F.TIER_NAMES[tn]}"] = {"slots": int(sm.sum()), "opponents_ban_main": float(any_opp[sm].mean()),
                                                     "expected_from_hero_ban_rate": float(base[sm].mean())}
    out["targeting"] = tgt
    print(json.dumps(tgt), flush=True)
    np.savez(os.path.join(C.CACHE, "x_ban_flags.npz"), treat=treat, ctrl=ctrl, taken=takenm,
             unavailable=elig & (opp_ban | own_ban | taken) & (d["hero"] != main), share=share)
    with open(os.path.join(C.RESULTS, "p3_x_ban_nat.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
