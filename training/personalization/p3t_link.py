"""
Paper 2 link check: does the missing player factor explain the hero-generic value function's overconfidence
on real games? (CPU only; judge window 2026-07-01..09-11; nothing from the sealed build or sealed ids.)

  b  = paper 1's main-effects logistic baseline (paper1_revision/overrating.py MainEffects, current split)
  I  = logit WP_pop - logit b               (the hero-generic model's interaction claim)
  P  = the drafter's personal term, pcal:  0.864 (b2 dS + b3 dO), logit units, drafter state as of 06-30
  team-row GLMs (both teams of each game; SE clustered by game), on half B of the window:
    y ~ offset(logit b) + alpha + beta I
    y ~ offset(logit b) + alpha + beta I + gamma P            (gamma free)
    y ~ offset(logit b + gamma_A P) + alpha + beta I          (gamma fixed from half A)
  overall, by own-team min3 (<25 / 25-99 / >=100) and by the team's count of off-pool or low-familiarity
  slots (0 / 1 / 2+; a slot is flagged when the player had fewer than 5 earlier games on the hero at game
  time or the hero is off his role). Composition: share of flagged slots by min3 bin on human teams.

Run (from training/): python personalization/p3t_link.py  -> results/p3t_link/link.json, LINK.md
Games are joined to map and tier through the worlds game sets (NODRIFT, T17) by their ten players, heroes
and day; replay ids are never read from the slot table.
"""
import os
import sys
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, TRAINING)
sys.path.insert(0, os.path.join(TRAINING, "worlds"))
os.environ.setdefault("P1_TIERS", "site")
RES = os.path.join(HERE, "results", "p3t_link")
H = 90
HEADLINE = [0.0, 1.2158510866657697, 4.242860855417209, -0.031338977228257625]
PSCALE = 0.864
LOWFAM = 5


def sig(x):
    return 1 / (1 + np.exp(-x))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def join_map_tier(src_rows):
    """(m, t, T0-is-src-team-0 flag) for judge-window games via the worlds NODRIFT / T17 sets.
    Key: the sorted heroes of each team + y; a game matches when its two hero sets and result agree
    with a unique worlds game on the same composition pair (ambiguous keys are dropped)."""
    import data as WD
    hero = src_rows["hero"]
    y0 = src_rows["y0"]
    A = np.sort(hero[:, :5], 1)
    B = np.sort(hero[:, 5:], 1)
    enc = lambda X: (X * (H ** np.arange(5)[::-1])).sum(1)        # 90^5 < 2^63
    kA, kB = enc(A), enc(B)
    lut = {}
    dup = set()
    for name in ("NODRIFT", "T17"):
        z = WD.load(name)
        e0, e1 = enc(z["T0"]), enc(z["T1"])
        for a, b, m, t, y in zip(e0.tolist(), e1.tolist(), z["m"].tolist(), z["t"].tolist(), z["y"].tolist()):
            k = (min(a, b), max(a, b))
            if k in lut:
                dup.add(k)
            lut[k] = (m, t, a, y)
    n = len(y0)
    m = np.full(n, -1)
    t = np.full(n, -1)
    for i, (a, b) in enumerate(zip(kA.tolist(), kB.tolist())):
        k = (min(a, b), max(a, b))
        if k in dup or k not in lut:
            continue
        mm, tt, a0, yy = lut[k]
        won_A = yy if a0 == a else 1.0 - yy                      # worlds y: team T0 won
        if won_A != y0[i]:
            continue
        m[i], t[i] = mm, tt
    return m, t


def glm(y, off, X, cl):
    import paper1_revision.overrating as OV
    w, se = OV.glm(y, off, X, cl=cl)
    return w, se


def main():
    import player as PL
    import p3t_hier as PH
    import data as WD
    from paper1_revision.overrating import MainEffects, base_path
    src = np.load(PL.SRC)
    cov = np.load(PL.COV)
    jw = np.flatnonzero(src["jw"])
    rows = {k: src[k][jw] for k in ("hero", "y0", "wp0", "nprior", "pid")}
    G = len(jw)
    m, t = join_map_tier(rows)
    ok = (m >= 0)
    maps, tiers = WD.meta()
    ME = MainEffects(base_path("current"))
    mname = np.array([maps[x] if 0 <= x < len(maps) else "?" for x in m])
    tname = np.array([tiers[x] if 0 <= x < len(tiers) else "?" for x in t])
    lb = ME.logit(rows["hero"][:, :5], rows["hero"][:, 5:], mname, tname)        # team 0 (src slots 0-4)
    lw = logit(rows["wp0"])
    I = lw - lb
    S, O = cov["j_s"].astype(float), cov["j_off"].astype(float)
    sg = np.r_[np.ones(5), -np.ones(5)]
    b2, b3 = PSCALE * HEADLINE[2], PSCALE * HEADLINE[3]
    P = b2 * (S @ sg) + b3 * (O @ sg)
    flag = (rows["nprior"] < LOWFAM) | (O > 0)
    nflag = np.stack([flag[:, :5].sum(1), flag[:, 5:].sum(1)], 1)
    zero = (rows["nprior"] == 0)
    min3 = src["j_min3"]
    y = rows["y0"]
    half = np.arange(G) % 2
    # team rows: (game, team) with own-team perspective
    gid = np.r_[np.arange(G), np.arange(G)]
    Y = np.r_[y, 1 - y]
    LB, II, PP = np.r_[lb, -lb], np.r_[I, -I], np.r_[P, -P]
    M3 = np.r_[min3[:, 0], min3[:, 1]]
    NF = np.r_[nflag[:, 0], nflag[:, 1]]
    OK = np.r_[ok, ok]
    HB = np.r_[half, half] == 1
    HA = ~HB
    # gamma on half A (team rows, with I)
    wA, _ = glm(Y[OK & HA], LB[OK & HA], np.column_stack([np.ones((OK & HA).sum()), II[OK & HA], PP[OK & HA]]), gid[OK & HA])
    gA = float(wA[2])

    def fits(sel):
        s = sel & OK & HB
        n = int(s.sum())
        if n < 500:
            return {"team_rows": n}
        X1 = np.column_stack([np.ones(n), II[s]])
        w1, e1 = glm(Y[s], LB[s], X1, gid[s])
        w2, e2 = glm(Y[s], LB[s], np.column_stack([X1, PP[s]]), gid[s])
        w3, e3 = glm(Y[s], LB[s] + gA * PP[s], X1, gid[s])
        return {"team_rows": n, "beta": [float(w1[1]), float(e1[1])], "alpha": [float(w1[0]), float(e1[0])],
                "beta_with_P": [float(w2[1]), float(e2[1])], "gamma_P": [float(w2[2]), float(e2[2])],
                "beta_with_P_fixedA": [float(w3[1]), float(e3[1])],
                "corr_I_P": float(np.corrcoef(II[s], PP[s])[0, 1]), "sd_I": float(II[s].std()), "sd_P": float(PP[s].std())}
    out = {"games": int(G), "joined": int(ok.sum()), "half_B_games": int((ok & (half == 1)).sum()),
           "gamma_halfA": gA, "lowfam_n": LOWFAM,
           "definitions": "I = logit WP_pop - logit main-effects b; P = pcal personal term (logit); team rows, "
                          "SE clustered by game; beta, gamma on half B; flagged slot = n_ph < 5 at game time or off-role",
           "overall": fits(np.ones_like(OK)),
           "by_min3": {"<25": fits(M3 < 25), "25-99": fits((M3 >= 25) & (M3 < 100)), ">=100": fits(M3 >= 100)},
           "by_flagged": {"0": fits(NF == 0), "1": fits(NF == 1), "2+": fits(NF >= 2)}}
    NZ = np.r_[zero[:, :5].sum(1), zero[:, 5:].sum(1)]
    out["by_never_played"] = {"0": fits(NZ == 0), "1": fits(NZ == 1), "2+": fits(NZ >= 2)}
    out["by_min3_x_flagged"] = {f"{a}|{b}": fits(ma & mb)
                                for a, ma in (("<25", M3 < 25), (">=25", M3 >= 25))
                                for b, mb in (("0", NF == 0), ("1+", NF >= 1))}
    # composition (all judge-window human teams, both halves)
    F = np.r_[flag[:, :5], flag[:, 5:]]
    Z = np.r_[zero[:, :5], zero[:, 5:]]
    comp = {}
    for lab, s in (("<25", M3 < 25), ("25-99", (M3 >= 25) & (M3 < 100)), ("100-499", (M3 >= 100) & (M3 < 500)), (">=500", M3 >= 500)):
        comp[lab] = {"teams": int(s.sum()), "share_teams": float(s.mean()),
                     "flagged_slot_share": float(F[s].mean()), "never_played_slot_share": float(Z[s].mean()),
                     "teams_with_2plus_flagged": float((F[s].sum(1) >= 2).mean()),
                     "mean_P_pts_logit": float(PP[s].mean())}
    out["composition"] = comp
    os.makedirs(RES, exist_ok=True)
    json.dump(out, open(os.path.join(RES, "link.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
