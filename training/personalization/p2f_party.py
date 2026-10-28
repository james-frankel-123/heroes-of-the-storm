"""
Paper 2 narrative facts, section 12: party (premade stack) structure.

Inputs: the full-panel slot table (party id per slot: 0 = solo, otherwise
members of one team share an id), the scored arms full_ms_layer (main model)
and A2S on the same games, real league tier per game, and causal at-game
player MMR (cache/mmr_at_game.npz).

  structure  per team: the sorted sizes of its parties of 2+ ("solo" = five
             solo players; "2", "2+2", "3", "3+2", "4", "5")
  describe   distribution over every corpus game with all ten slots, and how
             often the two teams' structures differ
  win        test games (V2 + OOT), one row per team: raw win rate, mean
             result beyond the draft-only WP (y - p_draft), and beyond the
             main model (y - p_full), by own structure, by matchups of the
             two teams' largest stack, and by tier for 3+ stacks
  value      the main model's combiner plus party terms (team differences in
             players in a 2+ party, players in a 3+ party, and a 5-stack
             flag), fit on V1, scored on V2 and OOT; paired by game
  check      the 3+ stack effect beyond the draft, with and without the
             team difference of player-model skill (D) and the team MMR gap

Usage (from training/): nice -n 19 taskset -c 48-51 python3 personalization/p2f_party.py --arms <dir>
Output: results/p2f/party.json
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from p2f_arms import game_meta, lookup  # noqa: E402

STRUCTS = ["solo", "2", "2+2", "3", "3+2", "4", "5"]


def ll(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit_logistic(X, y, iters=60):
    X = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        Hm = (X * (p * (1 - p))[:, None]).T @ X + 1e-9 * np.eye(len(w))
        w += np.linalg.solve(Hm, X.T @ (y - p))
    return w


def team_structures(z):
    """per (game index g, team): structure label index, players in 2+ party, in 3+ party, max stack."""
    g, team, party = z["g"].astype(np.int64), z["team"].astype(np.int64), z["party"].astype(np.int64)
    rid = z["replay_id"].astype(np.int64)
    ng = int(g.max()) + 1
    gt = g * 2 + team
    cnt10 = np.bincount(g, minlength=ng)
    # party sizes
    pm = party != 0
    key = gt[pm] * (1 << 33) + (party[pm] % (1 << 32))
    uk, inv, sz = np.unique(key, return_inverse=True, return_counts=True)
    gt_of = uk // (1 << 33)
    size_slot = np.ones(len(g), np.int64)
    size_slot[np.flatnonzero(pm)] = sz[inv]
    n2 = np.bincount(gt, weights=(size_slot >= 2), minlength=2 * ng)
    n3 = np.bincount(gt, weights=(size_slot >= 3), minlength=2 * ng)
    mx = np.zeros(2 * ng, np.int64)
    np.maximum.at(mx, gt, size_slot)
    # label from sorted party sizes
    lab = np.zeros(2 * ng, np.int64)    # solo
    parts = {}
    o = np.argsort(gt_of, kind="stable")
    for t, s in zip(gt_of[o], sz[o]):
        parts.setdefault(int(t), []).append(int(s))
    for t, ss in parts.items():
        nm = "+".join(str(x) for x in sorted([x for x in ss if x >= 2], reverse=True)) or "solo"
        lab[t] = STRUCTS.index(nm) if nm in STRUCTS else len(STRUCTS)
    grid = np.zeros(ng, np.int64)
    grid[g] = rid
    return {"ng": ng, "complete": cnt10 == 10, "rid": grid, "lab": lab.reshape(ng, 2), "n2": n2.reshape(ng, 2),
            "n3": n3.reshape(ng, 2), "max": mx.reshape(ng, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True)
    a_ = ap.parse_args()
    z = np.load(os.path.join(a_.arms, "x_slots_full.npz"))
    T = team_structures(z)
    out = {}
    comp = T["complete"]
    labs = T["lab"][comp]
    dist = {STRUCTS[k] if k < len(STRUCTS) else "other": float(np.mean(labs.ravel() == k)) for k in range(len(STRUCTS) + 1)}
    out["describe"] = {"games": int(comp.sum()), "team_structure_share": dist,
                       "share_games_structures_differ": float(np.mean(labs[:, 0] != labs[:, 1])),
                       "share_games_largest_stack_differs": float(np.mean(T["max"][comp][:, 0] != T["max"][comp][:, 1])),
                       "share_games_any_stack": float(np.mean((T["max"][comp] >= 2).any(1)))}
    print(json.dumps(out["describe"], indent=1))

    # test games
    M = {k: np.load(os.path.join(a_.arms, f"{k}.npz")) for k in ("full_ms_layer", "A2S")}
    F = M["full_ms_layer"]
    gr = F["g_replay_id"]
    o = np.argsort(T["rid"])
    j = np.minimum(np.searchsorted(T["rid"][o], gr), len(o) - 1)
    ok = T["rid"][o][j] == gr
    gi = o[j]
    assert ok.mean() > 0.99, ok.mean()
    lab, n2, n3, mx = (T[k][gi] for k in ("lab", "n2", "n3", "max"))
    keys, reg, tier, mmr = game_meta()
    gtier = lookup(keys, gr, tier)[0]
    y = F["g_y"].astype(float)
    p0, pf = F["g_p0"], F["g_p"]
    test = F["g_win"] >= 2
    # team rows: team 0 and team 1
    yy = np.r_[y, 1 - y]
    q0 = np.r_[p0, 1 - p0]
    qf = np.r_[pf, 1 - pf]
    own_lab, opp_lab = np.r_[lab[:, 0], lab[:, 1]], np.r_[lab[:, 1], lab[:, 0]]
    own_mx, opp_mx = np.r_[mx[:, 0], mx[:, 1]], np.r_[mx[:, 1], mx[:, 0]]
    tt = np.r_[test, test]
    trow_tier = np.r_[gtier, gtier]

    def cell(m):
        n = int(m.sum())
        if n < 50:
            return None
        se = float(np.sqrt(0.25 / n))
        return {"teams": n, "raw_win": float(yy[m].mean()), "vs_draft_pp": float(100 * (yy[m] - q0[m]).mean()),
                "vs_full_pp": float(100 * (yy[m] - qf[m]).mean()), "se_pp": 100 * se}

    out["by_own_structure"] = {STRUCTS[k]: cell(tt & (own_lab == k)) for k in range(len(STRUCTS))}
    out["by_largest_stack_matchup"] = {f"{a} vs {b}": cell(tt & (own_mx == a) & (opp_mx == b))
                                       for a in range(1, 6) for b in range(1, 6)}
    out["by_largest_stack_difference"] = {str(dd): cell(tt & (own_mx - opp_mx == dd)) for dd in range(-4, 5)}
    TN = {1: "Bronze", 2: "Silver", 3: "Gold", 4: "Platinum", 5: "Diamond", 6: "Master"}
    out["three_plus_by_tier"] = {}
    for t, tn in TN.items():
        m3 = tt & (trow_tier == t) & (own_mx >= 3)
        m1 = tt & (trow_tier == t) & (own_mx == 1)
        out["three_plus_by_tier"][tn] = {"3+ stack": cell(m3), "all solo": cell(m1)}
    for k, v in out["by_own_structure"].items():
        print(k, v and {a: round(b, 2) if isinstance(b, float) else b for a, b in v.items()})
    for k, v in out["three_plus_by_tier"].items():
        print(k, {a: (b and (round(b["vs_draft_pp"], 2), round(b["vs_full_pp"], 2), b["teams"])) for a, b in v.items()})

    # value: party terms in the combiner (fit on V1)
    d2 = (n2[:, 0] - n2[:, 1]).astype(float)
    d3 = (n3[:, 0] - n3[:, 1]).astype(float)
    d5 = ((mx[:, 0] == 5).astype(float) - (mx[:, 1] == 5))
    lo, D = F["g_lo"], F["g_D"]
    fit = F["g_win"] == 1
    Xb = np.column_stack([lo, D])
    Xp = np.column_stack([lo, D, d2, d3, d5])
    wb, wp = fit_logistic(Xb[fit], y[fit]), fit_logistic(Xp[fit], y[fit])
    pb = 1 / (1 + np.exp(-(wb[0] + Xb @ wb[1:])))
    pp = 1 / (1 + np.exp(-(wp[0] + Xp @ wp[1:])))
    Xd = np.column_stack([lo, d2, d3, d5])
    wd0 = fit_logistic(lo[fit][:, None], y[fit])
    wd = fit_logistic(Xd[fit], y[fit])
    pd0 = 1 / (1 + np.exp(-(wd0[0] + lo * wd0[1])))
    pdp = 1 / (1 + np.exp(-(wd[0] + Xd @ wd[1:])))
    val = {"coef_main_plus_party": wp.tolist(), "coef_names": ["const", "logit WP", "D skill", "d 2+ players", "d 3+ players", "d 5-stack"]}
    for wn, wv in (("V2", 2), ("OOT", 3)):
        m = F["g_win"] == wv
        for nm, (pa, pb_) in {"party on top of the main model": (pp, pb), "party on top of the draft only": (pdp, pd0)}.items():
            c1 = ((pa[m] > .5) == (y[m] == 1)).astype(float)
            c0 = ((pb_[m] > .5) == (y[m] == 1)).astype(float)
            dl = 1000 * (ll(pb_[m], y[m]) - ll(pa[m], y[m]))
            val[f"{wn}: {nm}"] = {"acc_diff_pts": float(100 * (c1 - c0).mean()), "se": float(100 * (c1 - c0).std() / np.sqrt(m.sum())),
                                   "ll_gain_x1000": float(dl.mean()), "ll_se": float(dl.std() / np.sqrt(m.sum()))}
    out["value"] = val
    print(json.dumps(val, indent=1))

    # check: 3+ stack effect beyond the draft with and without skill and MMR controls (test team rows)
    # collision-free (game, player) lookup (2026-10-28; see p3_mmr_feats.stamps_at)
    import p3_mmr_feats as MF
    S = M["full_ms_layer"]
    mm, _ = MF.stamps_at(os.path.join(HERE, "cache"), S["s_replay_id"].astype(np.int64), S["s_key"].astype(np.int64))
    sgi = np.minimum(np.searchsorted(gr, S["s_replay_id"]), len(gr) - 1)
    sgn = np.where(S["s_team"] == 0, 1.0, -1.0)
    okm = np.isfinite(mm)
    sm = np.zeros(len(gr))
    cnt = np.bincount(sgi, weights=okm, minlength=len(gr))
    np.add.at(sm, sgi, np.where(okm, sgn * mm, 0))
    dmmr = sm / 5 / 100
    full_mmr = cnt == 10
    r0 = y - p0
    chk = {}
    for wn, m in (("V2+OOT", test & full_mmr),):
        base = np.column_stack([np.ones(m.sum()), d3[m]])
        b0 = np.linalg.lstsq(base, r0[m], rcond=None)[0]
        b1 = np.linalg.lstsq(np.column_stack([base, D[m]]), r0[m], rcond=None)[0]
        b2 = np.linalg.lstsq(np.column_stack([base, D[m], dmmr[m]]), r0[m], rcond=None)[0]
        res = r0[m] - np.column_stack([base, D[m], dmmr[m]]) @ b2
        se = float(np.sqrt(np.mean(res ** 2) / np.sum((d3[m] - d3[m].mean()) ** 2)))
        chk[wn] = {"games": int(m.sum()), "pp_per_3plus_player_beyond_draft": float(100 * b0[1]),
                   "plus_skill_D": float(100 * b1[1]), "plus_skill_and_team_mmr_gap": float(100 * b2[1]), "se_pp": 100 * se,
                   "mean_D_pp_by_own_3plus": float(100 * np.polyfit(d3[m], D[m], 1)[0]),
                   "mmr_gap_per_3plus_player": float(100 * np.polyfit(d3[m], dmmr[m], 1)[0])}
    out["check"] = chk
    print(json.dumps(chk, indent=1))
    json.dump(out, open(os.path.join(HERE, "results", "p2f", "party.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
