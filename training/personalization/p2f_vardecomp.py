"""
Paper 2 narrative facts, section 11, check: how one game's outcome variance
(about 0.25) splits into the draft, each player's stable overall effect, each
player's stable hero-specific effect, and the rest; what the matchmaker
removes; and out-of-sample confirmation, with young accounts excluded.

  draft      variance of the draft-only WP across games, and its out-of-sample
             Brier skill (V2 + OOT games, arm A2S)
  stable     split-half covariances on corpus slots (full panel slot table,
             r = y - draft WP, which includes the lobby's average MMR), in
             182-day windows (one every 91 days): each cell's games split into
             an earlier and a later half; the covariance of the two half means
             across cells is the variance of the part that is stable over the
             window. Overall: cell = player (20+ games in each half). Hero-
             specific: cell = player x hero, value minus the player's own
             mean in the same half (10+ games in each half). Shares are of one
             game's variance 0.25; "10 players" multiplies by 10 (independent
             slots). Repeated for players with 300+ and 1,000+ corpus games
             before the window.
  matchmaker causal at-game player MMR (cache/mmr_at_game.npz): spread of team
             MMR difference in real lobbies vs random teams from the same
             lobby's ten players and vs random players; converted to
             win-probability points with the slope of y on the team MMR
             difference.
  oos        Brier skill (share of outcome variance explained out of sample)
             of draft-only and full models, all games and games whose ten
             players all have 300+ / 1,000+ earlier games (full history).

Usage (from training/): nice -n 19 taskset -c 44-47 python3 personalization/p2f_vardecomp.py --arms <dir>
Output: results/p2f/vardecomp.json
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def brier_skill(p, y):
    return float(1 - np.mean((y - p) ** 2) / np.var(y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True)
    a_ = ap.parse_args()
    out = {}
    # ---------------- draft and out of sample
    oos = {}
    for arm in ("A2S", "full_ms_layer"):
        M = np.load(os.path.join(a_.arms, f"{arm}.npz"))
        gi = np.minimum(np.searchsorted(M["g_replay_id"], M["s_replay_id"]), len(M["g_replay_id"]) - 1)
        ng = len(M["g_replay_id"])
        minp = np.full(ng, np.inf)
        np.minimum.at(minp, gi, M["s_np"].astype(float))
        res = {}
        for lab, thr in (("all games", 0), ("all ten players 300+ earlier games", 300), ("all ten players 1,000+", 1000)):
            m = (M["g_win"] >= 2) & (minp >= thr)
            y = M["g_y"][m]
            res[lab] = {"games": int(m.sum()), "var_y": float(np.var(y)),
                        "var_draft_wp": float(np.var(M["g_p0"][m])),
                        "brier_skill_draft": brier_skill(M["g_p0"][m], y), "brier_skill_full": brier_skill(M["g_p"][m], y),
                        "acc_draft": float(((M["g_p0"][m] > .5) == (y == 1)).mean()),
                        "acc_full": float(((M["g_p"][m] > .5) == (y == 1)).mean())}
        oos[arm] = res
        print(arm, json.dumps(res, indent=0))
    out["out_of_sample"] = oos

    # ---------------- stable shares by split halves
    z = np.load(os.path.join(a_.arms, "x_slots_full.npz"))
    day, pid, hero = z["day"].astype(np.int64), z["pid"].astype(np.int64), z["hero"].astype(np.int64)
    r = z["y"].astype(np.float64) - z["wp"].astype(np.float64)
    H = 90
    o = np.lexsort((day, pid))
    starts = np.arange(day.min(), day.max() - 182, 91)
    # corpus games before each window per player
    stable = {}
    for thr in (0, 300, 1000):
        acc = {"overall": [], "hero": []}
        for s in starts:
            m = (day >= s) & (day < s + 182)
            idx = np.flatnonzero(m)
            # earlier/later halves per player (overall) and per cell (hero), by time
            for kind in ("overall", "hero"):
                cell = pid[idx] if kind == "overall" else pid[idx] * H + hero[idx]
                oo = np.lexsort((day[idx], cell))
                cs = cell[oo]
                u, st, inv = np.unique(cs, return_index=True, return_inverse=True)
                cnt = np.bincount(inv)
                rank = np.arange(len(cs)) - st[inv]
                h2 = rank >= cnt[inv] / 2
                rr = r[idx][oo]
                pp = pid[idx][oo]
                n1 = np.bincount(inv, weights=~h2)
                n2 = np.bincount(inv, weights=h2)
                s1 = np.bincount(inv, weights=np.where(h2, 0, rr))
                s2 = np.bincount(inv, weights=np.where(h2, rr, 0))
                x1, x2 = s1 / np.maximum(n1, 1), s2 / np.maximum(n2, 1)
                pu = u if kind == "overall" else u // H
                if kind == "hero":
                    # the player's own mean in the same half
                    pa, pinv = np.unique(pp, return_inverse=True)
                    P1 = np.bincount(pinv, weights=np.where(h2, 0, rr)) / np.maximum(np.bincount(pinv, weights=~h2), 1)
                    P2 = np.bincount(pinv, weights=np.where(h2, rr, 0)) / np.maximum(np.bincount(pinv, weights=h2), 1)
                    j = np.searchsorted(pa, pu)
                    x1, x2 = x1 - P1[j], x2 - P2[j]
                    keep = (n1 >= 10) & (n2 >= 10)
                else:
                    keep = (n1 >= 20) & (n2 >= 20)
                if thr:
                    prior = np.bincount(pid[day < s], minlength=int(pid.max()) + 1)
                    keep &= prior[pu] >= thr
                if keep.sum() < 100:
                    continue
                c = np.cov(x1[keep], x2[keep])[0, 1]
                acc[kind].append((c, int(keep.sum())))
        res = {}
        for kind, v in acc.items():
            cv = np.array([a for a, _ in v])
            nn = np.array([b for _, b in v])
            tau2 = float(np.average(cv, weights=nn))
            res[kind] = {"windows": len(v), "cells_per_window_median": float(np.median(nn)),
                         "stable_sd_pp": float(100 * np.sqrt(max(tau2, 0))), "stable_var": tau2,
                         "share_of_one_game_per_player": tau2 / 0.25, "share_all_ten_players": 10 * tau2 / 0.25,
                         "sd_over_windows_pp2": float(1e4 * cv.std())}
        stable[f"players with {thr}+ earlier corpus games"] = res
        print(thr, json.dumps(res))
    out["stable"] = stable

    # ---------------- matchmaker
    # collision-free (game, player) lookup (2026-10-28; see p3_mmr_feats.stamps_at)
    import p3_mmr_feats as MF
    A = np.load(os.path.join(a_.arms, "A2S.npz"))
    q = A["s_win"] >= 2
    mm, _ = MF.stamps_at(os.path.join(HERE, "cache"), A["s_replay_id"][q].astype(np.int64), A["s_key"][q].astype(np.int64))
    g = A["s_replay_id"][q]
    team = A["s_team"][q]
    ug, gidx = np.unique(g, return_inverse=True)
    ok = np.isfinite(mm)
    full = np.bincount(gidx, weights=ok) == 10
    M10 = np.zeros((len(ug), 10))
    slot = np.zeros(len(g), np.int64)
    order = np.lexsort((team, gidx))
    pos_in = np.arange(len(g)) - np.searchsorted(gidx[order], gidx[order])
    slot[order] = pos_in
    sel = full[gidx] & (slot < 10)
    M10[gidx[sel], slot[sel]] = mm[sel]
    T = np.zeros((len(ug), 10), np.int8)
    T[gidx[sel], slot[sel]] = team[sel]
    gf = np.flatnonzero(full)
    M10, T = M10[gf], T[gf]
    dteam = (np.where(T == 0, M10, 0).sum(1) - np.where(T == 1, M10, 0).sum(1)) / 5
    gy = dict(zip(A["g_replay_id"], A["g_y"]))
    y = np.array([gy[x] for x in ug[gf]])
    b = np.polyfit(dteam / 100, y, 1)[0]
    rng = np.random.default_rng(0)
    perm = np.argsort(rng.random(M10.shape), 1)
    Mp = np.take_along_axis(M10, perm, 1)
    d_lobby = (Mp[:, :5].sum(1) - Mp[:, 5:].sum(1)) / 5
    pool = M10.ravel()
    Mr = pool[rng.integers(0, len(pool), M10.shape)]
    d_rand = (Mr[:, :5].sum(1) - Mr[:, 5:].sum(1)) / 5
    out["matchmaker"] = {"games": int(len(gf)), "slope_win_prob_per_100_mmr_team_diff": float(b),
                         "sd_player_mmr_population": float(np.std(pool)),
                         "sd_player_mmr_within_lobby": float(np.std(M10 - M10.mean(1, keepdims=True))),
                         "sd_team_diff_real_mmr": float(dteam.std()),
                         "sd_team_diff_shuffled_within_lobby": float(d_lobby.std()),
                         "sd_team_diff_random_players": float(d_rand.std()),
                         "var_share_real": float((b * dteam.std() / 100) ** 2 / 0.25),
                         "var_share_shuffled_within_lobby": float((b * d_lobby.std() / 100) ** 2 / 0.25),
                         "var_share_random_players": float((b * d_rand.std() / 100) ** 2 / 0.25)}
    print(json.dumps(out["matchmaker"], indent=0))
    json.dump(out, open(os.path.join(HERE, "results", "p2f", "vardecomp.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
