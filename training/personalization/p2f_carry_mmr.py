"""
Paper 2 narrative facts, section 11, check 2: does matchmaking feedback hide
carry-forward? The section 3 estimator (slope of the scored slot's r = y - WP
on the record to date, by games bin; arm A2S, V2 + OOT) with the at-game MMR
of both teams added.

MMR is the causal player rating (cache/mmr_at_game.npz: the latest Heroes
Profile stamp parsed before the game started, P3_MMR_AT_GAME). Controls:
  dTeam   own team's mean at-game MMR minus the opponents' (per 100)
  own     the player's at-game MMR minus own team mean (per 100)
  hero    the player's at-game hero MMR minus own player MMR (per 100), where known
Variants: slope on the record with dTeam + own; + hero MMR; and within bands
of the player's own at-game MMR (quintiles), and within bands of dTeam.
Also the "Max on Ana" example (key passed with --key, never stored).

Usage (from training/): nice -n 19 taskset -c 48-51 python3 personalization/p2f_carry_mmr.py --arms <dir> [--key K --hero NAME]
Output: results/p2f/carry_mmr.json
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BINS = [(30, 59), (60, 119), (120, 249), (250, 499), (500, 799), (800, 10 ** 9), (250, 10 ** 9)]
LABS = ["30-59", "60-119", "120-249", "250-499", "500-799", "800+", "250+ (all)"]


def ols(X, yv):
    X = np.column_stack([np.ones(len(yv)), X])
    return np.linalg.lstsq(X, yv, rcond=None)[0][1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True)
    ap.add_argument("--key", type=int)
    ap.add_argument("--hero")
    a_ = ap.parse_args()
    A = np.load(os.path.join(a_.arms, "A2S.npz"))
    q = A["s_win"] >= 2
    rid, key, team = A["s_replay_id"][q], A["s_key"][q], A["s_team"][q]
    nh = A["s_nh"][q].astype(float)
    npl = A["s_np"][q].astype(float)
    x = np.where(nh > 0, A["s_rec"][q] / np.maximum(nh, 1), np.nan)
    r = A["s_r"][q].astype(float)
    Z = np.load(os.path.join(HERE, "cache", "mmr_at_game.npz"))
    zk = (Z["region"].astype(np.int64) << 40) | Z["blizz_ids"]
    comp = Z["replay_ids"].astype(np.int64) * (1 << 44) + zk
    o = np.argsort(comp)
    comp = comp[o]
    qc = rid.astype(np.int64) * (1 << 44) + key
    j = np.minimum(np.searchsorted(comp, qc), len(comp) - 1)
    ok = comp[j] == qc
    pm = np.where(ok, Z["player_mmr_causal"][o][j], np.nan).astype(float)
    hm = np.where(ok, Z["hero_mmr_causal"][o][j], np.nan).astype(float)
    # team means per game
    gk = rid * 2 + team
    ug, gi = np.unique(gk, return_inverse=True)
    good = np.isfinite(pm)
    s = np.bincount(gi, weights=np.where(good, pm, 0), minlength=len(ug))
    c = np.bincount(gi, weights=good.astype(float), minlength=len(ug))
    tm = s / np.maximum(c, 1)
    opp = np.searchsorted(ug, rid * 2 + (1 - team))
    full = (c[gi] == 5) & (c[opp] == 5)
    dteam = (tm[gi] - tm[opp]) / 100
    own = (pm - tm[gi]) / 100
    hrel = (hm - pm) / 100
    out = {"slots_with_both_teams_mmr": float(full.mean())}
    res = {}
    for (lo, hi), lb in zip(BINS, LABS):
        m = (nh >= lo) & (nh <= hi) & full & np.isfinite(x)
        e = {"slots": int(m.sum()), "slope_no_mmr": float(ols(x[m][:, None], r[m])[0])}
        b = ols(np.column_stack([x[m], dteam[m], own[m]]), r[m])
        e["slope_with_team_and_own_mmr"] = float(b[0])
        e["coef_team_mmr_diff_per100"] = float(b[1])
        e["coef_own_mmr_per100"] = float(b[2])
        mh = m & np.isfinite(hrel)
        b2 = ols(np.column_stack([x[mh], dteam[mh], own[mh], hrel[mh]]), r[mh])
        e["slope_with_hero_mmr_too"] = float(b2[0])
        e["slots_hero_mmr"] = int(mh.sum())
        # how the record relates to the team MMR difference the player then gets
        e["dteam_on_record_per10pp"] = float(ols(x[m][:, None], dteam[m])[0] * 0.1 * 100)
        # within bands of own MMR and of team difference
        qs = np.nanpercentile(pm[m], [20, 40, 60, 80])
        band = np.searchsorted(qs, pm[m])
        e["slope_within_own_mmr_quintiles"] = [float(ols(x[m][band == k][:, None], r[m][band == k])[0]) for k in range(5)]
        qd = np.nanpercentile(dteam[m], [20, 40, 60, 80])
        bd = np.searchsorted(qd, dteam[m])
        e["slope_within_team_diff_quintiles"] = [float(ols(x[m][bd == k][:, None], r[m][bd == k])[0]) for k in range(5)]
        noise = 0.245 / nh[m]
        e["noise_only_prediction"] = float(1 - noise.mean() / x[m].var())
        e["implied_true_persistence"] = e["slope_no_mmr"] / e["noise_only_prediction"]
        # specialists vs mixed: share of the player's games on this hero
        share = nh / np.maximum(npl, 1)
        sp = {}
        for sl, (a0, a1) in {"under 15%": (0, 0.15), "15-50%": (0.15, 0.5), "over 50%": (0.5, 1.01)}.items():
            ms = m & (share >= a0) & (share < a1)
            if ms.sum() < 500:
                continue
            nz = 0.245 / nh[ms]
            rel = float(1 - nz.mean() / x[ms].var())
            b0 = float(ols(x[ms][:, None], r[ms])[0])
            b1 = float(ols(np.column_stack([x[ms], dteam[ms], own[ms]]), r[ms])[0])
            sp[sl] = {"slots": int(ms.sum()), "slope": b0, "slope_with_mmr": b1, "noise_only_prediction": rel,
                      "implied_true_persistence": b0 / rel if rel > 0 else None}
        e["by_hero_share"] = sp
        res[lb] = e
    out["by_bin"] = res
    for k, v in res.items():
        print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()})
    if a_.key:
        import p3_heroes  # noqa: F401
        names = [str(h) for h in np.load(os.path.join(HERE, "cache", "x_slots_ext.npz"))["hero_names"]]
        hi = names.index(a_.hero)
        ex = {}
        for arm in ("A2S", "full_ms_layer"):
            M = np.load(os.path.join(a_.arms, f"{arm}.npz"))
            mk = M["s_key"] == a_.key
            last = mk & (M["s_hero"] == hi)
            if not last.any():
                continue
            i = np.flatnonzero(last)[np.argmax(M["s_day"][last])]
            other = mk & (M["s_hero"] != hi)
            ex[arm] = {"day": str(np.datetime64(int(M["s_day"][i]), "D")), "games_on_hero": float(M["s_nh"][i]),
                       "games_total": float(M["s_np"][i]),
                       "record_pp": float(100 * M["s_rec"][i] / max(M["s_nh"][i], 1)) if "s_rec" in M.files else None,
                       "estimate_pp": float(100 * M["s_pred"][i]),
                       "mean_estimate_other_heroes_pp": float(100 * M["s_pred"][other].mean()) if other.any() else None,
                       "realized_on_hero_in_test_pp": float(100 * M["s_r"][last].mean()), "test_games_on_hero": int(last.sum()),
                       "realized_other_heroes_in_test_pp": float(100 * M["s_r"][other].mean()) if other.any() else None,
                       "test_games_other": int(other.sum())}
        out["example"] = ex
        print(json.dumps(ex, indent=1))
    json.dump(out, open(os.path.join(HERE, "results", "p2f", "carry_mmr.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
