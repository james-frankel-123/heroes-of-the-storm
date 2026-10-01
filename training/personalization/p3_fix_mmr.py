"""
Audit fix P3-07: the MMR lines of P3_HERO_STRENGTH, restated with causal
MMR (P3_MMR_AT_GAME: the rating of the latest stamp parsed before the game
started). The old lines used the naive lagged MMR, which leaks.

Game level, V2 full lobbies, combiner fit on V1 (same protocol as
P3_HERO_STRENGTH section 2): logit WP + team differences of
  hero / role / player causal MMR (each alone, and all three),
  skill (lag-1 offset with the table refit on lag-1 counts + GP mean of the
  static kernel, online lag 1: cache/fix_sd_pred_static_lag1.npz),
  skill + hero MMR, skill + all three.
Missing hero MMR falls back to role, then player, then the global mean (the
old fill rule); missing role to player. Values / 100.
CIs: paired replay bootstrap and a player-cluster bootstrap (one-way, the
player of the slot with the most V2 games in the game, P3-16 approximation:
games resampled through clusters of their first team-0 player).

Run (from training/): OMP_NUM_THREADS=4 nice -n 19 taskset -c 48-63 python3 personalization/p3_fix_mmr.py
Output: results/fix/p3_fix_mmr.json
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import p3_hs_core as C
import p3_fix_counts as F
from p3_mmr_at_game import match


def team_diff(values, g, team, n_games):
    sign = np.where(team == 0, 1.0, -1.0)
    return np.bincount(g, weights=sign * values, minlength=n_games)


def main():
    rng = np.random.RandomState(0)
    d = C.load_slots()
    _, n_p, n_ph, table, _ = F.prepare_l1(d)
    mu = table[C.exp_bins(n_p, n_ph)]
    m = np.load(os.path.join(C.CACHE, "fix_sd_pred_static_lag1.npz"))["m"]
    skill = mu + np.where(np.isnan(m), 0, m)
    z = np.load(os.path.join(C.CACHE, "mmr_at_game.npz"))
    pkey = z["region"].astype(np.int64) << 40 | z["blizz_ids"]
    idx = match(d["player_keys"][d["pid"]], d["replay_id"], pkey, z["replay_ids"])
    ok = idx >= 0
    gi = np.maximum(idx, 0)
    V = {k: np.where(ok, z[f"{k}_mmr_causal"][gi].astype(np.float64), np.nan) for k in ("player", "role", "hero")}
    gm = np.nanmean(V["player"])
    pl = np.where(np.isnan(V["player"]), gm, V["player"])
    ro = np.where(np.isnan(V["role"]), pl, V["role"])
    he = np.where(np.isnan(V["hero"]), ro, V["hero"])
    days = d["day"]
    post = ~d["in_sample"]
    _, fr = np.unique(d["g"][post], return_index=True)
    med = np.median(days[post][fr])
    n_games = int(d["g"].max()) + 1
    y = np.zeros(n_games)
    wg = np.full(n_games, 0.5)
    t0m = d["team"] == 0
    y[d["g"][t0m]] = d["y"][t0m]
    wg[d["g"][t0m]] = d["wp"][t0m]
    lo = np.log(wg / (1 - wg))
    full = np.bincount(d["g"][post], minlength=n_games) == 10
    gv1 = np.zeros(n_games, bool)
    gv1[d["g"][post & (days < med)]] = True
    gv2 = np.zeros(n_games, bool)
    gv2[d["g"][post & (days >= med)]] = True
    fit_m, test_m = gv1 & full, gv2 & full
    D = {"hero": team_diff(he / 100, d["g"], d["team"], n_games),
         "role": team_diff(ro / 100, d["g"], d["team"], n_games),
         "player": team_diff(pl / 100, d["g"], d["team"], n_games),
         "skill": team_diff(skill, d["g"], d["team"], n_games)}
    specs = {"M0": [], "hero MMR (causal) alone": ["hero"], "role MMR alone": ["role"],
             "player MMR alone": ["player"], "all three causal MMR": ["player", "role", "hero"],
             "skill": ["skill"], "skill + hero MMR": ["skill", "hero"],
             "skill + all three causal MMR": ["skill", "player", "role", "hero"]}
    eps = 1e-7
    nb = len(C.NPH_EDGES)
    tab = {f"n_p>={int(C.NP_EDGES[i])}": [round(float(100 * table[i * nb + j]), 2) for j in range(nb)]
           for i in range(len(C.NP_EDGES))}
    lls, out = {}, {"experience_table_lag1_pp (columns n_ph >= 0,1,2,3,5,10,20,50)": tab,
                    "V2_games": int(test_m.sum()), "V1_games": int(fit_m.sum()),
                    "coverage_V2_slots": {k: float(np.isfinite(V[k][post & (days >= med)]).mean()) for k in V}}
    for nm, cols in specs.items():
        X = np.column_stack([lo] + [D[c] for c in cols])
        w = C.fit_logistic(X[fit_m], y[fit_m])
        p = C.predict(w, X)
        lls[nm] = -(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        out[nm] = {**C.game_metrics(p[test_m], y[test_m]), "coef": w.tolist()}
    tidx = np.flatnonzero(test_m)
    # player clusters: the first team-0 slot's player of each test game
    first_pid = np.full(n_games, -1)
    o = np.flatnonzero(t0m)[::-1]
    first_pid[d["g"][o]] = d["pid"][o]
    cl = first_pid[tidx]
    ug, inv = np.unique(cl, return_inverse=True)
    Wr = [np.bincount(rng.randint(0, len(tidx), len(tidx)), minlength=len(tidx)).astype(float) for _ in range(300)]
    Wc = [np.bincount(rng.randint(0, len(ug), len(ug)), minlength=len(ug))[inv].astype(float) for _ in range(300)]

    def ci(dd, Ws):
        b = [np.average(dd, weights=w) for w in Ws]
        return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
    for nm in specs:
        for ref in ("M0", "skill"):
            dd = (lls[ref] - lls[nm])[tidx]
            out[nm][f"gain_vs_{ref}"] = [float(dd.mean())] + ci(dd, Wr)
            out[nm][f"gain_vs_{ref}_player_cluster_ci"] = ci(dd, Wc)
        print(f"{nm:32s} acc {out[nm]['acc']:.4f} ll {out[nm]['logloss']:.5f} "
              f"vs M0 {out[nm]['gain_vs_M0'][0]:+.5f} [{out[nm]['gain_vs_M0'][1]:+.5f},{out[nm]['gain_vs_M0'][2]:+.5f}] "
              f"vs skill {out[nm]['gain_vs_skill'][0]:+.5f} [{out[nm]['gain_vs_skill'][1]:+.5f},"
              f"{out[nm]['gain_vs_skill'][2]:+.5f}]", flush=True)
    os.makedirs(F.FIX_RESULTS, exist_ok=True)
    with open(os.path.join(F.FIX_RESULTS, "p3_fix_mmr.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
