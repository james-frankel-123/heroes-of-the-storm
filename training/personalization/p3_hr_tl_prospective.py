"""
P3 hero recs and replays, part B2.4: prospective incremental value of replay
timeline features over the scoreboard (stages `targets` and `prospective` of
p3_hr_tl_analyze.py).

Targets: every 2.57 Storm League game in the local research DB with ten
replay_players rows, minus the sealed holdout ids (never read beyond the id
list). Outcome: the team's win. History for a target game: the timeline games
of its players on EARLIER UTC days only (strictly before the target's day), so
no target ever sees its own row or a same-day game.

Nested logistic models on team differences, fit and scored by 10-fold
cross-validation over games (folds by game id hash), L2 penalty fixed:
  (a) base: team difference of the hero's 2.57 win rate (as of earlier days,
      shrunk) and, when cache/hr/state_0928.npz exists, of the pre-2.57 main
      model's estimate for the player on that hero (offset + posterior mean);
  (b) + the scoreboard (15 model metrics) as-of player means from the history
      timeline games;
  (c) + timeline features from the same history games;
  (c') + the 9 extra score-screen stats (CC, stun, root, silence, teamfight).
Player means are shrunk toward 0 with k = (1 - r) / r pseudo-games, r the
feature's player-level per-game reliability (results/hr/tl_reliability.json).
No game's own MMR stamp is used anywhere.
"""
import json
import os
import sys
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p3_hr_tl_analyze as A  # noqa: E402

SEALED = os.path.join(HERE, "..", "replays", "cohorts", "sealed_ids.txt")
DAY = 86400.0
# HP scoreboard keys for the model's 15 metrics and the extra score-screen stats
DBMAP = {"hero_damage": "hero_damage", "siege_damage": "siege_damage", "structure_damage": "structure_damage",
         "healing": "healing", "self_healing": "self_healing", "protection_allies": "protection_allies",
         "damage_taken": "damage_taken", "deaths": "deaths", "kills": "kills", "assists": "assists",
         "experience_contribution": "experience_contribution", "merc_camp_captures": "merc_camp_captures",
         "outnumbered_deaths": "outnumbered_deaths", "escapes": "escapes", "regen_globes": "regen_globes",
         "cc_time": "time_cc_enemy_heroes", "stun_time": "stunning_enemies", "root_time": "rooting_enemies",
         "silence_time": "silencing_enemies", "tf_hero_damage": "teamfight_hero_damage",
         "tf_damage_taken": "teamfight_damage_taken", "tf_healing": "teamfight_healing",
         "minion_damage": "minion_damage", "creep_damage": "creep_damage"}


def targets():
    import psycopg2
    env = {}
    for line in open(os.path.join(HERE, "..", "..", ".env")):
        if "=" in line and not line.startswith("#"):
            k, v = line.strip().split("=", 1)
            env[k] = v.strip().strip('"').strip("'")
    url = env["DATABASE_URL_RESEARCH"]
    assert "neon" not in url.lower()
    sealed = set(int(x) for x in open(SEALED) if x.strip())
    c = psycopg2.connect(url)
    c.set_session(readonly=True)
    cur = c.cursor()
    cur.execute("""select r.replay_id, extract(epoch from r.game_date), r.game_map, p.region, p.blizz_id, p.hero,
                          p.team, p.winner, r.game_length, p.scoreboard
                   from replay_draft_data r join replay_players p using (replay_id)
                   where r.game_version like '2.57%%'""")
    rows = cur.fetchall()
    rid = np.array([r[0] for r in rows], np.int64)
    keep = ~np.isin(rid, np.fromiter(sealed, np.int64))
    cnt = np.bincount(np.unique(rid, return_inverse=True)[1])
    full = cnt[np.unique(rid, return_inverse=True)[1]] == 10
    keep &= full
    sel = np.flatnonzero(keep)
    out = {"replay_id": rid[sel], "unix": np.array([float(rows[i][1]) for i in sel]),
           "map": np.array([rows[i][2] or "" for i in sel]), "region": np.array([rows[i][3] or -1 for i in sel], np.int32),
           "toon_id": np.array([rows[i][4] for i in sel], np.int64), "hero": np.array([rows[i][5] or "" for i in sel]),
           "team": np.array([rows[i][6] for i in sel], np.int32), "win": np.array([bool(rows[i][7]) for i in sel], np.int32),
           "minutes": np.array([max((rows[i][8] or 0) / 60.0, 1.0) for i in sel]),
           "n_sealed_excluded_games": np.int64(len(set(rid[np.isin(rid, np.fromiter(sealed, np.int64))])))}
    for f, dbk in DBMAP.items():
        out[f] = np.array([np.nan if (rows[i][9] or {}).get(dbk) is None else float(rows[i][9][dbk]) for i in sel])
    np.savez_compressed(A.TARGETS, **out)
    print(f"targets: {len(sel):,} slots, {len(np.unique(out['replay_id'])):,} games; "
          f"sealed games excluded {int(out['n_sealed_excluded_games'])}", flush=True)


def asof_player_means(hk, hday, Zh, qk, qday, k):
    """Shrunk mean of Zh rows of player hk on days < qday (per query)."""
    o = np.lexsort((hday, hk))
    hk, hday, Zh = hk[o], hday[o], Zh[o]
    F = Zh.shape[1]
    ok = np.isfinite(Zh)
    cs = np.vstack([np.zeros((1, F)), np.cumsum(np.where(ok, Zh, 0.0), 0)])
    cn = np.vstack([np.zeros((1, F)), np.cumsum(ok, 0)])
    comp = hk.astype(np.float64) * 1e5 + hday
    start = np.searchsorted(hk, qk, "left")
    end = np.searchsorted(comp, qk.astype(np.float64) * 1e5 + qday, "left")
    end = np.maximum(end, start)
    S = cs[end] - cs[start]
    N = cn[end] - cn[start]
    return S / (N + k[None, :]), (end - start)


def fit_l2(X, y, lam, off=None, iters=50):
    n, p = X.shape
    Xb = np.column_stack([np.ones(n), X])
    w = np.zeros(p + 1)
    off = np.zeros(n) if off is None else off
    R = lam * np.eye(p + 1)
    R[0, 0] = 0
    for _ in range(iters):
        eta = np.clip(off + Xb @ w, -30, 30)
        pr = 1 / (1 + np.exp(-eta))
        g = Xb.T @ (y - pr) - R @ w
        Hm = (Xb * (pr * (1 - pr))[:, None]).T @ Xb + R
        st = np.linalg.solve(Hm, g)
        w += st
        if np.abs(st).max() < 1e-9:
            break
    return w


def predict(w, X, off=None):
    eta = (0 if off is None else off) + np.column_stack([np.ones(len(X)), X]) @ w
    return 1 / (1 + np.exp(-np.clip(eta, -30, 30)))


def logloss(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def prospective(lam=1.0, nfold=10, hist_cohorts=(1, 2, 3), tag=""):
    rel = json.load(open(os.path.join(A.RES, "tl_reliability.json")))["features"]
    z = dict(np.load(A.ROWS))
    Z = A.normalize(z)
    hsel = np.isin(z["cohort"], hist_cohorts)
    t = dict(np.load(A.TARGETS))
    hk = A.pkey(z["region"], z["toon_id"])[hsel]
    hday = np.floor(z["unix"] / DAY)[hsel]
    Zh = Z[hsel]
    qk = A.pkey(t["region"], t["toon_id"])
    qday = np.floor(t["unix"] / DAY)
    kf = np.array([np.clip((1 - rel[f]["player"]["r"]) / max(rel[f]["player"]["r"], 1e-3), 0.2, 200)
                   for f in A.FEATS])
    out = {"lam": lam, "nfold": nfold, "history_cohorts": list(hist_cohorts)}
    ug, gi = np.unique(t["replay_id"], return_inverse=True)
    G = len(ug)
    gday = np.zeros(G)
    gday[gi] = qday
    y0 = np.zeros(G)
    m0 = t["team"] == 0
    y0[gi[m0]] = t["win"][m0]
    sgn = np.where(t["team"] == 0, 1.0, -1.0)

    def tdiff(v):
        return np.bincount(gi, weights=sgn * np.nan_to_num(v), minlength=G)

    # hero 2.57 win rate as of earlier days (shrunk with 200 games toward 0.5), logit
    uh, hi = np.unique(t["hero"], return_inverse=True)
    hw = np.zeros(len(t["win"]))
    for dday in np.unique(qday):
        prev = qday < dday
        wsum = np.bincount(hi[prev], weights=t["win"][prev], minlength=len(uh))
        nsum = np.bincount(hi[prev], minlength=len(uh))
        cur = qday == dday
        p = (wsum + 100) / (nsum + 200)
        hw[cur] = np.log(p / (1 - p))[hi[cur]]
    cols = {"hero_wr": tdiff(hw)}
    # the player's 2.57 win record on earlier days (residual vs the as-of hero rate), k = 74 games
    res_db = t["win"] - 1 / (1 + np.exp(-hw))
    wr, _ = asof_player_means(qk, qday, res_db[:, None], qk, qday, np.array([74.0]))
    cols["win_rec_257"] = tdiff(wr[:, 0])
    have_state = os.path.exists(A.STATE)
    out["state_0928_used"] = have_state
    if have_state:
        s = np.load(A.STATE, allow_pickle=True)
        sk = A.pkey(s["region"], s["blizz_id"])
        hn = [str(h) for h in s["hero_names"]]
        name_ix = {h: i for i, h in enumerate(hn)}
        hix = np.array([name_ix.get(h, -1) for h in t["hero"]])
        os_ = np.argsort(sk)
        j = np.minimum(np.searchsorted(sk[os_], qk), len(sk) - 1)
        hit = (sk[os_][j] == qk) & (hix >= 0)
        est = np.zeros(len(qk))
        rr = os_[j[hit]]
        est[hit] = s["mean_all"][rr, hix[hit]] + s["offset_all"][rr, hix[hit]]
        cols["model"] = tdiff(est)
        out["slots_with_state"] = float(hit.mean())
        out["unmatched_heroes"] = sorted(set(t["hero"][hix < 0].tolist()))
    # scoreboard (and extra score-screen stats) from ALL earlier DB 2.57 games, same normalization
    tdb = {"hero": t["hero"], "map": t["map"], "win": t["win"], "minutes": t["minutes"]}
    for f in A.FEATS:
        tdb[f] = t[f] if f in t else np.full(len(qk), np.nan)
    Zdb = A.normalize(tdb)
    sb_ix = [A.FEATS.index(f) for f in A.SB + A.EXTRA]
    Mdb, ndb = asof_player_means(qk, qday, Zdb[:, sb_ix], qk, qday, kf[sb_ix])
    # timeline-game features
    M, nh = asof_player_means(hk, hday, Zh, qk, qday, kf)
    has = nh > 0
    out.update(target_slots=int(len(qk)), target_games=int(G), slots_with_tl_history=int(has.sum()),
               players_with_tl_history=int(len(np.unique(qk[has]))), slots_with_db_history=int((ndb > 0).sum()),
               median_tl_history_games=float(np.median(nh[has])), median_db_history_games=float(np.median(ndb[ndb > 0])))
    ngh = np.bincount(gi, weights=has, minlength=G)
    out["games_by_players_with_tl_history"] = {str(k): int((ngh == k).sum()) for k in range(11)}
    TLp = [f for f in A.TLF if f not in A.F.TEAM_FEATS]
    D = {f: tdiff(M[:, k]) for k, f in enumerate(A.FEATS)}
    Ddb = {"db_" + f: tdiff(Mdb[:, k]) for k, f in enumerate(A.SB + A.EXTRA)}
    base = ["hero_wr", "win_rec_257"] + (["model"] if have_state else [])
    sbt = list(A.SB)
    sba = ["db_" + f for f in A.SB]
    exa = ["db_" + f for f in A.EXTRA if f != "cc_time"]
    arms = {"a_base": base,
            "b_sb_tlgames": base + sbt, "c_sb_tl_tlgames": base + sbt + TLp,
            "b_sb_all": base + sba, "b2_sb_extra_all": base + sba + exa,
            "d_sb_all_tl": base + sba + TLp, "d2_sb_extra_all_tl": base + sba + exa + TLp,
            "d3_all_with_team": base + sba + exa + TLp + [f for f in A.TLF if f in A.F.TEAM_FEATS]}
    for f in TLp:
        arms[f"d_plus_{f}"] = base + sba + [f]
    fold = np.array([zlib.crc32(str(int(r)).encode()) % nfold for r in ug])
    allcols = dict(cols, **D, **Ddb)
    ll = {}
    llf = {}
    split_day = float(np.median(gday))
    out["forward_split_day"] = split_day
    out["coef_full_fit"] = {}
    for nm, cl in arms.items():
        X = np.column_stack([allcols[c] for c in cl])
        mu, sd = X.mean(0), X.std(0) + 1e-12
        X = (X - mu) / sd
        p = np.zeros(G)
        for k in range(nfold):
            tr = fold != k
            p[~tr] = predict(fit_l2(X[tr], y0[tr], lam), X[~tr])
        ll[nm] = logloss(p, y0)
        tr = gday < split_day
        pf = predict(fit_l2(X[tr], y0[tr], lam), X[~tr])
        llf[nm] = logloss(pf, y0[~tr])
        out["coef_full_fit"][nm] = dict(zip(cl, (fit_l2(X, y0, lam)[1:]).round(5).tolist()))
    ud, di = np.unique(gday, return_inverse=True)
    rng = np.random.RandomState(0)
    B = [rng.multinomial(len(ud), np.ones(len(ud)) / len(ud))[di] for _ in range(200)]
    pairs = [("b_sb_tlgames", "a_base"), ("c_sb_tl_tlgames", "b_sb_tlgames"), ("b_sb_all", "a_base"),
             ("b2_sb_extra_all", "b_sb_all"), ("d_sb_all_tl", "b_sb_all"), ("d2_sb_extra_all_tl", "b2_sb_extra_all"),
             ("d3_all_with_team", "d2_sb_extra_all_tl")] + [(f"d_plus_{f}", "b_sb_all") for f in TLp]
    subsets = {"all games": np.ones(G, bool), "3+ players with timeline history": ngh >= 3,
               "6+ players with timeline history": ngh >= 6}
    res = {}
    for sn, sm in subsets.items():
        r = {"games": int(sm.sum()), "base_ll": float(ll["a_base"][sm].mean())}
        for a_, b_ in pairs:
            dlt = (ll[b_] - ll[a_])[sm]
            bs = [np.average(dlt, weights=w_[sm]) for w_ in B if w_[sm].sum() > 0]
            r[f"{a_} vs {b_}"] = {"gain_x1000": 1000 * float(dlt.mean()),
                                  "se_game_x1000": 1000 * float(dlt.std(ddof=1) / np.sqrt(len(dlt))),
                                  "se_dayblock_x1000": 1000 * float(np.std(bs))}
        res[sn] = r
    out["results"] = res
    fw = {"test_games": int((gday >= split_day).sum())}
    gd_te = gday[gday >= split_day]
    ud2, di2 = np.unique(gd_te, return_inverse=True)
    B2 = [rng.multinomial(len(ud2), np.ones(len(ud2)) / len(ud2))[di2] for _ in range(200)]
    for a_, b_ in pairs[:7]:
        dlt = llf[b_] - llf[a_]
        fw[f"{a_} vs {b_}"] = {"gain_x1000": 1000 * float(dlt.mean()),
                               "se_dayblock_x1000": 1000 * float(np.std([np.average(dlt, weights=w_) for w_ in B2]))}
    out["forward"] = fw
    print("forward", {k: (round(v["gain_x1000"], 2), round(v["se_dayblock_x1000"], 2)) for k, v in fw.items()
                      if isinstance(v, dict)}, flush=True)
    out["history_days"] = [float(hday.min()), float(hday.max())]
    out["target_days"] = [float(qday.min()), float(qday.max())]
    fn = f"tl_prospective{tag}{'' if have_state else '_nostate'}.json"
    with open(os.path.join(A.RES, fn), "w") as f:
        json.dump(out, f, indent=1)
    for sn, r in res.items():
        print(sn, r["games"], {k: (round(v["gain_x1000"], 2), round(v["se_dayblock_x1000"], 2))
                               for k, v in r.items() if isinstance(v, dict) and not k.startswith("d_plus")}, flush=True)
    print({k: v for k, v in res["all games"].items() if k.startswith("d_plus")})
    print(json.dumps({k: out[k] for k in out if k not in ("results", "coef_full_fit")}), flush=True)
