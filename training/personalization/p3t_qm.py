"""
P3 tournament: Quick Match check of the judge's personal term (coordinator,
2026-10-21; addition 1 of the validation checklist).

Quick Match has no draft: each player queues with a hero he chose, so the
hero-to-player map is known and no assignment is needed. If the personal
term the judge fits on Storm League games (c2 = 4.53 per unit of team skill
sum) is real skill and not an artifact of SL drafting, the same Storm League
skill state should predict Quick Match outcomes with a similar coefficient.

Causality: the judge's skill state is as of 2026-09-12 (evidence from the
judge window, 07-01 to 09-11, SL games only). The QM games are 2026-09-12 or
later, so every state is strictly earlier than every game. The population
part is paper 1's QM2026 judge, whose training games end 2026-06-18.

Stages:
  players (GPU host, CPU only)  the judge state of every player of a
           tournament-window lobby, keyed by (region, blizz_id):
           $P3_CACHE/p3t/qm_state_players.npz (player keys stay in caches)
  export (main box, local research DB, read-only role)  QM games from
           2026-09-12 with all ten per-player rows: replay, player key,
           hero, team, win, map, tier -> a cache npz (never committed)
  fit    (GPU host)  logit P(team 0 wins) = a logit p_QM2026 + c2 dS + c3 dO
           on games with enough known players; compare c2 with the SL fit.
Output: $P3_RESULTS/p3t/qm_check.json
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("P1_TIERS", "site")
import numpy as np

QM_START = "2026-09-12"


def players_file():
    import p3t_data as TD
    return os.path.join(TD.OUT, "qm_state_players.npz")


def rows_file():
    import p3t_data as TD
    return os.path.join(TD.OUT, "qm_rows.npz")


def stage_players():
    import p3_x_common as X
    import p3t_data as TD
    L = TD.load_lobbies()
    z = np.load(os.path.join(TD.OUT, "state_judge.npz"))
    gsel = z["gsel"]
    rows = L["slot_rows"][gsel].ravel()
    ext = np.load(X.EXT)
    pid = ext["pid"][rows].astype(np.int64)
    keys = ext["player_keys"][pid].astype(np.int64)
    u, first = np.unique(keys, return_index=True)
    # the judge state is as of 2026-09-12 for every row (fixed query day), so one row per player suffices
    np.savez(players_file(), key=u, s=z["s"][first].astype(np.float32), off=z["off"][first],
             n=z["n"][first].astype(np.uint16))
    print(f"players: {len(u):,} distinct (from {len(rows):,} slots)", flush=True)


def stage_export(out):
    import psycopg2
    sys.path.insert(0, HERE)
    from p3_keys import player_keys
    repo = os.path.dirname(os.path.dirname(HERE))
    url = [l.split("=", 1)[1].strip().strip('"') for l in open(os.path.join(repo, ".env"))
           if l.startswith("DATABASE_URL_RESEARCH=")][0]
    assert "neon" not in url and ("127.0.0.1" in url or "localhost" in url), "local research DB only"
    c = psycopg2.connect(url)
    cur = c.cursor()
    cur.execute("""
        with g as (select q.replay_id from qm_games q join replay_players rp on rp.replay_id = q.replay_id
                   where q.game_date >= %s group by 1 having count(*) = 10)
        select rp.replay_id, coalesce(rp.region, 0), coalesce(q.region, 0), rp.blizz_id, rp.hero, rp.team,
               rp.winner, q.game_map, q.league_tier, q.avg_mmr, extract(epoch from q.game_date)
        from g join replay_players rp on rp.replay_id = g.replay_id join qm_games q on q.replay_id = g.replay_id
        order by rp.replay_id, rp.team""", (QM_START,))
    R = cur.fetchall()
    rid = np.array([r[0] for r in R], np.int64)
    key = player_keys(np.array([r[1] for r in R]), np.array([r[3] for r in R], np.int64),
                      game_region=np.array([r[2] for r in R]), label="qm")
    np.savez(out, replay_id=rid, key=key, hero=np.array([r[4] for r in R]), team=np.array([int(r[5]) for r in R]),
             win=np.array([bool(r[6]) for r in R]), game_map=np.array([r[7] for r in R]),
             league_tier=np.array([-1 if r[8] is None else int(r[8]) for r in R]),
             avg_mmr=np.array([np.nan if r[9] is None else float(r[9]) for r in R]),
             ts=np.array([float(r[10]) for r in R]))
    print(f"export: {len(R):,} rows, {len(np.unique(rid)):,} games -> {out}", flush=True)


def stage_fit(min_known):
    import p3t_data as TD
    import p3t_judge as JG
    from shared import HEROES, MAPS
    from overfit2026 import judges_v2 as J
    P = dict(np.load(players_file()))
    pos = {int(k): i for i, k in enumerate(P["key"])}
    Q = dict(np.load(rows_file(), allow_pickle=True))         # load once (lazy npz access re-reads)
    judge = json.load(open(os.path.join(JG.RES, "judge.json")))
    c2_sl, c3_sl = judge["c2_skill"], judge["c3_offrole"]
    norm = lambda s: "".join(ch for ch in str(s).lower() if ch.isalnum())
    hidx = {norm(h): i for i, h in enumerate(HEROES)}
    rid = Q["replay_id"]
    order = np.argsort(rid, kind="stable")
    games = []
    unknown_hero = set()
    starts = np.flatnonzero(np.r_[True, rid[order][1:] != rid[order][:-1]])
    for s0, s1 in zip(starts, np.r_[starts[1:], len(order)]):
        ix = order[s0:s1]
        teams = Q["team"][ix]
        if sorted(teams.tolist()) != [0] * 5 + [1] * 5:
            continue
        hs = [hidx.get(norm(h)) for h in Q["hero"][ix]]
        if any(h is None for h in hs):
            unknown_hero.update(str(Q["hero"][i]) for i, h in zip(ix, hs) if h is None)
            continue
        gm = str(Q["game_map"][ix[0]])
        tier = TD.site_tier(int(Q["league_tier"][ix[0]]) if Q["league_tier"][ix[0]] >= 0 else None,
                            None if np.isnan(Q["avg_mmr"][ix[0]]) else float(Q["avg_mmr"][ix[0]]))
        if gm not in MAPS or tier == "unknown":
            continue
        S, O, kn = [0.0, 0.0], [0.0, 0.0], [0, 0]
        for i, h in zip(ix, hs):
            t = int(Q["team"][i])
            p = pos.get(int(Q["key"][i]))
            if p is None:
                continue
            S[t] += float(P["s"][p, h])
            O[t] += float(P["off"][p, h])
            kn[t] += 1
        w0 = bool(Q["win"][ix[np.flatnonzero(teams == 0)[0]]])
        games.append((int(rid[ix[0]]), [HEROES[h] for h, t in zip(hs, teams) if t == 0],
                      [HEROES[h] for h, t in zip(hs, teams) if t == 1], gm, tier, S[0] - S[1], O[0] - O[1],
                      min(kn), sum(kn), w0))
    n_all = len(games)
    # min_known = 0: every game with at least one known player (unknown players count as the population, s = 0)
    keep = [g for g in games if g[7] >= min_known and g[8] >= 1]
    rows = [(g[1], g[2], g[3], g[4]) for g in keep]
    base = J.score_base(rows, judges=["QM2026"], nproc=4)
    pq = np.clip(base["QM2026"], 1e-6, 1 - 1e-6)
    lq = np.log(pq / (1 - pq))
    dS = np.array([g[5] for g in keep])
    dO = np.array([g[6] for g in keep])
    y = np.array([g[9] for g in keep], float)
    w, se = JG.fit_offset_logistic(np.zeros(len(y)), np.c_[lq, dS, dO], y)
    w1, se1 = JG.fit_offset_logistic(np.zeros(len(y)), np.c_[lq], y)

    def ll(p):
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    sig = lambda z: 1 / (1 + np.exp(-z))
    out = {"qm_games_with_10_rows_from_2026_09_12": int(n_all), "min_known_players_per_team": int(min_known),
           "games_used": int(len(keep)), "unknown_hero_names": sorted(unknown_hero)[:20],
           "known_players_per_game_mean": round(float(np.mean([g[8] for g in keep])), 2) if keep else None,
           "games_by_known_players": np.bincount([g[8] for g in keep], minlength=11).tolist(),
           "fit": {"a_QM2026": [round(float(w[0]), 3), round(float(se[0]), 3)],
                   "c2_skill": [round(float(w[1]), 3), round(float(se[1]), 3)],
                   "c3_offrole": [round(float(w[2]), 4), round(float(se[2]), 4)]},
           "sl_judge": {"c2_skill": round(c2_sl, 3), "c3_offrole": round(c3_sl, 4)},
           "log_loss": {"coin": round(float(np.log(2)), 5), "QM2026 alone (refit scale)": round(ll(sig(w1[0] * lq)), 5),
                        "QM2026 + personal (refit)": round(ll(sig(np.c_[lq, dS, dO] @ w)), 5),
                        "QM2026 (refit scale) + SL judge's personal coefficients":
                            round(ll(sig(w1[0] * lq + c2_sl * dS + c3_sl * dO)), 5)},
           "dS_sd": round(float(dS.std()), 4)}
    import p3_hs_core as C
    os.makedirs(os.path.join(C.RESULTS, "p3t"), exist_ok=True)
    json.dump(out, open(os.path.join(C.RESULTS, "p3t", f"qm_check_k{min_known}.json"), "w"), indent=1)
    print(json.dumps(out, indent=1), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["players", "export", "fit"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-known", type=int, default=0)
    a = ap.parse_args()
    if a.stage == "players":
        stage_players()
    elif a.stage == "export":
        stage_export(a.out)
    else:
        stage_fit(a.min_known)


if __name__ == "__main__":
    main()
