"""
P3 extensions, task 1 data: every Storm League game of build 2.55.16.97039
and 2.55.17.* from the DB (read-only), with its ten player rows. These are
the games after the pinned 2026-05-22 snapshot (plus 97039 games already in
the snapshot, which are used to check that the WP rebuild reproduces the
cached scores). Games dated 2026-09-28 or later are dropped (no build after
2026-09-27 is touched; 98025 shipped 2026-09-12).

No battletags are read.

Usage (from training/, DATABASE_URL set):
  python3 personalization/p3_x_fetch_post.py
Outputs (personalization/cache/):
  x_post_games.json.gz   list of dicts with the fields extract_features needs
                         plus game_version, ts, game_length, draft_order
  x_post_players.npz     replay_ids, blizz_ids, region, hero (name index into
                         hero_names of players_2024q2.npz, -1 if unknown), team,
                         party, player_mmr, hero_mmr, role_mmr, hero_level
"""
import os
import gzip
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
OUT_G = os.path.join(CACHE, "x_post_games.json.gz")
OUT_P = os.path.join(CACHE, "x_post_players.npz")
WHERE = ("(d.game_version = '2.55.16.97039' OR d.game_version LIKE '2.55.17.%') "
         "AND d.game_date < '2026-09-28'")


def main():
    import psycopg2
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    conn.set_session(readonly=True)
    cur = conn.cursor(name="x_g")
    cur.itersize = 50000
    cur.execute(f"""
        SELECT d.replay_id, d.game_version, EXTRACT(EPOCH FROM d.game_date)::bigint,
               d.game_length, d.game_map, d.skill_tier, d.team0_heroes, d.team1_heroes,
               d.team0_bans, d.team1_bans, d.winner, d.avg_mmr, d.league_tier,
               d.draft_order, d.region
        FROM replay_draft_data d WHERE {WHERE}""")
    games = []
    for (rid, ver, ts, gl, gm, tier, t0, t1, b0, b1, w, am, lt, do, reg) in cur:
        lst = lambda x: json.loads(x) if isinstance(x, str) else x
        games.append({"replay_id": rid, "game_version": ver, "ts": ts,
                      "game_length": gl, "game_map": gm, "skill_tier": tier,
                      "team0_heroes": lst(t0), "team1_heroes": lst(t1),
                      "team0_bans": lst(b0), "team1_bans": lst(b1), "winner": w,
                      "avg_mmr": am, "league_tier": lt, "region": reg,
                      "draft_order": [[e.get("hero"), int(e.get("type")),
                                       int(e.get("pick_number", 0)),
                                       int(e.get("player_slot", -1))]
                                      for e in (lst(do) or [])]})
    cur.close()
    with gzip.open(OUT_G, "wt") as f:
        json.dump(games, f)
    print(f"wrote {OUT_G}: {len(games):,} games", flush=True)

    names = [str(x) for x in np.load(os.path.join(CACHE, "players_2024q2.npz"))["hero_names"]]
    hidx = {n: i for i, n in enumerate(names)}
    cur = conn.cursor(name="x_p")
    cur.itersize = 200000
    cur.execute(f"""
        SELECT p.replay_id, p.blizz_id, p.region, p.hero, p.team, p.party,
               p.player_mmr, p.hero_mmr, p.role_mmr, p.hero_level
        FROM replay_players p JOIN replay_draft_data d USING (replay_id)
        WHERE {WHERE}""")
    cols = {k: [] for k in ("replay_ids", "blizz_ids", "region", "hero", "team", "party",
                            "player_mmr", "hero_mmr", "role_mmr", "hero_level")}
    nan = float("nan")
    unknown = set()
    for r, b, rg, h, t, pa, pm, hm, rm, hl in cur:
        cols["replay_ids"].append(r)
        cols["blizz_ids"].append(b)
        cols["region"].append(-1 if rg is None else rg)
        if h not in hidx:
            unknown.add(h)
        cols["hero"].append(hidx.get(h, -1))
        cols["team"].append(t)
        cols["party"].append(pa or 0)
        cols["player_mmr"].append(nan if pm is None else pm)
        cols["hero_mmr"].append(nan if hm is None else hm)
        cols["role_mmr"].append(nan if rm is None else rm)
        cols["hero_level"].append(-1 if hl is None else hl)
    dt = {"replay_ids": np.int64, "blizz_ids": np.int64, "region": np.int16,
          "hero": np.int16, "team": np.int8, "party": np.int64,
          "player_mmr": np.float32, "hero_mmr": np.float32, "role_mmr": np.float32,
          "hero_level": np.int16}
    np.savez(OUT_P, hero_names=np.array(names),
             **{k: np.array(v, dt[k]) for k, v in cols.items()})
    print(f"wrote {OUT_P}: {len(cols['replay_ids']):,} rows; unknown heroes {unknown}")


if __name__ == "__main__":
    main()
