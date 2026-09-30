"""
P3 extensions, side information for tasks 2, 6 and 7 (DB read-only, no
battletags): for every snapshot Storm League game since 2024-04-01
(replay_id <= 63653039)

  games:   replay_id, game_map, skill_tier, region            -> x_side_games.npz
  players: replay_id, blizz_id, a fixed set of scoreboard numbers, and the
           seven talent choices coded as integers               -> x_side_players.npz

Talent codes index into `talent_names` ("<tier>|<talent name>").

Usage (from training/, DATABASE_URL set):
  python3 personalization/p3_x_fetch_side.py
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
SNAP = 63653039
E_START = "2024-04-01"
SB = ["kills", "deaths", "assists", "takedowns", "hero_damage", "siege_damage",
      "healing", "self_healing", "damage_taken", "experience_contribution",
      "time_spent_dead", "time_cc_enemy_heroes", "merc_camp_captures",
      "protection_allies", "teamfight_hero_damage", "structure_damage",
      "stunning_enemies", "escapes", "outnumbered_deaths", "regen_globes"]
TIERS = ["1", "4", "7", "10", "13", "16", "20"]


def main():
    import psycopg2
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    conn.set_session(readonly=True)
    cur = conn.cursor(name="xs_g")
    cur.itersize = 200000
    cur.execute(f"""SELECT replay_id, game_map, skill_tier, region FROM replay_draft_data
                    WHERE game_date >= '{E_START}' AND replay_id <= {SNAP}""")
    rid, mp, tr, rg = [], [], [], []
    maps, tiers = {}, {}
    for r, m, t, g in cur:
        rid.append(r)
        mp.append(maps.setdefault(m, len(maps)))
        tr.append(tiers.setdefault(t, len(tiers)))
        rg.append(-1 if g is None else g)
    cur.close()
    np.savez(os.path.join(CACHE, "x_side_games.npz"), replay_ids=np.array(rid, np.int64),
             map=np.array(mp, np.int16), tier=np.array(tr, np.int8),
             region=np.array(rg, np.int16), map_names=np.array(sorted(maps, key=maps.get)),
             tier_names=np.array(sorted(tiers, key=tiers.get)))
    print(f"games: {len(rid):,}", flush=True)

    sel = ", ".join(f"(p.scoreboard->>'{k}')::real" for k in SB)
    tal = ", ".join(f"p.talents->>'{t}'" for t in TIERS)
    cur = conn.cursor(name="xs_p")
    cur.itersize = 200000
    cur.execute(f"""
        SELECT p.replay_id, p.blizz_id, {sel}, {tal}
        FROM replay_players p JOIN replay_draft_data d USING (replay_id)
        WHERE d.game_date >= '{E_START}' AND d.replay_id <= {SNAP}""")
    n = 0
    cap = 12_000_000
    R = np.empty(cap, np.int64)
    B = np.empty(cap, np.int64)
    S = np.full((cap, len(SB)), np.nan, np.float32)
    T = np.full((cap, len(TIERS)), -1, np.int32)
    tnames = {}
    for row in cur:
        R[n] = row[0]
        B[n] = row[1]
        sb = row[2:2 + len(SB)]
        S[n] = [np.nan if x is None else x for x in sb]
        for j, t in enumerate(row[2 + len(SB):]):
            if t is not None:
                T[n, j] = tnames.setdefault(f"{TIERS[j]}|{t}", len(tnames))
        n += 1
        if n % 2_000_000 == 0:
            print(f"  {n:,}", flush=True)
    np.savez(os.path.join(CACHE, "x_side_players.npz"), replay_ids=R[:n], blizz_ids=B[:n],
             scoreboard=S[:n], sb_names=np.array(SB), talents=T[:n],
             talent_names=np.array(sorted(tnames, key=tnames.get)))
    print(f"players: {n:,}, talent names {len(tnames):,}")


if __name__ == "__main__":
    main()
