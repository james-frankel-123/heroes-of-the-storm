"""
P3 hero strength, stage 0: pull per-player hero_mmr, role_mmr and hero_level
for the same rows as players_2024q2.npz (Storm League, game_date >=
2024-04-01, replay_id <= snapshot bound).

These three columns are recorded when Heroes Profile parses the replay (after
the game), like player_mmr. Downstream code only ever uses LAGGED values
(latest value from a game at least G days earlier).

Usage (needs psycopg2 and DATABASE_URL):
  python3 personalization/p3_fetch_hero_mmr.py
Output: personalization/cache/hero_mmr_2024q2.npz
  replay_ids, blizz_ids, region, hero_mmr, role_mmr, hero_level
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cache", "hero_mmr_2024q2.npz")
E_START = "2024-04-01"
SNAPSHOT_BOUND = 63653039


def main():
    import psycopg2
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    cur = conn.cursor(name="p3hm")
    cur.itersize = 200000
    cur.execute(f"""
        SELECT p.replay_id, p.blizz_id, COALESCE(p.region, d.region), p.hero_mmr, p.role_mmr, p.hero_level
        FROM replay_players p JOIN replay_draft_data d USING (replay_id)
        WHERE d.game_date >= '{E_START}' AND d.replay_id <= {SNAPSHOT_BOUND}""")
    rid, bid, reg, hm, rm, hl = [], [], [], [], [], []
    nan = float("nan")
    for r, b, rg, h, ro, lv in cur:
        rid.append(r)
        bid.append(b)
        reg.append(-1 if rg is None else rg)
        hm.append(nan if h is None else h)
        rm.append(nan if ro is None else ro)
        hl.append(-1 if lv is None else lv)
        if len(rid) % 2000000 == 0:
            print(f"  {len(rid):,} rows", flush=True)
    np.savez(OUT, replay_ids=np.array(rid, np.int64), blizz_ids=np.array(bid, np.int64),
             region=np.array(reg, np.int16),
             hero_mmr=np.array(hm, np.float32), role_mmr=np.array(rm, np.float32),
             hero_level=np.array(hl, np.int16))
    print(f"wrote {OUT} ({len(rid):,} rows)")


if __name__ == "__main__":
    main()
