"""
P3: pull game_date (timestamp) and game_length for every snapshot game since
2024-04-01, used to order a player's same-day games by play time instead of
replay_id (upload order).

Usage (needs psycopg2 and DATABASE_URL):
  python3 personalization/p3_fetch_gametime.py
Output: personalization/cache/gametime_2024q2.npz  replay_ids, ts (epoch
seconds, UTC as stored), game_length (seconds, -1 if missing)
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cache", "gametime_2024q2.npz")


def main():
    import psycopg2
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    cur = conn.cursor(name="p3gt")
    cur.itersize = 200000
    cur.execute("""SELECT replay_id, EXTRACT(EPOCH FROM game_date)::bigint, game_length
                   FROM replay_draft_data
                   WHERE game_date >= '2024-04-01' AND replay_id <= 63653039""")
    rid, ts, gl = [], [], []
    for r, t, g in cur:
        rid.append(r)
        ts.append(t)
        gl.append(-1 if g is None else g)
    np.savez(OUT, replay_ids=np.array(rid, np.int64), ts=np.array(ts, np.int64),
             game_length=np.array(gl, np.int64))
    print(f"wrote {OUT} ({len(rid):,} games)")


if __name__ == "__main__":
    main()
