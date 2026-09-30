"""
P3: pick order of every hero pick (draft_order entries with type '1') for
snapshot games since 2024-04-01. Heroes are unique within a game, so
(replay_id, hero) identifies the slot.

Usage (needs psycopg2 and DATABASE_URL):
  python3 personalization/p3_fetch_pickorder.py
Output: personalization/cache/pickorder_2024q2.npz
  replay_ids, hero (name), pick_rank (0..9 among the game's picks)
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cache", "pickorder_2024q2.npz")


def main():
    import psycopg2
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    cur = conn.cursor(name="p3po")
    cur.itersize = 50000
    cur.execute("""SELECT replay_id, draft_order FROM replay_draft_data
                   WHERE game_date >= '2024-04-01' AND replay_id <= 63653039""")
    rid, hero, rank = [], [], []
    for r, do in cur:
        if not do:
            continue
        picks = sorted((e for e in do if str(e.get("type")) == "1"),
                       key=lambda e: int(e.get("pick_number", 0)))
        for k, e in enumerate(picks):
            rid.append(r)
            hero.append(e.get("hero") or "")
            rank.append(k)
    np.savez(OUT, replay_ids=np.array(rid, np.int64), hero=np.array(hero),
             pick_rank=np.array(rank, np.int8))
    print(f"wrote {OUT} ({len(rid):,} picks)")


if __name__ == "__main__":
    main()
