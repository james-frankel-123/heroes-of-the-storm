"""
Pick order (rank 0..9 among a game's picks) for the post-snapshot games on the
whitelisted builds (overfit2026.data.FUTURE_BUILDS plus the backfill builds:
replay_id > snapshot bound, game_version <= 2.55.17.98025 by whitelist). Read-only.
Snapshot-window pick order already exists in personalization/cache/pickorder_2024q2.npz.

Usage (from training/): python3 overfit2026/comp_fetch_pickorder.py
Output: overfit2026/cache/comp_pickorder_post.npz (replay_ids, hero, pick_rank)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np

from overfit2026 import data

OUT = os.path.join(HERE, "cache", "comp_pickorder_post.npz")


def main():
    import psycopg2
    for line in open(os.path.join(os.path.dirname(HERE), "..", ".env")):
        if line.startswith("DATABASE_URL=") and "DATABASE_URL" not in os.environ:
            os.environ["DATABASE_URL"] = line.split("=", 1)[1].strip().strip('"')
    keep = set()
    for v in data.load_future().values():
        keep.update(g[0] for g in v)
    for v in data.load_backfill().values():
        keep.update(g[0] for g in v)
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    conn.set_session(readonly=True)
    cur = conn.cursor(name="comp_po")
    cur.itersize = 50000
    cur.execute("SELECT replay_id, draft_order FROM replay_draft_data WHERE replay_id > %s",
                (data.SNAPSHOT_BOUND,))
    rid, hero, rank = [], [], []
    for r, do in cur:
        if r not in keep or not do:
            continue
        picks = sorted((e for e in do if str(e.get("type")) == "1"),
                       key=lambda e: int(e.get("pick_number", 0)))
        for k, e in enumerate(picks):
            rid.append(r)
            hero.append(e.get("hero") or "")
            rank.append(k)
    np.savez(OUT, replay_ids=np.array(rid, np.int64), hero=np.array(hero),
             pick_rank=np.array(rank, np.int8))
    print(f"wrote {OUT} ({len(rid):,} picks, {len(set(rid)):,} games of {len(keep):,})")


if __name__ == "__main__":
    main()
