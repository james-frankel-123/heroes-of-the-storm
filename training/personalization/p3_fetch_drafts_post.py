"""
P3 drafter: full draft_order (bans and picks), map and tier for the post-
cutoff snapshot games (game_date >= 2026-02-10, replay_id <= 63653039).

Usage (needs psycopg2 and DATABASE_URL):
  python3 personalization/p3_fetch_drafts_post.py
Output: personalization/cache/drafts_post.json.gz  {replay_id: {map, tier,
  order: [[hero, type, pick_number, player_slot], ...]}}
"""
import os
import gzip
import json

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cache", "drafts_post.json.gz")


def main():
    import psycopg2
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    cur = conn.cursor(name="p3dp")
    cur.itersize = 20000
    cur.execute("""SELECT replay_id, game_map, skill_tier, draft_order FROM replay_draft_data
                   WHERE game_date >= '2026-02-10' AND replay_id <= 63653039""")
    out = {}
    for r, m, t, do in cur:
        if not do:
            continue
        out[str(r)] = {"map": m, "tier": t,
                       "order": [[e.get("hero"), int(e.get("type")), int(e.get("pick_number", 0)),
                                  int(e.get("player_slot", -1))] for e in do]}
    with gzip.open(OUT, "wt") as f:
        json.dump(out, f)
    print(f"wrote {OUT} ({len(out):,} games)")


if __name__ == "__main__":
    main()
