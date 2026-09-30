"""
P3 at-game MMR: target panel for the HP MMR-history pull
(sync/fetch-mmr-history.ts).

Panel = accounts (region, blizz_id) with >= MIN_GAMES Storm League games in
the P3 window (game_date 2024-04-01 .. 2026-05-22, all games now stored),
counted live from replay_players x replay_draft_data. The
API looks players up by battletag + region, so we attach each account's
most recent battletag from replay_players (tags can change; the latest is
the one HP knows now).

Order: a fixed hash of (region, blizz_id), so any partial pull is a random
subset of the panel, not the heaviest players first.

Usage (from training/, needs psycopg2 and DATABASE_URL):
  python3 personalization/p3_mmr_panel.py [--min-games 100]
Output: personalization/cache/mmr_history/panel.tsv
  columns: order, region, blizz_id, battletag, n_games
"""
import argparse
import hashlib
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
OUT_DIR = os.path.join(CACHE, "mmr_history")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-games", type=int, default=100)
    ap.add_argument("--start", default="2024-04-01")
    ap.add_argument("--end", default="2026-05-23")  # exclusive; snapshot ends 2026-05-22
    a = ap.parse_args()
    import psycopg2
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    cur = conn.cursor()
    cur.execute("SET statement_timeout='30min'")
    # Count from the live tables, not the P3 snapshot cache: games in the
    # window kept arriving after the snapshot (26.1K accounts there vs
    # 29.6K now, 2026-09-30).
    cur.execute(f"""
        SELECT rp.blizz_id, d.region, count(*) FROM replay_players rp
        JOIN replay_draft_data d USING (replay_id)
        WHERE d.game_date >= '{a.start}' AND d.game_date < '{a.end}'
        GROUP BY 1, 2 HAVING count(*) >= {int(a.min_games)}""")
    got = cur.fetchall()
    bid = np.array([g[0] for g in got], np.int64)
    reg = np.array([g[1] for g in got], np.int64)
    cnt = np.array([g[2] for g in got], np.int64)
    print(f"accounts with >= {a.min_games} SL games in [{a.start}, {a.end}): {len(bid):,} "
          f"(slots {cnt.sum():,})")
    cur.execute("CREATE TEMP TABLE panel_ids (blizz_id bigint, region int)")
    from psycopg2.extras import execute_values
    execute_values(cur, "INSERT INTO panel_ids VALUES %s",
                   list(zip(bid.tolist(), reg.tolist())), page_size=5000)
    cur.execute("ANALYZE panel_ids")
    # latest tag per account; replay_players.region is NULL on some early
    # rows, so take region from the draft row
    cur.execute("""
        SELECT DISTINCT ON (rp.blizz_id, d.region) rp.blizz_id, d.region, rp.battletag
        FROM panel_ids pi
        JOIN replay_players rp ON rp.blizz_id = pi.blizz_id
        JOIN replay_draft_data d ON d.replay_id = rp.replay_id AND d.region = pi.region
        ORDER BY rp.blizz_id, d.region, d.game_date DESC""")
    tags = {(int(b), int(r)): t for b, r, t in cur.fetchall()}
    rows = []
    for b, r, n in zip(bid.tolist(), reg.tolist(), cnt.tolist()):
        t = tags.get((b, r))
        if not t or "#" not in t:
            continue
        h = hashlib.sha1(f"{r}:{b}".encode()).hexdigest()[:12]
        rows.append((h, r, b, t, n))
    rows.sort()
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "panel.tsv")
    with open(out + ".tmp", "w") as f:
        f.write("order\tregion\tblizz_id\tbattletag\tn_games\n")
        for i, (_, r, b, t, n) in enumerate(rows):
            f.write(f"{i}\t{r}\t{b}\t{t}\t{n}\n")
    os.replace(out + ".tmp", out)
    print(f"wrote {out}: {len(rows):,} accounts ({len(bid) - len(rows)} without a usable battletag); "
          f"regions {dict(zip(*np.unique([x[1] for x in rows], return_counts=True)))}")


if __name__ == "__main__":
    main()
