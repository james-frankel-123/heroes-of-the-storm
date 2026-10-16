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

Tranche 2 (--tranche2, 2026-10-16): accounts NOT yet pulled (not in
panel.tsv nor done.jsonl) with >= 20 SL games since 2025-01-01, ordered by
last game, most recent first, so a partial pull covers the players most
likely to appear in eval lobbies and future tranches. Reads the local store
through DATABASE_URL_RESEARCH (read-only; Neon refused).
  nice -n 19 python3 personalization/p3_mmr_panel.py --tranche2
Output: personalization/cache/mmr_history/panel2.tsv
  columns: order, region, blizz_id, battletag, n_games, last_game
The panels hold battletags: cache/ is gitignored, never commit them.
"""
import argparse
import hashlib
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
OUT_DIR = os.path.join(CACHE, "mmr_history")


def tranche2(min_games, since, out_name):
    import json
    url = os.environ.get("DATABASE_URL_RESEARCH")
    if not url:
        raise SystemExit("DATABASE_URL_RESEARCH is not set (local read-only role)")
    if "neon" in url.lower():
        raise SystemExit("refusing a Neon URL: P3 reads the local store only")
    pulled = set()
    with open(os.path.join(OUT_DIR, "panel.tsv")) as f:
        next(f)
        for line in f:
            c = line.split("\t")
            pulled.add((int(c[1]), int(c[2])))
    done_file = os.path.join(OUT_DIR, "done.jsonl")
    if os.path.exists(done_file):
        for line in open(done_file):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            pulled.add((int(d["region"]), int(d["blizz_id"])))
    import psycopg2
    conn = psycopg2.connect(url)
    conn.set_session(readonly=True)
    cur = conn.cursor()
    cur.execute("SET statement_timeout='60min'")
    # latest battletag and last game per account in one pass; region from
    # the draft row (replay_players.region is NULL on some early rows)
    cur.execute(f"""
        SELECT blizz_id, region, n, last_ts, tag FROM (
          SELECT rp.blizz_id, d.region,
                 count(*) OVER w AS n,
                 EXTRACT(EPOCH FROM max(d.game_date) OVER w)::bigint AS last_ts,
                 rp.battletag AS tag,
                 row_number() OVER (PARTITION BY rp.blizz_id, d.region ORDER BY d.game_date DESC) AS k
          FROM replay_players rp JOIN replay_draft_data d USING (replay_id)
          WHERE d.game_date >= %s
          WINDOW w AS (PARTITION BY rp.blizz_id, d.region)) x
        WHERE k = 1 AND n >= %s""", (since, int(min_games)))
    got = cur.fetchall()
    print(f"accounts with >= {min_games} SL games since {since}: {len(got):,}")
    rows, skipped_pulled, skipped_tag = [], 0, 0
    for b, r, n, last_ts, t in got:
        if (int(r), int(b)) in pulled:
            skipped_pulled += 1
            continue
        if not t or "#" not in t:
            skipped_tag += 1
            continue
        rows.append((-int(last_ts), -int(n), int(r), int(b), t, int(n), int(last_ts)))
    rows.sort()
    out = os.path.join(OUT_DIR, out_name)
    import datetime
    with open(out + ".tmp", "w") as f:
        f.write("order\tregion\tblizz_id\tbattletag\tn_games\tlast_game\n")
        for i, (_, _, r, b, t, n, ts) in enumerate(rows):
            day = datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime("%Y-%m-%d")
            f.write(f"{i}\t{r}\t{b}\t{t}\t{n}\t{day}\n")
    os.replace(out + ".tmp", out)
    n_arr = np.array([x[5] for x in rows])
    print(f"wrote {out}: {len(rows):,} accounts (skipped {skipped_pulled:,} already pulled, "
          f"{skipped_tag} without a usable battletag); games since {since}: median {np.median(n_arr):.0f}, "
          f"p90 {np.percentile(n_arr, 90):.0f}; regions "
          f"{dict(zip(*[u.tolist() for u in np.unique([x[2] for x in rows], return_counts=True)]))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tranche2", action="store_true",
                    help="build panel2.tsv: unpulled accounts, >= 20 SL games since 2025-01-01, recent first")
    ap.add_argument("--since", default="2025-01-01")
    ap.add_argument("--min-games", type=int, default=None)
    ap.add_argument("--start", default="2024-04-01")
    ap.add_argument("--end", default="2026-05-23")  # exclusive; snapshot ends 2026-05-22
    a = ap.parse_args()
    if a.tranche2:
        return tranche2(a.min_games or 20, a.since, "panel2.tsv")
    a.min_games = a.min_games or 100
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
