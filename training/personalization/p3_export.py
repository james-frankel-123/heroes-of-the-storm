"""
P3 frozen exports from the LOCAL research store (read-only role,
DATABASE_URL_RESEARCH). Never Neon. Runs on the main box as a streaming
COPY to gzip (no parsing here); the remotes parse the files.

Every export keeps Storm League drafts (replay_draft_data) of builds released
on or before 2026-09-27 only: game_date < 2026-09-28 and no 2.57 build. Rows
carry blizz_id but no battletag.

  stamps   one row per player-game: ids, region (player region, else the
           game's; flag when the player region was NULL), hero, hero_level
           (number) and hero_level_band (v1 API string), HP parse time
           (UTC epoch from mmr_date_parsed, US Eastern), game end epoch,
           game length, build, team, party, winner, the three MMR stamps
  games    one row per draft with what extract_features needs

Usage (from repo root):
  set -a; . ./.env; set +a
  nice -n 19 python3 training/personalization/p3_export.py stamps --tag oct05
Output: training/personalization/cache/export/<tag>/<what>.csv.gz plus
<what>.json (query, row count, sha256, time).
"""
import argparse
import datetime
import gzip
import hashlib
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WHERE = ("d.game_date < '2026-09-28' AND d.game_version NOT LIKE '2.57%' "
         "AND d.game_version NOT LIKE '2.58%' AND d.game_version NOT LIKE '3.%'")

QUERIES = {
    "stamps": f"""
        SELECT rp.replay_id, rp.blizz_id, COALESCE(rp.region, d.region) AS region,
               (rp.region IS NULL)::int AS region_was_null, rp.hero, rp.hero_level,
               rp.raw_extras->>'hero_level_band' AS hero_level_band,
               EXTRACT(EPOCH FROM ((rp.raw_extras->>'mmr_date_parsed')::timestamp
                                   AT TIME ZONE 'America/New_York'))::bigint AS parse_ts,
               EXTRACT(EPOCH FROM d.game_date)::bigint AS end_ts, d.game_length,
               d.game_version, rp.team, rp.party, rp.winner::int,
               rp.player_mmr, rp.hero_mmr, rp.role_mmr,
               EXTRACT(EPOCH FROM rp.fetched_at)::bigint AS fetched_ts
        FROM replay_players rp JOIN replay_draft_data d USING (replay_id)
        WHERE {WHERE}""",
    "games": f"""
        SELECT d.replay_id, d.game_version, EXTRACT(EPOCH FROM d.game_date)::bigint AS end_ts,
               d.game_length, d.game_map, d.skill_tier, d.league_tier, d.region,
               d.team0_heroes, d.team1_heroes, d.team0_bans, d.team1_bans, d.winner,
               d.avg_mmr, d.draft_order, EXTRACT(EPOCH FROM d.fetched_at)::bigint AS fetched_ts
        FROM replay_draft_data d WHERE {WHERE}""",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=sorted(QUERIES))
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    url = os.environ.get("DATABASE_URL_RESEARCH")
    if not url:
        raise SystemExit("DATABASE_URL_RESEARCH is not set (local read-only role)")
    if "neon" in url.lower():
        raise SystemExit("refusing a Neon URL: P3 reads the local store only")
    import psycopg2
    out_dir = os.path.join(HERE, "cache", "export", a.tag)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{a.what}.csv.gz")
    t0 = time.time()
    conn = psycopg2.connect(url)
    conn.set_session(readonly=True)
    with conn.cursor() as c:
        c.execute("SET statement_timeout = 0")
        with gzip.open(path + ".tmp", "wb", compresslevel=4) as f:
            c.copy_expert(f"COPY ({QUERIES[a.what]}) TO STDOUT WITH (FORMAT csv, HEADER true)", f)
    os.replace(path + ".tmp", path)
    h = hashlib.sha256()
    n = -1
    with gzip.open(path, "rb") as f:
        for line in f:
            n += 1
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    meta = {"what": a.what, "rows": n, "sha256": h.hexdigest(), "query": " ".join(QUERIES[a.what].split()),
            "exported_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "seconds": round(time.time() - t0)}
    with open(os.path.join(out_dir, f"{a.what}.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
