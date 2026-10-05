"""
RETIRED 2026-10-05: full Neon dumps cost ~100+ GB of egress per run. Replaced by
sync/backup_local.py (bundled backups of the local primary store). Kept for reference.

Local backup of the Neon database (no pg_dump on this box; uses psycopg2
COPY). Every user table is exported to gzipped CSV under
/archive-backup/hots-db/<YYYY-MM-DD>/ (NAS, mirrored), plus a schema.sql-ish
column manifest. All tables are read in one REPEATABLE READ snapshot. A run
writes to <date>.partial/ and renames on success, so only complete backups
count toward KEEP. Refuses to run if the NAS is not mounted.

The big tables (replay_players ~7M rows with jsonb scoreboards) dominate;
expect a few GB compressed and tens of minutes on the first run.

Usage:
    source .env first, then: python3 sync/backup_db.py
Cron (Sundays 03:00). Cron's /usr/bin/python3 has no psycopg2, so the
linuxbrew interpreter is named explicitly:
    0 3 * * 0 cd /home/max/heroes-of-the-storm && set -a && . ./.env && set +a && /home/linuxbrew/.linuxbrew/bin/python3 sync/backup_db.py >> sync/logs/backup-db.log 2>&1
"""
import datetime
import gzip
import json
import os
import shutil
import sys
import time

import psycopg2

KEEP = 12
MOUNT = "/archive-backup"
BASE = os.path.join(MOUNT, "hots-db")


def main():
    url = os.environ.get("DATABASE_URL")
    if not url:
        sys.exit("DATABASE_URL required (source .env)")
    if not os.path.ismount(MOUNT):
        sys.exit(f"{MOUNT} is not mounted; refusing to back up to local disk")
    today = datetime.date.today().isoformat()
    final_dir = os.path.join(BASE, today)
    out_dir = final_dir + ".partial"
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)

    conn = psycopg2.connect(url)
    conn.set_session(isolation_level="REPEATABLE READ", readonly=True)
    cur = conn.cursor()
    cur.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
        ORDER BY table_name""")
    tables = [r[0] for r in cur.fetchall()]
    manifest = {"date": today, "tables": {}}
    print(f"[{today}] backing up {len(tables)} tables -> {out_dir}", flush=True)

    for t in tables:
        cur.execute("""
            SELECT column_name, data_type FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = %s
            ORDER BY ordinal_position""", (t,))
        cols = cur.fetchall()
        path = os.path.join(out_dir, f"{t}.csv.gz")
        t0 = time.time()
        with gzip.open(path, "wt", compresslevel=6) as f:
            cur.copy_expert(
                f'COPY "{t}" TO STDOUT WITH (FORMAT csv, HEADER true)', f)
        cur.execute(f'SELECT count(*) FROM "{t}"')
        n = cur.fetchone()[0]
        manifest["tables"][t] = {
            "rows": n, "bytes_gz": os.path.getsize(path),
            "columns": [{"name": c, "type": ty} for c, ty in cols]}
        print(f"  {t}: {n:,} rows, {os.path.getsize(path) / 1e6:.1f} MB gz "
              f"({time.time() - t0:.0f}s)", flush=True)

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    cur.close()
    conn.close()
    shutil.rmtree(final_dir, ignore_errors=True)
    os.rename(out_dir, final_dir)

    dated = sorted(d for d in os.listdir(BASE)
                   if os.path.isdir(os.path.join(BASE, d))
                   and d[:2] == "20" and not d.endswith(".partial"))
    for old in dated[:-KEEP]:
        shutil.rmtree(os.path.join(BASE, old))
        print(f"pruned {old}", flush=True)
    print("backup complete", flush=True)


if __name__ == "__main__":
    main()
