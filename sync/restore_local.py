"""
Restore the local store from the NAS bundles written by sync/backup_local.py.

    python3 sync/restore_local.py --db hots_restore [--base base_<stamp>.tar] [--upto <stamp>]

1. Picks the newest base bundle (or --base), extracts it to the local RAID
   staging dir and checks hots.dump against the manifest sha256.
2. Creates database --db in the hots-pg container (refuses if it exists) and
   runs pg_restore -j4 into it.
3. Applies every incr bundle whose snapshot is after the base snapshot, oldest
   first (up to --upto): sha256-checks each member, upserts BIG-table deltas on
   the primary key, replaces SMALL tables with their full daily copy.
   neon-study/* members are not loaded (Neon is their home); restore them into
   a Neon branch by hand if Neon itself is lost.
4. Prints per-table row counts. Compare content with:
       python3 sync/verify_store.py --ref <env> --other <env of the restored DB>

To promote a restored DB: stop cron, rename databases inside hots-pg
(ALTER DATABASE), re-run sync/sql/local-store.sql, re-enable cron.
Re-apply privacy afterwards if PRIVACY_RESEARCH_MODE=pseudonymize was ever used
(run sync/privacy-sync.ts; it re-applies player_privacy rows).

Needs DATABASE_URL (local store; its database name is swapped for --db).
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from urllib.parse import urlsplit, urlunsplit

import psycopg2

ROOT = "/archive-backup/hots-db"
STAGE = "/home/max/pgdata/staging/backup"
CONTAINER = "hots-pg"
# cron runs with a minimal PATH; zstd comes from linuxbrew on this box.
ZSTD = shutil.which("zstd") or "/home/linuxbrew/.linuxbrew/bin/zstd"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dsn_for(db):
    u = urlsplit(os.environ["DATABASE_URL"])
    return urlunsplit(u._replace(path="/" + db))


def psql(db, sql):
    subprocess.run(["docker", "exec", CONTAINER, "psql", "-U", "hots", "-d", db,
                    "-v", "ON_ERROR_STOP=1", "-qc", sql], check=True)


def extract(tar_path, dest):
    os.makedirs(dest, exist_ok=True)
    with tarfile.open(tar_path) as t:
        t.extractall(dest, filter="data")
    return json.load(open(os.path.join(dest, "manifest.json")))


def pk_columns(cur, table):
    cur.execute("""SELECT a.attname FROM pg_index i
                   JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
                   WHERE i.indrelid = %s::regclass AND i.indisprimary""", (f'public."{table}"',))
    return [r[0] for r in cur.fetchall()]


def copy_in(cur, target, cols, path):
    collist = ", ".join(f'"{c}"' for c in cols)
    z = subprocess.Popen([ZSTD, "-dcq", path], stdout=subprocess.PIPE)
    cur.copy_expert(f"COPY {target} ({collist}) FROM STDIN WITH (FORMAT csv, HEADER true)", z.stdout)
    if z.wait() != 0:
        raise RuntimeError(f"zstd failed on {path}")


def apply_incr(conn, work, man):
    cur = conn.cursor()
    for name, meta in man["tables"].items():
        if meta["mode"] == "neon-study":
            continue
        f = os.path.join(work, f"{name}.csv.zst")
        if sha256(f) != meta["sha256"]:
            raise RuntimeError(f"sha256 mismatch: {f}")
    replace = [t for t, m in man["tables"].items() if m["mode"] == "replace"]
    # FK parents load before their children (e.g. users before tracked_battletags).
    cur.execute("""SELECT conrelid::regclass::text, confrelid::regclass::text
                   FROM pg_constraint WHERE contype = 'f'""")
    parents = {}
    for child, parent in cur.fetchall():
        parents.setdefault(child.strip('"'), set()).add(parent.strip('"'))
    ordered = []
    def visit(t, seen=()):
        if t in ordered or t not in replace:
            return
        for p in parents.get(t, ()):
            if p not in seen:
                visit(p, seen + (t,))
        ordered.append(t)
    for t in replace:
        visit(t)
    replace = ordered
    if replace:
        cur.execute("TRUNCATE " + ", ".join(f'public."{t}"' for t in replace))
        for t in replace:
            copy_in(cur, f'public."{t}"', man["tables"][t]["columns"], os.path.join(work, f"{t}.csv.zst"))
    for t, m in man["tables"].items():
        if m["mode"] != "upsert":
            continue
        cols, pk = m["columns"], pk_columns(cur, t)
        collist = ", ".join(f'"{c}"' for c in cols)
        sets = ", ".join(f'"{c}" = excluded."{c}"' for c in cols if c not in pk)
        cur.execute(f'CREATE TEMP TABLE _d (LIKE public."{t}" INCLUDING DEFAULTS) ON COMMIT DROP')
        copy_in(cur, "_d", cols, os.path.join(work, f"{t}.csv.zst"))
        cur.execute(f'INSERT INTO public."{t}" ({collist}) SELECT {collist} FROM _d '
                    f'ON CONFLICT ({", ".join(pk)}) DO UPDATE SET {sets}')
        print(f"    {t}: {cur.rowcount:,} rows upserted", flush=True)
        conn.commit()
    conn.commit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--base")
    ap.add_argument("--upto")
    ap.add_argument("--incr-only", action="store_true",
                    help="--db already holds the restored base: only apply increments")
    a = ap.parse_args()
    if a.db == "hots":
        sys.exit("refusing to restore over the live database 'hots'")

    bases = sorted(f for f in os.listdir(os.path.join(ROOT, "base")) if f.endswith(".tar"))
    base = a.base or bases[-1]
    work = os.path.join(STAGE, f"restore_{a.db}")
    shutil.rmtree(work, ignore_errors=True)
    print(f"base {base}", flush=True)
    if a.incr_only:
        os.makedirs(work, exist_ok=True)
        with tarfile.open(os.path.join(ROOT, "base", base)) as t:
            man = json.load(t.extractfile("manifest.json"))
    else:
        man = extract(os.path.join(ROOT, "base", base), work)
        if sha256(os.path.join(work, "hots.dump")) != man["files"]["hots.dump"]["sha256"]:
            sys.exit("base dump sha256 mismatch")
        print("  sha256 ok", flush=True)
        psql("postgres", f'CREATE DATABASE "{a.db}"')
        r = subprocess.run(["docker", "exec", CONTAINER, "pg_restore", "-U", "hots", "-d", a.db,
                            "-j", "4", "--no-owner", f"/staging/backup/restore_{a.db}/hots.dump"])
        if r.returncode != 0:
            sys.exit("pg_restore failed")
        print("  pg_restore ok", flush=True)
        os.remove(os.path.join(work, "hots.dump"))
    base_snap = man["snapshot"]

    conn = psycopg2.connect(dsn_for(a.db))
    # Triggers off while applying: keep each row's backed-up updated_at instead of
    # re-stamping it, so increments taken after a promote stay contiguous.
    conn.cursor().execute("SET TimeZone = 'UTC'; SET session_replication_role = replica")
    incrs = sorted(f for f in os.listdir(os.path.join(ROOT, "incr")) if f.endswith(".tar"))
    for f in incrs:
        stamp = f[len("incr_"):-len(".tar")]
        if a.upto and stamp > a.upto:
            break
        iw = os.path.join(work, f[:-4])
        im = extract(os.path.join(ROOT, "incr", f), iw)
        if im["snapshot"] <= base_snap:
            shutil.rmtree(iw)
            continue
        print(f"  applying {f}", flush=True)
        apply_incr(conn, iw, im)
        shutil.rmtree(iw)

    cur = conn.cursor()
    cur.execute("""SELECT table_name FROM information_schema.tables
                   WHERE table_schema='public' AND table_type='BASE TABLE' ORDER BY 1""")
    for (t,) in cur.fetchall():
        cur.execute(f'SELECT count(*) FROM public."{t}"')
        print(f"  {t:42} {cur.fetchone()[0]:>12,}")
    conn.close()
    shutil.rmtree(work, ignore_errors=True)
    print("restore complete", flush=True)


if __name__ == "__main__":
    main()
