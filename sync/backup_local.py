"""
Bundled backups of the LOCAL primary store to the NAS (no Neon bulk reads).

Every run writes exactly ONE archive under /archive-backup/hots-db/:

  base/base_<UTC stamp>.tar   weekly. Members: hots.dump (pg_dump -Fc, zstd),
                              manifest.json (row counts, sha256, snapshot time).
  incr/incr_<UTC stamp>.tar   every 4 h. Members: <table>.csv.zst for
                              - BIG tables: rows with updated_at in (wm - 10 min, run start]
                              - SMALL tables: full copy, once a day
                              - study tables: full copy read from Neon, once a day (KB)
                              plus manifest.json (rows, sha256, watermark range, columns).

state.json (one small file next to them) holds the per-table watermarks and
the last daily copy time; the manifest of each bundle repeats them, so a lost
state.json can be rebuilt from the newest bundle.

A run writes <name>.tar.partial and renames on success. Retention: 4 newest
weekly bases + the first base of each of the last 6 months; increments older
than the oldest kept base are pruned.

Usage (source .env; DATABASE_URL = local store, NEON_DATABASE_URL = Neon):
    python3 sync/backup_local.py incr
    python3 sync/backup_local.py base
Restore: sync/restore_local.py. Plan: sync/docs/data-architecture-plan.md.
"""
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time

import psycopg2

MOUNT = "/archive-backup"
ROOT = os.path.join(MOUNT, "hots-db")
STAGE = "/home/max/pgdata/staging/backup"   # local RAID; /staging inside the container
CONTAINER = "hots-pg"
# cron runs with a minimal PATH; zstd comes from linuxbrew on this box.
ZSTD = shutil.which("zstd") or "/home/linuxbrew/.linuxbrew/bin/zstd"
OVERLAP = dt.timedelta(minutes=10)
DAILY = dt.timedelta(hours=20)

# Mirror of sync/table-groups.ts plus the local-only groupings.
BIG = ["replay_players", "replay_draft_data", "replay_extras", "qm_games"]
STUDY = ["rating_items", "draft_ratings"]          # Neon is their only home
STATIC = ["replay_draft_skill_tier_backup_20260930"]  # frozen; lives in bases only
KEEP_WEEKLY, KEEP_MONTHLY = 4, 6


def log(msg):
    print(f"[{dt.datetime.now(dt.timezone.utc):%Y-%m-%dT%H:%M:%SZ}] {msg}", flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def connect(env, snapshot=False):
    """Read-only session in UTC; snapshot=True gives one REPEATABLE READ snapshot."""
    c = psycopg2.connect(os.environ[env])
    c.set_session(readonly=True, isolation_level="REPEATABLE READ" if snapshot else None)
    c.cursor().execute("SET TimeZone = 'UTC'")
    return c


def tables(cur):
    cur.execute("""SELECT table_name FROM information_schema.tables
                   WHERE table_schema='public' AND table_type='BASE TABLE' ORDER BY 1""")
    return [r[0] for r in cur.fetchall()]


def columns(cur, t):
    cur.execute("""SELECT column_name FROM information_schema.columns
                   WHERE table_schema='public' AND table_name=%s ORDER BY ordinal_position""", (t,))
    return [r[0] for r in cur.fetchall()]


def export(cur, table, where, path):
    """COPY rows to <path> (CSV with header) through zstd; returns (rows, columns)."""
    cols = columns(cur, table)
    collist = ", ".join(f'"{c}"' for c in cols)
    z = subprocess.Popen([ZSTD, "-q", "-3", "-T4", "-o", path], stdin=subprocess.PIPE)
    cur.copy_expert(f'COPY (SELECT {collist} FROM public."{table}" {where}) '
                    f"TO STDOUT WITH (FORMAT csv, HEADER true)", z.stdin)
    z.stdin.close()
    if z.wait() != 0:
        raise RuntimeError(f"zstd failed for {table}")
    cur.execute(f'SELECT count(*) FROM public."{table}" {where}')
    return cur.fetchone()[0], cols


def write_bundle(kind, stamp, members, manifest):
    """tar the staged members + manifest into ONE archive on the NAS."""
    os.makedirs(os.path.join(ROOT, kind), exist_ok=True)
    final = os.path.join(ROOT, kind, f"{kind}_{stamp}.tar")
    partial = final + ".partial"
    mpath = os.path.join(STAGE, stamp, "manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=1, default=str)
    with tarfile.open(partial, "w") as tar:   # members are already compressed
        tar.add(mpath, arcname="manifest.json")
        for m in members:
            tar.add(os.path.join(STAGE, stamp, m), arcname=m)
    os.rename(partial, final)
    return final


def load_state():
    p = os.path.join(ROOT, "state.json")
    return json.load(open(p)) if os.path.exists(p) else {"watermarks": {}, "last_daily": None}


def save_state(state):
    p = os.path.join(ROOT, "state.json")
    with open(p + ".tmp", "w") as f:
        json.dump(state, f, indent=1)
    os.replace(p + ".tmp", p)


def run_incr(stamp, run_start):
    state = load_state()
    if not state["watermarks"]:
        sys.exit("no watermarks yet: take a base first (its snapshot time seeds them)")
    work = os.path.join(STAGE, stamp)
    os.makedirs(work)
    local = connect("DATABASE_URL", snapshot=True)   # one snapshot for the whole bundle
    cur = local.cursor()
    cur.execute("SELECT now()")      # pins the snapshot; rows visible == rows <= this
    snap = cur.fetchone()[0]
    manifest = {"kind": "incr", "stamp": stamp, "snapshot": snap.isoformat(), "tables": {}}
    members = []

    for t in BIG:
        lo = dt.datetime.fromisoformat(state["watermarks"][t]) - OVERLAP
        where = cur.mogrify("WHERE updated_at > %s", (lo,)).decode()
        name = f"{t}.csv.zst"
        n, cols = export(cur, t, where, os.path.join(work, name))
        manifest["tables"][t] = {"mode": "upsert", "rows": n, "since": lo.isoformat(),
                                 "until": snap.isoformat(), "columns": cols,
                                 "sha256": sha256(os.path.join(work, name))}
        members.append(name)

    last = state.get("last_daily")
    daily = last is None or snap - dt.datetime.fromisoformat(last) >= DAILY
    if daily:
        for t in tables(cur):
            if t in BIG or t in STUDY or t in STATIC:
                continue
            name = f"{t}.csv.zst"
            n, cols = export(cur, t, "", os.path.join(work, name))
            manifest["tables"][t] = {"mode": "replace", "rows": n, "columns": cols,
                                     "sha256": sha256(os.path.join(work, name))}
            members.append(name)
        neon = connect("NEON_DATABASE_URL")
        ncur = neon.cursor()
        for t in STUDY:
            name = f"neon-study/{t}.csv.zst"
            os.makedirs(os.path.join(work, "neon-study"), exist_ok=True)
            n, cols = export(ncur, t, "", os.path.join(work, name))
            manifest["tables"][f"neon:{t}"] = {"mode": "neon-study", "rows": n, "columns": cols,
                                               "sha256": sha256(os.path.join(work, name))}
            members.append(name)
        neon.close()
    local.close()

    path = write_bundle("incr", stamp, members, manifest)
    for t in BIG:
        state["watermarks"][t] = snap.isoformat()
    if daily:
        state["last_daily"] = snap.isoformat()
    save_state(state)
    rows = {k: v["rows"] for k, v in manifest["tables"].items()}
    log(f"incr bundle {path} ({os.path.getsize(path)/1e6:.1f} MB) rows={rows}")


def run_base(stamp, run_start):
    work = os.path.join(STAGE, stamp)
    os.makedirs(work)
    local = connect("DATABASE_URL")
    cur = local.cursor()
    # Snapshot time taken BEFORE pg_dump's own snapshot: increments from here on
    # (with the overlap) cover anything the dump might miss.
    cur.execute("SELECT now()")
    snap = cur.fetchone()[0]
    local.rollback()
    t0 = time.time()
    r = subprocess.run(["docker", "exec", CONTAINER, "pg_dump", "-U", "hots", "-d", "hots",
                        "-Fc", "-Z", "zstd:3", "-f", f"/staging/backup/{stamp}/hots.dump"])
    if r.returncode != 0:
        raise RuntimeError("pg_dump failed")
    log(f"pg_dump done in {time.time()-t0:.0f}s")
    # Row counts taken after the dump (rows only grow; exact counts are
    # recomputed from the restored DB by restore_local.py --verify).
    counts = {}
    for t in tables(cur):
        cur.execute(f'SELECT count(*) FROM public."{t}"')
        counts[t] = cur.fetchone()[0]
    local.close()
    dump = os.path.join(work, "hots.dump")
    manifest = {"kind": "base", "stamp": stamp, "snapshot": snap.isoformat(),
                "rows_after_dump": counts, "files": {"hots.dump": {
                    "bytes": os.path.getsize(dump), "sha256": sha256(dump)}}}
    path = write_bundle("base", stamp, ["hots.dump"], manifest)
    state = load_state()
    for t in BIG:   # never move a watermark forward past rows an increment hasn't covered
        cur_wm = state["watermarks"].get(t)
        if cur_wm is None:
            state["watermarks"][t] = snap.isoformat()
    save_state(state)
    log(f"base bundle {path} ({os.path.getsize(path)/1e9:.2f} GB)")
    prune()


def prune():
    bdir, idir = os.path.join(ROOT, "base"), os.path.join(ROOT, "incr")
    bases = sorted(f for f in os.listdir(bdir) if f.endswith(".tar"))
    keep = set(bases[-KEEP_WEEKLY:])
    months = {}
    for b in bases:
        months.setdefault(b[5:12], b)       # base_YYYY-MM... -> first base of the month
    keep |= set(sorted(months.values())[-KEEP_MONTHLY:])
    for b in bases:
        if b not in keep:
            os.remove(os.path.join(bdir, b))
            log(f"pruned {b}")
    oldest = min(keep)[5:]                  # stamp of the oldest kept base
    for f in sorted(os.listdir(idir)) if os.path.isdir(idir) else []:
        if f.endswith(".tar") and f[5:] < oldest:
            os.remove(os.path.join(idir, f))
            log(f"pruned {f}")


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("incr", "base"):
        sys.exit(__doc__)
    if not os.path.ismount(MOUNT):
        sys.exit(f"{MOUNT} is not mounted; refusing to back up to local disk")
    if "neon.tech" in os.environ.get("DATABASE_URL", ""):
        sys.exit("DATABASE_URL points at Neon; backups read the local store only")
    run_start = dt.datetime.now(dt.timezone.utc)
    stamp = run_start.strftime("%Y-%m-%dT%H%M%SZ")
    try:
        (run_incr if sys.argv[1] == "incr" else run_base)(stamp, run_start)
    finally:
        shutil.rmtree(os.path.join(STAGE, stamp), ignore_errors=True)


if __name__ == "__main__":
    main()
