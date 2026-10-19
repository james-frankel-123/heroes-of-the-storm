"""
Compare tables between two Postgres databases by row count and content hash,
without moving rows: each side returns one (count, hash) per bucket.

Bucketed tables (those with a replay_id column) group by replay_id / 100000;
others are hashed whole. The hash is sum(hashtextextended(ROW(cols)::text, 0))
over the columns the REFERENCE side has, in its order, so local-only columns
(updated_at) never enter it. Both sessions run with TimeZone=UTC.

Usage (source .env first):
    python3 sync/verify_store.py --ref NEON_DATABASE_URL --other DATABASE_URL \
        [--tables a,b] [--out result.json] [--reuse-ref ref.json]

--ref-cache DIR keeps each finished reference-side table hash, so a rerun
after a failure does not rescan Neon.
--ref/--other name environment variables holding DSNs (never DSNs on the
command line). --reuse-ref loads the reference side from an earlier result
file instead of querying it again (Neon scans cost compute; the reference
side of a frozen table does not change). Exit code 1 on any mismatch.
"""
import argparse
import concurrent.futures as cf
import json
import os
import sys
import time

import psycopg2

BUCKET = 100000
SKIP_DEFAULT = {"rating_items", "draft_ratings"}  # study tables: Neon only


def connect(dsn):
    c = psycopg2.connect(dsn)
    c.set_session(readonly=True, autocommit=True)
    cur = c.cursor()
    cur.execute("SET TimeZone = 'UTC'; SET statement_timeout = 0")
    return c


def tables_of(dsn):
    with connect(dsn) as c, c.cursor() as cur:
        cur.execute("""SELECT table_name FROM information_schema.tables
                       WHERE table_schema='public' AND table_type='BASE TABLE'""")
        return sorted(r[0] for r in cur.fetchall())


def columns_of(dsn, table):
    """[(name, data_type)] in table order."""
    with connect(dsn) as c, c.cursor() as cur:
        cur.execute("""SELECT column_name, data_type FROM information_schema.columns
                       WHERE table_schema='public' AND table_name=%s
                       ORDER BY ordinal_position""", (table,))
        return [list(r) for r in cur.fetchall()]


def table_hash(dsn, table, coltypes, cache=None):
    if cache and os.path.exists(cache):
        return json.load(open(cache))
    cols = [c for c, _ in coltypes]
    row = "ROW(" + ", ".join(f'"{c}"' for c in cols) + ")::text"
    bucketed = ["replay_id", "integer"] in coltypes
    key = f"replay_id / {BUCKET}" if bucketed else "0"
    q = (f'SELECT {key} AS b, count(*), coalesce(sum(hashtextextended({row}, 0)), 0)::text '
         f'FROM public."{table}" GROUP BY 1 ORDER BY 1')
    t0 = time.time()
    c = connect(dsn)
    try:
        cur = c.cursor()
        cur.execute(q)
        res = {str(b): [n, h] for b, n, h in cur.fetchall()}
    finally:
        c.close()
    out = {"columns": coltypes, "bucketed": bucketed, "buckets": res,
           "rows": sum(v[0] for v in res.values()), "seconds": round(time.time() - t0, 1)}
    if cache:
        with open(cache + ".tmp", "w") as f:
            json.dump(out, f)
        os.replace(cache + ".tmp", cache)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--other", required=True)
    ap.add_argument("--tables")
    ap.add_argument("--out")
    ap.add_argument("--reuse-ref")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--ref-cache", help="dir: keep each finished reference table here and "
                    "reuse it on a rerun (saves repeat Neon scans after a failure)")
    a = ap.parse_args()
    if a.ref_cache:
        os.makedirs(a.ref_cache, exist_ok=True)
    ref_dsn, other_dsn = os.environ[a.ref], os.environ[a.other]

    prior = json.load(open(a.reuse_ref))["ref"] if a.reuse_ref else {}
    tables = a.tables.split(",") if a.tables else [
        t for t in tables_of(ref_dsn) if t not in SKIP_DEFAULT]
    cols = {t: (prior[t]["columns"] if t in prior else columns_of(ref_dsn, t)) for t in tables}

    ref, other = {}, {}
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        fr = {t: ex.submit(table_hash, ref_dsn, t, cols[t],
                           os.path.join(a.ref_cache, f"{t}.json") if a.ref_cache else None)
              for t in tables if t not in prior}
        fo = {t: ex.submit(table_hash, other_dsn, t, cols[t]) for t in tables}
        for t in tables:
            ref[t] = prior[t] if t in prior else fr[t].result()
            other[t] = fo[t].result()

    bad = 0
    report = {}
    for t in tables:
        r, o = ref[t]["buckets"], other[t]["buckets"]
        diff = sorted((k for k in set(r) | set(o) if r.get(k) != o.get(k)), key=int)
        report[t] = {"ref_rows": ref[t]["rows"], "other_rows": other[t]["rows"],
                     "buckets": len(r), "mismatched_buckets": diff}
        bad += bool(diff)
        print(f"{'OK  ' if not diff else 'DIFF'} {t:42} ref={ref[t]['rows']:>12,} "
              f"other={other[t]['rows']:>12,} buckets={len(r):>4} "
              f"mismatch={len(diff)}{' ' + ','.join(diff[:12]) if diff else ''}", flush=True)

    if a.out:
        with open(a.out, "w") as f:
            json.dump({"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "ref_env": a.ref, "other_env": a.other,
                       "report": report, "ref": ref, "other": other}, f)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
