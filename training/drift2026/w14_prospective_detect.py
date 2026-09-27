"""
W14 — prospective runner for pre-registered claim C1 (PREREG_FORWARD.md).

The retrospective detector (w3_changepoints.py) reads a daily count cache
that w3_recovery.py built from the pinned May-2026 snapshot, and its
ground truth comes from fetch_patch_notes.py with a hard-coded build list.
This runner points the SAME frozen code at builds from 2.55.17.98025 on,
read from the live database, and writes to separate paths so nothing
retrospective is overwritten. It changes no decision logic:

  counts  DB -> per-day counts via w3_recovery.walk_build (frozen), same
          row tuple and ordering (date, replay_id) as w3_recovery.main
  notes   fetch_patch_notes.main (frozen) with its build list replaced by
          the prospective builds and their first game dates
  detect  w3_changepoints.main (frozen): consecutive builds with at least
          common.SIZABLE_GAMES (20,000) games, smaller builds folded in,
          per-hero two-sided two-proportion WR tests (>= 200 games/side),
          Benjamini-Hochberg q = 0.05 within each boundary

The first boundary is 2.55.17.98025 -> first new build. 98025 and every
earlier build are excluded from confirmation (they only serve as the
earlier side of that first boundary).

Usage (from training/, with DATABASE_URL set):
  python3 drift2026/w14_prospective_detect.py counts
  python3 drift2026/w14_prospective_detect.py notes
  python3 drift2026/w14_prospective_detect.py detect
Outputs: patch_stats/prospective_daily_counts.pkl.gz,
         prospective_patch_notes.json, results/prospective/
"""
import os
import sys
import gzip
import json
import pickle
import argparse
import datetime
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drift2026 import common

common.setup()

FIRST_BUILD = "2.55.17.98025"   # last excluded build; earlier side of boundary 1
CACHE = os.path.join(common.STATS_DIR, "prospective_daily_counts.pkl.gz")
GT = os.path.join(common.DRIFT_DIR, "prospective_patch_notes.json")
OUT_DIR = os.path.join(common.RESULTS_DIR, "prospective")
LAST_BUILD = None   # set only by --validate


def set_validation():
    """Reproduction check on retrospective builds (2.55.13.95301 through
    2.55.16.96881), compared with results/w3_changepoints.json. Separate
    paths; never touches 2.55.17 builds."""
    global FIRST_BUILD, LAST_BUILD, CACHE, GT, OUT_DIR
    FIRST_BUILD, LAST_BUILD = "2.55.13.95301", "2.55.16.96881"
    CACHE = os.path.join(common.STATS_DIR, "validation_daily_counts.pkl.gz")
    GT = os.path.join(common.DRIFT_DIR, "patch_notes_ground_truth.json")
    OUT_DIR = os.path.join(common.RESULTS_DIR, "prospective_validation")


def fetch_rows():
    import psycopg2
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    cur = conn.cursor()
    cur.execute("SET statement_timeout='20min'")
    cur.execute("""
        SELECT game_version, game_date, replay_id, skill_tier, game_map,
               team0_heroes, team1_heroes, team0_bans, team1_bans, winner
        FROM replay_draft_data WHERE game_version LIKE '2.55.%'""")
    key0 = common.build_sort_key(FIRST_BUILD)

    def lst(x):
        return json.loads(x) if isinstance(x, str) else x
    epoch = datetime.date(1970, 1, 1)
    by_build = defaultdict(list)
    for v, gd, rid, tier, gmap, t0, t1, b0, b1, w in cur.fetchall():
        if common.build_sort_key(v) < key0:
            continue
        if LAST_BUILD and common.build_sort_key(v) > common.build_sort_key(LAST_BUILD):
            continue
        by_build[v].append(((gd.date() - epoch).days, rid, tier, gmap,
                            tuple(lst(t0)), tuple(lst(t1)),
                            tuple(lst(b0)) + tuple(lst(b1)), w))
    return by_build


def counts():
    from drift2026.w3_recovery import walk_build
    by_build = fetch_rows()
    builds = sorted(by_build, key=common.build_sort_key)
    daily_cache, build_games, first_date = {}, {}, {}
    for bi, b in enumerate(builds):
        rows = sorted(by_build[b], key=lambda t: (t[0], t[1]))
        slim = [(t[0], t[2], t[3], t[4], t[5], t[6], t[7]) for t in rows]
        _, n, _, plain = walk_build((bi, slim, False))
        daily_cache[bi], build_games[bi] = plain, n
        first_date[b] = (datetime.date(1970, 1, 1)
                         + datetime.timedelta(days=rows[0][0])).isoformat()
        print(f"{b}: {n:,} games from {first_date[b]}")
    with gzip.open(CACHE, "wb") as f:
        pickle.dump({"builds": builds, "per_build": daily_cache,
                     "build_games": build_games, "first_date": first_date},
                    f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"wrote {CACHE}")


def notes():
    from drift2026 import fetch_patch_notes as fpn
    with gzip.open(CACHE, "rb") as f:
        first_date = pickle.load(f)["first_date"]
    fpn.CORPUS_BUILDS = dict(first_date)
    fpn.BUILDNUM_TO_CORPUS = {v.rsplit(".", 1)[1]: v for v in fpn.CORPUS_BUILDS}
    fpn.OUT = Path(GT)
    sys.argv = sys.argv[:1]   # the frozen mains parse argv themselves
    fpn.main()


def detect():
    from drift2026 import w3_changepoints as w3
    os.makedirs(OUT_DIR, exist_ok=True)
    w3.DAILY_CACHE = CACHE
    w3.GT_PATH = GT
    common.RESULTS_DIR = OUT_DIR
    sys.argv = sys.argv[:1]
    w3.main()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["counts", "notes", "detect"])
    ap.add_argument("--validate", action="store_true",
                    help="reproduction check on retrospective builds")
    args = ap.parse_args()
    if args.validate:
        assert args.stage != "notes", "validation uses the frozen ground truth"
        set_validation()
    {"counts": counts, "notes": notes, "detect": detect}[args.stage]()


if __name__ == "__main__":
    main()
