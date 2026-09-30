"""
overfit2026 data layer: compact game tables for the proxy/gold experiments.

Games are stored as tuples (replay_id, tier, map, t0, t1, bans, winner)
with hero-name tuples (the drift2026 "slim" format, so drift2026's
count_chunk/stats_from_counts consume them directly).

Sources
  snapshot   the pinned paper-1 corpus (rerun2026.common.load_data():
             1,949,087 replays, patch 2.55, replay_id <= 63,653,039).
  future     replay_draft_data rows with replay_id > 63,653,039 on the
             whitelisted post-snapshot builds 2.55.16.97039 and
             2.55.17.{97605,97650,97771,98025}. Every one of these builds
             shipped before the drift pre-registration date (2026-09-27);
             anything newer is excluded by construction (whitelist, not a
             date filter) so no pre-registered build is ever read.
  backfill   replay_id > 63,653,039 on OLDER 2.55 builds (<= 2.55.16.96881):
             games uploaded to the DB after the snapshot was pinned but played
             in the snapshot's era. Unseen by every paper-1 model's weights.

Splits
  half(rid)  deterministic replay-level hash split of the snapshot into
             A / B (splitmix64 of replay_id, low bit). Parity of the raw id is
             avoided because W13 already uses id parity on the future set.

Usage: python3 overfit2026/data.py build
"""
import os
import sys
import json
import gzip
import pickle

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)

CACHE = os.path.join(HERE, "cache")
SNAP_PKL = os.path.join(CACHE, "snapshot_games.pkl.gz")
FUT_PKL = os.path.join(CACHE, "future_games.pkl.gz")
BACKFILL_PKL = os.path.join(CACHE, "backfill_games.pkl.gz")

SNAPSHOT_BOUND = 63653039
FUTURE_BUILDS = ["2.55.16.97039", "2.55.17.97605", "2.55.17.97650",
                 "2.55.17.97771", "2.55.17.98025"]
LAST_PRE_SNAPSHOT_BUILD = (2, 55, 16, 96881)


def splitmix64(x):
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return x ^ (x >> 31)


def half(rid, salt=0):
    """0 = half A, 1 = half B (salt gives independent sub-splits)."""
    return splitmix64(int(rid) * 2654435761 + salt) & 1


def frac_bucket(rid, n=8, salt=17):
    """Deterministic bucket in [0, n) for nested data-size subsets."""
    return splitmix64(int(rid) * 40503 + salt) % n


def _dump(obj, path):
    with gzip.open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def load_snapshot():
    return _load(SNAP_PKL)


def load_future():
    """dict build -> list of games."""
    return _load(FUT_PKL)


def load_backfill():
    return _load(BACKFILL_PKL)


def _slim(rid, tier, gmap, t0, t1, b0, b1, w):
    def lst(x):
        return json.loads(x) if isinstance(x, str) else (x or [])
    return (int(rid), tier, gmap, tuple(lst(t0)), tuple(lst(t1)),
            tuple(lst(b0)) + tuple(lst(b1)), int(w))


def build_snapshot():
    from rerun2026 import common
    common.setup()
    rows = common.load_data()
    games = []
    for r in rows:
        t0, t1 = r["team0_heroes"] or [], r["team1_heroes"] or []
        if len(t0) != 5 or len(t1) != 5 or r["winner"] not in (0, 1):
            continue
        games.append(_slim(r["replay_id"], r["skill_tier"], r["game_map"], t0, t1,
                           r.get("team0_bans"), r.get("team1_bans"), r["winner"]))
    assert max(g[0] for g in games) <= SNAPSHOT_BOUND
    _dump(games, SNAP_PKL)
    print(f"snapshot: {len(games):,} games -> {SNAP_PKL}")


def build_db():
    import psycopg2
    for line in open(os.path.join(TRAINING_DIR, "..", ".env")):
        if line.startswith("DATABASE_URL=") and "DATABASE_URL" not in os.environ:
            os.environ["DATABASE_URL"] = line.split("=", 1)[1].strip().strip('"')
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    conn.set_session(readonly=True)
    cur = conn.cursor()
    cur.execute("SET statement_timeout='30min'")
    cur.execute("""
        SELECT replay_id, skill_tier, game_map, team0_heroes, team1_heroes,
               team0_bans, team1_bans, winner, game_version
        FROM replay_draft_data
        WHERE replay_id > %s AND game_version LIKE '2.55.%%'""", (SNAPSHOT_BOUND,))
    fut = {b: [] for b in FUTURE_BUILDS}
    back = {}
    skipped = {}
    for rid, tier, gmap, t0, t1, b0, b1, w, ver in cur:
        g = _slim(rid, tier, gmap, t0, t1, b0, b1, w)
        if len(g[3]) != 5 or len(g[4]) != 5 or g[6] not in (0, 1):
            continue
        if ver in fut:
            fut[ver].append(g)
            continue
        key = tuple(int(p) for p in ver.split("."))
        if key <= LAST_PRE_SNAPSHOT_BUILD:
            back.setdefault(ver, []).append(g)
        else:
            skipped[ver] = skipped.get(ver, 0) + 1   # never analyzed
    _dump(fut, FUT_PKL)
    _dump(back, BACKFILL_PKL)
    print("future:", {b: len(v) for b, v in fut.items()})
    print(f"backfill: {sum(len(v) for v in back.values()):,} games over {len(back)} builds")
    print("skipped (not whitelisted, not read further):", skipped)


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        build_db()
        build_snapshot()
