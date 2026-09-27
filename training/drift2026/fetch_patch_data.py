"""
PHASE D0: patch-aware sidecar for the pinned replay snapshot.

The pinned snapshot loader (shared.load_replay_data) omits game_version and
game_date. This script runs ONE DB query fetching replay_id ->
(game_version, game_date) for every replay_id <= SNAPSHOT_BOUND (63653039,
the snapshot's max id) and saves a compact sidecar:

  drift2026/patch_sidecar.npz    replay_ids (int64), build_idx (int16),
                                 date_days (int32, days since 1970-01-01)
  drift2026/patch_index.json     ordered build list (chronological =
                                 numeric version order) + per-build DB counts
                                 and date ranges; also raw game_version ->
                                 build index map

Full replays are NEVER refetched — this is two columns.

--inventory then joins the sidecar with the pinned snapshot (via
drift2026.common.load_data_with_patches) and writes PATCH_INVENTORY.md with
games per build and per minor patch (2.55.x) *of the dataset of record*.

Usage:
    set -a && source .env && set +a
    python drift2026/fetch_patch_data.py            # fetch (needs DATABASE_URL)
    python drift2026/fetch_patch_data.py --inventory  # join + PATCH_INVENTORY.md
"""
import os
import sys
import json
import argparse
import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drift2026 import common

EPOCH = datetime.date(1970, 1, 1)


def fetch():
    import numpy as np
    import psycopg2
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL required")
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    cur.execute(
        "SELECT replay_id, game_version, game_date FROM replay_draft_data "
        "WHERE replay_id <= %s ORDER BY replay_id", (common.SNAPSHOT_BOUND,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    print(f"Fetched {len(rows)} (replay_id, game_version, game_date) rows")

    versions = sorted({r[1] for r in rows}, key=common.build_sort_key)
    vidx = {v: i for i, v in enumerate(versions)}
    n = len(rows)
    ids = np.empty(n, dtype=np.int64)
    bidx = np.empty(n, dtype=np.int16)
    days = np.empty(n, dtype=np.int32)
    meta = defaultdict(lambda: {"n_db": 0, "min_date": None, "max_date": None})
    for i, (rid, ver, dt) in enumerate(rows):
        ids[i] = rid
        bidx[i] = vidx[ver]
        d = dt.date()
        days[i] = (d - EPOCH).days
        m = meta[ver]
        m["n_db"] += 1
        m["min_date"] = min(m["min_date"] or d, d)
        m["max_date"] = max(m["max_date"] or d, d)

    np.savez_compressed(common.SIDECAR_NPZ, replay_ids=ids, build_idx=bidx,
                        date_days=days)
    common.write_json(common.PATCH_INDEX_JSON, {
        "snapshot_bound": common.SNAPSHOT_BOUND,
        "n_rows": n,
        "builds": versions,
        "build_meta": {v: {"n_db": meta[v]["n_db"],
                           "min_date": str(meta[v]["min_date"]),
                           "max_date": str(meta[v]["max_date"])}
                       for v in versions},
    })
    sz = os.path.getsize(common.SIDECAR_NPZ) / 1e6
    print(f"Sidecar: {common.SIDECAR_NPZ} ({sz:.1f} MB), {len(versions)} builds")


def inventory():
    rows, builds = common.load_data_with_patches()
    per_build = defaultdict(lambda: [0, None, None])
    for r in rows:
        e = per_build[r["build"]]
        e[0] += 1
        e[1] = r["date_days"] if e[1] is None else min(e[1], r["date_days"])
        e[2] = r["date_days"] if e[2] is None else max(e[2], r["date_days"])

    def dstr(days):
        return str(EPOCH + datetime.timedelta(days=int(days)))

    ordered = [b for b in builds if b in per_build]
    per_minor = defaultdict(lambda: [0, None, None, 0])
    for b in ordered:
        n, lo, hi = per_build[b]
        mk = common.minor_key(b)
        m = per_minor[mk]
        m[0] += n
        m[1] = lo if m[1] is None else min(m[1], lo)
        m[2] = hi if m[2] is None else max(m[2], hi)
        m[3] += 1

    cutoff = common.cutoff_idx(builds)
    n_sizable = sum(1 for b in ordered if per_build[b][0] >= common.SIZABLE_GAMES)
    n_sizable_minor = sum(1 for m in per_minor.values()
                          if m[0] >= common.SIZABLE_GAMES)
    n_train = sum(per_build[b][0] for b in ordered if builds.index(b) <= cutoff)
    n_test = sum(per_build[b][0] for b in ordered if builds.index(b) > cutoff)

    lines = [
        "# drift2026 patch inventory (dataset of record)",
        "",
        f"Corpus: pinned snapshot 2026-05-22, 2.55-filtered (rerun2026), joined with",
        f"`patch_sidecar.npz` -> **{len(rows):,} replays** across "
        f"**{len(ordered)} builds** / **{len(per_minor)} minor patches**.",
        "",
        f"- Patch unit for all drift2026 experiments: the **build** (full game_version).",
        f"  research_focus_v2's \"~50 minor patches\" corresponds to the {len(ordered)}",
        f"  builds; there are only {len(per_minor)} minor 2.55.x versions (2.55.11 was",
        f"  never shipped), and minor patches are far too coarse (2.55.3 spans 17",
        f"  months / 8 builds).",
        f"- Builds with >= {common.SIZABLE_GAMES//1000}K games: **{n_sizable} of {len(ordered)}**"
        f" (minor patches: {n_sizable_minor} of {len(per_minor)}).",
        f"- Temporal split: train = builds through **{common.TRAIN_CUTOFF_BUILD}**"
        f" ({n_train:,} replays), strictly-future test = everything after"
        f" ({n_test:,} replays; headline builds: {', '.join(common.SIZABLE_TEST_BUILDS)}).",
        "",
        "## Games per build",
        "",
        "| idx | build | games | first game | last game | role |",
        "|---|---|---|---|---|---|",
    ]
    for b in ordered:
        n, lo, hi = per_build[b]
        i = builds.index(b)
        role = "TEST" if i > cutoff else "train"
        if n < common.SIZABLE_GAMES:
            role += " (small)"
        lines.append(f"| {i} | {b} | {n:,} | {dstr(lo)} | {dstr(hi)} | {role} |")

    lines += ["", "## Games per minor patch (2.55.x)", "",
              "| minor | builds | games | first game | last game |",
              "|---|---|---|---|---|"]
    for mk in sorted(per_minor, key=common.build_sort_key):
        n, lo, hi, nb = per_minor[mk]
        lines.append(f"| {mk} | {nb} | {n:,} | {dstr(lo)} | {dstr(hi)} |")

    lines += [
        "",
        "## Notes",
        "",
        "- Adjacent builds overlap by ~1-2 days at the boundary (rollout lag);",
        "  patch assignment is by game_version, never by date, so this is benign.",
        "- Several hotfix builds are tiny (<5K games: e.g. 2.55.3.91020,",
        "  2.55.14.95883, 2.55.9.93565, 2.55.16.96846/96870) — too small to",
        "  condition on alone; per-patch stats exist for them but experiments",
        "  threshold on sizable builds, and min-games feature thresholds",
        "  (>=30 pairwise / >=50 hero-map) mostly blank them out.",
        "- Early 2.55 builds are months long (80K+ games); post-2024 cadence is",
        "  ~6-9 weeks, 20-100K games per build (~250K/week arriving now).",
    ]
    path = os.path.join(common.DRIFT_DIR, "PATCH_INVENTORY.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inventory", action="store_true",
                    help="join sidecar with pinned snapshot, write PATCH_INVENTORY.md")
    args = ap.parse_args()
    common.setup()
    if args.inventory:
        inventory()
    else:
        fetch()


if __name__ == "__main__":
    main()
