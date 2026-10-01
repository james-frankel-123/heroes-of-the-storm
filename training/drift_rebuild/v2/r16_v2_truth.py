"""
v2 out-of-sample truth: the W12 scoring games (build 2.55.16.97039 and the
four 2.55.17 builds, game_date <= 2026-09-27) re-fetched read-only with
league_tier and avg_mmr and labeled with the site tier rule (v2env.site_tier),
'unknown' dropped.

Writes, in the exact formats the frozen w12_clean_truth.truth_sets() and
r10_net / r15_net_struct read:
  drift_v2/patch_stats/clean_truth_counts.pkl.gz   (builds, per_build, meta)
  drift_v2/feature_cache/r10_clean_games.json      (rid, tier, map, t0, t1, bans, w)
No build released after 2026-09-27 is queried (explicit build list).

Usage: nice -n 19 taskset -c 48-63 python3 drift_rebuild/v2/run.py drift_rebuild/v2/r16_v2_truth.py
"""
import gzip
import json
import os
import pickle

import v2env
from drift2026 import common
from drift2026.build_patch_stats import count_chunk, merge_cell, _new_cell

BUILDS = ["2.55.16.97039", "2.55.17.97605", "2.55.17.97650", "2.55.17.97771", "2.55.17.98025"]


def main():
    import psycopg2
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    conn.set_session(readonly=True)
    cur = conn.cursor()
    cur.execute("SET statement_timeout='30min'")
    cur.execute("""
        SELECT game_version, league_tier, avg_mmr, game_map, team0_heroes, team1_heroes,
               team0_bans, team1_bans, winner, replay_id, game_date
        FROM replay_draft_data
        WHERE game_version = ANY(%s) AND game_date < '2026-09-28'""", (BUILDS,))
    rows = cur.fetchall()
    bidx = {b: i for i, b in enumerate(BUILDS)}

    def lst(x):
        return json.loads(x) if isinstance(x, str) else (x or [])
    slim, games, dropped = [], [], 0
    meta = {b: {"games": 0, "in_snapshot": 0, "min_date": None, "max_date": None} for b in BUILDS}
    for v, lt, mmr, gmap, t0, t1, b0, b1, w, rid, gd in rows:
        tier = v2env.site_tier(lt, mmr)
        t0, t1 = tuple(lst(t0)), tuple(lst(t1))
        if tier == "unknown" or len(t0) != 5 or len(t1) != 5 or w not in (0, 1):
            dropped += 1
            continue
        bans = tuple(lst(b0)) + tuple(lst(b1))
        slim.append((bidx[v], tier, gmap, t0, t1, bans, w))
        games.append((int(rid), tier, gmap, list(t0), list(t1), list(bans), int(w)))
        m = meta[v]
        m["games"] += 1
        m["in_snapshot"] += int(rid <= common.SNAPSHOT_BOUND)
        d = gd.date().isoformat()
        m["min_date"] = min(m["min_date"] or d, d)
        m["max_date"] = max(m["max_date"] or d, d)
    per_build = {}
    for i in range(0, len(slim), 20000):
        for (bi, tier), cell in count_chunk(slim[i:i + 20000]).items():
            dst = per_build.setdefault(bi, {}).get(tier)
            if dst is None:
                dst = per_build[bi][tier] = _new_cell()
            merge_cell(dst, cell)
    for by_tier in per_build.values():
        for tier, cell in by_tier.items():
            plain = {"games": cell["games"], "bans": dict(cell["bans"])}
            for k in ("hero", "hmap", "with", "against", "comp"):
                plain[k] = {kk: list(vv) for kk, vv in cell[k].items()}
            by_tier[tier] = plain
    path = os.path.join(common.STATS_DIR, "clean_truth_counts.pkl.gz")
    with gzip.open(path, "wb") as f:
        pickle.dump({"builds": BUILDS, "per_build": per_build, "meta": meta,
                     "tier_rule": "site scheme (v2env.site_tier)", "dropped": dropped},
                    f, protocol=pickle.HIGHEST_PROTOCOL)
    json.dump(games, open(os.path.join(common.CACHE_DIR, "r10_clean_games.json"), "w"))
    print(f"{len(games):,} games kept, {dropped:,} dropped; wrote {path}")
    for b in BUILDS:
        print(b, meta[b])


if __name__ == "__main__":
    main()
