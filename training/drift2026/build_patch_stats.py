"""
PHASE D1: per-patch aggregate statistics from OUR OWN corpus (the pinned
snapshot), NOT the Heroes Profile API.

For every 2.55 build (the patch unit, see common.py) this computes, per skill
tier (low/mid/high):
  - hero WR / pick rate / ban rate (+ games)
  - hero-per-map WR (+ games)
  - pairwise with/against WRs (+ games)
  - role-composition WRs (+ games), using the exact fine-role -> HP-role
    mapping of StatsCache.get_comp_wr so lookups are drop-in

in two flavors:
  patch_stats/per_patch/<build>.json.gz     that build's games only
  patch_stats/cumulative/<build>.json.gz    builds 0..N only — the stats a
                                            system deployed at the END of
                                            build N could have computed
                                            (no future leakage)

plus patch_stats/patch_counts.pkl.gz: the raw per-build count tables
(hero/map/pair/comp wins+games) consumed by signal_decay.py.

Format matches what drift2026.common.load_patch_stats -> PatchStats expects,
which duck-types sweep_enriched_wp.StatsCache, so the paper-1 enriched
feature extractor consumes per-patch stats as a drop-in.

STORAGE THRESHOLDS (consumers apply their own reliability thresholds on top:
get_counter/get_synergy >= 30 games, get_hero_map_wr >= 50 games):
  - hero_stats rows require >= 20 games (else omitted -> 50.0 default; a
    3-game 0% WR is worse than the prior)
  - hero_map_stats / comp_stats rows: >= 5 games
  - pairwise rows: >= 5 games (per-patch), >= 10 (cumulative)

Usage:
    python drift2026/build_patch_stats.py
"""
import os
import sys
import json
import gzip
import time
import pickle
import argparse
import multiprocessing as mp
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drift2026 import common

NUM_WORKERS = min(mp.cpu_count(), 56)

HP_ROLE_MAP = {  # must match StatsCache.get_comp_wr exactly
    "tank": "Tank", "bruiser": "Bruiser", "healer": "Healer",
    "ranged_aa": "Ranged Assassin", "ranged_mage": "Ranged Assassin",
    "melee_assassin": "Melee Assassin", "support_utility": "Support",
    "varian": "Bruiser", "pusher": "Ranged Assassin", "unknown": "Ranged Assassin",
}


def comp_key(heroes):
    from shared import HERO_ROLE_FINE
    roles = sorted(HERO_ROLE_FINE.get(h, "unknown") for h in heroes)
    return ",".join(sorted(HP_ROLE_MAP.get(r, "Ranged Assassin") for r in roles))


def _new_cell():
    return {"games": 0,
            "hero": defaultdict(lambda: [0, 0]),      # hero -> [games, wins]
            "bans": defaultdict(int),                  # hero -> ban games
            "hmap": defaultdict(lambda: [0, 0]),       # (map, hero) -> [g, w]
            "with": defaultdict(lambda: [0, 0]),       # (a<b) -> [g, w] (pair's team)
            "against": defaultdict(lambda: [0, 0]),    # (a<b) -> [g, wins_of_a]
            "comp": defaultdict(lambda: [0, 0])}       # roles_key -> [g, w]


def count_chunk(chunk):
    """chunk: list of (build_idx, tier, map, t0_heroes, t1_heroes, bans, winner).
    Returns {(build_idx, tier): plain-dict cell}."""
    out = {}
    for bidx, tier, gmap, t0, t1, bans, winner in chunk:
        key = (bidx, tier)
        cell = out.get(key)
        if cell is None:
            cell = out[key] = _new_cell()
        cell["games"] += 1
        for ti, team in ((0, t0), (1, t1)):
            won = 1 if winner == ti else 0
            for h in team:
                e = cell["hero"][h]
                e[0] += 1
                e[1] += won
                e2 = cell["hmap"][(gmap, h)]
                e2[0] += 1
                e2[1] += won
            for i in range(len(team)):
                for j in range(i + 1, len(team)):
                    a, b = team[i], team[j]
                    if a > b:
                        a, b = b, a
                    e = cell["with"][(a, b)]
                    e[0] += 1
                    e[1] += won
            e = cell["comp"][comp_key(team)]
            e[0] += 1
            e[1] += won
        for h in set(bans):
            cell["bans"][h] += 1
        t0won = 1 if winner == 0 else 0
        for ha in t0:
            for hb in t1:
                if ha < hb:
                    e = cell["against"][(ha, hb)]
                    e[0] += 1
                    e[1] += t0won
                else:
                    e = cell["against"][(hb, ha)]
                    e[0] += 1
                    e[1] += 1 - t0won
    # convert defaultdicts to plain dicts for pickling
    for cell in out.values():
        for k in ("hero", "hmap", "with", "against", "comp"):
            cell[k] = dict(cell[k])
        cell["bans"] = dict(cell["bans"])
    return out


def merge_cell(dst, src):
    dst["games"] += src["games"]
    for k in ("hero", "hmap", "with", "against", "comp"):
        d = dst[k]
        for key, (g, w) in src[k].items():
            e = d[key]
            e[0] += g
            e[1] += w
    for h, n in src["bans"].items():
        dst["bans"][h] += n


def derive_stats_file(build, counts_by_tier, kind, pair_min):
    """counts_by_tier: {tier: cell}. Writes <build>.json.gz in frozen-stats-
    like format (+ comp_stats)."""
    hero_stats, hero_map_stats, pairwise_stats, comp_stats = [], [], [], []
    for tier, cell in counts_by_tier.items():
        total = cell["games"]
        if total == 0:
            continue
        for h, (g, w) in cell["hero"].items():
            if g < 20:
                continue
            hero_stats.append({
                "hero": h, "tier": tier, "games": g,
                "win_rate": round(100.0 * w / g, 3),
                "pick_rate": round(100.0 * g / total, 3),
                "ban_rate": round(100.0 * cell["bans"].get(h, 0) / total, 3)})
        for (m, h), (g, w) in cell["hmap"].items():
            if g < 5:
                continue
            hero_map_stats.append({
                "hero": h, "map": m, "tier": tier, "games": g,
                "win_rate": round(100.0 * w / g, 3)})
        for (a, b), (g, w) in cell["with"].items():
            if g < pair_min:
                continue
            wr = round(100.0 * w / g, 3)
            pairwise_stats.append({"hero_a": a, "hero_b": b, "tier": tier,
                                   "relationship": "with", "win_rate": wr, "games": g})
            pairwise_stats.append({"hero_a": b, "hero_b": a, "tier": tier,
                                   "relationship": "with", "win_rate": wr, "games": g})
        for (a, b), (g, wa) in cell["against"].items():
            if g < pair_min:
                continue
            wr_a = round(100.0 * wa / g, 3)
            pairwise_stats.append({"hero_a": a, "hero_b": b, "tier": tier,
                                   "relationship": "against", "win_rate": wr_a,
                                   "games": g})
            pairwise_stats.append({"hero_a": b, "hero_b": a, "tier": tier,
                                   "relationship": "against",
                                   "win_rate": round(100.0 - wr_a, 3), "games": g})
        for ck, (g, w) in cell["comp"].items():
            if g < 5:
                continue
            comp_stats.append({"roles": ck, "tier": tier, "games": g,
                               "win_rate": round(100.0 * w / g, 3)})

    path = common.stats_path(kind, build)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt") as f:
        json.dump({
            "_meta": {"build": build, "kind": kind,
                      "games": {t: c["games"] for t, c in counts_by_tier.items()},
                      "source": "pinned snapshot 2026-05-22 (own corpus)"},
            "hero_stats": hero_stats, "hero_map_stats": hero_map_stats,
            "pairwise_stats": pairwise_stats, "comp_stats": comp_stats,
        }, f)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    common.setup()

    rows, builds = common.load_data_with_patches()
    present = sorted({r["build_idx"] for r in rows})
    print(f"{len(rows)} rows across {len(present)} builds")
    if args.dry_run:
        return

    slim = [(r["build_idx"], r["skill_tier"], r["game_map"],
             tuple(r["team0_heroes"]), tuple(r["team1_heroes"]),
             tuple(r["team0_bans"]) + tuple(r["team1_bans"]), r["winner"])
            for r in rows]
    del rows
    size = max(1, len(slim) // (NUM_WORKERS * 4))
    chunks = [slim[i:i + size] for i in range(0, len(slim), size)]
    print(f"Counting: {len(chunks)} chunks on {NUM_WORKERS} workers")
    t0 = time.time()
    merged = defaultdict(_new_cell)   # (bidx, tier) -> cell
    with mp.Pool(NUM_WORKERS) as pool:
        for i, res in enumerate(pool.imap_unordered(count_chunk, chunks)):
            for key, cell in res.items():
                merge_cell(merged[key], cell)
            if (i + 1) % 20 == 0:
                print(f"  merged {i+1}/{len(chunks)} ({time.time()-t0:.0f}s)",
                      flush=True)
    print(f"Counting done in {time.time()-t0:.0f}s; {len(merged)} (build,tier) cells")

    # Raw per-build counts for signal_decay.py
    per_build = {}   # bidx -> {tier: cell(plain dicts)}
    for (bidx, tier), cell in merged.items():
        plain = {"games": cell["games"], "bans": dict(cell["bans"])}
        for k in ("hero", "hmap", "with", "against", "comp"):
            plain[k] = {kk: list(vv) for kk, vv in cell[k].items()}
        per_build.setdefault(bidx, {})[tier] = plain
    os.makedirs(common.STATS_DIR, exist_ok=True)
    with gzip.open(common.COUNTS_PKL, "wb") as f:
        pickle.dump({"builds": builds, "per_build": per_build}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote {common.COUNTS_PKL} "
          f"({os.path.getsize(common.COUNTS_PKL)/1e6:.1f} MB)")

    # Per-patch files + running cumulative
    cum = {}   # tier -> cell
    t0 = time.time()
    for bidx in present:
        build = builds[bidx]
        by_tier = {t: c for (bi, t), c in merged.items() if bi == bidx}
        derive_stats_file(build, by_tier, "per_patch", pair_min=5)
        for tier, cell in by_tier.items():
            if tier not in cum:
                cum[tier] = _new_cell()
            merge_cell(cum[tier], cell)
        derive_stats_file(build, cum, "cumulative", pair_min=10)
        print(f"  {build}: per_patch + cumulative written "
              f"({sum(c['games'] for c in by_tier.values()):,} games; "
              f"cum {sum(c['games'] for c in cum.values()):,})", flush=True)
    print(f"Stats files done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
