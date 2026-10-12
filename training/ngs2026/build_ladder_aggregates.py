"""
One-off: compute ladder-side aggregates for the NGS meta-ness comparison,
from the paper's dataset of record (pinned 2026-05-22 snapshot, 2.55-filtered
via rerun2026.common.load_data()).

Writes ngs2026/ladder_aggregates.json:
  n_replays, pick_counts (per hero, both teams), ban_counts,
  role_comp_counts (sorted Blizzard-role 5-tuples per team),
  map_counts, tier_counts
"""
import os
import sys
import json
from collections import Counter

TRAINING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TRAINING_DIR)
from rerun2026 import common  # noqa: E402

_TAG = ("_" + os.environ["NGS_TAG"]) if os.environ.get("NGS_TAG") else ""   # see analyze_meta.py
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   f"ladder_aggregates{_TAG}.json")


def main():
    common.setup()
    from shared import HERO_ROLE_FINE, FINE_TO_BLIZZ_ROLE
    rows = common.load_data()

    pick_counts = Counter()
    ban_counts = Counter()
    role_comp_counts = Counter()
    map_counts = Counter()
    tier_counts = Counter()
    n = 0
    known_only = os.environ.get("NGS_KNOWN_TIERS") == "1"   # site corpus: ranked games only
    for r in rows:
        if known_only and r.get("skill_tier") not in ("low", "mid", "high"):
            continue
        t0, t1 = r["team0_heroes"], r["team1_heroes"]
        if len(t0) != 5 or len(t1) != 5:
            continue
        n += 1
        for h in t0:
            pick_counts[h] += 1
        for h in t1:
            pick_counts[h] += 1
        for h in (r.get("team0_bans") or []):
            ban_counts[h] += 1
        for h in (r.get("team1_bans") or []):
            ban_counts[h] += 1
        for team in (t0, t1):
            roles = sorted(FINE_TO_BLIZZ_ROLE.get(HERO_ROLE_FINE.get(h, ""),
                                                  "Ranged Assassin")
                           for h in team)
            role_comp_counts[",".join(roles)] += 1
        map_counts[r["game_map"]] += 1
        tier_counts[r.get("skill_tier") or "unknown"] += 1

    with open(OUT, "w") as f:
        json.dump({
            "n_replays": n,
            "pick_counts": dict(pick_counts),
            "ban_counts": dict(ban_counts),
            "role_comp_counts": dict(role_comp_counts),
            "map_counts": dict(map_counts),
            "tier_counts": dict(tier_counts),
        }, f)
    print(f"Wrote {OUT}: {n} replays, {len(pick_counts)} heroes picked, "
          f"{len(role_comp_counts)} distinct role comps")


if __name__ == "__main__":
    main()
