"""
Audit fix P3-26: guard against mixing skill_tier schemes on a rerun.

On 2026-09-30 (commit 4848ca5) the DB's historical skill_tier was relabeled
into the site's scheme. Every P3 result uses the old research scheme:
  stored league_tier = real tier + 1 (NULL = Master);
  research labels: low = Bronze; mid = Silver + Gold + Master;
  high = Platinum + Diamond.
The caches cache/x_post_games.json.gz and cache/x_side_games.npz were fetched
before the relabel and carry the old labels. A refetch now returns site
labels (Master high, Platinum mid, Silver low), which p3_pgd_data and
p3_x_robust would mix with the snapshot.

This script checks a games file against the old rule (every row with a
league_tier must carry the old research label) and, for an npz side file,
compares its labels with the cached copy on shared replay_ids. Exit code 1
on any mismatch. Run it after any refetch and before any rerun:

  /usr/bin/python3 personalization/p3_fix_tier_guard.py        # check the caches
  /usr/bin/python3 personalization/p3_fix_tier_guard.py NEW.json.gz   # check a refetch
"""
import os
import sys
import gzip
import json
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import p3_hs_core as C

# stored league_tier -> research label (old scheme); None = Master
OLD_RULE = {2: "low", 3: "mid", 4: "mid", 5: "high", 6: "high", None: "mid"}


def check_games(path):
    g = json.load(gzip.open(path, "rt"))
    c = Counter((x.get("league_tier"), x.get("skill_tier")) for x in g)
    bad = {k: v for k, v in c.items() if k[0] in OLD_RULE and OLD_RULE[k[0]] != k[1]}
    print(f"{path}: {len(g):,} games; (league_tier, skill_tier) counts {sorted(c.items(), key=str)}")
    if bad:
        print(f"  MISMATCH with the old research scheme: {bad}")
        return False
    print("  ok: labels follow the old research scheme")
    return True


def check_side(path, ref=os.path.join(C.CACHE, "x_side_games.npz")):
    a = np.load(path, allow_pickle=True)
    if os.path.abspath(path) == os.path.abspath(ref):
        print(f"{path}: reference copy, tier names {list(a['tier_names'])}")
        return True
    b = np.load(ref, allow_pickle=True)
    ia = {int(r): str(a["tier_names"][t]) for r, t in zip(a["replay_ids"], a["tier"])}
    n, bad = 0, 0
    for r, t in zip(b["replay_ids"], b["tier"]):
        r = int(r)
        if r in ia:
            n += 1
            bad += ia[r] != str(b["tier_names"][t])
    print(f"{path}: {n:,} shared replay_ids with the cached copy, {bad:,} with a different label")
    return bad == 0


def main():
    ok = True
    paths = sys.argv[1:] or [os.path.join(C.CACHE, "x_post_games.json.gz"), os.path.join(C.CACHE, "x_side_games.npz")]
    for p in paths:
        ok &= check_games(p) if p.endswith(".json.gz") else check_side(p)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
