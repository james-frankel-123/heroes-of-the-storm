"""Golden vectors for browser/trainer feature parity (partial + full states).
Regenerate when either feature implementation or the stats artifact changes:
    python3 scripts/gen-feature-goldens.py && npx tsx scripts/check-feature-parity.ts
"""
import os, sys, json, random
TRAINING = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training")
sys.path.insert(0, TRAINING)
os.environ.setdefault("WP_STATS_PATH", os.path.join(TRAINING, "production_refresh/2026-07-14/stats_decayed90.json"))
import numpy as np
from sweep_enriched_wp import StatsCache, extract_features, FEATURE_GROUPS
from train_partial_wp import WP_GROUPS
from shared import HEROES, MAPS, SKILL_TIERS

GROUPS_MASK = [g in WP_GROUPS for g in FEATURE_GROUPS]
rng = random.Random(20260810)
stats = StatsCache()
cases = []
for _ in range(200):
    n0 = rng.randint(0, 5); n1 = rng.randint(max(0, n0 - 1), min(5, n0 + 1))
    picks = rng.sample(HEROES, n0 + n1)
    t0, t1 = picks[:n0], picks[n0:]
    gmap = rng.choice(MAPS); tier = rng.choice(SKILL_TIERS)
    _, enr = extract_features({"team0_heroes": t0, "team1_heroes": t1,
                               "game_map": gmap, "skill_tier": tier}, stats, GROUPS_MASK)
    cases.append({"t0": t0, "t1": t1, "map": gmap, "tier": tier,
                  "enriched": [round(float(x), 4) for x in np.asarray(enr)]})
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feature-parity-goldens.json")
json.dump(cases, open(out, "w"))
print(f"wrote {len(cases)} goldens -> {out}")
