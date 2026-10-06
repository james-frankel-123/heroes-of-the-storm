"""Golden vectors for browser/trainer feature parity (partial + full states).
Regenerate when either feature implementation or the stats artifact changes:
    GOLDEN_RUN_DIR=training/production_refresh/<run> python3 scripts/gen-feature-goldens.py
    npx tsx scripts/check-feature-parity.ts
GOLDEN_RUN_DIR is the production refresh run whose stats and compositions
snapshot the site serves (src/lib/data/draft-stats-decayed.json and
model-compositions.json come from its export). Encoding: HOTS_HERO_SET=v2.
"""
import os, sys, json, random
TRAINING = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training")
sys.path.insert(0, TRAINING)
os.environ["HOTS_HERO_SET"] = "v2"
RUN = os.environ["GOLDEN_RUN_DIR"]
os.environ["WP_STATS_PATH"] = os.path.join(RUN, "stats_decayed90.json")
os.environ["WP_COMPOSITIONS_PATH"] = os.path.join(RUN, "compositions.json")
import numpy as np
from sweep_enriched_wp import StatsCache, extract_features, FEATURE_GROUPS
from train_partial_wp import WP_GROUPS
from shared import HEROES, MAPS, SKILL_TIERS

GROUPS_MASK = [g in WP_GROUPS for g in FEATURE_GROUPS]
NEW_HERO, NEW_MAP = HEROES[-1], MAPS[-1]   # appended in v2
rng = random.Random(20260810)
stats = StatsCache()
cases = []
for i in range(300):
    n0 = rng.randint(0, 5); n1 = rng.randint(max(0, n0 - 1), min(5, n0 + 1))
    picks = rng.sample(HEROES, n0 + n1)
    if i % 3 == 0 and n0 + n1 and NEW_HERO not in picks:   # a third of cases carry her
        picks[rng.randrange(n0 + n1)] = NEW_HERO
    t0, t1 = picks[:n0], picks[n0:]
    gmap = NEW_MAP if i % 4 == 0 else rng.choice(MAPS)
    tier = rng.choice(SKILL_TIERS)
    base, enr = extract_features({"team0_heroes": t0, "team1_heroes": t1,
                                  "game_map": gmap, "skill_tier": tier}, stats, GROUPS_MASK)
    cases.append({"t0": t0, "t1": t1, "map": gmap, "tier": tier,
                  "base_ones": [int(j) for j in np.flatnonzero(np.asarray(base))],
                  "enriched": [round(float(x), 4) for x in np.asarray(enr)]})
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feature-parity-goldens.json")
json.dump({"heroes": HEROES, "maps": MAPS, "tiers": SKILL_TIERS, "base_dim": len(base),
           "cases": cases}, open(out, "w"))
print(f"wrote {len(cases)} goldens -> {out}")
