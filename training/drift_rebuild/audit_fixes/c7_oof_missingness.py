"""Consolidated audit C7: train/deploy missingness mismatch of the out-of-fold
statistics. Training rows see cutoff statistics built from 4/5 of the
training-period games (their own fold removed); deployment rows see all of
them. Cells near the storage/lookup thresholds (pair >= 10 stored and >= 30 for
get_synergy/get_counter, map >= 50 for get_hero_map_wr, comp >= 5, hero >= 20)
are therefore missing slightly more often for training rows.

Measured on a 60,000-game sample of fold-0 training rows: the fraction of
synergy pairs, counter pairs, hero-map cells and composition cells available
under (a) fold-0-out statistics (what the row was trained with) and (b) full
cutoff statistics (what a deployment row of the same draft would see).
Output: drift_rebuild/results/c7_oof_missingness.json"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rb_common as rb  # noqa: E402
from drift2026 import common  # noqa: E402
from drift2026.build_patch_stats import count_chunk, merge_cell, _new_cell, comp_key  # noqa: E402

common.setup()
rows, builds = common.load_data_with_patches()
cut = builds.index(common.TRAIN_CUTOFF_BUILD)
train = [r for r in rows if r["build_idx"] <= cut]
del rows
full, oof = {}, {}
for i in range(0, len(train), 50000):
    chunk = [(rb.fold_of(r["replay_id"]), r["skill_tier"], r["game_map"], tuple(r["team0_heroes"]),
              tuple(r["team1_heroes"]), tuple(r["team0_bans"]) + tuple(r["team1_bans"]), r["winner"])
             for r in train[i:i + 50000]]
    for (f, tier), cell in count_chunk(chunk).items():
        for dst_map, ok in ((full, True), (oof, f != 0)):
            if not ok:
                continue
            dst = dst_map.get(tier)
            if dst is None:
                dst = dst_map[tier] = _new_cell()
            merge_cell(dst, cell)
S = {"full": rb.stats_from_counts_rounded(full), "fold0_out": rb.stats_from_counts_rounded(oof)}
f0 = [r for r in train if rb.fold_of(r["replay_id"]) == 0]
random.Random(0).shuffle(f0)
sample = f0[:60000]
out = {"n_games": len(sample)}
for name, st in S.items():
    n = {"syn": [0, 0], "ctr": [0, 0], "map": [0, 0], "comp": [0, 0]}
    for r in sample:
        t, m = r["skill_tier"], r["game_map"]
        for team, opp in ((r["team0_heroes"], r["team1_heroes"]), (r["team1_heroes"], r["team0_heroes"])):
            for i, a in enumerate(team):
                for b in team[i + 1:]:
                    n["syn"][0] += 1
                    n["syn"][1] += st.get_synergy(a, b, t) is not None
                for b in opp:
                    n["ctr"][0] += 1
                    n["ctr"][1] += st.get_counter(a, b, t) is not None
                d = st.hero_map_wr.get(t, {}).get(m, {}).get(a)
                n["map"][0] += 1
                n["map"][1] += bool(d and d[1] >= 50)
            n["comp"][0] += 1
            n["comp"][1] += comp_key(team) in st.comp_data.get(t, {})
    out[name] = {k: round(v[1] / v[0], 5) for k, v in n.items()}
    print(name, out[name], flush=True)
json.dump(out, open(os.path.join(rb.RESULTS_DIR, "c7_oof_missingness.json"), "w"), indent=1)
