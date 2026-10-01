"""
Committed artifact for the leak figures in the paper (audit P1-C15):

  overlap   share of the games behind each Heroes Profile hero statistic
            (frozen_stats_2026-05-19.json, games per hero and tier label) that
            are games of our snapshot with the same hero and label; reported
            as the median over heroes, per tier label.
  leave-test-out
            the submitted enriched WP (wp_enriched_256.pt) on the paper's
            38,981 test replays (both team orders) with (a) the external
            statistics as published, (b) every test game's own contribution
            subtracted from every hero, hero-map, pairwise and composition
            cell, (c) the same number of random training games subtracted
            (control; 3 draws).
Subtraction is approximate: Heroes Profile publishes rounded win rates and
its own tier groups, and our labels group ranks differently (§III-A), so a
cell's own-game count is removed under our label.

Usage (from training/): nice -n 19 taskset -c 48-53 python3 paper1_revision/leak_measure.py
Output: results/leak_results.json
"""
import os
import sys
import json
import copy
import random

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)

import numpy as np

from paper1_revision import core
from overfit2026 import feats, split

HP = os.path.join(TRAINING_DIR, "frozen_stats_2026-05-19.json")


def subtract(raw, comps, cnt):
    """raw: frozen-schema dict; comps: {tier: {key: (wr, g)}}; cnt: counts_for(games)."""
    raw = copy.deepcopy(raw)

    def adj(wr, g, n, w):
        g2 = g - n
        if g2 <= 0:
            return wr, g
        return round(100.0 * (wr / 100.0 * g - w) / g2, 3), g2
    for r in raw["hero_stats"]:
        c = cnt.get(r["tier"], {}).get("hero", {}).get(r["hero"])
        if c:
            r["win_rate"], r["games"] = adj(r["win_rate"], r["games"], c[0], c[1])
    for r in raw["hero_map_stats"]:
        c = cnt.get(r["tier"], {}).get("hmap", {}).get((r["map"], r["hero"]))
        if c:
            r["win_rate"], r["games"] = adj(r["win_rate"], r["games"], c[0], c[1])
    for r in raw["pairwise_stats"]:
        a, b = r["hero_a"], r["hero_b"]
        cell = cnt.get(r["tier"], {})
        if r["relationship"] == "with":
            c = cell.get("with", {}).get(tuple(sorted((a, b))))
            n, w = (c[0], c[1]) if c else (0, 0)
        else:
            c = cell.get("against", {}).get((a, b))
            if c:
                n, w = c[0], c[1]
            else:
                c = cell.get("against", {}).get((b, a))
                n, w = (c[0], c[0] - c[1]) if c else (0, 0)
        if n:
            r["win_rate"], r["games"] = adj(r["win_rate"], r["games"], n, w)
    comps2 = {}
    for tier, d in comps.items():
        comps2[tier] = {}
        cc = cnt.get(tier, {}).get("comp", {})
        for k, (wr, g) in d.items():
            c = cc.get(k)
            comps2[tier][k] = adj(wr, g, c[0], c[1]) if c else (wr, g)
    return raw, comps2


def accuracy(model, cols, rows, y, st):
    from paper1_revision.bench_mcts import sym_predict
    Xf, Xs = core.featurize(rows, st)
    import torch
    with torch.no_grad():
        a = model(torch.tensor(Xf[:, cols])).view(-1).numpy()
        b = model(torch.tensor(Xs[:, cols])).view(-1).numpy()
    # both team orders, the original Table I convention
    p = np.r_[a, b]
    yy = np.r_[y, 1 - y]
    return float(np.mean((p > 0.5) == (yy > 0.5)))


def main():
    from paper1_revision.bench_mcts import sub_enriched
    from experiment_synthetic_augmentation import ENRICHED_GROUPS
    raw = json.load(open(HP))
    comps = feats.pinned_hp_comps()
    sp = core.split_games()
    test, train = sp["test"], sp["train"]
    out = {}
    # overlap
    snap = split.counts_for(train + sp["val"] + test)
    shares = {}
    for r in raw["hero_stats"]:
        c = snap.get(r["tier"], {}).get("hero", {}).get(r["hero"])
        if c and r["games"]:
            shares.setdefault(r["tier"], []).append(c[0] / r["games"])
    out["overlap_median_by_tier"] = {t: float(np.median(v)) for t, v in shares.items()}
    out["overlap_range_over_heroes"] = {t: [float(np.percentile(v, 10)), float(np.percentile(v, 90))]
                                        for t, v in shares.items()}
    print("overlap", out["overlap_median_by_tier"], flush=True)
    m = sub_enriched()
    cols = core.group_cols(ENRICHED_GROUPS)
    rows = [(g[3], g[4], g[2], g[1]) for g in test]
    y = np.array([1.0 if g[6] == 0 else 0.0 for g in test])

    def st_of(r, c):
        st = feats.stats_from_json(r, compositions=False)
        st.comp_data = c
        return st
    out["acc_published"] = accuracy(m, cols, rows, y, st_of(raw, comps))
    r2, c2 = subtract(raw, comps, split.counts_for(test))
    out["acc_test_removed"] = accuracy(m, cols, rows, y, st_of(r2, c2))
    ctrl = []
    for s in range(3):
        rnd = random.Random(s).sample(train, len(test))
        r3, c3 = subtract(raw, comps, split.counts_for(rnd))
        ctrl.append(accuracy(m, cols, rows, y, st_of(r3, c3)))
    out["acc_random_train_removed"] = ctrl
    print({k: v for k, v in out.items() if k.startswith("acc")}, flush=True)
    json.dump(out, open(os.path.join(core.RESULTS, "leak_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
