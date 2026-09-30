"""Compute across-draft standard errors on consensus WP for tournament matchups.

Consensus WP per draft = mean over the four evaluators (naive, herostrength,
enriched, augmented) of the symmetrized team-0 WP. For each ordered pair we
report mean +- SE across the 200 per-draft records; for each strategy we also
report its overall consensus WP with an SE across all of its per-draft
observations (each ordered pair contributes 200 drafts).

Data: training/rerun2026/results/roundrobin/*.json (72 ordered pairs) +
training/rerun2026/results/constrained/roundrobin/*.json (38 ordered pairs
involving the two constrained strategies).
"""
import glob
import json
import os

import numpy as np

BASE = "/home/max/heroes-of-the-storm/training/rerun2026/results/"
EVS = ["naive", "herostrength", "enriched", "augmented"]


def per_draft_consensus(path):
    d = json.load(open(path))
    vals = np.array([[r["wp"][ev]["sym"] for ev in EVS] for r in d["records"]])
    return d["team0_strategy"], d["team1_strategy"], vals.mean(axis=1)


def main():
    files = sorted(glob.glob(BASE + "roundrobin/*.json")
                   + glob.glob(BASE + "constrained/roundrobin/*.json"))
    strat_samples = {}
    pair_stats = {}
    for f in files:
        a, b, cons = per_draft_consensus(f)
        key = f"{a}__{b}"
        pair_stats[key] = (cons.mean(), cons.std(ddof=1) / np.sqrt(len(cons)),
                           len(cons))
        strat_samples.setdefault(a, []).append(cons)
        strat_samples.setdefault(b, []).append(1.0 - cons)

    print(f"{len(files)} ordered pairs loaded")
    print("\n-- Key matchups (mean consensus WP +- SE, n drafts) --")
    for k in ["constrained_mcts__mcts", "mcts__constrained_mcts",
              "constrained_greedy__enriched", "enriched__constrained_greedy"]:
        m, se, n = pair_stats[k]
        print(f"  {k:<38} {m:.3f} +- {se:.3f}  (n={n})")

    print("\n-- Per-strategy overall consensus WP +- SE (pooled drafts) --")
    rows = []
    for s, chunks in strat_samples.items():
        allv = np.concatenate(chunks)
        m = allv.mean()
        se_pool = allv.std(ddof=1) / np.sqrt(len(allv))
        # SE on the mean of per-pair means (20 matchup means)
        pm = np.array([c.mean() for c in chunks])
        se_pair = pm.std(ddof=1) / np.sqrt(len(pm))
        rows.append((m, s, se_pool, se_pair, len(allv), len(pm)))
    for m, s, se_pool, se_pair, n, npair in sorted(rows, reverse=True):
        print(f"  {s:<22} {m:.3f}  SE(drafts)={se_pool:.4f} (n={n})"
              f"  SE(pair means)={se_pair:.4f} (pairs={npair})")

    print("\n-- SE range across all 110 ordered pairs --")
    ses = [v[1] for v in pair_stats.values()]
    print(f"  min={min(ses):.4f} max={max(ses):.4f} median={np.median(ses):.4f}")


if __name__ == "__main__":
    main()
