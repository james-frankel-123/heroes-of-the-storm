"""Dual-snapshot stability table for the supplement.

Compares the interaction metrics (synergy exploitation, counter
responsiveness, early resilience) of every MCTS configuration re-benchmarked
under two heroesprofile.com statistics snapshots:

  - training/mcts_experiment_results_v2_prefrozen.json : evaluated with the
    aggregate statistics in use before the May 2026 re-pull (earlier pull)
  - training/mcts_experiment_results_v2.json           : identical policies
    and evaluation protocol re-scored under the frozen 2026-05-19 snapshot

Same trained policy networks, same 600-draft/seed benchmark; the only change
is the statistics snapshot feeding the enriched features and the metric
lookups. Reports per-config means (15 seeds) under both snapshots and the
Spearman rank correlation of the config ordering.
"""
import json

import numpy as np

BASE = "/home/max/heroes-of-the-storm/training/"


def config_means(path, field):
    d = json.load(open(path))
    cfgs = {}
    for k, v in d.items():
        cfgs.setdefault(v["experiment"], []).append(v[field])
    return {c: float(np.mean(vs)) for c, vs in cfgs.items()}


def spearman(x, y):
    def rank(a):
        order = np.argsort(a)
        r = np.empty(len(a))
        r[order] = np.arange(len(a))
        return r
    rx, ry = rank(np.array(x)), rank(np.array(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    pre = BASE + "mcts_experiment_results_v2_prefrozen.json"
    fro = BASE + "mcts_experiment_results_v2.json"
    for field, label in [("synergy", "Synergy"), ("counter", "Counter"),
                         ("resil_early", "Resil. early")]:
        a = config_means(pre, field)
        b = config_means(fro, field)
        common = sorted(set(a) & set(b))
        av = [a[c] for c in common]
        bv = [b[c] for c in common]
        rho = spearman(av, bv)
        print(f"\n== {label} (earlier pull vs frozen 2026-05-19) "
              f"Spearman rho = {rho:.3f}, n configs = {len(common)} ==")
        for c in sorted(common, key=lambda c: -b[c]):
            print(f"  {c:<14} pre={a[c]:+.3f}  frozen={b[c]:+.3f}  "
                  f"shift={b[c]-a[c]:+.3f}")


if __name__ == "__main__":
    main()
