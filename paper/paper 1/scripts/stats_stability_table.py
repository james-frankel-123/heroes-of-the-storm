"""Statistics-set stability of the interaction metrics (supplement table).

The same benchmark drafts of every site-tier MCTS configuration (bench_mcts,
1,000 drafts per seed, root temperature 1) are scored for synergy and counter
responsiveness under two statistics sets: our own deploy statistics (training
split) and the external Heroes Profile aggregates (frozen 2026-05-19). Same
policies, same drafts; only the statistics behind the metric lookups differ.
Reports per-config means over seeds under both sets and the Spearman rank
correlation of the configuration ordering.

Usage: python3 "paper/paper 1/scripts/stats_stability_table.py"
"""
import glob
import json
import os

import numpy as np

BENCH = "/home/max/heroes-of-the-storm/training/paper1_revision/site/results/mcts_bench"
OUT = "/home/max/heroes-of-the-storm/training/paper1_revision/site/results/stats_stability.json"
ORDER = ["B_oof", "F_oof", "J_oof", "E_oof", "K_oof", "B_leak", "F_leak", "J_leak"]


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main(temp=1.0):
    per = {}
    for p in glob.glob(os.path.join(BENCH, "*.json")):
        d = json.load(open(p))
        if d["temp"] != temp:
            continue
        cfg = d["run"].split(":", 1)[1].rsplit("_s", 1)[0]
        per.setdefault(cfg, []).append(d["means"])
    res = {"temp": temp, "configs": {}, "spearman": {}}
    for metric in ("syn", "ctr"):
        own = [np.mean([m[f"{metric}_own"] for m in per[c]]) for c in ORDER]
        hp = [np.mean([m[f"{metric}_hp"] for m in per[c]]) for c in ORDER]
        res["spearman"][metric] = spearman(own, hp)
        for c, a, b in zip(ORDER, own, hp):
            res["configs"].setdefault(c, {"n_seeds": len(per[c])})[metric] = {"own": float(a), "hp": float(b)}
    json.dump(res, open(OUT, "w"), indent=1)
    for c in ORDER:
        v = res["configs"][c]
        print(f"{c:8s} n={v['n_seeds']} syn own {v['syn']['own']:+.2f} hp {v['syn']['hp']:+.2f} | "
              f"ctr own {v['ctr']['own']:+.3f} hp {v['ctr']['hp']:+.3f}")
    print("spearman", res["spearman"])


if __name__ == "__main__":
    main()
