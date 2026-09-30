"""
Held-out accuracy and calibration slope of every independent reference, on
games none of them trained on: the paper test set (38,981 snapshot replays;
the references were fit on post-snapshot games or on Quick Match), plus the
drifted 2.55.17 games for the post-snapshot judges.
Output: results/judge_calibration.json
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)
from paper1_revision import core
from paper1_revision.train_wp import fit_slope

import numpy as np


def metrics(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return {"n": int(len(y)), "acc": float(np.mean((p > .5) == (y > .5))),
            "ll": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
            "slope": fit_slope(p, y)}


def main():
    from paper1_revision.score_tournament import score_rows
    out = {}
    sets = {"test": core.split_games()["test"], "T17": core.gold_games("T17")}
    for name, games in sets.items():
        rows = [(list(g[3]), list(g[4]), g[2], g[1]) for g in games]
        y = np.array([1.0 if g[6] == 0 else 0.0 for g in games])
        sc = score_rows(rows)
        out[name] = {k: metrics(v, y) for k, v in sc.items()}
        for k, v in out[name].items():
            print(name, k, {a: round(b, 4) for a, b in v.items()}, flush=True)
    json.dump(out, open(os.path.join(core.RESULTS, "judge_calibration.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
