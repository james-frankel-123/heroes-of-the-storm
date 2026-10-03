"""
Held-out accuracy of the reproduced Gourdeau win-rate estimator
(rerun2026/models/gourdeau_wp.pt; hero multi-hot + map, no statistics) on the
paper test set and on the post-snapshot no-drift games, symmetrized.
Output: results/gourdeau_eval.json
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
import torch


def main():
    from train_gourdeau_baseline import GourdeauWPModel
    from shared import HERO_TO_IDX, NUM_HEROES, map_to_one_hot
    m = GourdeauWPModel()
    from rerun2026 import common as rcommon     # honours RERUN_NS (p1site rebuild)
    m.load_state_dict(torch.load(os.path.join(rcommon.MODELS_DIR, "gourdeau_wp.pt"),
                                 map_location="cpu", weights_only=True))
    m.eval()
    out = {}
    sets = {"test": core.split_games()["test"], "NODRIFT": core.gold_games("NODRIFT")}
    for name, games in sets.items():
        def hot(teams):
            X = np.zeros((len(teams), NUM_HEROES), np.float32)
            for i, t in enumerate(teams):
                for h in t:
                    X[i, HERO_TO_IDX[h]] = 1.0
            return torch.tensor(X)
        A, B = hot([g[3] for g in games]), hot([g[4] for g in games])
        M = torch.tensor(np.array([map_to_one_hot(g[2]) for g in games], np.float32))
        with torch.no_grad():
            pf = m.predict_wp(A, B, M).view(-1).numpy()
            ps = m.predict_wp(B, A, M).view(-1).numpy()
        y = np.array([1.0 if g[6] == 0 else 0.0 for g in games])
        p = np.clip(0.5 * (pf + 1 - ps), 1e-6, 1 - 1e-6)
        out[name] = {"n": len(y), "acc_sym": float(np.mean((p > .5) == (y > .5))),
                     "acc": float(0.5 * (np.mean((pf > .5) == (y > .5)) + np.mean((ps > .5) == (y < .5)))),
                     "ll": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
                     "slope": fit_slope(p, y)}
        print(name, out[name], flush=True)
    json.dump(out, open(os.path.join(core.RESULTS, "gourdeau_eval.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
