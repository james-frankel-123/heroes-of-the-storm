"""
Retrain the 20-member specification-diverse leak-free WP ensemble of
deferred.py job_ensemble (same specs, same data halves, same seeds, same
train_wp.train_one loop) for the site rebuild, into a NEW directory
site/models/ensemble_lcb/ (the site run's member checkpoints are not on any
host we use; its results file results/deferred/ensemble_uncertainty.json is
not touched). Used as the pessimistic leaf of mcts_lcb (search_actor.py).

Usage (from training/, P1_TIERS=site, GPU): python3 paper1_revision/train_ensemble_site.py
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)
from paper1_revision import core

import numpy as np

MDIR = os.path.join(core.MODEL_DIR, "ensemble_lcb")


def main():
    import torch
    from overfit2026 import data as odata
    from paper1_revision import train_wp
    from paper1_revision.deferred import _ensemble_specs
    assert odata.TIER_SCHEME == "site"
    dev = torch.device("cuda")
    os.makedirs(MDIR, exist_ok=True)
    specs = _ensemble_specs()
    if all(os.path.exists(os.path.join(MDIR, sp["name"] + ".pt")) for sp in specs):
        print("all members present")
        return
    Xf, Xs, y, rid, tier = core.load_features("train_oof")
    half = np.array([odata.splitmix64(int(r) * 7 + 101) & 1 for r in rid])
    for sp in specs:
        path = os.path.join(MDIR, sp["name"] + ".pt")
        if os.path.exists(path):
            continue
        cols = core.group_cols(sp["groups"])
        m = half == sp["half"]
        tX = torch.tensor(np.concatenate([Xf[m][:, cols], Xs[m][:, cols]]), device=dev)
        tY = torch.tensor(np.concatenate([y[m], 1 - y[m]]), device=dev)
        vX, vY = train_wp._tensors(cols, "val", dev)
        model, vl, hist = train_wp.train_one(sp["name"], sp, sp["seed"], dev, (tX, tY, vX, vY))
        torch.save(model.state_dict(), path + ".tmp")
        os.replace(path + ".tmp", path)
        json.dump({"spec": sp, "val_loss": vl, "epochs": len(hist),
                   "test": train_wp.evaluate(model, cols, "test", dev)},
                  open(path[:-3] + ".json", "w"), indent=1, default=float)
        del tX, tY
        torch.cuda.empty_cache()
        print("trained", sp["name"], round(vl, 5), flush=True)


if __name__ == "__main__":
    main()
