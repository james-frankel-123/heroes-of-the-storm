"""
NGS transfer check (supplement) with the revision's leak-free models: the
11,034 NGS drafts (ngs2026/ngs_drafts.json), tier "mid" as in ngs2026/eval_wp.py,
features from own deploy statistics. NGS games are in neither our corpus nor the
Storm League aggregates, so neither model family has self-inclusion here.
Submitted models are scored with the external statistics for comparison.
Metrics: game-level accuracy of the symmetrized prediction, log-loss, slope.
Output: results/ngs.json
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)
from paper1_revision import core
from paper1_revision.train_wp import fit_slope, load

import numpy as np
import torch


def main():
    d = json.load(open(os.path.join(TRAINING_DIR, "ngs2026", "ngs_drafts.json")))["drafts"]
    rows = [(g["team0_heroes"], g["team1_heroes"], g["game_map"], "mid") for g in d]
    y = np.array([1.0 if int(g["winner"]) == 0 else 0.0 for g in d])
    out = {"n": len(rows)}

    def metrics(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return {"acc": float(np.mean((p > .5) == (y > .5))),
                "ll": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
                "slope": fit_slope(p, y)}
    Xf, Xs = core.featurize(rows, core.load_stats("deploy"), nproc=8)
    for n in ("naive", "enriched", "aug_wr10_512"):
        m, cols = load(n)
        with torch.no_grad():
            a = m(torch.tensor(Xf[:, cols])).view(-1).numpy()
            b = m(torch.tensor(Xs[:, cols])).view(-1).numpy()
        out[f"rev_{n}"] = metrics(0.5 * (a + 1 - b))
    from sweep_enriched_wp import WinProbEnrichedModel
    from experiment_synthetic_augmentation import ENRICHED_GROUPS
    Xf, Xs = core.featurize(rows, core.load_stats("hp"), nproc=8)
    for n, fn, arch, g in (("naive", "wp_naive.pt", [256, 128], []),
                           ("enriched_256", "wp_enriched_256.pt", [256, 128], ENRICHED_GROUPS),
                           ("aug_v2_512", "wp_aug_v2_512.pt", [512, 256, 128], ENRICHED_GROUPS)):
        cols = core.group_cols(g)
        m = WinProbEnrichedModel(len(cols), arch, dropout=0.3)
        m.load_state_dict(torch.load(os.path.join(TRAINING_DIR, "rerun2026", "models", fn),
                                     map_location="cpu", weights_only=True))
        m.eval()
        with torch.no_grad():
            a = m(torch.tensor(Xf[:, cols])).view(-1).numpy()
            b = m(torch.tensor(Xs[:, cols])).view(-1).numpy()
        out[f"sub_{n}"] = metrics(0.5 * (a + 1 - b))
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(core.RESULTS, "ngs.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
