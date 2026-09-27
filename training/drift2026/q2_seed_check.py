"""Seed-robustness check for the Q2 vintage matrix: retrain the 2024-07,
2025-07 and 2026-build judges with 2 extra seeds each and rescore the W6
records. Prints maintained WP per (vintage, seed)."""
import os
import sys
import json

TRAINING_DIR = "/home/max/heroes-of-the-storm/training"
sys.path.insert(0, TRAINING_DIR)
sys.path.insert(0, os.path.join(TRAINING_DIR, "drift2026"))

import numpy as np
import torch

from drift2026 import common
import drift2026.q2_vintage_judges as q2

EXTRA_SEEDS = [7, 991]
CHECK = ["2024-07", "2025-07", "2026-build"]


def main():
    common.setup()
    device = torch.device("cuda")
    with open(q2.W6_PATH) as f:
        records = json.load(f)["records"]
    rows, builds = common.load_data_with_patches()
    cut_idx = common.cutoff_idx(builds)

    vint = {name: (kind, val) for name, kind, val in q2.VINTAGES}
    pools = {}
    for name in CHECK:
        kind, val = vint[name]
        pools[name] = ([r for r in rows if r["date_days"] <= val] if kind == "date"
                       else [r for r in rows if r["build_idx"] <= cut_idx])
    cap = 314761

    out = {}
    for name in CHECK:
        for seed in EXTRA_SEEDS:
            q2.SEED = seed
            model, meta = q2.train_judge(f"{name}_s{seed}", pools[name], device, cap)
            scores, _ = q2.score_judge(model, device, records)
            out[f"{name}_s{seed}"] = {"holdout_acc": meta["holdout_acc"], **scores}
            print(f">>> {name} seed {seed}: acc {meta['holdout_acc']:.4f}  "
                  f"maintained WP {scores['maintained_wp_mean']:.4f}  "
                  f"win share {scores['win_share']:.3f}")
            del model
            torch.cuda.empty_cache()
    print(json.dumps(out, indent=2))
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "q2_seed_check.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
