"""
Sanity suite (28 hand-built tests; the first 21 are the original suite) and
degenerate-composition probes (5 tanks, 5 healers, ... vs a standard team,
Cursed Hollow, mid) for the revision's models (selected seeds, own deploy
statistics) and, for reference, the submitted models (external statistics).
Uses experiment_synthetic_augmentation.evaluate_config unchanged.
Output: results/sanity.json
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)
from paper1_revision import core, train_wp

import torch


def main():
    from experiment_synthetic_augmentation import evaluate_config, ENRICHED_GROUPS
    from sweep_enriched_wp import WinProbEnrichedModel
    dev = torch.device("cpu")
    out = {}
    for n in ("naive", "herostrength", "enriched", "enriched_512", "aug_wr0_512", "aug_wr5_512",
              "aug_wr10_512", "aug_wr50_512", "enriched_leak"):
        if not os.path.exists(os.path.join(core.MODEL_DIR, f"{n}.json")):
            continue
        m, cols = train_wp.load(n)
        r = evaluate_config(m, list(cols[197:] - 197), core.load_stats("deploy"), dev)
        out[f"rev_{n}"] = {"passed": r["sanity_passed"], "total": r["sanity_total"],
                           "degen_scores": r["degen_scores"]}
        print(n, r["sanity_passed"], {k: round(v, 3) for k, v in r["degen_scores"].items()},
              flush=True)
    S = train_wp.specs()
    for n, fn, arch, g in (("naive", "wp_naive.pt", [256, 128], []),
                           ("herostrength", "wp_herostrength.pt", [256, 128], ["hero_wr", "team_avg_wr"]),
                           ("enriched_256", "wp_enriched_256.pt", [256, 128], ENRICHED_GROUPS),
                           ("enriched_512", "wp_enriched_512.pt", [512, 256, 128], ENRICHED_GROUPS),
                           ("aug_v2_512", "wp_aug_v2_512.pt", [512, 256, 128], ENRICHED_GROUPS)):
        cols = core.group_cols(g)
        m = WinProbEnrichedModel(len(cols), arch, dropout=0.3)
        m.load_state_dict(torch.load(os.path.join(TRAINING_DIR, "rerun2026", "models", fn),
                                     map_location="cpu", weights_only=True))
        m.eval()
        r = evaluate_config(m, list(cols[197:] - 197), core.load_stats("hp"), dev)
        out[f"sub_{n}"] = {"passed": r["sanity_passed"], "total": r["sanity_total"],
                           "degen_scores": r["degen_scores"]}
        print("sub", n, r["sanity_passed"], {k: round(v, 3) for k, v in r["degen_scores"].items()},
              flush=True)
    json.dump(out, open(os.path.join(core.RESULTS, "sanity.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
