"""
Held-out quality of the SUBMITTED win-probability models (rerun2026/models),
with the external statistics they were trained on:
  test_hp      the paper test set (38,981) - what Table I reported; its
               features contain the test games' own outcomes
  NODRIFT_hp   291,837 post-snapshot no-drift games (uploaded after the
               snapshot, so absent from the 2026-05-19 aggregates)
Same metric definitions as train_wp.evaluate. Output:
results/submitted_models.json
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from paper1_revision import core
from paper1_revision.train_wp import evaluate

import torch

MODELS = {  # name -> (file, arch, groups preset)
    "naive": ("wp_naive.pt", [256, 128], []),
    "true_base": ("wp_true_base.pt", [256, 128], []),
    "herostrength": ("wp_herostrength.pt", [256, 128], ["hero_wr", "team_avg_wr"]),
    "enriched_256": ("wp_enriched_256.pt", [256, 128], "E"),
    "enriched_512": ("wp_enriched_512.pt", [512, 256, 128], "E"),
    "aug_v2_512": ("wp_aug_v2_512.pt", [512, 256, 128], "E"),
    "relational_256": ("wp_relational_256.pt", [256, 128], "R"),
    "absolute_256": ("wp_absolute_256.pt", [256, 128], "A"),
}


def main():
    from sweep_enriched_wp import WinProbEnrichedModel
    from paper1_revision.train_wp import specs
    S = specs()
    gmap = {"E": S["enriched"]["groups"], "R": S["relational"]["groups"],
            "A": S["absolute"]["groups"]}
    out = {}
    dev = torch.device("cpu")
    torch.set_num_threads(16)
    for name, (fn, arch, groups) in MODELS.items():
        p = os.path.join(core.TRAINING_DIR, "rerun2026", "models", fn)
        if not os.path.exists(p):
            continue
        groups = gmap.get(groups, groups) if isinstance(groups, str) else groups
        cols = core.group_cols(groups)
        m = WinProbEnrichedModel(len(cols), arch, dropout=0.3)
        m.load_state_dict(torch.load(p, map_location="cpu", weights_only=True))
        m.eval()
        out[name] = {fs: evaluate(m, cols, fs, dev) for fs in ("test_hp", "NODRIFT_hp")}
        print(name, {fs: {k: round(v, 4) for k, v in r.items()} for fs, r in out[name].items()},
              flush=True)
    json.dump(out, open(os.path.join(core.RESULTS, "submitted_models.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
