"""
Reproducible check of the MCQ "dead neurons" sentence (audit B21, no script
existed): for the MCQ tau=0.5 checkpoint and, as controls, CQL alpha=1.0 and
MCQ tau=0.6, count ReLU units in each hidden layer that output zero for every
state in a sample of held-out draft states (rerun2026 CQL naive test cache).
Output: results/mcq_dead_units.json
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)
from paper1_revision import core

import numpy as np
import torch


def main():
    from rerun2026 import common
    from experiment_cql_draft import CQLDraftAgent
    mm = common.memmap_open(common.CQL_NAIVE_TEST)   # held-out states of the active namespace
    S = mm["states"]
    rng = np.random.RandomState(0)
    idx = np.sort(rng.choice(len(S), size=min(200000, len(S)), replace=False))
    X = torch.tensor(np.asarray(S[idx], dtype=np.float32))
    out = {"n_states": int(len(idx))}
    from rerun2026 import common as rcommon     # honours RERUN_NS (p1site rebuild)
    M = rcommon.MODELS_DIR
    for name, path in (("mcq_t0.5", os.path.join(M, "mcq", "_mcq_temp_t0.5.pt")),
                       ("mcq_t0.6", os.path.join(M, "mcq", "_mcq_temp_t0.6.pt")),
                       ("cql_a1.0", os.path.join(M, "cql", "_cql_temp_a1.0.pt"))):
        net = CQLDraftAgent()
        net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        net.eval()
        mods = list(net.net)
        h = X
        res = {"params": int(sum(p.numel() for p in net.parameters()))}
        layer = 0
        with torch.no_grad():
            for m in mods:
                h = m(h)
                if isinstance(m, torch.nn.ReLU):
                    layer += 1
                    active = (h > 0).any(0)
                    res[f"layer{layer}_dead"] = int((~active).sum())
                    res[f"layer{layer}_units"] = int(h.shape[1])
        out[name] = res
        print(name, res, flush=True)
    json.dump(out, open(os.path.join(core.RESULTS, "mcq_dead_units.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
