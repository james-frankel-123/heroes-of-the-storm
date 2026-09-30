"""
Check that the CUDA kernel's in-kernel enriched features (built from the own
deploy statistics and own composition table) reproduce the Python feature
pipeline: generate 200 drafts with the revision leaf WP and compare the
kernel's terminal WP with the Python WP of the same terminal teams.
Output: results/verify_kernel.json
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)
sys.path.insert(0, os.path.join(TRAINING_DIR, "cuda_mcts"))
from paper1_revision import core, bench_mcts, train_wp

import numpy as np
import torch


def main():
    from overfit2026 import search
    kernel = search.load_kernel("ofit")
    run = "old:J_800sim_s0"
    wcfg = bench_mcts.leaf_config("new:x_s0")          # revision leaf WP + own LUTs
    drafts = search.run(kernel, bench_mcts.policy_flat(run), wcfg, n=200, sims=200,
                        seed=7, root_temp=1.0, guard=1, batch=100)
    m, cols = train_wp.load("enriched")
    rows = [(d["our"], d["opp"], d["map"], d["tier"]) for d in drafts]
    Xf, Xs = core.featurize(rows, core.load_stats("deploy"), nproc=4)
    with torch.no_grad():
        a = m(torch.tensor(Xf[:, cols])).view(-1).numpy()
        b = m(torch.tensor(Xs[:, cols])).view(-1).numpy()
    sym = 0.5 * (a + 1 - b)
    k = np.array([d["kernel_wp"] for d in drafts])
    res = {"n": len(drafts), "max_abs_diff_raw": float(np.abs(k - a).max()),
           "max_abs_diff_sym": float(np.abs(k - sym).max()),
           "corr_raw": float(np.corrcoef(k, a)[0, 1]), "corr_sym": float(np.corrcoef(k, sym)[0, 1]),
           "mean_kernel": float(k.mean()), "mean_python_sym": float(sym.mean())}
    print(res)
    json.dump(res, open(os.path.join(core.RESULTS, "verify_kernel.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
