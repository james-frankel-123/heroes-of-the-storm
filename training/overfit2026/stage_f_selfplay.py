"""
Stage F: benchmark the self-play MCTS policies trained against split-A
proxies (train_mcts_split.py) with the paper's benchmark protocol (the trained
network as prior, its own proxy at the leaves, 200 inference sims, root T=1;
plus T=0), 400 paired draft configs, scored on side-B gold and post-snapshot
gold exactly like stage E.
Output: results/selfplay/<run>__inf<sims>__T<T>.json
"""
import os
import sys
import glob
import json
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)

import numpy as np

from overfit2026 import search, score
from overfit2026.stage_e import N_DRAFTS, SEED

OUT = os.path.join(HERE, "results", "selfplay")


def main():
    os.makedirs(OUT, exist_ok=True)
    kernel = search.load_kernel("ofit")
    dirs = sorted(glob.glob(os.path.join(HERE, "mcts_runs", "*"))) + \
        sorted(glob.glob(os.path.join(HERE, "mcts_runs_partial", "*")))
    for d in dirs:
        ck = os.path.join(d, "draft_policy.pt")
        name = os.path.basename(d)
        if "_at" not in name:
            log = os.path.join(HERE, "logs", f"mcts_{name}.log")
            if not os.path.exists(ck) or "Complete." not in open(log).read():
                continue
        proxy = name.split("_")[0] + "_" + name.split("_")[1]   # pA8_leak / pA8_oof
        for inf, T in ((200, 1.0), (200, 0.0)):
            path = os.path.join(OUT, f"{name}__inf{inf}__T{T}.json")
            if os.path.exists(path):
                continue
            t0 = time.time()
            m = score.model(proxy)[0]
            wcfg = search.wp_config([m], score._stats("A8"))
            drafts = search.run(kernel, search.policy_flat(f"path:{ck}"), wcfg, n=N_DRAFTS,
                                sims=inf, seed=SEED, root_temp=T, guard=1)
            sc = score.score(drafts, proxies=(proxy,))
            res = {"spec": {"proxies": [proxy], "prior": f"selfplay:{name}", "sims": inf,
                            "temp": T, "lcb": 0.0, "train_sims": int(name.split("_")[2][:-3])},
                   "n": len(drafts), "summary": score.summarize(sc),
                   "diversity": score.diversity(drafts), "drafts": drafts,
                   "per_draft": {k: np.round(np.asarray(v, float), 5).tolist() for k, v in sc.items()},
                   "secs": time.time() - t0}
            json.dump(res, open(path, "w"))
            s = res["summary"]
            print(f"{name:28s} inf{inf} T{T}: proxy={s[f'proxy:{proxy}'][0]:.4f} gB={s['gB'][0]:.4f} "
                  f"RB={s['RB'][0]:.4f} RN={s['RN'][0]:.4f} QM={s['QM2026'][0]:.4f} "
                  f"deg={s['degen'][0]:.3f} H={res['diversity']['entropy_bits']:.2f}", flush=True)


if __name__ == "__main__":
    main()
