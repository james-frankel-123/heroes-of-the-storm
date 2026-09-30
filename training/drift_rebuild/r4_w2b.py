"""
R4 — the W2b greedy-drafting policy evaluation (drift2026/eval_policy_w2b.py
protocol, reused by import: 500 drafts per seed, greedy VF drafting with
GD-argmax rollouts against the rerun2026 GD pool, same seeds and configs) for
value functions trained in the rebuild.

  --cell   model file stem under drift_rebuild/models (without _s<seed>) or
           'd2:<stem>' for a drift2026 model
  --stats  deployment statistics: 'cutoff' (cumulative at the 2026 cutoff),
           'cumprev' (cumulative through the last completed build), or
           'cum:<build>' (cumulative at <build>)
  --gd-pool frozen | cutoff (as in eval_policy_w2b.py)

Output: drift_rebuild/results/r4_w2b/<cell>_s<seed>[_<stats>][_gdcutoff].json
"""
import os
import sys
import json
import time
import random
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rb_common as rb  # noqa: E402

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch  # noqa: E402

torch.set_num_threads(4)
from drift2026 import eval_policy_w2b as E  # noqa: E402
from drift2026 import common  # noqa: E402
import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--stats", default="cutoff")
    ap.add_argument("--gd-pool", default="frozen")
    ap.add_argument("--drafts", type=int, default=500)
    args = ap.parse_args()
    t0 = time.time()
    dev = torch.device("cpu")

    tag = args.cell.replace("d2:", "d2_")
    suffix = "" if args.stats == "cutoff" else "_" + args.stats.replace(":", "")
    suffix += "" if args.gd_pool == "frozen" else f"_gd{args.gd_pool}"
    out = os.path.join(rb.RESULTS_DIR, "r4_w2b", f"{tag}_s{args.seed}{suffix}.json")
    if os.path.exists(out):
        print("exists", out)
        return

    if args.cell.startswith("d2:"):
        model = E.load_cell_model(args.cell[3:], args.seed, dev)
    else:
        saved = common.MODELS_DIR
        common.MODELS_DIR = rb.MODELS_DIR
        model = E.load_cell_model(args.cell, args.seed, dev)
        common.MODELS_DIR = saved

    if args.stats == "cutoff":
        stats = E.deployment_stats("cutoff")
    elif args.stats == "cumprev":
        stats = E.deployment_stats("cumulative_prev")
    else:
        stats = common.load_patch_stats("cumulative", args.stats.split(":", 1)[1])

    gd = E.load_gd_models(args.gd_pool)
    truth = E.future_truth_stats()
    strat = E.make_wp_greedy_strategy(model, E.ENRICHED_GROUPS, stats,
                                      E.compute_group_indices(), dev)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = [(i, random.choice(E.MAPS), random.choice(E.SKILL_TIERS), i % 2)
           for i in range(args.drafts)]
    dr = E.run_drafts_with_strategy(strat, cfg, gd, truth, dev)
    n = len(dr)
    res = {"cell": args.cell, "seed": args.seed, "stats": args.stats,
           "gd_pool": args.gd_pool, "drafts": n,
           "healer_rate": round(sum(d["has_healer"] for d in dr) / n * 100, 2),
           "degen_rate": round(sum(d["is_degen"] for d in dr) / n * 100, 2),
           "synergy_future": float(np.mean([E.synergy_exploitation(
               d["our_picks"], truth, d["tier"]) for d in dr])),
           "counter_future": float(np.mean([E.counter_responsiveness(
               d["our_picks"], d["opp_picks"], truth, d["tier"]) for d in dr])),
           "minutes": round((time.time() - t0) / 60, 1)}
    common.write_json(out, res)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
