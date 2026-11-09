"""
Table IV row (5 seeds x 1000 drafts, the rich-evaluation protocol of every
Q-learning baseline: experiment_iql_draft._rich_run, GD opponents, frozen
statistics) for a td_offline.py checkpoint.

Usage (from training/, RICH env of swapfix_queue):
  python3 paper1_revision/td_rich.py cql_td_a1.0 [--seeds 5 --drafts 1000]
Output: rerun2026/ns/p1site/results/rich_eval/<name>.json (skipped if present).
"""
import os
import sys
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from rerun2026 import common
from rerun2026.common import MODELS_DIR, RESULTS_DIR
from rerun2026 import experiment_iql_draft as iql


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--drafts", type=int, default=1000)
    a = ap.parse_args()
    sub = "rich_eval" if (a.seeds, a.drafts) == (5, 1000) else "td_screen"
    out = os.path.join(RESULTS_DIR, sub, f"{a.name}.json")
    if os.path.exists(out):
        print("exists", out)
        return
    from experiment_rich_evaluation import make_cql_strategy
    device = iql._device()
    strat = make_cql_strategy(os.path.join(MODELS_DIR, "td", f"{a.name}.pt"), device)
    r = iql._rich_run(strat, a.name, a.name, a.seeds, a.drafts, device, common.stats_cache())
    tl = os.path.join(MODELS_DIR, "td", f"{a.name}.json")
    if os.path.exists(tl):
        r["train_log_last"] = json.load(open(tl))["log"][-1]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(r, open(out + ".tmp", "w"), indent=1, default=str)
    os.replace(out + ".tmp", out)
    print(json.dumps(r["metrics"]))


if __name__ == "__main__":
    main()
