"""
W7 — staleness gradient: MCTS arms for the 1yr- and 2yr-stale VFs, then
head-to-head vs the maintained agent (Q6 of WORK_QUEUE_PREWRITING.md).

Reuses the W4 recipe verbatim (200 sims, 300K episodes, linear head, base
net, same exclusions, same GD paths) — the ONLY differences per arm are the
value function checkpoint and MCTS_STATS_BUILD (a stale deployment has its
own era's stats, so each arm gets cumulative stats through ITS cutoff).

Stages:
  train  — export best-seed stale VFs, run 2 arms x 5 seeds MCTS
           (restrict GPUs via CUDA_VISIBLE_DEVICES before invoking)
  h2h    — each stale arm head-to-head vs w4_d2c_cumprev (W6 protocol via
           w6_head2head machinery), judge-free future-truth scoring, and
           vintage-matrix rescoring with the Q2 judges
Usage:
  CUDA_VISIBLE_DEVICES=1,3 python3 drift2026/w7_staleness.py --stage train
  python3 drift2026/w7_staleness.py --stage h2h
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drift2026 import common

common.setup()

import numpy as np
import torch

from drift2026.phase_w4_mcts import (build_exclude_ids, MCTS_RUNS_DIR,
                                     RERUN_MODELS, WORKER, SIMS, EPISODES,
                                     SEEDS_PER_ARM)

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = {  # cell name -> (stale cutoff build, human label)
    "stale1yr": ("2.55.9.93613", "~1yr stale (trained through 2025-02)"),
    "stale2yr": ("2.55.4.91418", "~2yr stale (trained through 2024-02)"),
}
VF_SEEDS = [42, 123, 777]


def export_stale_vf(cell):
    build, _ = ARMS[cell]
    accs = {}
    for s in VF_SEEDS:
        j = json.load(open(os.path.join(common.RESULTS_DIR, "w7",
                                        f"w7_stale_{build}_s{s}.json")))
        accs[s] = j["test_acc_future"] if j["test_acc_future"] is not None \
            else j["val_acc"]
    best = max(accs, key=accs.get)
    ck = torch.load(os.path.join(common.MODELS_DIR,
                                 f"w7_stale_{build}_s{best}.pt"),
                    weights_only=True, map_location="cpu")
    out = os.path.join(common.MODELS_DIR, f"w7_vf_{cell}.pt")
    torch.save(ck["state_dict"], out)
    print(f"exported {cell} VF (seed {best}, sel-metric {accs[best]:.2f}) -> {out}")
    return out


def stage_train():
    exclude_path = build_exclude_ids()
    jobs = []
    for cell, (build, _) in ARMS.items():
        vf = export_stale_vf(cell)
        for seed in range(SEEDS_PER_ARM):
            name = f"w7_{cell}_s{seed}"
            save_dir = os.path.join(MCTS_RUNS_DIR, name)
            os.makedirs(save_dir, exist_ok=True)
            env = {
                "MCTS_SAVE_DIR": save_dir,
                "MCTS_WP_MODEL": "enriched_full",
                "MCTS_WP_PATH": vf,
                "MCTS_GD_PATH": os.path.join(RERUN_MODELS, "generic_draft_0.pt"),
                "MCTS_NUM_EPISODES": str(EPISODES),
                "MCTS_NUM_SIMS": str(SIMS),
                "MCTS_BATCH_EPISODES": "128",
                "MCTS_FRESH": "1",
                "MCTS_POLICY_HEAD": "linear",
                "MCTS_NET_SIZE": "base",
                "MCTS_STATS_BUILD": build,
                "MCTS_EXCLUDE_IDS": exclude_path,
                "WANDB_RUN_NAME": f"drift2026_{name}",
                # No wandb cloud for bulk sweeps (kills run-finished emails).
                "WANDB_MODE": "offline",
            }
            jobs.append(common.Job(f"w7_mcts_{name}", [WORKER],
                                   [os.path.join(save_dir, "draft_policy.pt")],
                                   weight="light", env=env))
    if common.run_pool(jobs):
        sys.exit(1)
    print("W7 MCTS TRAINING COMPLETE")


def stage_h2h():
    import subprocess
    for cell in ARMS:
        cmd = [sys.executable, os.path.join(HERE, "w6_head2head.py"),
               "--drafts-per-cell", "40",
               "--arm-b-prefix", "w7", "--arm-b", cell,
               "--out-suffix", f"_{cell}"]
        print("running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=os.path.dirname(HERE))
    print("W7 head-to-heads complete; run judge-free + vintage rescoring "
          "via w6_judgefree.py --in w6_head2head_<cell>.json and "
          "q2_vintage_judges.py rescore mode.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["train", "h2h"])
    args = ap.parse_args()
    if args.stage == "train":
        stage_train()
    else:
        stage_h2h()


if __name__ == "__main__":
    main()
