"""
W8 — second-review hardening: MCTS pool for W8a/W8b
(WORK_QUEUE_PREWRITING.md, W8 section).

Runs, in ONE run_pool call (30 jobs, ~5h on 4 GPUs):
  W8a  w8_volmatch_s0..4     maintained-config VF trained on a random
                             subsample volume-matched to stale-2yr's n_train
                             (1,594,630 rows; VF = w8_vf_volmatch.pt, best of
                             3 seeds by future acc), cumprev deploy stats
                             (cumulative @ last completed build).
  W8b  w8_champion_s0..14    champion config decayed90k100 (Q7 57.22 cell;
                             VF = w8_vf_champion.pt = q7_decayed90k100_s777),
                             deploy stats decayed90k100 @ last completed
                             build (MCTS_STATS_KIND override in
                             train_mcts_worker.py).
  W8b  w8_d2b_allhist_s5..14 stale-4mo seed extension (same VF + env recipe
                             as w4_d2b_allhist_s0..4: w4_vf_d2b_allhist.pt,
                             cumulative @ cutoff build); pooled with the w4_
                             runs at head-to-head time via the multi-prefix
                             loader in w6_head2head.py ("w4,w8").

Everything else = the W4 recipe verbatim: 200 sims, 300K episodes, linear
head, base net, MCTS_BATCH_EPISODES=128, MCTS_FRESH=1, w4 exclusions.

Usage:
  python3 drift2026/w8_hardening.py --dry-run
  python3 drift2026/w8_hardening.py --smoke      # 2K-episode champion smoke
  nohup python3 -u drift2026/w8_hardening.py > drift2026/logs/w8_pool.log 2>&1 &
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drift2026 import common

common.setup()

from drift2026.phase_w4_mcts import (build_exclude_ids, MCTS_RUNS_DIR,
                                     RERUN_MODELS, WORKER, SIMS, EPISODES)


def builds_255():
    return [b for b in common.load_patch_index()["builds"]
            if b.startswith("2.55")]


def base_env(save_dir, vf_path, stats_build, name, episodes=EPISODES):
    return {
        "MCTS_SAVE_DIR": save_dir,
        "MCTS_WP_MODEL": "enriched_full",
        "MCTS_WP_PATH": vf_path,
        "MCTS_GD_PATH": os.path.join(RERUN_MODELS, "generic_draft_0.pt"),
        "MCTS_NUM_EPISODES": str(episodes),
        "MCTS_NUM_SIMS": str(SIMS),
        "MCTS_BATCH_EPISODES": "128",
        "MCTS_FRESH": "1",
        "MCTS_POLICY_HEAD": "linear",
        "MCTS_NET_SIZE": "base",
        "MCTS_STATS_BUILD": stats_build,
        "MCTS_EXCLUDE_IDS": build_exclude_ids(),
        "WANDB_RUN_NAME": f"drift2026_{name}",
    }


def make_jobs():
    b = builds_255()
    last_completed = b[-2]                 # cumprev/decayed deploy convention
    cutoff = common.TRAIN_CUTOFF_BUILD     # d2b_allhist deploy convention
    vf_champion = os.path.join(common.MODELS_DIR, "w8_vf_champion.pt")
    vf_volmatch = os.path.join(common.MODELS_DIR, "w8_vf_volmatch.pt")
    vf_allhist = os.path.join(common.MODELS_DIR, "w4_vf_d2b_allhist.pt")
    for p in (vf_champion, vf_volmatch, vf_allhist):
        assert os.path.exists(p), p

    jobs = []

    def add(name, vf, stats_build, stats_kind=None):
        save_dir = os.path.join(MCTS_RUNS_DIR, name)
        os.makedirs(save_dir, exist_ok=True)
        env = base_env(save_dir, vf, stats_build, name)
        if stats_kind:
            env["MCTS_STATS_KIND"] = stats_kind
        jobs.append(common.Job(f"w8_mcts_{name}", [WORKER],
                               [os.path.join(save_dir, "draft_policy.pt")],
                               weight="light", env=env))

    # champion first (15 seeds needed for the biggest head-to-head)
    for s in range(15):
        add(f"w8_champion_s{s}", vf_champion, last_completed,
            stats_kind="decayed90k100")
    for s in range(5, 15):   # stale-4mo extension, pools with w4_ s0..4
        add(f"w8_d2b_allhist_s{s}", vf_allhist, cutoff)
    for s in range(5):
        add(f"w8_volmatch_s{s}", vf_volmatch, last_completed)
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="single 2K-episode champion-config run (end-to-end "
                         "check of the MCTS_STATS_KIND path)")
    args = ap.parse_args()

    if args.smoke:
        b = builds_255()
        name = "w8_smoke_champion"
        save_dir = os.path.join(MCTS_RUNS_DIR, name)
        os.makedirs(save_dir, exist_ok=True)
        env = base_env(save_dir, os.path.join(common.MODELS_DIR,
                                              "w8_vf_champion.pt"),
                       b[-2], name, episodes=2048)
        env["MCTS_STATS_KIND"] = "decayed90k100"
        env["WANDB_MODE"] = "disabled"
        job = common.Job(name, [WORKER],
                         [os.path.join(save_dir, "draft_policy.pt")],
                         weight="light", env=env)
        sys.exit(common.run_pool([job], num_gpus=1))

    jobs = make_jobs()
    if common.run_pool(jobs, dry_run=args.dry_run):
        sys.exit(1)
    print("W8 MCTS POOL COMPLETE")


if __name__ == "__main__":
    main()
