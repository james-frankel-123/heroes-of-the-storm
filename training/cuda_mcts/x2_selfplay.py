"""
Short self-play comparison for the X2 fix: the paper-1 revision MCTS run
(paper1_revision/train_mcts.py config F: 400 sims, leak-free enriched WP, own
deploy statistics and composition table, GD opponent, value pretraining on the
filtered snapshot) through its unchanged --child entry, with a shorter episode
budget and the kernel search mode as the only difference.

Usage (from training/):
  python3 cuda_mcts/x2_selfplay.py <legacy|chance> [--episodes 50000] [--seed 0]
      [--gpu 0] [--pause-file PATH] [--resume]
Output: cuda_mcts/x2_results/selfplay/F<eps/1000>k_<mode>_s<seed>/ (draft_policy.pt,
kernel_info.json, run_meta.json, train.log)
"""
import os
import sys
import json
import argparse
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["legacy", "chance", "rollfwd"])
    ap.add_argument("--episodes", type=int, default=50000)
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--pause-file", default="")
    ap.add_argument("--ckpt-every", default="0")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, TRAINING_DIR)
    from paper1_revision import core, train_mcts
    meta = json.load(open(os.path.join(core.MODEL_DIR, "enriched.json")))
    wp = os.path.join(core.MODEL_DIR, f"enriched_s{meta['selected_seed']}.pt")
    name = f"F{a.episodes // 1000}k_{a.mode}_s{a.seed}"
    save = os.path.join(HERE, "x2_results", "selfplay", name)
    os.makedirs(save, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(a.gpu),
        "MCTS_SAVE_DIR": save,
        "MCTS_WP_MODEL": "enriched_full",
        "MCTS_WP_PATH": wp,
        "WP_STATS_PATH": core.stats_path("deploy"),
        "P1R_COMP_PATH": core.comp_path("deploy"),
        "MCTS_GD_PATH": os.path.join(TRAINING_DIR, "rerun2026", "models", "generic_draft_0.pt"),
        "MCTS_NUM_EPISODES": str(a.episodes),
        "MCTS_NUM_SIMS": str(a.sims),
        "MCTS_BATCH_EPISODES": "128",
        "MCTS_FRESH": "0" if a.resume else "1",
        "MCTS_POLICY_HEAD": "linear",
        "MCTS_NET_SIZE": "base",
        "MCTS_EXCLUDE_IDS": train_mcts.exclude_path(),
        "MCTS_SEARCH_MODE": a.mode,
        "REPLAY_SNAPSHOT": "1",
        "WANDB_MODE": "disabled",
        "WANDB_RUN_NAME": f"x2_{name}",
        "PYTHONHASHSEED": str(a.seed),
        "P1R_RUN_SEED": str(a.seed),
        "OMP_NUM_THREADS": env.get("OMP_NUM_THREADS", "4"),
        "MKL_NUM_THREADS": env.get("MKL_NUM_THREADS", "4"),
    })
    if a.pause_file:
        env["MCTS_PAUSE_FILE"] = a.pause_file
    if a.ckpt_every != "0":
        env["MCTS_CKPT_EVERY_SEC"] = a.ckpt_every
    json.dump({"config": "F", "sims": a.sims, "episodes": a.episodes, "seed": a.seed,
               "search_mode": a.mode, "wp": wp, "stats": core.stats_path("deploy")},
              open(os.path.join(save, "run_meta.json"), "w"), indent=1)
    log = open(os.path.join(save, "train.log"), "a" if a.resume else "w")
    p = subprocess.Popen([sys.executable, "-u", os.path.join(TRAINING_DIR, "paper1_revision", "train_mcts.py"),
                          "--child"], cwd=TRAINING_DIR, env=env, stdout=log, stderr=subprocess.STDOUT)
    sys.exit(p.wait())


if __name__ == "__main__":
    main()
