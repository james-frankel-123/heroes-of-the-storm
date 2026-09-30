"""
Launch the UNCHANGED paper MCTS self-play worker (training/train_mcts_worker.py)
against a split-A proxy, so the full paper pipeline (self-play training +
best-by-proxy checkpoint selection) is replicated with a value function that
has never seen half B.

  proxy weights   overfit2026/models/<proxy>.pt   (283-d, 256x128, like wp_enriched_256)
  LUT statistics  WP_STATS_PATH=overfit2026/cache/split_stats/A8.json
  value-head pretraining data: snapshot minus pre-2.55 minus every half-B replay
  opponent / bootstrap: rerun2026/models/generic_draft_0.pt (behavior only)

Usage: python3 overfit2026/train_mcts_split.py <proxy> <sims> <seed_tag> <gpu>
"""
import os
import sys
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)


def main():
    proxy, sims, tag, gpu = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
    episodes = int(sys.argv[5]) if len(sys.argv) > 5 else 300000
    stats = sys.argv[6] if len(sys.argv) > 6 else "A8"
    name = f"{proxy}_{sims}sim_{tag}"
    save = os.path.join(HERE, "mcts_runs", name)
    os.makedirs(save, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "MCTS_SAVE_DIR": save,
        "MCTS_WP_MODEL": "enriched_full",
        "MCTS_WP_PATH": os.path.join(HERE, "models", f"{proxy}.pt"),
        "WP_STATS_PATH": os.path.join(HERE, "cache", "split_stats", f"{stats}.json"),
        "MCTS_GD_PATH": os.path.join(TRAINING_DIR, "rerun2026", "models", "generic_draft_0.pt"),
        "MCTS_NUM_EPISODES": str(episodes),
        "MCTS_NUM_SIMS": str(sims),
        "MCTS_BATCH_EPISODES": "128",
        "MCTS_FRESH": "1",
        "MCTS_POLICY_HEAD": "linear",
        "MCTS_NET_SIZE": "base",
        "MCTS_EXCLUDE_IDS": os.path.join(HERE, "cache", "exclude_pre255_and_B.json"),
        "REPLAY_SNAPSHOT": "1",
        "WANDB_MODE": "disabled",
        "WANDB_RUN_NAME": f"overfit2026_{name}",
    })
    log = open(os.path.join(HERE, "logs", f"mcts_{name}.log"), "w")
    p = subprocess.Popen([sys.executable, "-u", os.path.join(TRAINING_DIR, "train_mcts_worker.py")],
                         cwd=TRAINING_DIR, env=env, stdout=log, stderr=subprocess.STDOUT)
    sys.exit(p.wait())


if __name__ == "__main__":
    main()
