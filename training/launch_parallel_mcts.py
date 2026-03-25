"""
Launch 4 parallel MCTS training runs, each on a dedicated GPU with 16 workers.

Usage:
    set -a && source .env && set +a
    python3 training/launch_parallel_mcts.py
"""
import os
import sys
import subprocess
import time

RUNS = [
    {
        "name": "run_G_augmented",
        "gpu": 0,
        "workers": 32,
        "wp_model": "augmented",
        "episodes": 300_000,
        "sims": 200,
        "fresh": False,          # Resume from checkpoint ~228K
        "log": "/tmp/mcts_run_G.log",
    },
    {
        "name": "run_A_base",
        "gpu": 1,
        "workers": 32,
        "wp_model": "base",
        "episodes": 300_000,
        "sims": 200,
        "fresh": True,
        "log": "/tmp/mcts_run_A.log",
    },
]


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    worker_script = os.path.join(script_dir, "train_draft_policy_worker.py")

    # Create the per-run worker script
    create_worker_script(worker_script)

    processes = []
    for run in RUNS:
        # Each run gets its own save directory
        save_dir = os.path.join(script_dir, f"mcts_runs/{run['name']}")
        os.makedirs(save_dir, exist_ok=True)

        # For resume run, copy checkpoint to save dir
        if not run["fresh"]:
            src_ckpt = os.path.join(script_dir, "draft_policy_checkpoint.pt")
            src_weights = os.path.join(script_dir, "draft_policy.pt")
            if os.path.exists(src_ckpt):
                import shutil
                shutil.copy2(src_ckpt, os.path.join(save_dir, "draft_policy_checkpoint.pt"))
                shutil.copy2(src_weights, os.path.join(save_dir, "draft_policy.pt"))
                print(f"  {run['name']}: copied checkpoint to {save_dir}")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(run["gpu"])
        env["MCTS_SAVE_DIR"] = save_dir
        env["MCTS_NUM_WORKERS"] = str(run["workers"])
        env["MCTS_WP_MODEL"] = run["wp_model"]
        env["MCTS_NUM_EPISODES"] = str(run["episodes"])
        env["MCTS_NUM_SIMS"] = str(run["sims"])
        env["MCTS_FRESH"] = "1" if run["fresh"] else "0"
        env["WANDB_RUN_NAME"] = run["name"]

        cmd = [sys.executable, "-u", worker_script]
        log_file = open(run["log"], "w")
        proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        processes.append((run["name"], proc, log_file, run["log"]))
        print(f"Started {run['name']} on GPU {run['gpu']} (PID {proc.pid}, log: {run['log']})")

    print(f"\n{'='*60}")
    print(f"All {len(processes)} runs launched. Monitor with:")
    for name, _, _, log in processes:
        print(f"  tail -f {log}")
    print(f"\nOr check all at once:")
    print(f"  for f in /tmp/mcts_run_*.log; do echo \"=== $f ===\"; tail -3 $f; done")
    print(f"{'='*60}")

    # Wait for all to finish
    for name, proc, log_file, log_path in processes:
        proc.wait()
        log_file.close()
        print(f"{name}: finished with exit code {proc.returncode}")


def create_worker_script(path):
    """Create the per-run training script that reads config from env vars."""
    script = '''#!/usr/bin/env python3
"""Single MCTS training run. Config via environment variables."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import random
import time
import multiprocessing as mp

# Read config from environment
SAVE_DIR = os.environ.get("MCTS_SAVE_DIR", os.path.dirname(__file__))
NUM_WORKERS = int(os.environ.get("MCTS_NUM_WORKERS", "16"))
WP_MODEL_TYPE = os.environ.get("MCTS_WP_MODEL", "augmented")
NUM_EPISODES = int(os.environ.get("MCTS_NUM_EPISODES", "300000"))
NUM_SIMS = int(os.environ.get("MCTS_NUM_SIMS", "200"))
FRESH = os.environ.get("MCTS_FRESH", "1") == "1"
RUN_NAME = os.environ.get("WANDB_RUN_NAME", "mcts_run")

from train_draft_policy import (
    AlphaZeroDraftNet, DraftState, DRAFT_ORDER,
    _flatten_state_dict, _write_to_shared, _worker_init, _run_episode,
    _predict_fn, _evaluate_wp, mcts_search, simulate_draft_with_mcts,
    pretrain_value_head, bootstrap_from_generic_draft,
    STATE_DIM, NUM_HEROES, HEROES, load_pretrained_models,
)
from shared import MAPS, SKILL_TIERS, load_replay_data, heroes_to_multi_hot, map_to_one_hot, tier_to_one_hot
from gpu_batch_server import GPUBatchServer
import torch.nn as nn
import torch.nn.functional as F

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def main():
    mp.set_start_method('spawn', force=True)
    device = torch.device("cpu")
    print(f"Run: {RUN_NAME}")
    print(f"Config: workers={NUM_WORKERS}, wp={WP_MODEL_TYPE}, episodes={NUM_EPISODES}, "
          f"sims={NUM_SIMS}, fresh={FRESH}, save_dir={SAVE_DIR}")
    print(f"GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'none')}")

    # Load WP model based on type
    wp_enriched_config = None
    if WP_MODEL_TYPE == "base":
        from train_win_probability import WinProbModel
        wp_path = os.path.join(os.path.dirname(__file__), "win_probability.pt")
        wp_model = WinProbModel()
        wp_model.load_state_dict(torch.load(wp_path, weights_only=True, map_location="cpu"))
        wp_model.eval()
        print(f"Loaded base WP model (197 dims)")
    elif WP_MODEL_TYPE in ("enriched", "augmented"):
        from sweep_enriched_wp import WinProbEnrichedModel, StatsCache, compute_group_indices, FEATURE_GROUP_DIMS
        WP_ENRICHED_GROUPS = ['role_counts', 'team_avg_wr', 'map_delta',
                              'pairwise_counters', 'pairwise_synergies', 'counter_detail',
                              'meta_strength', 'draft_diversity', 'comp_wr']
        enriched_dim = sum(FEATURE_GROUP_DIMS[g] for g in WP_ENRICHED_GROUPS)
        wp_input_dim = 197 + enriched_dim

        if WP_MODEL_TYPE == "augmented":
            wp_path = os.path.join(os.path.dirname(__file__), "wp_enriched_winner.pt")
            wp_hidden = [512, 256, 128]
        else:
            wp_path = os.path.join(os.path.dirname(__file__), "wp_experiment_enriched.pt")
            wp_hidden = [256, 128]

        wp_model = WinProbEnrichedModel(wp_input_dim, wp_hidden, dropout=0.3)
        wp_model.load_state_dict(torch.load(wp_path, weights_only=True, map_location="cpu"))
        wp_model.eval()

        _stats = StatsCache()
        wp_enriched_config = {
            'input_dim': wp_input_dim, 'hidden_dims': wp_hidden, 'dropout': 0.3,
            'groups': WP_ENRICHED_GROUPS,
            '_stats_hero_wr': _stats.hero_wr, '_stats_hero_map_wr': _stats.hero_map_wr,
            '_stats_pairwise': _stats.pairwise, '_stats_hero_meta': _stats.hero_meta,
            '_stats_comp_data': _stats.comp_data,
        }
        print(f"Loaded {WP_MODEL_TYPE} WP model ({wp_input_dim} dims, {wp_hidden})")

    # Load GD models
    from train_generic_draft import GenericDraftModel
    gd_models = []
    for i in range(10):
        gd_path = os.path.join(os.path.dirname(__file__), f"generic_draft_{i}.pt")
        if not os.path.exists(gd_path):
            break
        gd = GenericDraftModel()
        gd.load_state_dict(torch.load(gd_path, weights_only=True, map_location="cpu"))
        gd.eval()
        gd_models.append(gd)
    if not gd_models:
        gd_path = os.path.join(os.path.dirname(__file__), "generic_draft.pt")
        gd = GenericDraftModel()
        gd.load_state_dict(torch.load(gd_path, weights_only=True, map_location="cpu"))
        gd.eval()
        gd_models.append(gd)
    print(f"Loaded {len(gd_models)} GD models")

    # Initialize network
    network = AlphaZeroDraftNet().to(device)
    print(f"Network: {sum(p.numel() for p in network.parameters()):,} params")

    # Checkpoint handling
    start_episode = 0
    best_eval_wp = 0.0
    ckpt_path = os.path.join(SAVE_DIR, "draft_policy_checkpoint.pt")
    weights_path = os.path.join(SAVE_DIR, "draft_policy.pt")

    if not FRESH and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
        network.load_state_dict(ckpt['model_state_dict'])
        start_episode = ckpt.get('episode', 0)
        best_eval_wp = ckpt.get('best_eval_wp', 0.0)
        print(f"Resumed from checkpoint: episode {start_episode}, best_wp={best_eval_wp:.4f}")
    else:
        bootstrap_from_generic_draft(network, device)
        pretrain_value_head(network, device)

    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150000, eta_min=1e-5)

    if not FRESH and os.path.exists(ckpt_path) and start_episode > 0:
        try:
            ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            print("Restored optimizer + scheduler")
        except Exception:
            print("Could not restore optimizer/scheduler")

    # Pre-extract state dicts
    wp_sd = wp_model.state_dict()
    gd_sds = [gd.state_dict() for gd in gd_models]

    # Shared memory for network weights
    net_sd_cpu = {k: v.cpu() for k, v in network.state_dict().items()}
    flat, net_shapes, net_keys = _flatten_state_dict(net_sd_cpu)
    shared_net_flat = mp.Array('f', flat.numpy(), lock=False)
    _write_to_shared(shared_net_flat, net_sd_cpu, net_keys)

    # GPU batch server
    gpu_server = None
    GPU_SLOTS = 256
    gpu_init_args = (None, None, None, None, GPU_SLOTS)
    if torch.cuda.is_available():
        gpu_server = GPUBatchServer(net_sd_cpu, STATE_DIM, num_slots=GPU_SLOTS, max_batch=128,
                                     device='cuda:0')
        gpu_server.start(AlphaZeroDraftNet)
        gpu_init_args = (gpu_server.slot_flags, gpu_server.req_buf,
                         gpu_server.resp_buf, STATE_DIM, GPU_SLOTS)
        time.sleep(2)
        print(f"GPU batch server: started")

    # Worker pool
    print(f"Creating pool with {NUM_WORKERS} workers...")
    pool = mp.Pool(
        NUM_WORKERS,
        initializer=_worker_init,
        initargs=(wp_sd, gd_sds, wp_enriched_config,
                  shared_net_flat, net_shapes, net_keys, False,
                  *gpu_init_args),
    )
    print("Pool ready.")

    # W&B
    if HAS_WANDB:
        wandb.init(project="hots-draft-policy", name=RUN_NAME,
                   config={"wp_model": WP_MODEL_TYPE, "workers": NUM_WORKERS,
                           "episodes": NUM_EPISODES, "sims": NUM_SIMS})

    # Training loop
    buffer = []
    BUFFER_SIZE = 150_000
    BATCH_SIZE = 512
    EVAL_EVERY = 500
    EVAL_DRAFTS = 200

    train_start = time.time()
    episode = start_episode

    while episode < NUM_EPISODES:
        batch_size = min(NUM_WORKERS, NUM_EPISODES - episode)
        net_sd_cpu = {k: v.cpu() for k, v in network.state_dict().items()}
        _write_to_shared(shared_net_flat, net_sd_cpu, net_keys)
        if gpu_server:
            gpu_server.update_weights(net_sd_cpu)

        worker_args = [
            (random.choice(MAPS), random.choice(SKILL_TIERS), NUM_SIMS)
            for _ in range(batch_size)
        ]
        results = pool.map(_run_episode, worker_args)

        last_wp = 0.0
        for win_prob, examples in results:
            for state_feat, mcts_policy, valid in examples:
                buffer.append((state_feat, mcts_policy, valid, win_prob))
                if len(buffer) > BUFFER_SIZE:
                    buffer.pop(0)
            last_wp = win_prob
            episode += 1

        # Train on buffer
        if len(buffer) >= BATCH_SIZE:
            network.train()
            num_train_steps = max(1, batch_size // 2)
            for _ in range(num_train_steps):
                batch = random.sample(buffer, BATCH_SIZE)
                states = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32).to(device)
                target_policies = torch.tensor(np.array([b[1] for b in batch]), dtype=torch.float32).to(device)
                masks = torch.tensor(np.array([b[2] for b in batch]), dtype=torch.float32).to(device)
                target_values = torch.tensor(np.array([b[3] for b in batch], dtype=np.float32)).to(device)

                pred_logits, pred_values = network(states, masks)
                pred_log_probs = F.log_softmax(pred_logits, dim=1)
                policy_loss = -(target_policies * pred_log_probs).sum(dim=1).mean()
                value_loss = F.mse_loss(pred_values, target_values)
                loss = policy_loss + value_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(network.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

        # Logging
        if episode % NUM_WORKERS == 0 or episode >= NUM_EPISODES:
            elapsed = time.time() - train_start
            eps = (episode - start_episode) / elapsed if elapsed > 0 else 0
            eta = (NUM_EPISODES - episode) / eps / 3600 if eps > 0 else 0
            print(f"Episode {episode}: wp={last_wp:.4f} buffer={len(buffer)} "
                  f"lr={scheduler.get_last_lr()[0]:.6f} [{eps:.1f} ep/s, ETA {eta:.1f}h]")
            if HAS_WANDB and wandb.run:
                wandb.log({"episode": episode, "last_wp": last_wp,
                           "buffer_size": len(buffer), "lr": scheduler.get_last_lr()[0],
                           "episodes_per_sec": eps}, step=episode)

        # Eval
        if episode % EVAL_EVERY < NUM_WORKERS or episode >= NUM_EPISODES:
            network.eval()
            net_sd_eval = {k: v.cpu() for k, v in network.state_dict().items()}
            _write_to_shared(shared_net_flat, net_sd_eval, net_keys)
            if gpu_server:
                gpu_server.update_weights(net_sd_eval)

            eval_args = [
                (random.choice(MAPS), random.choice(SKILL_TIERS), NUM_SIMS // 2)
                for _ in range(EVAL_DRAFTS)
            ]
            eval_results = pool.map(_run_episode, eval_args)
            eval_wps = [r[0] for r in eval_results]
            avg_wp = np.mean(eval_wps)
            std_wp = np.std(eval_wps)
            win_rate = np.mean([1.0 if w > 0.5 else 0.0 for w in eval_wps])
            print(f"\\n  EVAL @ {episode}: avg_wp={avg_wp:.4f} +/- {std_wp:.4f} "
                  f"win_rate={win_rate:.1%} (vs {len(gd_models)} opponents)")

            if HAS_WANDB and wandb.run:
                wandb.log({"eval/avg_wp": avg_wp, "eval/win_rate": win_rate}, step=episode)

            if avg_wp > best_eval_wp:
                best_eval_wp = avg_wp
                torch.save(network.state_dict(), weights_path)
                torch.save({
                    'model_state_dict': network.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'episode': episode,
                    'best_eval_wp': best_eval_wp,
                }, ckpt_path)
                print(f"  New best! Saved to {SAVE_DIR}")
            print()

    pool.close(); pool.join()
    if gpu_server:
        print(f"GPU server stats: {gpu_server.stats()}")
        gpu_server.shutdown()
    if HAS_WANDB and wandb.run:
        wandb.finish()
    print(f"Training complete. {episode} episodes in {(time.time()-train_start)/3600:.1f}h")


if __name__ == "__main__":
    main()
'''
    with open(path, 'w') as f:
        f.write(script)
    os.chmod(path, 0o755)
    print(f"Created worker script: {path}")


if __name__ == "__main__":
    main()
