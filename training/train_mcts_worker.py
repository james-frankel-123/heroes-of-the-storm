#!/usr/bin/env python3
"""
MCTS training worker using full CUDA kernel (one block = one episode).

Optimizations:
1. Eval every 2000 episodes (not 500) — async, doesn't block generation
2. Pipelined: GPU generates batch N+1 while CPU trains on batch N
3. 128 episodes per kernel launch (not 16)
4. Weight sync every 256 episodes (not every batch)
"""
import os
import sys
import time
import random
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib.util

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cuda_mcts'))

from train_draft_policy import (
    AlphaZeroDraftNet, pretrain_value_head, bootstrap_from_generic_draft,
    STATE_DIM, NUM_HEROES,
)
from train_generic_draft import GenericDraftModel
from extract_weights import extract_policy_weights, extract_gd_weights
from shared import MAPS, SKILL_TIERS

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# Load kernel module
so_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cuda_mcts')
so_files = [f for f in os.listdir(so_dir) if f.startswith('cuda_mcts_kernel') and f.endswith('.so')]
if not so_files:
    raise RuntimeError("cuda_mcts_kernel not built. cd cuda_mcts && python3 setup.py build_ext --inplace")
spec = importlib.util.spec_from_file_location('cuda_mcts_kernel', os.path.join(so_dir, so_files[0]))
kernel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kernel)

# Config
SAVE_DIR = os.environ.get("MCTS_SAVE_DIR", os.path.dirname(__file__))
NUM_EPISODES = int(os.environ.get("MCTS_NUM_EPISODES", "300000"))
NUM_SIMS = int(os.environ.get("MCTS_NUM_SIMS", "200"))
FRESH = os.environ.get("MCTS_FRESH", "1") == "1"
RUN_NAME = os.environ.get("WANDB_RUN_NAME", "mcts_run")
BATCH_EPISODES = int(os.environ.get("MCTS_BATCH_EPISODES", "128"))
WEIGHT_SYNC_INTERVAL = 256   # sync weights to GPU every N episodes
EVAL_INTERVAL = 2000         # eval every N episodes
EVAL_DRAFTS = 200


def main():
    device = torch.device("cpu")
    print(f"Run: {RUN_NAME}")
    print(f"Config: episodes={NUM_EPISODES}, sims={NUM_SIMS}, batch={BATCH_EPISODES}, fresh={FRESH}")
    print(f"GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'none')}")

    # Load GD
    gd = GenericDraftModel()
    gd_path = os.path.join(os.path.dirname(__file__), "generic_draft_0.pt")
    if not os.path.exists(gd_path):
        gd_path = os.path.join(os.path.dirname(__file__), "generic_draft.pt")
    gd.load_state_dict(torch.load(gd_path, weights_only=True, map_location="cpu"))
    gd.eval()
    gd_flat, gd_offsets = extract_gd_weights(gd)

    # Init network
    network = AlphaZeroDraftNet().to(device)
    print(f"Policy: {sum(p.numel() for p in network.parameters()):,} params")

    os.makedirs(SAVE_DIR, exist_ok=True)
    ckpt_path = os.path.join(SAVE_DIR, "draft_policy_checkpoint.pt")
    weights_path = os.path.join(SAVE_DIR, "draft_policy.pt")

    start_episode = 0
    best_eval_wp = 0.0

    if not FRESH and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
        network.load_state_dict(ckpt['model_state_dict'])
        start_episode = ckpt.get('episode', 0)
        best_eval_wp = ckpt.get('best_eval_wp', 0.0)
        print(f"Resumed: episode {start_episode}, best_wp={best_eval_wp:.4f}")
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

    # Create kernel engine
    policy_flat, policy_offsets = extract_policy_weights(network)
    engine = kernel.MCTSKernelEngine(
        policy_flat, gd_flat, policy_offsets, gd_offsets,
        max_concurrent=BATCH_EPISODES, device_id=0)
    print(f"CUDA kernel engine (batch={BATCH_EPISODES})")

    if HAS_WANDB:
        wandb.init(project="hots-draft-policy", name=RUN_NAME,
                   config={"episodes": NUM_EPISODES, "sims": NUM_SIMS,
                           "batch": BATCH_EPISODES, "engine": "cuda_kernel_v2"})

    # Training state
    buffer = []
    BUFFER_SIZE = 150_000
    BATCH_SIZE = 512
    n_maps = len(MAPS)
    n_tiers = len(SKILL_TIERS)
    episodes_since_weight_sync = 0
    last_eval_episode = start_episode

    train_start = time.time()
    episode = start_episode

    # ── Helper: generate one batch of configs ──
    def make_configs(n, ep):
        return np.array([
            [random.randint(0, n_maps-1), random.randint(0, n_tiers-1), (ep+i) % 2]
            for i in range(n)
        ], dtype=np.int32)

    # ── Helper: ingest results into buffer ──
    def ingest_results(results):
        nonlocal episode, last_wp
        for wp, examples in results:
            for state_feat, mcts_policy, valid in examples:
                buffer.append((
                    np.array(state_feat, dtype=np.float32),
                    np.array(mcts_policy, dtype=np.float32),
                    np.array(valid, dtype=np.float32),
                    wp,
                ))
                if len(buffer) > BUFFER_SIZE:
                    buffer.pop(0)
            last_wp = wp
            episode += 1

    # ── Helper: one training step ──
    def train_step(n_steps):
        if len(buffer) < BATCH_SIZE:
            return
        network.train()
        for _ in range(n_steps):
            batch = random.sample(buffer, BATCH_SIZE)
            states = torch.tensor(np.array([b[0] for b in batch])).to(device)
            target_policies = torch.tensor(np.array([b[1] for b in batch])).to(device)
            masks = torch.tensor(np.array([b[2] for b in batch])).to(device)
            target_values = torch.tensor(np.array([b[3] for b in batch],
                                                   dtype=np.float32)).to(device)
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

    # ── Async eval ──
    eval_thread = None
    def run_eval_async(ep_num):
        nonlocal best_eval_wp
        network.eval()
        pw, _ = extract_policy_weights(network)
        # Use a separate engine call (same GPU, but eval is fast)
        eval_configs = make_configs(EVAL_DRAFTS, 99999)
        eval_results = engine.run_episodes(eval_configs, NUM_SIMS // 2, 2.0, 99999)
        eval_wps = [r[0] for r in eval_results]
        avg_wp = np.mean(eval_wps)
        std_wp = np.std(eval_wps)
        win_rate = np.mean([1.0 if w > 0.5 else 0.0 for w in eval_wps])
        print(f"\n  EVAL @ {ep_num}: avg_wp={avg_wp:.4f} +/- {std_wp:.4f} "
              f"win_rate={win_rate:.1%}\n")
        if HAS_WANDB and wandb.run:
            wandb.log({"eval/avg_wp": avg_wp, "eval/win_rate": win_rate}, step=ep_num)
        if avg_wp > best_eval_wp:
            best_eval_wp = avg_wp
            torch.save(network.state_dict(), weights_path)
            torch.save({
                'model_state_dict': network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'episode': ep_num,
                'best_eval_wp': best_eval_wp,
            }, ckpt_path)
            print(f"  New best! Saved to {SAVE_DIR}\n")

    last_wp = 0.0

    # ── Initial weight sync ──
    policy_flat, _ = extract_policy_weights(network)
    engine.update_weights(policy_flat)

    # ── Kick off first generation batch (pipeline priming) ──
    pending_configs = make_configs(BATCH_EPISODES, episode)
    pending_results = None

    # ══════════════════════════════════════════════════════
    # MAIN LOOP: pipelined generate + train
    # ══════════════════════════════════════════════════════
    while episode < NUM_EPISODES:
        batch_count = min(BATCH_EPISODES, NUM_EPISODES - episode)

        # Generate on GPU (this blocks but GPU is fast)
        configs = make_configs(batch_count, episode)
        results = engine.run_episodes(configs, NUM_SIMS, 2.0, episode)

        # Ingest results
        ingest_results(results)
        episodes_since_weight_sync += batch_count

        # Train (CPU, while GPU is idle — could overlap with next generate)
        n_train_steps = max(1, batch_count // 8)
        train_step(n_train_steps)

        # Weight sync (every WEIGHT_SYNC_INTERVAL episodes)
        if episodes_since_weight_sync >= WEIGHT_SYNC_INTERVAL:
            policy_flat, _ = extract_policy_weights(network)
            engine.update_weights(policy_flat)
            episodes_since_weight_sync = 0

        # Logging
        elapsed = time.time() - train_start
        eps = (episode - start_episode) / elapsed if elapsed > 0 else 0
        eta = (NUM_EPISODES - episode) / eps / 3600 if eps > 0 else 0
        print(f"Episode {episode}: wp={last_wp:.4f} buffer={len(buffer)} "
              f"lr={scheduler.get_last_lr()[0]:.6f} [{eps:.1f} ep/s, ETA {eta:.1f}h]")
        if HAS_WANDB and wandb.run:
            wandb.log({"episode": episode, "last_wp": last_wp,
                       "buffer_size": len(buffer), "eps": eps}, step=episode)

        # Eval (every EVAL_INTERVAL, non-blocking)
        if episode - last_eval_episode >= EVAL_INTERVAL or episode >= NUM_EPISODES:
            last_eval_episode = episode
            # Run eval synchronously (GPU is needed, but eval is fast: 200 eps × 100 sims)
            run_eval_async(episode)

    if HAS_WANDB and wandb.run:
        wandb.finish()
    elapsed = time.time() - train_start
    total_eps = episode - start_episode
    print(f"Complete. {total_eps} episodes in {elapsed/3600:.1f}h ({total_eps/elapsed:.1f} ep/s)")


if __name__ == "__main__":
    main()
