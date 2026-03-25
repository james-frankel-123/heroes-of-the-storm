#!/usr/bin/env python3
"""
MCTS training worker with batched virtual-loss MCTS + direct GPU inference.

Each worker process:
1. Loads policy network on GPU (shared across workers via CUDA)
2. Loads WP model on CPU (for terminal evaluation)
3. Loads GD models on CPU (for opponent moves)
4. Runs batched MCTS episodes, submitting batch-32 leaf evals to GPU
5. Returns training examples to main process

Config via environment variables:
    CUDA_VISIBLE_DEVICES  — which GPU this run uses
    MCTS_SAVE_DIR         — checkpoint directory
    MCTS_NUM_WORKERS      — number of parallel workers
    MCTS_WP_MODEL         — "base", "enriched", or "augmented"
    MCTS_NUM_EPISODES     — total episodes to train
    MCTS_NUM_SIMS         — MCTS simulations per move (default 200)
    MCTS_BATCH_SIZE       — virtual loss batch size (default 32)
    MCTS_FRESH            — "1" for fresh start, "0" to resume
    WANDB_RUN_NAME        — W&B run name
"""
import os
import sys
import time
import random
import multiprocessing as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_draft_policy import (
    AlphaZeroDraftNet, DRAFT_ORDER,
    pretrain_value_head, bootstrap_from_generic_draft,
    STATE_DIM, NUM_HEROES, HEROES,
    _flatten_state_dict, _write_to_shared, _load_net_from_shared,
)
from mcts_batched import (
    DraftStateFast, state_from_strings, mcts_search_batched,
    simulate_episode_batched, NUM_HEROES as NH,
)
from train_generic_draft import GenericDraftModel
from shared import MAPS, SKILL_TIERS, load_replay_data, heroes_to_multi_hot

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# ── Config ──
SAVE_DIR = os.environ.get("MCTS_SAVE_DIR", os.path.dirname(__file__))
NUM_WORKERS = int(os.environ.get("MCTS_NUM_WORKERS", "16"))
WP_MODEL_TYPE = os.environ.get("MCTS_WP_MODEL", "augmented")
NUM_EPISODES = int(os.environ.get("MCTS_NUM_EPISODES", "300000"))
NUM_SIMS = int(os.environ.get("MCTS_NUM_SIMS", "200"))
MCTS_BATCH_SIZE = int(os.environ.get("MCTS_BATCH_SIZE", "32"))
FRESH = os.environ.get("MCTS_FRESH", "1") == "1"
RUN_NAME = os.environ.get("WANDB_RUN_NAME", "mcts_run")

# ── Worker state ──
_ws = {}


def _worker_init(net_sd, wp_sd, wp_type, wp_hidden, wp_enriched_config,
                 gd_sds, shared_net_flat, net_shapes, net_keys):
    """Worker process initializer. Loads all models once."""
    _ws['shared_net_flat'] = shared_net_flat
    _ws['net_shapes'] = net_shapes
    _ws['net_keys'] = net_keys

    # Policy network on GPU for batched inference
    gpu_available = torch.cuda.is_available()
    _ws['policy_device'] = torch.device('cuda:0') if gpu_available else torch.device('cpu')

    policy_net = AlphaZeroDraftNet().to(_ws['policy_device'])
    policy_net.eval()
    _ws['policy_net'] = policy_net

    # GD models on CPU
    gd_models = []
    for sd in gd_sds:
        gd = GenericDraftModel()
        gd.load_state_dict(sd)
        gd.eval()
        gd_models.append(gd)
    _ws['gd_models'] = gd_models

    # WP model on CPU (terminal evaluation)
    if wp_type == 'base':
        from train_win_probability import WinProbModel
        wp = WinProbModel()
    else:
        from sweep_enriched_wp import WinProbEnrichedModel
        wp = WinProbEnrichedModel(wp_enriched_config['input_dim'], wp_hidden, dropout=0.3)
    wp.load_state_dict(wp_sd)
    wp.eval()
    _ws['wp_model'] = wp
    _ws['wp_type'] = wp_type
    _ws['wp_enriched_config'] = wp_enriched_config


def _load_policy_weights():
    """Load latest policy weights from shared memory into GPU model."""
    shared = _ws['shared_net_flat']
    shapes = _ws['net_shapes']
    keys = _ws['net_keys']
    sd = {}
    offset = 0
    for key, shape in zip(keys, shapes):
        numel = 1
        for s in shape:
            numel *= s
        sd[key] = torch.tensor(shared[offset:offset + numel]).reshape(shape)
        offset += numel
    _ws['policy_net'].load_state_dict(sd)


def _batch_predict(states_np, masks_np):
    """Batched policy network inference on GPU. Returns (priors, values)."""
    net = _ws['policy_net']
    dev = _ws['policy_device']
    s_t = torch.from_numpy(states_np).float().to(dev)
    m_t = torch.from_numpy(masks_np).float().to(dev)
    with torch.no_grad():
        logits, values = net(s_t, m_t)
        priors = F.softmax(logits, dim=1).cpu().numpy()
        vals = values.cpu().numpy().flatten()
    return priors, vals


def _gd_predict(state_np, mask_np):
    """Single GD forward pass on CPU for opponent moves."""
    gd = random.choice(_ws['gd_models'])
    s_t = torch.from_numpy(state_np).float().unsqueeze(0)
    m_t = torch.from_numpy(mask_np).float().unsqueeze(0)
    with torch.no_grad():
        return gd(s_t, m_t).squeeze(0).numpy()


def _wp_eval(t0_heroes, t1_heroes, game_map, skill_tier):
    """Symmetrized WP evaluation on CPU."""
    wp = _ws['wp_model']
    wp_type = _ws['wp_type']
    wp_cfg = _ws['wp_enriched_config']

    def _run(t0h, t1h):
        if wp_type != 'base' and wp_cfg:
            from sweep_enriched_wp import extract_features, FEATURE_GROUPS, compute_group_indices
            d = {'team0_heroes': t0h, 'team1_heroes': t1h,
                 'game_map': game_map, 'skill_tier': skill_tier, 'winner': 0}
            all_mask = [True] * len(FEATURE_GROUPS)
            base, enriched = extract_features(d, wp_cfg.get('_stats_cache_obj'), all_mask)
            cols = []
            gi = compute_group_indices()
            for g in wp_cfg['groups']:
                s, e = gi[g]
                cols.extend(range(s, e))
            x_np = np.concatenate([base, enriched[cols]])
            x = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0)
        else:
            from shared import heroes_to_multi_hot, map_to_one_hot, tier_to_one_hot
            t0 = torch.tensor(heroes_to_multi_hot(t0h), dtype=torch.float32).unsqueeze(0)
            t1 = torch.tensor(heroes_to_multi_hot(t1h), dtype=torch.float32).unsqueeze(0)
            m = torch.tensor(map_to_one_hot(game_map), dtype=torch.float32).unsqueeze(0)
            t = torch.tensor(tier_to_one_hot(skill_tier), dtype=torch.float32).unsqueeze(0)
            x = torch.cat([t0, t1, m, t], dim=1)
        with torch.no_grad():
            return wp(x).item()

    # Symmetrized
    wp_normal = _run(t0_heroes, t1_heroes)
    wp_swapped = _run(t1_heroes, t0_heroes)
    return (wp_normal + (1.0 - wp_swapped)) / 2.0


def _run_episode(args):
    """Worker entry point. Loads weights, runs one episode."""
    game_map, skill_tier, num_sims, batch_size = args
    _load_policy_weights()
    our_team = random.randint(0, 1)
    return simulate_episode_batched(
        game_map, skill_tier, our_team,
        _batch_predict, _gd_predict, _wp_eval,
        num_simulations=num_sims, batch_size=batch_size,
    )


# ── Main Training Loop ──

def main():
    mp.set_start_method('spawn', force=True)
    device = torch.device('cpu')
    print(f"Run: {RUN_NAME}")
    print(f"Config: workers={NUM_WORKERS}, wp={WP_MODEL_TYPE}, episodes={NUM_EPISODES}, "
          f"sims={NUM_SIMS}, batch={MCTS_BATCH_SIZE}, fresh={FRESH}")
    print(f"GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'none')}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # ── Load models ──
    wp_enriched_config = None
    wp_hidden = [256, 128]
    if WP_MODEL_TYPE == 'base':
        from train_win_probability import WinProbModel
        wp_path = os.path.join(os.path.dirname(__file__), "win_probability.pt")
        wp_model = WinProbModel()
        wp_model.load_state_dict(torch.load(wp_path, weights_only=True, map_location='cpu'))
        wp_model.eval()
    else:
        from sweep_enriched_wp import WinProbEnrichedModel, StatsCache, compute_group_indices, FEATURE_GROUP_DIMS
        WP_GROUPS = ['role_counts', 'team_avg_wr', 'map_delta',
                     'pairwise_counters', 'pairwise_synergies', 'counter_detail',
                     'meta_strength', 'draft_diversity', 'comp_wr']
        enriched_dim = sum(FEATURE_GROUP_DIMS[g] for g in WP_GROUPS)
        wp_input_dim = 197 + enriched_dim
        wp_hidden = [512, 256, 128] if WP_MODEL_TYPE == 'augmented' else [256, 128]
        wp_path = os.path.join(os.path.dirname(__file__),
                               "wp_enriched_winner.pt" if WP_MODEL_TYPE == 'augmented'
                               else "wp_experiment_enriched.pt")
        wp_model = WinProbEnrichedModel(wp_input_dim, wp_hidden, dropout=0.3)
        wp_model.load_state_dict(torch.load(wp_path, weights_only=True, map_location='cpu'))
        wp_model.eval()
        _stats = StatsCache()
        wp_enriched_config = {
            'input_dim': wp_input_dim, 'groups': WP_GROUPS,
            '_stats_cache_obj': _stats,
        }
    print(f"WP model: {WP_MODEL_TYPE} ({wp_path})")

    gd_models = []
    for i in range(10):
        p = os.path.join(os.path.dirname(__file__), f"generic_draft_{i}.pt")
        if not os.path.exists(p):
            break
        gd = GenericDraftModel()
        gd.load_state_dict(torch.load(p, weights_only=True, map_location='cpu'))
        gd.eval()
        gd_models.append(gd)
    if not gd_models:
        p = os.path.join(os.path.dirname(__file__), "generic_draft.pt")
        gd = GenericDraftModel()
        gd.load_state_dict(torch.load(p, weights_only=True, map_location='cpu'))
        gd.eval()
        gd_models.append(gd)
    print(f"GD models: {len(gd_models)}")

    # ── Network ──
    network = AlphaZeroDraftNet().to(device)
    print(f"Policy network: {sum(p.numel() for p in network.parameters()):,} params")

    start_episode = 0
    best_eval_wp = 0.0
    os.makedirs(SAVE_DIR, exist_ok=True)
    ckpt_path = os.path.join(SAVE_DIR, "draft_policy_checkpoint.pt")
    weights_path = os.path.join(SAVE_DIR, "draft_policy.pt")

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

    # ── Shared memory for policy weights ──
    net_sd = {k: v.cpu() for k, v in network.state_dict().items()}
    flat, net_shapes, net_keys = _flatten_state_dict(net_sd)
    shared_net_flat = mp.Array('f', flat.numpy(), lock=False)
    _write_to_shared(shared_net_flat, net_sd, net_keys)
    print(f"Shared memory: {len(flat)*4/1024/1024:.1f} MB")

    # ── Worker pool ──
    wp_sd = wp_model.state_dict()
    gd_sds = [g.state_dict() for g in gd_models]

    # Serialize StatsCache for workers if using enriched WP
    wp_cfg_for_workers = None
    if wp_enriched_config is not None:
        _stats = wp_enriched_config['_stats_cache_obj']
        wp_cfg_for_workers = {
            'input_dim': wp_enriched_config['input_dim'],
            'groups': wp_enriched_config['groups'],
            '_stats_cache_obj': _stats,  # StatsCache is picklable
        }

    print(f"Creating pool with {NUM_WORKERS} workers...")
    pool = mp.Pool(
        NUM_WORKERS,
        initializer=_worker_init,
        initargs=(net_sd, wp_sd, WP_MODEL_TYPE, wp_hidden, wp_cfg_for_workers,
                  gd_sds, shared_net_flat, net_shapes, net_keys),
    )
    print("Pool ready.")

    # ── W&B ──
    if HAS_WANDB:
        wandb.init(project="hots-draft-policy", name=RUN_NAME,
                   config={"wp_model": WP_MODEL_TYPE, "workers": NUM_WORKERS,
                           "episodes": NUM_EPISODES, "sims": NUM_SIMS,
                           "batch_size": MCTS_BATCH_SIZE})

    # ── Training loop ──
    buffer = []
    BUFFER_SIZE = 150_000
    BATCH_SIZE = 512
    EVAL_EVERY = 500
    EVAL_DRAFTS = 200

    train_start = time.time()
    episode = start_episode

    while episode < NUM_EPISODES:
        batch_count = min(NUM_WORKERS, NUM_EPISODES - episode)

        # Sync weights to shared memory
        net_sd = {k: v.cpu() for k, v in network.state_dict().items()}
        _write_to_shared(shared_net_flat, net_sd, net_keys)

        # Generate episodes
        worker_args = [
            (random.choice(MAPS), random.choice(SKILL_TIERS), NUM_SIMS, MCTS_BATCH_SIZE)
            for _ in range(batch_count)
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
            for _ in range(max(1, batch_count // 2)):
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
                           "buffer_size": len(buffer), "eps": eps}, step=episode)

        # Eval
        if episode % EVAL_EVERY < NUM_WORKERS or episode >= NUM_EPISODES:
            network.eval()
            net_sd = {k: v.cpu() for k, v in network.state_dict().items()}
            _write_to_shared(shared_net_flat, net_sd, net_keys)

            eval_args = [
                (random.choice(MAPS), random.choice(SKILL_TIERS), NUM_SIMS // 2, MCTS_BATCH_SIZE)
                for _ in range(EVAL_DRAFTS)
            ]
            eval_results = pool.map(_run_episode, eval_args)
            eval_wps = [r[0] for r in eval_results]
            avg_wp = np.mean(eval_wps)
            std_wp = np.std(eval_wps)
            win_rate = np.mean([1.0 if w > 0.5 else 0.0 for w in eval_wps])
            print(f"\n  EVAL @ {episode}: avg_wp={avg_wp:.4f} +/- {std_wp:.4f} "
                  f"win_rate={win_rate:.1%}\n")

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
                print(f"  New best! Saved to {SAVE_DIR}\n")

    pool.close()
    pool.join()
    if HAS_WANDB and wandb.run:
        wandb.finish()
    elapsed = time.time() - train_start
    print(f"Training complete. {episode - start_episode} episodes in {elapsed/3600:.1f}h "
          f"({(episode - start_episode)/elapsed:.1f} ep/s)")


if __name__ == "__main__":
    main()
