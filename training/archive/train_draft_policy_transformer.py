"""
Draft Policy — Transformer variant.

Replaces the ResNet backbone with a transformer that treats each
picked/banned hero as a token with learned embeddings. Self-attention
learns hero-hero interactions natively.

Uses the same MCTS training loop as train_draft_policy.py.

Quick 10K episode test to compare learning curves.

Usage:
    set -a && source .env && set +a
    python3 -u training/train_draft_policy_transformer.py
"""
import os
import sys
import math
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

sys.path.insert(0, os.path.dirname(__file__))
from shared import (
    NUM_HEROES, NUM_MAPS, NUM_TIERS, HEROES, MAPS, SKILL_TIERS,
    heroes_to_multi_hot, map_to_one_hot, tier_to_one_hot,
    load_replay_data, split_data, embed_onnx_weights,
)

# Import MCTS machinery from existing training
from train_draft_policy import (
    DraftState, DRAFT_ORDER, MCTSNode,
    _evaluate_wp, load_pretrained_models,
    simulate_draft_with_mcts,
)

STATE_DIM = NUM_HEROES * 3 + NUM_MAPS + NUM_TIERS + 2 + 1  # 290


class TransformerDraftNet(nn.Module):
    """
    Transformer-based draft policy + value network.

    Instead of a flat 290-dim input, we tokenize the draft state:
    - Each picked/banned hero → learned hero embedding
    - Token types: team0_pick, team1_pick, ban, map, tier, step_info
    - Self-attention learns hero-hero interactions

    But we still accept the same flat input for compatibility with MCTS
    infrastructure. We decode the multi-hot vectors back to token sequences
    internally.

    For MCTS compatibility, forward(x, mask) interface is preserved.
    """
    def __init__(self, d_model=256, nhead=8, num_layers=6, dropout=0.2):
        super().__init__()
        self.d_model = d_model

        # Token embeddings
        # 90 heroes + special tokens for map(14) + tier(3) + step_info(2) + our_team(1) + CLS
        self.hero_embed = nn.Embedding(NUM_HEROES, d_model)
        self.map_embed = nn.Embedding(NUM_MAPS, d_model)
        self.tier_embed = nn.Embedding(NUM_TIERS, d_model)

        # Token type embeddings: 0=CLS, 1=team0_pick, 2=team1_pick, 3=ban, 4=context
        self.type_embed = nn.Embedding(5, d_model)

        # Continuous features projected to d_model
        self.step_proj = nn.Linear(3, d_model)  # step_norm, step_type, our_team

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Policy head: from CLS embedding → hero logits
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, NUM_HEROES),
        )

        # Value head: from CLS embedding → win probability
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
        )

    def _decode_flat_input(self, x):
        """Convert flat 290-dim input back to token sequences.
        Returns (hero_indices, type_ids, map_idx, tier_idx, step_features) per batch item.
        """
        B = x.shape[0]
        t0 = x[:, :NUM_HEROES]           # team0 picks multi-hot
        t1 = x[:, NUM_HEROES:2*NUM_HEROES]  # team1 picks
        bans = x[:, 2*NUM_HEROES:3*NUM_HEROES]  # bans
        map_oh = x[:, 3*NUM_HEROES:3*NUM_HEROES+NUM_MAPS]
        tier_oh = x[:, 3*NUM_HEROES+NUM_MAPS:3*NUM_HEROES+NUM_MAPS+NUM_TIERS]
        step_feats = x[:, 3*NUM_HEROES+NUM_MAPS+NUM_TIERS:]  # step_norm, step_type, our_team

        return t0, t1, bans, map_oh, tier_oh, step_feats

    def forward(self, x, mask=None):
        B = x.shape[0]
        t0, t1, bans, map_oh, tier_oh, step_feats = self._decode_flat_input(x)

        # Build token sequences dynamically
        # Start with CLS token
        tokens = [self.cls_token.expand(B, -1, -1)]
        type_ids = [torch.zeros(B, 1, dtype=torch.long, device=x.device)]  # 0 = CLS

        # Team 0 picks
        for i in range(NUM_HEROES):
            active = t0[:, i] > 0.5  # (B,)
            if active.any():
                emb = self.hero_embed(torch.tensor(i, device=x.device)).unsqueeze(0).expand(B, -1)
                # Zero out inactive batch items
                emb = emb * active.float().unsqueeze(-1)
                tokens.append(emb.unsqueeze(1))
                type_ids.append(torch.ones(B, 1, dtype=torch.long, device=x.device))  # 1 = team0

        # Team 1 picks
        for i in range(NUM_HEROES):
            active = t1[:, i] > 0.5
            if active.any():
                emb = self.hero_embed(torch.tensor(i, device=x.device)).unsqueeze(0).expand(B, -1)
                emb = emb * active.float().unsqueeze(-1)
                tokens.append(emb.unsqueeze(1))
                type_ids.append(2 * torch.ones(B, 1, dtype=torch.long, device=x.device))  # 2 = team1

        # Bans
        for i in range(NUM_HEROES):
            active = bans[:, i] > 0.5
            if active.any():
                emb = self.hero_embed(torch.tensor(i, device=x.device)).unsqueeze(0).expand(B, -1)
                emb = emb * active.float().unsqueeze(-1)
                tokens.append(emb.unsqueeze(1))
                type_ids.append(3 * torch.ones(B, 1, dtype=torch.long, device=x.device))  # 3 = ban

        # Map token
        map_idx = map_oh.argmax(dim=-1)  # (B,)
        map_emb = self.map_embed(map_idx).unsqueeze(1)
        tokens.append(map_emb)
        type_ids.append(4 * torch.ones(B, 1, dtype=torch.long, device=x.device))  # 4 = context

        # Tier token
        tier_idx = tier_oh.argmax(dim=-1)
        tier_emb = self.tier_embed(tier_idx).unsqueeze(1)
        tokens.append(tier_emb)
        type_ids.append(4 * torch.ones(B, 1, dtype=torch.long, device=x.device))

        # Step info token
        step_emb = self.step_proj(step_feats).unsqueeze(1)
        tokens.append(step_emb)
        type_ids.append(4 * torch.ones(B, 1, dtype=torch.long, device=x.device))

        # Concatenate all tokens
        h = torch.cat(tokens, dim=1)  # (B, T, d_model)
        type_tensor = torch.cat(type_ids, dim=1)  # (B, T)

        # Add type embeddings
        h = h + self.type_embed(type_tensor)

        # Transformer encode
        h = self.encoder(h)
        h = self.norm(h)

        # CLS token output
        cls = h[:, 0]  # (B, d_model)

        # Policy head
        policy_logits = self.policy_head(cls)
        if mask is not None:
            policy_logits = policy_logits + (1 - mask) * (-1e9)

        # Value head (tanh → [0, 1])
        value = torch.tanh(self.value_head(cls)) * 0.5 + 0.5

        return policy_logits, value.squeeze(-1)

    def predict(self, state: DraftState, device) -> tuple:
        """Get policy priors and value for a single state (MCTS compatibility)."""
        self.eval()
        x = state.to_tensor(device)
        m = state.valid_mask(device)
        with torch.no_grad():
            logits, value = self(x, m)
            priors = F.softmax(logits, dim=1).cpu().numpy()[0]
        return priors, value.item()


# Override _run_episode to use TransformerDraftNet
def _run_episode_transformer(args):
    """Worker function using transformer network."""
    net_state_dict, wp_state_dict, gd_state_dicts, game_map, skill_tier, num_sims = args
    device = torch.device('cpu')

    from train_generic_draft import GenericDraftModel
    from train_win_probability import WinProbModel

    network = TransformerDraftNet(d_model=256, nhead=8, num_layers=6)
    network.load_state_dict(net_state_dict)
    network.eval()

    wp_model = WinProbModel()
    wp_model.load_state_dict(wp_state_dict)
    wp_model.eval()

    gd_models = []
    for sd in gd_state_dicts:
        gd = GenericDraftModel()
        gd.load_state_dict(sd)
        gd.eval()
        gd_models.append(gd)

    our_team = random.randint(0, 1)

    win_prob, examples = simulate_draft_with_mcts(
        network, wp_model, gd_models, game_map, skill_tier,
        device, our_team=our_team, num_simulations=num_sims,
    )
    return win_prob, examples


def train():
    # Use GPU for pre-training and gradient updates, CPU workers for MCTS
    gpu_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cpu_device = torch.device("cpu")
    print(f"GPU: {gpu_device}, MCTS workers: CPU")

    NUM_WORKERS = min(58, os.cpu_count() or 1)
    print(f"Workers: {NUM_WORKERS}")

    print("Loading pre-trained models...")
    wp_model, gd_models = load_pretrained_models(cpu_device)

    network = TransformerDraftNet(d_model=256, nhead=8, num_layers=6).to(gpu_device)
    params = sum(p.numel() for p in network.parameters())
    print(f"Transformer params: {params:,}")

    # Pre-train value head on GPU
    print("Pre-training value head on GPU...")
    data = load_replay_data()
    if len(data) > 100:
        train_data, _ = split_data(data)
        X_list, y_list = [], []
        for d in train_data:
            t0 = heroes_to_multi_hot(d["team0_heroes"])
            t1 = heroes_to_multi_hot(d["team1_heroes"])
            ban_vec = np.zeros(NUM_HEROES, dtype=np.float32)
            for h in d.get("team0_bans", []) + d.get("team1_bans", []):
                from shared import HERO_TO_IDX
                idx = HERO_TO_IDX.get(h)
                if idx is not None:
                    ban_vec[idx] = 1.0
            m = map_to_one_hot(d["game_map"])
            t = tier_to_one_hot(d["skill_tier"])
            team0_won = float(d["winner"] == 0)
            x0 = np.concatenate([t0, t1, ban_vec, m, t, [1.0, 1.0, 0.0]])
            X_list.append(x0)
            y_list.append(team0_won)
            x1 = np.concatenate([t0, t1, ban_vec, m, t, [1.0, 1.0, 1.0]])
            X_list.append(x1)
            y_list.append(1.0 - team0_won)

        X = torch.tensor(np.array(X_list[:100000], dtype=np.float32)).to(gpu_device)
        y = torch.tensor(np.array(y_list[:100000], dtype=np.float32)).to(gpu_device)

        value_params = list(network.value_head.parameters())
        opt = torch.optim.Adam(value_params, lr=1e-3)
        network.train()
        for epoch in range(10):
            perm = torch.randperm(len(X), device=gpu_device)
            total_loss = 0
            n = 0
            for i in range(0, len(X), 1024):
                idx = perm[i:i+1024]
                bx, by = X[idx], y[idx]
                _, vp = network(bx)
                loss = F.mse_loss(vp, by)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()
                n += 1
            if (epoch + 1) % 5 == 0:
                print(f"  Value pre-train epoch {epoch+1}: loss={total_loss/n:.4f}")

    # Move to CPU for state dict serialization to workers
    network = network.to(cpu_device)

    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5000, eta_min=1e-5)

    buffer = []
    BUFFER_SIZE = 75_000
    BATCH_SIZE = 256
    NUM_EPISODES = 10_000
    MCTS_SIMULATIONS = 50  # reduced for speed (transformer is 6x slower per call)
    EVAL_EVERY = 500
    EVAL_DRAFTS = 100
    PARALLEL_BATCH = NUM_WORKERS

    best_eval_wp = 0.0
    save_dir = os.path.dirname(__file__)

    wp_sd = wp_model.state_dict()
    gd_sds = [gd.state_dict() for gd in gd_models]

    # Network stays on CPU between episodes for state dict serialization
    # Moves to GPU only for gradient updates
    print(f"Training transformer for {NUM_EPISODES} episodes (MCTS on CPU, gradients on {gpu_device})...")

    if HAS_WANDB:
        wandb.init(
            project="hots-draft-policy",
            name="transformer-10k",
            config={
                "architecture": "transformer",
                "d_model": 256,
                "nhead": 8,
                "num_layers": 6,
                "num_episodes": NUM_EPISODES,
                "params": params,
            },
        )
        print(f"wandb run: {wandb.run.url}")

    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    train_start = time.time()
    episode = 0
    while episode < NUM_EPISODES:
        batch_size = min(PARALLEL_BATCH, NUM_EPISODES - episode)
        net_sd = network.state_dict()

        worker_args = [
            (net_sd, wp_sd, gd_sds,
             random.choice(MAPS), random.choice(SKILL_TIERS),
             MCTS_SIMULATIONS)
            for _ in range(batch_size)
        ]

        with mp.Pool(NUM_WORKERS) as pool:
            results = pool.map(_run_episode_transformer, worker_args)

        last_wp = 0.0
        for win_prob, examples in results:
            for state_feat, mcts_policy, valid in examples:
                buffer.append((state_feat, mcts_policy, valid, win_prob))
                if len(buffer) > BUFFER_SIZE:
                    buffer.pop(0)
            last_wp = win_prob
            episode += 1

        if len(buffer) >= BATCH_SIZE:
            # Move to GPU for gradient updates
            network = network.to(gpu_device)
            network.train()
            num_train_steps = max(1, batch_size // 2)
            for _ in range(num_train_steps):
                batch = random.sample(buffer, BATCH_SIZE)
                states = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32).to(gpu_device)
                target_policies = torch.tensor(np.array([b[1] for b in batch]), dtype=torch.float32).to(gpu_device)
                masks = torch.tensor(np.array([b[2] for b in batch]), dtype=torch.float32).to(gpu_device)
                target_values = torch.tensor(np.array([b[3] for b in batch], dtype=np.float32)).to(gpu_device)

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
            # Move back to CPU for next round of worker serialization
            network = network.to(cpu_device)

        if episode % 58 < PARALLEL_BATCH or episode >= NUM_EPISODES:
            elapsed = time.time() - train_start
            eps_per_sec = episode / elapsed if elapsed > 0 else 0
            eta_h = (NUM_EPISODES - episode) / eps_per_sec / 3600 if eps_per_sec > 0 else 0
            print(f"Episode {episode}: wp={last_wp:.4f} buffer={len(buffer)} "
                  f"lr={scheduler.get_last_lr()[0]:.6f} "
                  f"[{eps_per_sec:.1f} ep/s, ETA {eta_h:.1f}h]")

            if HAS_WANDB and wandb.run:
                wandb.log({
                    "episode": episode,
                    "last_wp": last_wp,
                    "buffer_size": len(buffer),
                    "lr": scheduler.get_last_lr()[0],
                    "episodes_per_sec": eps_per_sec,
                }, step=episode)

        if episode % EVAL_EVERY < PARALLEL_BATCH or episode >= NUM_EPISODES:
            network.eval()
            net_sd_eval = network.state_dict()
            eval_args = [
                (net_sd_eval, wp_sd, gd_sds,
                 random.choice(MAPS), random.choice(SKILL_TIERS),
                 MCTS_SIMULATIONS // 2)
                for _ in range(EVAL_DRAFTS)
            ]
            with mp.Pool(NUM_WORKERS) as pool:
                eval_results = pool.map(_run_episode_transformer, eval_args)
            eval_wps = [r[0] for r in eval_results]
            avg_wp = np.mean(eval_wps)
            std_wp = np.std(eval_wps)
            win_rate = np.mean([1.0 if w > 0.5 else 0.0 for w in eval_wps])
            print(f"\n  EVAL @ {episode}: avg_wp={avg_wp:.4f} +/- {std_wp:.4f} "
                  f"win_rate={win_rate:.1%} (vs 5 opponents)")

            if HAS_WANDB and wandb.run:
                wandb.log({
                    "eval/avg_wp": avg_wp,
                    "eval/std_wp": std_wp,
                    "eval/win_rate": win_rate,
                    "eval/best_wp": max(best_eval_wp, avg_wp),
                }, step=episode)

            if avg_wp > best_eval_wp:
                best_eval_wp = avg_wp
                torch.save(network.state_dict(), os.path.join(save_dir, "draft_policy_transformer.pt"))
                print(f"  New best! Saved draft_policy_transformer.pt")
            print()

    elapsed = time.time() - train_start
    print(f"Done. Best eval WP: {best_eval_wp:.4f}, time: {elapsed/3600:.1f}h")
    if HAS_WANDB and wandb.run:
        wandb.log({"final/best_eval_wp": best_eval_wp})
        wandb.finish()


if __name__ == "__main__":
    train()
