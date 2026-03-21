"""
Win Probability Model — predicts P(team0 wins) given both team compositions + map + tier.

Input: team0_heroes (90) + team1_heroes (90) + map (14) + tier (3) = 197 features
Output: sigmoid probability of team 0 winning

Usage:
    export DATABASE_URL=...
    python training/train_win_probability.py           # train current model
    python training/train_win_probability.py --sweep    # full embedding sweep (320 variants)
    python training/train_win_probability.py --sweep --variants 20  # quick test
"""
import os
import sys
import csv
import json
import itertools
import argparse
import filelock
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from shared import (
    NUM_HEROES, NUM_MAPS, NUM_TIERS, HERO_TO_IDX,
    heroes_to_multi_hot, map_to_one_hot, tier_to_one_hot,
    load_replay_data, split_data, embed_onnx_weights,
    optimize_onnx, quantize_onnx, verify_quantized_model,
)

INPUT_DIM = NUM_HEROES * 2 + NUM_MAPS + NUM_TIERS  # 90+90+14+3 = 197


# ── Datasets ─────────────────────────────────────────────────────────

class WinProbDataset(Dataset):
    """Original multi-hot dataset (no augmentation)."""
    def __init__(self, data: list[dict]):
        self.X = []
        self.y = []
        for d in data:
            t0 = heroes_to_multi_hot(d["team0_heroes"])
            t1 = heroes_to_multi_hot(d["team1_heroes"])
            m = map_to_one_hot(d["game_map"])
            t = tier_to_one_hot(d["skill_tier"])
            x = np.concatenate([t0, t1, m, t])
            self.X.append(x)
            self.y.append(float(d["winner"] == 0))
        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])


class WinProbAugDataset(Dataset):
    """Multi-hot dataset with team swap augmentation (doubles data)."""
    def __init__(self, data: list[dict]):
        self.X = []
        self.y = []
        for d in data:
            t0 = heroes_to_multi_hot(d["team0_heroes"])
            t1 = heroes_to_multi_hot(d["team1_heroes"])
            m = map_to_one_hot(d["game_map"])
            t = tier_to_one_hot(d["skill_tier"])
            # Original
            self.X.append(np.concatenate([t0, t1, m, t]))
            self.y.append(float(d["winner"] == 0))
            # Swapped teams, flipped label
            self.X.append(np.concatenate([t1, t0, m, t]))
            self.y.append(float(d["winner"] == 1))
        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])


class EmbeddingWPDataset(Dataset):
    """Dataset for embedding models: stores hero indices + map/tier + label.
    Includes team swap augmentation."""
    def __init__(self, data: list[dict]):
        self.samples = []
        for d in data:
            t0_idx = [HERO_TO_IDX[h] for h in d["team0_heroes"] if h in HERO_TO_IDX]
            t1_idx = [HERO_TO_IDX[h] for h in d["team1_heroes"] if h in HERO_TO_IDX]
            m = map_to_one_hot(d["game_map"])
            t = tier_to_one_hot(d["skill_tier"])
            ctx = np.concatenate([m, t]).astype(np.float32)
            y0 = float(d["winner"] == 0)
            # Original + swapped
            self.samples.append((t0_idx, t1_idx, ctx, y0))
            self.samples.append((t1_idx, t0_idx, ctx, 1.0 - y0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        t0_idx, t1_idx, ctx, y = self.samples[idx]
        # Pad to 5 heroes (some games might have fewer due to data issues)
        t0 = torch.zeros(5, dtype=torch.long)
        t1 = torch.zeros(5, dtype=torch.long)
        for i, h in enumerate(t0_idx[:5]):
            t0[i] = h
        for i, h in enumerate(t1_idx[:5]):
            t1[i] = h
        t0_mask = torch.zeros(5, dtype=torch.float32)
        t1_mask = torch.zeros(5, dtype=torch.float32)
        t0_mask[:len(t0_idx[:5])] = 1.0
        t1_mask[:len(t1_idx[:5])] = 1.0
        return t0, t1, t0_mask, t1_mask, torch.from_numpy(ctx), torch.tensor(y)


def collate_embedding(batch):
    t0s, t1s, t0ms, t1ms, ctxs, ys = zip(*batch)
    return (torch.stack(t0s), torch.stack(t1s), torch.stack(t0ms), torch.stack(t1ms),
            torch.stack(ctxs), torch.stack(ys))


# ── Models ────────────────────────────────────────────────────────────

class WinProbModel(nn.Module):
    """Original multi-hot MLP (unchanged)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class WinProbEmbeddingModel(nn.Module):
    """Embedding-based WP model with configurable interaction features."""

    def __init__(self, embed_dim=16, interaction_mode="concat_product",
                 hidden_dims=None, dropout=0.2):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.embed_dim = embed_dim
        self.interaction_mode = interaction_mode
        self.hero_embed = nn.Embedding(NUM_HEROES, embed_dim)

        # Attention layer for full_attention mode
        if interaction_mode == "full_attention":
            self.attn_query = nn.Linear(embed_dim, embed_dim)
            self.attn_key = nn.Linear(embed_dim, embed_dim)

        # Compute feature dimension based on interaction mode
        e = embed_dim
        if interaction_mode == "concat":
            feat_dim = 2 * e
        elif interaction_mode == "product":
            feat_dim = e
        elif interaction_mode == "concat_product":
            feat_dim = 3 * e
        elif interaction_mode == "full":
            feat_dim = 5 * e  # t0_sum, t1_sum, product, t0_pair, t1_pair
        elif interaction_mode == "full_attention":
            feat_dim = 5 * e
        else:
            raise ValueError(f"Unknown interaction_mode: {interaction_mode}")

        feat_dim += NUM_MAPS + NUM_TIERS  # append map + tier

        # Build MLP
        layers = []
        in_dim = feat_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        layers.extend([nn.Linear(in_dim, 1), nn.Sigmoid()])
        self.mlp = nn.Sequential(*layers)

    def _aggregate_team(self, hero_indices, hero_mask):
        """Embed heroes and aggregate per team.
        hero_indices: (B, 5) long tensor
        hero_mask: (B, 5) float tensor (1 where hero exists, 0 for padding)
        Returns: (sum_embed, individual_embeds, mask)
        """
        embeds = self.hero_embed(hero_indices)  # (B, 5, E)
        masked = embeds * hero_mask.unsqueeze(-1)  # zero out padding
        return masked.sum(dim=1), embeds, hero_mask  # (B, E), (B, 5, E), (B, 5)

    def _attention_aggregate(self, embeds, mask):
        """Single-head dot-product attention aggregation.
        embeds: (B, 5, E), mask: (B, 5)
        Returns: (B, E)
        """
        Q = self.attn_query(embeds)  # (B, 5, E)
        K = self.attn_key(embeds)    # (B, 5, E)
        # Dot product attention scores
        scores = (Q * K).sum(dim=-1) / (self.embed_dim ** 0.5)  # (B, 5)
        # Mask out padding
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # (B, 5, 1)
        return (embeds * weights).sum(dim=1)  # (B, E)

    def _pairwise_interaction(self, sum_embed, embeds, mask):
        """Factorization machine pairwise trick:
        (sum² - sum_of_squares) / 2
        """
        masked = embeds * mask.unsqueeze(-1)
        sq_of_sum = sum_embed ** 2
        sum_of_sq = (masked ** 2).sum(dim=1)
        return (sq_of_sum - sum_of_sq) / 2  # (B, E)

    def forward(self, t0_idx, t1_idx, t0_mask, t1_mask, ctx):
        """
        t0_idx, t1_idx: (B, 5) hero indices
        t0_mask, t1_mask: (B, 5) masks
        ctx: (B, 17) map one-hot + tier one-hot
        """
        t0_sum, t0_embeds, t0_m = self._aggregate_team(t0_idx, t0_mask)
        t1_sum, t1_embeds, t1_m = self._aggregate_team(t1_idx, t1_mask)

        if self.interaction_mode == "full_attention":
            t0_agg = self._attention_aggregate(t0_embeds, t0_m)
            t1_agg = self._attention_aggregate(t1_embeds, t1_m)
            product = t0_agg * t1_agg
            t0_pair = self._pairwise_interaction(t0_agg, t0_embeds, t0_m)
            t1_pair = self._pairwise_interaction(t1_agg, t1_embeds, t1_m)
            features = torch.cat([t0_agg, t1_agg, product, t0_pair, t1_pair, ctx], dim=1)
        elif self.interaction_mode == "concat":
            features = torch.cat([t0_sum, t1_sum, ctx], dim=1)
        elif self.interaction_mode == "product":
            features = torch.cat([t0_sum * t1_sum, ctx], dim=1)
        elif self.interaction_mode == "concat_product":
            features = torch.cat([t0_sum, t1_sum, t0_sum * t1_sum, ctx], dim=1)
        elif self.interaction_mode == "full":
            product = t0_sum * t1_sum
            t0_pair = self._pairwise_interaction(t0_sum, t0_embeds, t0_m)
            t1_pair = self._pairwise_interaction(t1_sum, t1_embeds, t1_m)
            features = torch.cat([t0_sum, t1_sum, product, t0_pair, t1_pair, ctx], dim=1)
        else:
            raise ValueError(f"Unknown: {self.interaction_mode}")

        return self.mlp(features).squeeze(-1)


# ── Training ──────────────────────────────────────────────────────────

def train_original(data=None):
    """Train the original WinProbModel."""
    print("Loading data...")
    if data is None:
        data = load_replay_data()
    print(f"Loaded {len(data)} replays")

    if len(data) < 100:
        print("Not enough data.")
        return

    train_data, test_data = split_data(data)
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")

    train_ds = WinProbDataset(train_data)
    test_ds = WinProbDataset(test_data)
    batch_size = 512 if len(train_data) > 100_000 else 256
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = WinProbModel().to(device)
    print(f"WP params: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.BCELoss()

    best_test_loss = float("inf")
    patience = 20
    patience_counter = 0

    for epoch in range(200):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        for X, y in train_dl:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y)
            train_correct += ((pred > 0.5).float() == y).sum().item()
            train_total += len(y)

        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for X, y in test_dl:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                loss = criterion(pred, y)
                test_loss += loss.item() * len(y)
                test_correct += ((pred > 0.5).float() == y).sum().item()
                test_total += len(y)

        train_acc = train_correct / train_total * 100
        test_acc = test_correct / test_total * 100
        avg_test_loss = test_loss / test_total

        if (epoch + 1) % 5 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:3d}: train_loss={train_loss/train_total:.4f} "
                  f"train_acc={train_acc:.1f}% test_loss={avg_test_loss:.4f} test_acc={test_acc:.1f}% lr={lr:.6f}")

        lr_scheduler.step(avg_test_loss)

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), "win_probability.pt"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Export to ONNX
    model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "win_probability.pt"),
                                     weights_only=True, map_location="cpu"))
    model.cpu().eval()
    dummy_input = torch.randn(1, INPUT_DIM)
    onnx_path = os.path.join(os.path.dirname(__file__), "win_probability.onnx")
    torch.onnx.export(
        model, dummy_input, onnx_path,
        input_names=["input"],
        output_names=["win_probability"],
        dynamic_axes={"input": {0: "batch"}, "win_probability": {0: "batch"}},
    )
    embed_onnx_weights(onnx_path)
    print(f"Exported ONNX model to {onnx_path}")
    print(f"Model size: {os.path.getsize(onnx_path) / 1024:.1f} KB")

    print("Optimizing ONNX graph...")
    optimize_onnx(onnx_path)

    print("Quantizing to INT8...")
    calib_data = load_replay_data(limit=2000)
    quant_path = quantize_onnx(onnx_path, calib_data, model_type="wp")

    print("Verifying quantization...")
    verify_quantized_model(onnx_path, quant_path, calib_data, model_type="wp")


# ── Sweep ─────────────────────────────────────────────────────────────

SWEEP = {
    "embed_dim": [8, 16, 32, 64],
    "interaction_mode": ["concat", "product", "concat_product", "full", "full_attention"],
    "hidden_dims": [
        [128, 64],
        [256, 128],
        [256, 128, 64],
        [512, 256, 128],
    ],
    "dropout": [0.1, 0.2],
    "lr": [1e-3, 5e-4],
}


def train_single_variant(variant_id, config, train_data, test_data, gpu_id, csv_path):
    """Train one embedding variant on a specific GPU."""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device('cuda')

    embed_dim = config['embed_dim']
    interaction_mode = config['interaction_mode']
    hidden_dims = config['hidden_dims']
    dropout = config['dropout']
    lr = config['lr']
    is_baseline = config.get('baseline', False)
    augmented = config.get('augmented', True)

    # Create datasets
    if is_baseline:
        if augmented:
            train_ds = WinProbAugDataset(train_data)
        else:
            train_ds = WinProbDataset(train_data)
        test_ds = WinProbDataset(test_data)
        train_dl = DataLoader(train_ds, batch_size=512, shuffle=True)
        test_dl = DataLoader(test_ds, batch_size=512)
        model = WinProbModel().to(device)
    else:
        train_ds = EmbeddingWPDataset(train_data)
        test_ds = EmbeddingWPDataset(test_data)
        train_dl = DataLoader(train_ds, batch_size=512, shuffle=True, collate_fn=collate_embedding)
        test_dl = DataLoader(test_ds, batch_size=512, collate_fn=collate_embedding)
        model = WinProbEmbeddingModel(
            embed_dim=embed_dim,
            interaction_mode=interaction_mode,
            hidden_dims=hidden_dims,
            dropout=dropout,
        ).to(device)

    params = sum(p.numel() for p in model.parameters())
    name = config.get('name', f"v{variant_id}")
    print(f"[GPU {gpu_id}] {name}: {params:,} params, "
          f"embed={embed_dim}, mode={interaction_mode}, "
          f"hidden={hidden_dims}, drop={dropout}, lr={lr}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
    criterion = nn.BCELoss()

    best_test_loss = float('inf')
    best_test_acc = 0.0
    best_epoch = 0
    patience = 15
    patience_counter = 0

    for epoch in range(200):
        model.train()
        train_correct = 0
        train_total = 0

        for batch in train_dl:
            if is_baseline:
                X, y = batch
                X, y = X.to(device), y.to(device)
                pred = model(X)
            else:
                t0, t1, t0m, t1m, ctx, y = batch
                t0, t1 = t0.to(device), t1.to(device)
                t0m, t1m = t0m.to(device), t1m.to(device)
                ctx, y = ctx.to(device), y.to(device)
                pred = model(t0, t1, t0m, t1m, ctx)

            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_correct += ((pred > 0.5).float() == y).sum().item()
            train_total += len(y)

        scheduler.step()

        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for batch in test_dl:
                if is_baseline:
                    X, y = batch
                    X, y = X.to(device), y.to(device)
                    pred = model(X)
                else:
                    t0, t1, t0m, t1m, ctx, y = batch
                    t0, t1 = t0.to(device), t1.to(device)
                    t0m, t1m = t0m.to(device), t1m.to(device)
                    ctx, y = ctx.to(device), y.to(device)
                    pred = model(t0, t1, t0m, t1m, ctx)

                loss = criterion(pred, y)
                test_loss += loss.item() * len(y)
                test_correct += ((pred > 0.5).float() == y).sum().item()
                test_total += len(y)

        train_acc = train_correct / train_total * 100
        test_acc = test_correct / test_total * 100
        avg_test_loss = test_loss / test_total

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [{name}] ep {epoch+1}: train={train_acc:.1f}% test={test_acc:.1f}% "
                  f"loss={avg_test_loss:.6f}")

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            best_test_acc = test_acc
            best_epoch = epoch + 1
            patience_counter = 0
            # Save checkpoint
            save_path = os.path.join(os.path.dirname(__file__), f"wp_sweep_{variant_id}.pt")
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    print(f"  [{name}] DONE: acc={best_test_acc:.2f}% epoch={best_epoch} params={params:,}")

    # Write result to CSV with file lock
    lock = filelock.FileLock(csv_path + ".lock")
    with lock:
        write_header = not os.path.exists(csv_path)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['variant_id', 'name', 'embed_dim', 'interaction_mode',
                                'hidden_dims', 'dropout', 'lr', 'best_test_acc',
                                'best_test_loss', 'epochs_trained', 'params'])
            writer.writerow([variant_id, name, embed_dim, interaction_mode,
                           json.dumps(hidden_dims), dropout, lr,
                           f"{best_test_acc:.4f}", f"{best_test_loss:.6f}",
                           best_epoch, params])

    return variant_id, name, best_test_acc, best_test_loss, best_epoch, params


def _sweep_worker(gpu_id, queue, csv_path):
    """Worker function for sweep — runs on a single GPU. Loads data in-process."""
    data = load_replay_data()
    train_data, test_data = split_data(data)
    for variant_id, config in queue:
        try:
            train_single_variant(variant_id, config, train_data, test_data, gpu_id, csv_path)
        except Exception as e:
            print(f"[GPU {gpu_id}] Variant {variant_id} failed: {e}")
            import traceback
            traceback.print_exc()


def run_sweep(max_variants=None):
    """Run the full embedding model sweep across all GPUs."""
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    num_gpus = torch.cuda.device_count()
    print(f"GPUs available: {num_gpus}")
    if num_gpus == 0:
        print("No GPUs found, using CPU (will be slow)")
        num_gpus = 1

    # Load data once
    print("Loading data...")
    data = load_replay_data()
    train_data, test_data = split_data(data)
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")

    # Build variant list
    variants = []

    # Baselines first
    variants.append({
        'embed_dim': 0, 'interaction_mode': 'baseline_no_aug',
        'hidden_dims': [], 'dropout': 0, 'lr': 1e-3,
        'baseline': True, 'augmented': False, 'name': 'baseline_no_aug',
    })
    variants.append({
        'embed_dim': 0, 'interaction_mode': 'baseline_aug',
        'hidden_dims': [], 'dropout': 0, 'lr': 1e-3,
        'baseline': True, 'augmented': True, 'name': 'baseline_aug',
    })

    # Embedding sweep
    keys = list(SWEEP.keys())
    for combo in itertools.product(*[SWEEP[k] for k in keys]):
        config = dict(zip(keys, combo))
        config['baseline'] = False
        config['augmented'] = True
        config['name'] = (f"e{config['embed_dim']}_"
                         f"{config['interaction_mode']}_"
                         f"h{'_'.join(str(d) for d in config['hidden_dims'])}_"
                         f"d{config['dropout']}_lr{config['lr']}")
        variants.append(config)

    total = len(variants)
    if max_variants:
        variants = variants[:max_variants]
    print(f"Total variants: {total}, running: {len(variants)}")

    csv_path = os.path.join(os.path.dirname(__file__), "win_prob_sweep_results.csv")
    # Clear previous results
    if os.path.exists(csv_path):
        os.remove(csv_path)

    # Assign variants to GPUs round-robin
    gpu_assignments = [[] for _ in range(num_gpus)]
    for i, config in enumerate(variants):
        gpu_assignments[i % num_gpus].append((i, config))

    # Write assignments to temp files, launch one subprocess per GPU
    # Each subprocess loads ALL its ~80 models onto GPU simultaneously,
    # shares one data loader, and trains them all each epoch.
    import json as _json, subprocess, tempfile
    assignment_files = []
    for gpu_id in range(num_gpus):
        if not gpu_assignments[gpu_id]:
            continue
        path = os.path.join(tempfile.gettempdir(), f"wp_sweep_gpu{gpu_id}.json")
        with open(path, 'w') as f:
            _json.dump(gpu_assignments[gpu_id], f)
        assignment_files.append((gpu_id, path))
        print(f"  GPU {gpu_id}: {len(gpu_assignments[gpu_id])} models (batched training)")

    procs = []
    for gpu_id, path in assignment_files:
        cmd = [sys.executable, "-u", __file__, "--sweep-worker", str(gpu_id), path, csv_path]
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        p = subprocess.Popen(cmd, env=env, stdout=sys.stdout, stderr=sys.stderr)
        procs.append(p)
        print(f"  Launched GPU {gpu_id} batched worker (PID {p.pid})")

    for p in procs:
        p.wait()

    # Print leaderboard
    print(f"\n{'='*80}")
    print("LEADERBOARD")
    print(f"{'='*80}")

    if not os.path.exists(csv_path):
        print("No results found!")
        return

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        results = list(reader)

    results.sort(key=lambda r: -float(r['best_test_acc']))

    print(f"{'Rank':<5} {'Name':<50} {'Acc':>7} {'Loss':>10} {'Ep':>4} {'Params':>10}")
    print(f"{'-'*86}")
    for i, r in enumerate(results[:30]):
        print(f"{i+1:<5} {r['name']:<50} {float(r['best_test_acc']):>6.2f}% "
              f"{float(r['best_test_loss']):>10.6f} {r['epochs_trained']:>4} "
              f"{int(r['params']):>10,}")

    # Save top 5 models
    print(f"\nSaving top 5 models...")
    for rank, r in enumerate(results[:5]):
        vid = r['variant_id']
        src = os.path.join(os.path.dirname(__file__), f"wp_sweep_{vid}.pt")
        dst = os.path.join(os.path.dirname(__file__), f"win_probability_embed_v{rank+1}.pt")
        if os.path.exists(src):
            import shutil
            shutil.copy(src, dst)
            print(f"  v{rank+1}: {r['name']} ({float(r['best_test_acc']):.2f}%)")

    # Hyperparameter analysis
    print(f"\n{'='*80}")
    print("HYPERPARAMETER ANALYSIS (avg test acc per value)")
    print(f"{'='*80}")

    for param in ['embed_dim', 'interaction_mode', 'dropout', 'lr']:
        from collections import defaultdict
        buckets = defaultdict(list)
        for r in results:
            if r.get('interaction_mode', '').startswith('baseline'):
                continue
            buckets[r[param]].append(float(r['best_test_acc']))
        print(f"\n{param}:")
        for val in sorted(buckets.keys(), key=lambda v: -np.mean(buckets[v])):
            accs = buckets[val]
            print(f"  {val}: {np.mean(accs):.2f}% +/- {np.std(accs):.2f}% (n={len(accs)})")

    # Clean up sweep checkpoints (keep only top 5)
    for i, r in enumerate(results[5:], start=5):
        vid = r['variant_id']
        path = os.path.join(os.path.dirname(__file__), f"wp_sweep_{vid}.pt")
        if os.path.exists(path):
            os.remove(path)


def train_batch_sweep_worker(gpu_id, variants_with_ids, csv_path):
    """Train many models simultaneously on one GPU.
    One data loader, one epoch loop, all models do forward/backward on each batch.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device('cuda')

    print(f"[GPU {gpu_id}] Loading data...")
    data = load_replay_data()
    train_data, test_data = split_data(data)

    # Single shared dataset + dataloader
    train_ds = EmbeddingWPDataset(train_data)
    test_ds = EmbeddingWPDataset(test_data)
    train_dl = DataLoader(train_ds, batch_size=512, shuffle=True, collate_fn=collate_embedding,
                          pin_memory=True, num_workers=2)
    test_dl = DataLoader(test_ds, batch_size=512, collate_fn=collate_embedding,
                         pin_memory=True, num_workers=2)

    # Also need multi-hot loaders for baselines
    train_ds_mh = WinProbDataset(train_data)
    test_ds_mh = WinProbDataset(test_data)
    train_ds_aug = WinProbAugDataset(train_data)
    train_dl_mh = DataLoader(train_ds_mh, batch_size=512, shuffle=True, pin_memory=True)
    train_dl_aug = DataLoader(train_ds_aug, batch_size=512, shuffle=True, pin_memory=True)
    test_dl_mh = DataLoader(test_ds_mh, batch_size=512, pin_memory=True)

    # Build all models + optimizers
    slots = []  # list of dicts with model, optimizer, scheduler, config, state
    for variant_id, config in variants_with_ids:
        is_baseline = config.get('baseline', False)
        if is_baseline:
            model = WinProbModel().to(device)
        else:
            model = WinProbEmbeddingModel(
                embed_dim=config['embed_dim'],
                interaction_mode=config['interaction_mode'],
                hidden_dims=config['hidden_dims'],
                dropout=config['dropout'],
            ).to(device)

        lr = config['lr']
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
        params = sum(p.numel() for p in model.parameters())

        slots.append({
            'variant_id': variant_id,
            'config': config,
            'name': config.get('name', f'v{variant_id}'),
            'model': model,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'params': params,
            'best_test_loss': float('inf'),
            'best_test_acc': 0.0,
            'best_epoch': 0,
            'patience_counter': 0,
            'done': False,
            'is_baseline': is_baseline,
            'augmented': config.get('augmented', True),
        })

    n_active = len(slots)
    print(f"[GPU {gpu_id}] {n_active} models loaded on GPU")

    criterion = nn.BCELoss()

    for epoch in range(200):
        if all(s['done'] for s in slots):
            break

        # ── Train phase ──
        # One pass through embedding data for all embedding models
        for s in slots:
            if not s['done'] and not s['is_baseline']:
                s['model'].train()

        embed_train_correct = {i: 0 for i, s in enumerate(slots) if not s['done'] and not s['is_baseline']}
        embed_train_total = 0

        for batch in train_dl:
            t0, t1, t0m, t1m, ctx, y = [b.to(device, non_blocking=True) for b in batch]
            embed_train_total += len(y)

            for i, s in enumerate(slots):
                if s['done'] or s['is_baseline']:
                    continue
                pred = s['model'](t0, t1, t0m, t1m, ctx)
                loss = criterion(pred, y)
                s['optimizer'].zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(s['model'].parameters(), 1.0)
                s['optimizer'].step()
                embed_train_correct[i] += ((pred > 0.5).float() == y).sum().item()

        # Baselines: separate data loaders
        for s in slots:
            if s['done'] or not s['is_baseline']:
                continue
            s['model'].train()
            dl = train_dl_aug if s['augmented'] else train_dl_mh
            correct = 0
            total = 0
            for X, yb in dl:
                X, yb = X.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                pred = s['model'](X)
                loss = criterion(pred, yb)
                s['optimizer'].zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(s['model'].parameters(), 1.0)
                s['optimizer'].step()
                correct += ((pred > 0.5).float() == yb).sum().item()
                total += len(yb)

        # Step schedulers
        for s in slots:
            if not s['done']:
                s['scheduler'].step()

        # ── Eval phase ──
        for s in slots:
            if not s['done']:
                s['model'].eval()

        embed_test_loss = {i: 0.0 for i, s in enumerate(slots) if not s['done'] and not s['is_baseline']}
        embed_test_correct = {i: 0 for i, s in enumerate(slots) if not s['done'] and not s['is_baseline']}
        embed_test_total = 0

        with torch.no_grad():
            for batch in test_dl:
                t0, t1, t0m, t1m, ctx, y = [b.to(device, non_blocking=True) for b in batch]
                embed_test_total += len(y)

                for i, s in enumerate(slots):
                    if s['done'] or s['is_baseline']:
                        continue
                    pred = s['model'](t0, t1, t0m, t1m, ctx)
                    loss = criterion(pred, y)
                    embed_test_loss[i] += loss.item() * len(y)
                    embed_test_correct[i] += ((pred > 0.5).float() == y).sum().item()

        # Baselines eval
        for s in slots:
            if s['done'] or not s['is_baseline']:
                continue
            test_loss = 0
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for X, yb in test_dl_mh:
                    X, yb = X.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                    pred = s['model'](X)
                    loss = criterion(pred, yb)
                    test_loss += loss.item() * len(yb)
                    test_correct += ((pred > 0.5).float() == yb).sum().item()
                    test_total += len(yb)
            s['_epoch_test_loss'] = test_loss / test_total
            s['_epoch_test_acc'] = test_correct / test_total * 100

        # Update tracking
        newly_done = 0
        for i, s in enumerate(slots):
            if s['done']:
                continue

            if s['is_baseline']:
                avg_loss = s['_epoch_test_loss']
                test_acc = s['_epoch_test_acc']
            else:
                if embed_test_total == 0:
                    continue
                avg_loss = embed_test_loss[i] / embed_test_total
                test_acc = embed_test_correct[i] / embed_test_total * 100

            if avg_loss < s['best_test_loss']:
                s['best_test_loss'] = avg_loss
                s['best_test_acc'] = test_acc
                s['best_epoch'] = epoch + 1
                s['patience_counter'] = 0
                torch.save(s['model'].state_dict(),
                          os.path.join(os.path.dirname(__file__), f"wp_sweep_{s['variant_id']}.pt"))
            else:
                s['patience_counter'] += 1
                if s['patience_counter'] >= 15:
                    s['done'] = True
                    newly_done += 1

        active = sum(1 for s in slots if not s['done'])
        if (epoch + 1) % 5 == 0 or epoch == 0 or newly_done > 0:
            best_so_far = max((s['best_test_acc'] for s in slots), default=0)
            print(f"[GPU {gpu_id}] epoch {epoch+1}: {active}/{len(slots)} active, "
                  f"best_acc={best_so_far:.2f}%")

    # Write all results to CSV
    lock = filelock.FileLock(csv_path + ".lock")
    with lock:
        write_header = not os.path.exists(csv_path)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['variant_id', 'name', 'embed_dim', 'interaction_mode',
                                'hidden_dims', 'dropout', 'lr', 'best_test_acc',
                                'best_test_loss', 'epochs_trained', 'params'])
            for s in slots:
                c = s['config']
                writer.writerow([s['variant_id'], s['name'], c['embed_dim'],
                               c['interaction_mode'], json.dumps(c.get('hidden_dims', [])),
                               c['dropout'], c['lr'],
                               f"{s['best_test_acc']:.4f}", f"{s['best_test_loss']:.6f}",
                               s['best_epoch'], s['params']])

    print(f"[GPU {gpu_id}] Done. Best: {max(s['best_test_acc'] for s in slots):.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true", help="Run embedding model sweep")
    parser.add_argument("--variants", type=int, default=None, help="Limit number of variants")
    parser.add_argument("--sweep-worker", nargs=3, metavar=("GPU_ID", "ASSIGNMENTS_JSON", "CSV_PATH"),
                        help="Internal: run batched sweep worker for one GPU")
    args = parser.parse_args()

    if args.sweep_worker:
        gpu_id = int(args.sweep_worker[0])
        import json as _json
        with open(args.sweep_worker[1]) as f:
            assignments = _json.load(f)
        train_batch_sweep_worker(gpu_id, assignments, args.sweep_worker[2])
    elif args.sweep:
        run_sweep(max_variants=args.variants)
    else:
        train_original()
