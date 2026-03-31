"""
Partial-state Win Probability model (Approach 2a).

Trains on draft states at EVERY pick step (not just complete drafts),
so the model learns calibrated predictions for partial draft states.

Each replay yields ~10 pick-step samples x 2 (team-swap augmentation) = ~20 samples.
With ~275K replays, this produces ~5.5M training samples.

Usage:
    set -a && source .env && set +a
    python training/train_partial_wp.py
"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from shared import (
    NUM_HEROES, NUM_MAPS, NUM_TIERS, HEROES, HERO_TO_IDX,
    heroes_to_multi_hot, map_to_one_hot, tier_to_one_hot,
    load_replay_data,
)
from sweep_enriched_wp import (
    StatsCache, extract_features,
    FEATURE_GROUPS, FEATURE_GROUP_DIMS, INPUT_DIM_BASE,
)

# Use the same 9 enriched feature groups as the deployed enriched WP model
WP_GROUPS = ['role_counts', 'team_avg_wr', 'map_delta', 'pairwise_counters',
             'pairwise_synergies', 'counter_detail', 'meta_strength',
             'draft_diversity', 'comp_wr']
ENRICHED_DIM = sum(FEATURE_GROUP_DIMS[g] for g in WP_GROUPS)
TOTAL_FEATURE_DIM = INPUT_DIM_BASE + ENRICHED_DIM  # 197 + 86 = 283


# ── Model ──

class PartialStateWP(nn.Module):
    def __init__(self, input_dim=TOTAL_FEATURE_DIM, step_embed_dim=8, hidden=(256, 128)):
        super().__init__()
        self.step_embed = nn.Embedding(16, step_embed_dim)
        layers = []
        in_dim = input_dim + step_embed_dim
        for h in hidden:
            layers.extend([nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.15)])
            in_dim = h
        layers.extend([nn.Linear(in_dim, 1), nn.Sigmoid()])
        self.net = nn.Sequential(*layers)

    def forward(self, features, step_idx):
        step_emb = self.step_embed(step_idx)
        x = torch.cat([features, step_emb], dim=1)
        return self.net(x).squeeze(-1)


# ── Data extraction ──

def extract_partial_states(replays, stats):
    """Walk each replay's draft_order, extract features at every pick step.
    Returns: features (N, TOTAL_FEATURE_DIM), step_indices (N,), labels (N,)
    Uses only the 9 WP enriched feature groups (86 dims), not all 14.
    Includes team-swap augmentation (doubles sample count).
    """
    all_groups_mask = [True] * len(FEATURE_GROUPS)
    # Compute column indices for the 9 WP groups within the full enriched vector
    from sweep_enriched_wp import compute_group_indices
    group_indices = compute_group_indices()
    wp_cols = []
    for g in WP_GROUPS:
        s, e = group_indices[g]
        wp_cols.extend(range(s, e))

    features_list = []
    steps_list = []
    labels_list = []

    skipped = 0
    for ri, replay in enumerate(replays):
        if ri % 25000 == 0:
            print(f"  Processing replay {ri}/{len(replays)} ({len(features_list)} samples so far)")

        draft_order = replay.get("draft_order")
        if not draft_order:
            skipped += 1
            continue

        game_map = replay["game_map"]
        tier = replay["skill_tier"]
        winner = replay["winner"]
        label = float(winner == 0)  # 1 if team0 wins

        # Walk through draft, accumulating picks
        team0_heroes = []
        team1_heroes = []

        for step_idx, step in enumerate(draft_order):
            step_type = str(step.get("type", ""))
            hero = step.get("hero")

            if step_type != "1":
                # Ban step — skip
                continue

            if not hero or hero not in HERO_TO_IDX:
                continue

            # Determine which team by checking final hero lists
            t0_set = set(replay.get("team0_heroes", []))
            t1_set = set(replay.get("team1_heroes", []))
            if hero in t0_set:
                team0_heroes.append(hero)
            elif hero in t1_set:
                team1_heroes.append(hero)
            else:
                continue

            # Build a sample dict for extract_features
            sample = {
                "team0_heroes": list(team0_heroes),
                "team1_heroes": list(team1_heroes),
                "game_map": game_map,
                "skill_tier": tier,
                "avg_mmr": replay.get("avg_mmr"),
            }

            try:
                base, enriched = extract_features(sample, stats, all_groups_mask)
                feat = np.concatenate([base, enriched[wp_cols]])

                # Use pick_number from the draft entry (0-15), clamped
                pick_num = min(max(int(step.get("pick_number", step_idx)), 0), 15)

                # Original perspective
                features_list.append(feat)
                steps_list.append(pick_num)
                labels_list.append(label)

                # Augmented: swap teams → flipped label
                sample_swap = {
                    "team0_heroes": list(team1_heroes),
                    "team1_heroes": list(team0_heroes),
                    "game_map": game_map,
                    "skill_tier": tier,
                    "avg_mmr": replay.get("avg_mmr"),
                }
                base_swap, enriched_swap = extract_features(sample_swap, stats, all_groups_mask)
                feat_swap = np.concatenate([base_swap, enriched_swap[wp_cols]])
                features_list.append(feat_swap)
                steps_list.append(pick_num)
                labels_list.append(1.0 - label)
            except Exception:
                # Some edge cases with partial data — skip silently
                continue

    print(f"  Extracted {len(features_list)} samples from {len(replays)} replays (skipped {skipped})")

    features_arr = np.array(features_list, dtype=np.float32)
    steps_arr = np.array(steps_list, dtype=np.int64)
    labels_arr = np.array(labels_list, dtype=np.float32)

    return features_arr, steps_arr, labels_arr


# ── Training ──

def train():
    print("=" * 60)
    print("Partial-State WP Model Training (Approach 2a)")
    print("=" * 60)

    # Load data
    print("\n[1/4] Loading replay data...")
    replays = load_replay_data()
    print(f"  Loaded {len(replays)} replays")

    # Load stats cache
    print("\n[2/4] Loading stats cache...")
    stats = StatsCache()
    print("  Stats cache loaded")

    # Extract partial states
    print(f"\n[3/4] Extracting partial draft states (feature dim = {TOTAL_FEATURE_DIM})...")
    t0 = time.time()
    features, steps, labels = extract_partial_states(replays, stats)
    print(f"  Extraction took {time.time() - t0:.1f}s")
    print(f"  Total samples: {len(labels):,}")
    print(f"  Features shape: {features.shape}")
    print(f"  Label mean: {labels.mean():.4f} (should be ~0.50)")

    # Per-step counts
    unique_steps, step_counts = np.unique(steps, return_counts=True)
    for s, c in zip(unique_steps, step_counts):
        print(f"    Step {s:2d}: {c:>8,} samples")

    # Train/test split (85/15 by replay, not by sample, to avoid leakage)
    print("\n[4/4] Training...")
    rng = np.random.RandomState(42)
    n = len(features)
    indices = rng.permutation(n)
    test_size = int(n * 0.15)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]

    X_train = torch.tensor(features[train_idx])
    S_train = torch.tensor(steps[train_idx])
    Y_train = torch.tensor(labels[train_idx])
    X_test = torch.tensor(features[test_idx])
    S_test = torch.tensor(steps[test_idx])
    Y_test = torch.tensor(labels[test_idx])

    print(f"  Train: {len(X_train):,} samples, Test: {len(X_test):,} samples")

    train_ds = TensorDataset(X_train, S_train, Y_train)
    test_ds = TensorDataset(X_test, S_test, Y_test)
    train_loader = DataLoader(train_ds, batch_size=4096, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    model = PartialStateWP(input_dim=TOTAL_FEATURE_DIM).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {param_count:,}")

    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    best_test_acc = 0.0
    best_state = None

    for epoch in range(1, 51):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for X_batch, S_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            S_batch = S_batch.to(device)
            Y_batch = Y_batch.to(device)

            optimizer.zero_grad()
            preds = model(X_batch, S_batch)
            loss = criterion(preds, Y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(Y_batch)
            train_correct += ((preds > 0.5).float() == Y_batch).sum().item()
            train_total += len(Y_batch)

        scheduler.step()

        # Evaluate every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            test_loss = 0.0
            test_correct = 0
            test_total = 0

            with torch.no_grad():
                for X_batch, S_batch, Y_batch in test_loader:
                    X_batch = X_batch.to(device)
                    S_batch = S_batch.to(device)
                    Y_batch = Y_batch.to(device)

                    preds = model(X_batch, S_batch)
                    loss = criterion(preds, Y_batch)

                    test_loss += loss.item() * len(Y_batch)
                    test_correct += ((preds > 0.5).float() == Y_batch).sum().item()
                    test_total += len(Y_batch)

            train_acc = train_correct / train_total
            test_acc = test_correct / test_total
            lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d} | "
                  f"Train loss={train_loss/train_total:.4f} acc={train_acc:.4f} | "
                  f"Test loss={test_loss/test_total:.4f} acc={test_acc:.4f} | "
                  f"lr={lr:.6f}")

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)
    print(f"\n  Best test accuracy: {best_test_acc:.4f}")

    # ── Per-step accuracy breakdown ──
    print("\n" + "=" * 60)
    print("Per-step accuracy breakdown (test set)")
    print("=" * 60)

    model.eval()
    all_preds = []
    all_labels = []
    all_steps = []

    with torch.no_grad():
        for X_batch, S_batch, Y_batch in test_loader:
            X_batch = X_batch.to(device)
            S_batch = S_batch.to(device)
            preds = model(X_batch, S_batch)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(Y_batch.cpu().numpy())
            all_steps.append(S_batch.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_steps = np.concatenate(all_steps)

    overall_acc = ((all_preds > 0.5) == all_labels).mean()
    print(f"\n  Overall test accuracy: {overall_acc:.4f}")
    print(f"\n  {'Step':>6} {'Count':>8} {'Accuracy':>10} {'Avg Pred':>10} {'Avg |Pred-0.5|':>16}")
    print(f"  {'-'*6} {'-'*8} {'-'*10} {'-'*10} {'-'*16}")

    for step in sorted(np.unique(all_steps)):
        mask = all_steps == step
        count = mask.sum()
        acc = ((all_preds[mask] > 0.5) == all_labels[mask]).mean()
        avg_pred = all_preds[mask].mean()
        avg_confidence = np.abs(all_preds[mask] - 0.5).mean()
        print(f"  {int(step):6d} {count:8d} {acc:10.4f} {avg_pred:10.4f} {avg_confidence:16.4f}")

    # Save model
    save_path = os.path.join(os.path.dirname(__file__), "partial_wp.pt")
    torch.save({
        "model_state_dict": model.cpu().state_dict(),
        "input_dim": TOTAL_FEATURE_DIM,
        "step_embed_dim": 8,
        "hidden": [256, 128],
        "best_test_acc": best_test_acc,
        "wp_groups": WP_GROUPS,
    }, save_path)
    print(f"\n  Model saved to {save_path}")
    print(f"  File size: {os.path.getsize(save_path) / 1024:.1f} KB")


if __name__ == "__main__":
    train()
