"""
Generic Draft Model — predicts the next pick/ban given the current draft state.

For each step in the draft sequence, the model sees:
- Current draft state: which heroes are picked/banned (multi-hot vectors)
- Map (one-hot, 14)
- Tier (one-hot, 3)
- Draft step number (scalar, 0-15)
- Step type (ban=0, pick=1)

Input: team0_picks(90) + team1_picks(90) + bans(90) + map(14) + tier(3) + step(1) + type(1) = 289
Output: softmax over 90 heroes (masked to valid/available heroes)

Architecture: 289 → 256 → 128 → 90

Trains 3-5 models with different random seeds and slight hyperparameter variation
to create an opponent pool for AlphaZero training.

Usage:
    export DATABASE_URL=...
    pip install psycopg2-binary
    python training/train_generic_draft.py
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from shared import (
    NUM_HEROES, NUM_MAPS, NUM_TIERS, HERO_TO_IDX,
    map_to_one_hot, tier_to_one_hot,
    load_replay_data, split_data,
)

INPUT_DIM = NUM_HEROES * 3 + NUM_MAPS + NUM_TIERS + 2  # 90*3+14+3+2 = 289

# Hyperparameter variants for the opponent pool
MODEL_VARIANTS = [
    {"seed": 42, "lr": 1e-3, "dropout1": 0.2, "dropout2": 0.1},
    {"seed": 123, "lr": 8e-4, "dropout1": 0.25, "dropout2": 0.15},
    {"seed": 777, "lr": 1.2e-3, "dropout1": 0.15, "dropout2": 0.05},
    {"seed": 2024, "lr": 7e-4, "dropout1": 0.3, "dropout2": 0.1},
    {"seed": 31415, "lr": 1.5e-3, "dropout1": 0.2, "dropout2": 0.2},
]


def replay_to_training_samples(replay: dict) -> list[tuple[np.ndarray, int, np.ndarray]]:
    """
    Convert a single replay's draft_order into training samples.
    Each step produces (input_features, target_hero_idx, valid_mask).
    """
    draft_order = replay["draft_order"]
    if not draft_order or len(draft_order) != 16:
        return []

    game_map = map_to_one_hot(replay["game_map"])
    tier = tier_to_one_hot(replay["skill_tier"])

    # Track draft state as we step through
    team0_picks = np.zeros(NUM_HEROES, dtype=np.float32)
    team1_picks = np.zeros(NUM_HEROES, dtype=np.float32)
    bans = np.zeros(NUM_HEROES, dtype=np.float32)
    taken = set()  # all heroes picked or banned so far

    samples = []
    for step in draft_order:
        hero = step["hero"]
        hero_idx = HERO_TO_IDX.get(hero)
        if hero_idx is None:
            # Unknown hero, skip entire replay
            return []

        step_type = float(step["type"])  # 0=ban, 1=pick
        step_num = float(step["pick_number"]) / 15.0  # normalize to [0,1]

        # Build input vector from current state (before this action)
        x = np.concatenate([
            team0_picks, team1_picks, bans,
            game_map, tier,
            [step_num, step_type],
        ])

        # Valid mask: all heroes not yet taken
        valid_mask = np.ones(NUM_HEROES, dtype=np.float32)
        for idx in taken:
            valid_mask[idx] = 0.0

        samples.append((x, hero_idx, valid_mask))

        # Update state
        taken.add(hero_idx)
        if step_type == 0:  # ban
            bans[hero_idx] = 1.0
        else:  # pick
            # Determine which team picked based on player_slot
            slot = step.get("player_slot", 0)
            if slot <= 4 or slot == 1:
                team0_picks[hero_idx] = 1.0
            else:
                team1_picks[hero_idx] = 1.0

    return samples


class DraftDataset(Dataset):
    def __init__(self, data: list[dict]):
        self.X = []
        self.y = []
        self.masks = []
        for d in data:
            samples = replay_to_training_samples(d)
            for x, target, mask in samples:
                self.X.append(x)
                self.y.append(target)
                self.masks.append(mask)
        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.int64)
        self.masks = np.array(self.masks, dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx]),
            torch.tensor(self.y[idx]),
            torch.from_numpy(self.masks[idx]),
        )


class GenericDraftModel(nn.Module):
    def __init__(self, dropout1=0.2, dropout2=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 256),
            nn.ReLU(),
            nn.Dropout(dropout1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(128, NUM_HEROES),
        )

    def forward(self, x, mask=None):
        logits = self.net(x)
        if mask is not None:
            # Set taken heroes to -inf so softmax gives them 0 probability
            logits = logits + (1 - mask) * (-1e9)
        return logits


def train_single_model(
    variant_idx: int,
    variant: dict,
    train_ds: DraftDataset,
    test_ds: DraftDataset,
    device: torch.device,
) -> float:
    """Train a single Generic Draft model variant. Returns best test loss."""
    seed = variant["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n{'='*60}")
    print(f"Training variant {variant_idx}: seed={seed} lr={variant['lr']} "
          f"dropout=({variant['dropout1']}, {variant['dropout2']})")
    print(f"{'='*60}")

    train_dl = DataLoader(train_ds, batch_size=512, shuffle=True,
                          generator=torch.Generator().manual_seed(seed))
    test_dl = DataLoader(test_ds, batch_size=512)

    model = GenericDraftModel(
        dropout1=variant["dropout1"],
        dropout2=variant["dropout2"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=variant["lr"], weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    pt_path = os.path.join(os.path.dirname(__file__), f"generic_draft_{variant_idx}.pt")
    best_test_loss = float("inf")
    patience = 10
    patience_counter = 0

    for epoch in range(200):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        for X, y, mask in train_dl:
            X, y, mask = X.to(device), y.to(device), mask.to(device)
            logits = model(X, mask)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y)
            train_correct += (logits.argmax(dim=1) == y).sum().item()
            train_total += len(y)

        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        test_top5 = 0
        with torch.no_grad():
            for X, y, mask in test_dl:
                X, y, mask = X.to(device), y.to(device), mask.to(device)
                logits = model(X, mask)
                loss = criterion(logits, y)
                test_loss += loss.item() * len(y)
                test_correct += (logits.argmax(dim=1) == y).sum().item()
                top5 = logits.topk(5, dim=1).indices
                test_top5 += (top5 == y.unsqueeze(1)).any(dim=1).sum().item()
                test_total += len(y)

        train_acc = train_correct / train_total * 100
        test_acc = test_correct / test_total * 100
        test_top5_acc = test_top5 / test_total * 100
        avg_test_loss = test_loss / test_total

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: train_acc={train_acc:.1f}% "
                  f"test_acc={test_acc:.1f}% test_top5={test_top5_acc:.1f}% "
                  f"test_loss={avg_test_loss:.4f}")

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            patience_counter = 0
            torch.save(model.state_dict(), pt_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Export to ONNX
    model.load_state_dict(torch.load(pt_path, weights_only=True))
    model.eval()
    dummy_x = torch.randn(1, INPUT_DIM)
    dummy_mask = torch.ones(1, NUM_HEROES)
    onnx_path = os.path.join(os.path.dirname(__file__), f"generic_draft_{variant_idx}.onnx")
    torch.onnx.export(
        model, (dummy_x, dummy_mask), onnx_path,
        input_names=["state", "valid_mask"],
        output_names=["hero_logits"],
        dynamic_axes={
            "state": {0: "batch"},
            "valid_mask": {0: "batch"},
            "hero_logits": {0: "batch"},
        },
    )
    print(f"  Exported ONNX: {onnx_path} ({os.path.getsize(onnx_path) / 1024:.1f} KB)")
    print(f"  Best test loss: {best_test_loss:.4f}")

    # Also save as generic_draft.pt/onnx for variant 0 (backwards compat)
    if variant_idx == 0:
        import shutil
        shutil.copy(pt_path, os.path.join(os.path.dirname(__file__), "generic_draft.pt"))
        shutil.copy(onnx_path, os.path.join(os.path.dirname(__file__), "generic_draft.onnx"))

    return best_test_loss


def train():
    print("Loading data...")
    data = load_replay_data()
    print(f"Loaded {len(data)} replays")

    if len(data) < 100:
        print("Not enough data. Run the replay daemon first.")
        return

    train_data, test_data = split_data(data)
    print(f"Train: {len(train_data)} replays, Test: {len(test_data)} replays")

    train_ds = DraftDataset(train_data)
    test_ds = DraftDataset(test_data)
    print(f"Train samples: {len(train_ds)}, Test samples: {len(test_ds)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results = []
    for i, variant in enumerate(MODEL_VARIANTS):
        loss = train_single_model(i, variant, train_ds, test_ds, device)
        results.append((i, loss))

    print(f"\n{'='*60}")
    print("All variants trained:")
    for i, loss in results:
        print(f"  Variant {i}: test_loss={loss:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    train()
