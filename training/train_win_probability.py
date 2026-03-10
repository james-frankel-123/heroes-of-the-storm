"""
Win Probability Model — predicts P(team0 wins) given both team compositions + map + tier.

Input: team0_heroes (90) + team1_heroes (90) + map (14) + tier (3) = 197 features
Output: sigmoid probability of team 0 winning

Architecture: 197 → 256 → 128 → 1 (wider to maximize accuracy — this model is the
quality ceiling for everything downstream)

Usage:
    export DATABASE_URL=...
    pip install psycopg2-binary
    python training/train_win_probability.py
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from shared import (
    NUM_HEROES, NUM_MAPS, NUM_TIERS,
    heroes_to_multi_hot, map_to_one_hot, tier_to_one_hot,
    load_replay_data, split_data,
)

INPUT_DIM = NUM_HEROES * 2 + NUM_MAPS + NUM_TIERS  # 90+90+14+3 = 197


class WinProbDataset(Dataset):
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
            self.y.append(float(d["winner"] == 0))  # P(team0 wins)
        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])


class WinProbModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train():
    print("Loading data...")
    data = load_replay_data()
    print(f"Loaded {len(data)} replays")

    if len(data) < 100:
        print("Not enough data to train (need at least 100 replays). Run the replay daemon first.")
        return

    train_data, test_data = split_data(data)
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")

    train_ds = WinProbDataset(train_data)
    test_ds = WinProbDataset(test_data)
    train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=256)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = WinProbModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.BCELoss()

    best_test_loss = float("inf")
    patience = 10
    patience_counter = 0

    for epoch in range(200):
        # Train
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

        # Test
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

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}: train_loss={train_loss/train_total:.4f} "
                  f"train_acc={train_acc:.1f}% test_loss={avg_test_loss:.4f} test_acc={test_acc:.1f}%")

        # Early stopping
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), "win_probability.pt"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Export to ONNX (on CPU to avoid device mismatch)
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
    print(f"Exported ONNX model to {onnx_path}")
    print(f"Model size: {os.path.getsize(onnx_path) / 1024:.1f} KB")


if __name__ == "__main__":
    train()
