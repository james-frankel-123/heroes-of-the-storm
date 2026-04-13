#!/usr/bin/env python3
"""
Reimplementation of Gourdeau & Archambault's HotS draft WP estimator in PyTorch.

Architecture (from https://github.com/HairyBlob/HotS-Drafter):
  - Siamese shared-weight network: each team processed identically
  - Multiplicative map conditioning: map embedding * hero embedding at first layer
  - Shared layers: 90→1024 (heroes) * 14→1024 (maps), then 1024→512→256→128
  - Classification head: concat both teams (256) → 256→256→256→2

Input: team0 multi-hot (90), team1 multi-hot (90), map one-hot (14)
Output: P(team0 wins) via softmax over 2 classes

Reference: "Discriminative Neural Network for Hero Selection in Professional
Heroes of the Storm and Dota 2", IEEE Trans. Games, 2021.
"""
import os, sys, time, random
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared import (NUM_HEROES, NUM_MAPS, HEROES, HERO_TO_IDX, MAPS, MAP_TO_IDX,
                    load_replay_data, split_data, heroes_to_multi_hot, map_to_one_hot)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class GourdeauWPModel(nn.Module):
    """Faithful PyTorch reimplementation of the Gourdeau Siamese WP estimator.

    Key architectural choices from the original:
    - Absolute value on hero and map weight matrices (lines 63-66 in estimator.py)
    - Multiplicative conditioning: hero_embed * map_embed (not additive)
    - Shared weights process each team identically (Siamese)
    - 4 shared layers → concat → 3 classification layers → 2 outputs
    """
    def __init__(self, n_heroes=NUM_HEROES, n_maps=NUM_MAPS):
        super().__init__()
        # Shared Siamese layers (process each team identically)
        self.hero_w = nn.Parameter(torch.empty(n_heroes, 1024))
        self.map_w = nn.Parameter(torch.empty(n_maps, 1024))
        nn.init.xavier_uniform_(self.hero_w)
        nn.init.xavier_uniform_(self.map_w)

        self.shared_b1 = nn.Parameter(torch.zeros(1024))
        self.shared_fc2 = nn.Linear(1024, 512)
        self.shared_fc3 = nn.Linear(512, 256)
        self.shared_fc4 = nn.Linear(256, 128)

        # Classification head (after concatenating both team embeddings)
        self.cls_fc1 = nn.Linear(256, 256)  # 128*2 = 256 input
        self.cls_fc2 = nn.Linear(256, 256)
        self.cls_fc3 = nn.Linear(256, 256)
        self.cls_out = nn.Linear(256, 2)

        self.dropout = nn.Dropout(0.3)

    def _team_forward(self, team_hot, map_hot):
        """Process one team through shared Siamese layers."""
        # Multiplicative map conditioning with absolute value weights
        hero_embed = torch.matmul(team_hot, torch.abs(self.hero_w))
        map_embed = torch.matmul(map_hot, torch.abs(self.map_w))
        x = hero_embed * map_embed + self.shared_b1
        x = torch.relu(x)
        x = self.dropout(x)

        x = torch.relu(self.shared_fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.shared_fc3(x))
        x = self.dropout(x)
        x = torch.relu(self.shared_fc4(x))
        x = self.dropout(x)
        return x  # (batch, 128)

    def forward(self, team0, team1, map_oh):
        """Forward pass.

        Args:
            team0: (batch, n_heroes) multi-hot
            team1: (batch, n_heroes) multi-hot
            map_oh: (batch, n_maps) one-hot
        Returns:
            logits: (batch, 2) — class 0 = team0 wins, class 1 = team1 wins
        """
        t0_embed = self._team_forward(team0, map_oh)
        t1_embed = self._team_forward(team1, map_oh)

        x = torch.cat([t0_embed, t1_embed], dim=1)  # (batch, 256)
        x = torch.relu(self.cls_fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.cls_fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.cls_fc3(x))
        x = self.dropout(x)
        logits = self.cls_out(x)
        return logits

    def predict_wp(self, team0, team1, map_oh):
        """Return P(team0 wins) as a scalar per sample."""
        logits = self.forward(team0, team1, map_oh)
        probs = torch.softmax(logits, dim=1)
        return probs[:, 0]  # P(team0 wins)


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    data = load_replay_data()
    train_data, test_data = split_data(data, test_frac=0.15)
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")

    # Prepare tensors
    def make_tensors(samples):
        t0_list, t1_list, map_list, label_list = [], [], [], []
        for d in samples:
            t0h = d.get('team0_heroes', [])
            t1h = d.get('team1_heroes', [])
            if len(t0h) != 5 or len(t1h) != 5:
                continue
            t0_list.append(heroes_to_multi_hot(t0h))
            t1_list.append(heroes_to_multi_hot(t1h))
            map_list.append(map_to_one_hot(d['game_map']))
            label_list.append(0 if d['winner'] == 0 else 1)
            # Team-swap augmentation
            t0_list.append(heroes_to_multi_hot(t1h))
            t1_list.append(heroes_to_multi_hot(t0h))
            map_list.append(map_to_one_hot(d['game_map']))
            label_list.append(1 if d['winner'] == 0 else 0)

        return (torch.tensor(np.array(t0_list)),
                torch.tensor(np.array(t1_list)),
                torch.tensor(np.array(map_list)),
                torch.tensor(np.array(label_list, dtype=np.int64)))

    print("Preparing tensors...")
    tr_t0, tr_t1, tr_map, tr_y = make_tensors(train_data)
    te_t0, te_t1, te_map, te_y = make_tensors(test_data)
    print(f"  Train samples: {len(tr_y):,}, Test samples: {len(te_y):,}")

    tr_t0, tr_t1, tr_map, tr_y = tr_t0.to(device), tr_t1.to(device), tr_map.to(device), tr_y.to(device)
    te_t0, te_t1, te_map, te_y = te_t0.to(device), te_t1.to(device), te_map.to(device), te_y.to(device)

    # Train
    model = GourdeauWPModel().to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    best_state = None
    batch_size = 4096
    n = len(tr_y)

    for epoch in range(100):
        model.train()
        perm = torch.randperm(n, device=device)
        epoch_loss = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            logits = model(tr_t0[idx], tr_t1[idx], tr_map[idx])
            loss = criterion(logits, tr_y[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)

        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                te_logits = model(te_t0, te_t1, te_map)
                te_pred = te_logits.argmax(dim=1)
                te_acc = (te_pred == te_y).float().mean().item() * 100
            if te_acc > best_acc:
                best_acc = te_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  Epoch {epoch+1}: loss={epoch_loss/n:.4f}, test_acc={te_acc:.2f}%, best={best_acc:.2f}%")

    model.load_state_dict(best_state)
    save_path = os.path.join(SCRIPT_DIR, 'gourdeau_wp.pt')
    torch.save(model.cpu().state_dict(), save_path)
    print(f"\nSaved to {save_path}")
    print(f"Best test accuracy: {best_acc:.2f}%")
    print(f"Parameters: {params:,}")

    # Quick sanity: evaluate 5-tank comp
    model.eval().to(device)
    tanks = ['Muradin', 'Johanna', 'Diablo', 'E.T.C.', "Mal'Ganis"]
    standard = ['Muradin', 'Brightwing', 'Valla', 'Sonya', 'Jaina']

    t0 = torch.tensor(heroes_to_multi_hot(tanks)).unsqueeze(0).to(device)
    t1 = torch.tensor(heroes_to_multi_hot(standard)).unsqueeze(0).to(device)
    m = torch.tensor(map_to_one_hot('Cursed Hollow')).unsqueeze(0).to(device)

    with torch.no_grad():
        wp = model.predict_wp(t0, t1, m).item()
    print(f"\n5-tank vs standard: {wp*100:.1f}% WP (should be low, Gourdeau original was 52.8%)")


if __name__ == '__main__':
    train()
