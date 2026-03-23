"""
Experiment: Independent baseline reproducing HairyBlob/HotS-Drafter architecture.

Tests whether a siamese multi-hot NN (no compositional features) exhibits the same
OOD pathology as our naive model. This strengthens the paper's claim that the problem
is about feature representation, not model architecture.

Architecture (from HairyBlob/HotS-Drafter):
  - Siamese: each team's multi-hot (90 dims) → shared layers → 128-d embedding
  - Multiplicative map conditioning: hero_emb * map_emb (element-wise)
  - Classification head: concat(team0_128, team1_128) → 256 → 256 → 256 → 2 (softmax)
  - Non-negative first-layer weights (abs constraint)
  - Dropout 0.3, Adam lr=1e-4, batch 4096

Usage:
    set -a && source .env && set +a && python training/experiment_independent_baseline.py
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from shared import (
    NUM_HEROES, NUM_MAPS, NUM_TIERS, HEROES, HERO_TO_IDX, MAPS, SKILL_TIERS,
    heroes_to_multi_hot, map_to_one_hot, tier_to_one_hot,
    load_replay_data, split_data,
    HERO_ROLE_FINE, FINE_ROLE_NAMES,
)
from sweep_enriched_wp import (
    StatsCache, WinProbEnrichedModel, FEATURE_GROUPS, FEATURE_GROUP_DIMS,
    compute_group_indices, extract_features,
)
from test_wp_sanity import TESTS, run_tests


# ── Siamese Model (HairyBlob architecture) ──

class SiameseWinProbModel(nn.Module):
    """
    Reproduces HairyBlob/HotS-Drafter estimator architecture in PyTorch.

    Siamese branch (shared weights, per-team):
      heroes (90) → abs(W) → 1024 (ReLU) → 512 (ReLU) → 256 (ReLU) → 128 (ReLU)
      map (14) → abs(W) → 1024, then element-wise multiply with hero embedding at layer 1.

    Classification head:
      concat(team0_128, team1_128) = 256 → 256 (ReLU) → 256 (ReLU) → 256 (ReLU) → 1 (sigmoid)
    """
    def __init__(self, n_heroes=NUM_HEROES, n_maps=NUM_MAPS, n_tiers=NUM_TIERS, dropout=0.3):
        super().__init__()
        # Siamese branch (shared)
        self.hero_w = nn.Linear(n_heroes, 1024)
        self.map_w = nn.Linear(n_maps + n_tiers, 1024)  # map + tier for conditioning
        self.branch_fc2 = nn.Linear(1024, 512)
        self.branch_fc3 = nn.Linear(512, 256)
        self.branch_fc4 = nn.Linear(256, 128)

        # Classification head
        self.head_fc1 = nn.Linear(256, 256)
        self.head_fc2 = nn.Linear(256, 256)
        self.head_fc3 = nn.Linear(256, 256)
        self.head_out = nn.Linear(256, 1)

        self.dropout = nn.Dropout(dropout)

        # Initialize with Xavier
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def _branch(self, heroes, map_ctx):
        """Process one team through the siamese branch."""
        # Non-negative first-layer weights (abs constraint)
        h = F.relu(F.linear(heroes, self.hero_w.weight.abs(), self.hero_w.bias))
        m = F.linear(map_ctx, self.map_w.weight.abs(), self.map_w.bias)
        # Multiplicative map conditioning
        h = h * torch.sigmoid(m)  # sigmoid to keep stable (original uses raw multiply)
        h = self.dropout(h)
        h = self.dropout(F.relu(self.branch_fc2(h)))
        h = self.dropout(F.relu(self.branch_fc3(h)))
        h = self.dropout(F.relu(self.branch_fc4(h)))
        return h

    def forward(self, t0_heroes, t1_heroes, map_ctx):
        """
        t0_heroes: (batch, n_heroes) multi-hot
        t1_heroes: (batch, n_heroes) multi-hot
        map_ctx: (batch, n_maps + n_tiers)
        Returns: win probability for team 0, shape (batch,)
        """
        e0 = self._branch(t0_heroes, map_ctx)
        e1 = self._branch(t1_heroes, map_ctx)
        combined = torch.cat([e0, e1], dim=1)  # (batch, 256)
        x = self.dropout(F.relu(self.head_fc1(combined)))
        x = self.dropout(F.relu(self.head_fc2(x)))
        x = self.dropout(F.relu(self.head_fc3(x)))
        return torch.sigmoid(self.head_out(x)).squeeze(-1)


# ── Dataset ──

class SiameseWinProbDataset(Dataset):
    """Dataset with team-swap augmentation for siamese model."""
    def __init__(self, data):
        self.t0_heroes = []
        self.t1_heroes = []
        self.map_ctx = []
        self.labels = []

        for d in data:
            t0 = heroes_to_multi_hot(d["team0_heroes"])
            t1 = heroes_to_multi_hot(d["team1_heroes"])
            m = list(map_to_one_hot(d["game_map"])) + list(tier_to_one_hot(d.get("skill_tier", "mid")))
            y = float(d["winner"])

            # Original
            self.t0_heroes.append(t0)
            self.t1_heroes.append(t1)
            self.map_ctx.append(m)
            self.labels.append(y)

            # Team-swap augmentation
            self.t0_heroes.append(t1)
            self.t1_heroes.append(t0)
            self.map_ctx.append(m)
            self.labels.append(1.0 - y)

        self.t0_heroes = torch.tensor(np.array(self.t0_heroes), dtype=torch.float32)
        self.t1_heroes = torch.tensor(np.array(self.t1_heroes), dtype=torch.float32)
        self.map_ctx = torch.tensor(np.array(self.map_ctx), dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.t0_heroes[idx], self.t1_heroes[idx], self.map_ctx[idx], self.labels[idx]


# ── Training ──

def train_siamese_model(train_data, test_data, device, lr=1e-4, batch_size=4096,
                        max_epochs=200, patience=25):
    """Train the siamese model. Returns (model, best_acc)."""
    train_ds = SiameseWinProbDataset(train_data)
    test_ds = SiameseWinProbDataset(test_data)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False,
                             num_workers=4, pin_memory=True)

    model = SiameseWinProbModel().to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Siamese model: {params:,} parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    best_acc = 0
    best_loss = float('inf')
    patience_counter = 0
    save_path = os.path.join(os.path.dirname(__file__), "wp_independent_siamese.pt")

    for epoch in range(max_epochs):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        for t0h, t1h, mc, y in train_loader:
            t0h, t1h, mc, y = t0h.to(device), t1h.to(device), mc.to(device), y.to(device)
            pred = model(t0h, t1h, mc)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y)
            train_correct += ((pred > 0.5).float() == y).sum().item()
            train_total += len(y)

        # Eval
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for t0h, t1h, mc, y in test_loader:
                t0h, t1h, mc, y = t0h.to(device), t1h.to(device), mc.to(device), y.to(device)
                pred = model(t0h, t1h, mc)
                loss = criterion(pred, y)
                test_loss += loss.item() * len(y)
                test_correct += ((pred > 0.5).float() == y).sum().item()
                test_total += len(y)

        train_acc = train_correct / train_total * 100
        test_acc = test_correct / test_total * 100
        avg_test_loss = test_loss / test_total

        if epoch % 10 == 0 or epoch < 5:
            print(f"  Epoch {epoch+1:3d}: train_acc={train_acc:.2f}% test_acc={test_acc:.2f}% "
                  f"test_loss={avg_test_loss:.4f}")

        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Reload best
    model.load_state_dict(torch.load(save_path, weights_only=True, map_location=device))
    model.eval()
    print(f"  Best test accuracy: {best_acc:.2f}%")
    return model, best_acc


# ── Evaluation helpers ──

def make_siamese_eval_fn(model, device):
    """Create eval_fn compatible with test_wp_sanity.run_tests."""
    def eval_fn(t0_heroes, t1_heroes, game_map="Cursed Hollow", tier="mid"):
        t0 = torch.tensor([heroes_to_multi_hot(t0_heroes)], dtype=torch.float32).to(device)
        t1 = torch.tensor([heroes_to_multi_hot(t1_heroes)], dtype=torch.float32).to(device)
        mc = list(map_to_one_hot(game_map)) + list(tier_to_one_hot(tier))
        mc = torch.tensor([mc], dtype=torch.float32).to(device)
        with torch.no_grad():
            return model(t0, t1, mc).item()
    return eval_fn


def make_enriched_eval_fn(model_path, groups, group_indices, stats, device):
    """Create eval_fn for our enriched/naive/herostrength models."""
    cols = []
    for g in groups:
        s, e = group_indices[g]
        cols.extend(range(s, e))
    dim = 197 + len(cols)
    model = WinProbEnrichedModel(dim, [256, 128], dropout=0.3)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model.to(device).eval()
    all_mask = [True] * len(FEATURE_GROUPS)

    def eval_fn(t0_heroes, t1_heroes, game_map="Cursed Hollow", tier="mid"):
        d = {"team0_heroes": t0_heroes, "team1_heroes": t1_heroes,
             "game_map": game_map, "skill_tier": tier, "winner": 0}
        base, enriched = extract_features(d, stats, all_mask)
        enriched_sel = enriched[cols] if cols else np.array([], dtype=np.float32)
        x = np.concatenate([base, enriched_sel]) if len(enriched_sel) > 0 else base
        with torch.no_grad():
            return model(torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)).item()
    return eval_fn


# ── Main ──

COMPOSITIONS = {
    "5 tanks vs standard": {
        "t0": ["Muradin", "Johanna", "Diablo", "E.T.C.", "Mal'Ganis"],
        "t1": ["Muradin", "Brightwing", "Valla", "Sonya", "Jaina"],
    },
    "5 healers vs standard": {
        "t0": ["Brightwing", "Malfurion", "Rehgar", "Uther", "Anduin"],
        "t1": ["Muradin", "Brightwing", "Valla", "Sonya", "Jaina"],
    },
    "5 ranged assassins vs std": {
        "t0": ["Valla", "Jaina", "Li-Ming", "Falstad", "Raynor"],
        "t1": ["Muradin", "Brightwing", "Valla", "Sonya", "Jaina"],
    },
    "5 melee assassins vs std": {
        "t0": ["Zeratul", "Illidan", "Kerrigan", "Malthael", "Qhira"],
        "t1": ["Muradin", "Brightwing", "Valla", "Sonya", "Jaina"],
    },
    "3 tanks 2 healers vs std": {
        "t0": ["Muradin", "Johanna", "Diablo", "Brightwing", "Malfurion"],
        "t1": ["Muradin", "Brightwing", "Valla", "Sonya", "Jaina"],
    },
    "No healer high WR vs std": {
        "t0": ["Muradin", "Johanna", "Valla", "Falstad", "Li-Ming"],
        "t1": ["Muradin", "Brightwing", "Valla", "Sonya", "Jaina"],
    },
}


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    print("Loading replay data...")
    data = load_replay_data()
    train_data, test_data = split_data(data)
    print(f"  Train: {len(train_data)}, Test: {len(test_data)}")

    stats = StatsCache()
    group_indices = compute_group_indices()

    # ═══════════════════════════════════════════════════════════════
    # Step 1: Train siamese model
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 1: TRAIN SIAMESE MODEL (HairyBlob architecture)")
    print("=" * 70)

    save_path = os.path.join(os.path.dirname(__file__), "wp_independent_siamese.pt")
    if os.path.exists(save_path):
        print("  Loading existing model...")
        model = SiameseWinProbModel().to(device)
        model.load_state_dict(torch.load(save_path, weights_only=True, map_location=device))
        model.eval()
        # Quick accuracy check
        test_ds = SiameseWinProbDataset(test_data)
        test_loader = DataLoader(test_ds, batch_size=8192, shuffle=False, num_workers=4)
        correct = 0
        total = 0
        with torch.no_grad():
            for t0h, t1h, mc, y in test_loader:
                t0h, t1h, mc, y = t0h.to(device), t1h.to(device), mc.to(device), y.to(device)
                pred = model(t0h, t1h, mc)
                correct += ((pred > 0.5).float() == y).sum().item()
                total += len(y)
        siamese_acc = correct / total * 100
        print(f"  Loaded. Test accuracy: {siamese_acc:.2f}%")
    else:
        model, siamese_acc = train_siamese_model(train_data, test_data, device)

    siamese_eval = make_siamese_eval_fn(model, device)

    # ═══════════════════════════════════════════════════════════════
    # Step 2: Load our naive and enriched models
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 2: LOAD OUR NAIVE AND ENRICHED MODELS")
    print("=" * 70)

    model_configs = {
        "naive": {"groups": [], "path": "wp_experiment_naive.pt"},
        "enriched": {
            "groups": ["role_counts", "team_avg_wr", "map_delta", "pairwise_counters",
                       "pairwise_synergies", "counter_detail", "meta_strength",
                       "draft_diversity", "comp_wr"],
            "path": "wp_experiment_enriched.pt",
        },
    }

    eval_fns = {"independent": siamese_eval}
    for name, cfg in model_configs.items():
        path = os.path.join(os.path.dirname(__file__), cfg["path"])
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping {name}")
            continue
        eval_fns[name] = make_enriched_eval_fn(path, cfg["groups"], group_indices, stats, device)
        print(f"  Loaded {name} model from {cfg['path']}")

    # ═══════════════════════════════════════════════════════════════
    # Step 3: Sanity tests
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 3: SANITY TESTS")
    print("=" * 70)

    sanity_results = {}
    for name, eval_fn in eval_fns.items():
        print(f"\n--- {name.upper()} ---")
        passed, total, results_list = run_tests(eval_fn, verbose=True)

        cats = {}
        cat_totals = {}
        for t, p in zip(TESTS, results_list):
            cat = t.get("category", "")
            if cat:
                cat_totals[cat] = cat_totals.get(cat, 0) + 1
                cats[cat] = cats.get(cat, 0) + (1 if p else 0)

        sanity_results[name] = {"passed": passed, "total": total, "cats": cats, "cat_totals": cat_totals}

    print("\n" + "=" * 70)
    print("SANITY TEST COMPARISON")
    print("=" * 70)
    print(f"{'Model':<15} {'Absurd':<10} {'Trap':<10} {'Normal':<10} {'Symmetry':<10} {'Total':<10}")
    print("-" * 65)
    for name in ["independent", "naive", "enriched"]:
        if name not in sanity_results:
            continue
        r = sanity_results[name]
        cats = r["cats"]
        ct = r["cat_totals"]
        print(f"{name:<15} {cats.get('absurd',0)}/{ct.get('absurd',0):<8} "
              f"{cats.get('trap',0)}/{ct.get('trap',0):<8} "
              f"{cats.get('normal',0)}/{ct.get('normal',0):<8} "
              f"{cats.get('symmetry',0)}/{ct.get('symmetry',0):<8} "
              f"{r['passed']}/{r['total']}")

    # ═══════════════════════════════════════════════════════════════
    # Step 4: Composition comparison table
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 4: DEGENERATE COMPOSITION WP COMPARISON")
    print("=" * 70)

    model_names = [n for n in ["independent", "naive", "enriched"] if n in eval_fns]
    header = f"{'Composition':<28}" + "".join(f"{n:>14}" for n in model_names)
    print(header)
    print("-" * len(header))

    comp_results = {}
    for comp_name, comp in COMPOSITIONS.items():
        row = {}
        for name in model_names:
            wp = eval_fns[name](comp["t0"], comp["t1"], "Cursed Hollow", "mid")
            row[name] = wp
        comp_results[comp_name] = row

        vals = "".join(f"{row[n]:>14.3f}" for n in model_names)
        print(f"{comp_name:<28}{vals}")

    # ═══════════════════════════════════════════════════════════════
    # Save results
    # ═══════════════════════════════════════════════════════════════
    results_path = os.path.join(os.path.dirname(__file__), "experiment_results",
                                "independent_baseline_results.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    output = {
        "siamese_test_accuracy": siamese_acc,
        "sanity_tests": {
            name: {"passed": r["passed"], "total": r["total"],
                   "categories": r["cats"], "category_totals": r["cat_totals"]}
            for name, r in sanity_results.items()
        },
        "composition_comparison": comp_results,
    }
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
