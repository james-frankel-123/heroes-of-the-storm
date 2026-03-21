"""
Experiment: Win Probability model architecture scaling.
Tests deep residual nets + transformer on separate GPUs.

Usage:
    set -a && source .env && set +a
    CUDA_VISIBLE_DEVICES=0 python3 -u training/experiment_wp_size.py resnet_4x
    CUDA_VISIBLE_DEVICES=1 python3 -u training/experiment_wp_size.py resnet_18x
    CUDA_VISIBLE_DEVICES=2 python3 -u training/experiment_wp_size.py resnet_48x
    CUDA_VISIBLE_DEVICES=3 python3 -u training/experiment_wp_size.py transformer
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from shared import (
    NUM_HEROES, NUM_MAPS, NUM_TIERS,
    heroes_to_multi_hot, map_to_one_hot, tier_to_one_hot,
    load_replay_data, split_data,
)

INPUT_DIM = NUM_HEROES * 2 + NUM_MAPS + NUM_TIERS  # 197


# ── Datasets ──

class WinProbDataset(Dataset):
    """Flat multi-hot encoding for MLP/ResNet models."""
    def __init__(self, data):
        self.X = []
        self.y = []
        for d in data:
            t0 = heroes_to_multi_hot(d["team0_heroes"])
            t1 = heroes_to_multi_hot(d["team1_heroes"])
            m = map_to_one_hot(d["game_map"])
            t = tier_to_one_hot(d["skill_tier"])
            self.X.append(np.concatenate([t0, t1, m, t]))
            self.y.append(float(d["winner"] == 0))
        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])


HERO_TO_IDX = {}
HEROES = [
    "Abathur","Alarak","Alexstrasza","Ana","Anduin","Anub'arak","Artanis",
    "Arthas","Auriel","Azmodan","Blaze","Brightwing","Cassia","Chen","Cho",
    "Chromie","D.Va","Deathwing","Deckard","Dehaka","Diablo","E.T.C.",
    "Falstad","Fenix","Gall","Garrosh","Gazlowe","Genji","Greymane",
    "Gul'dan","Hanzo","Hogger","Illidan","Imperius","Jaina","Johanna",
    "Junkrat","Kael'thas","Kel'Thuzad","Kerrigan","Kharazim","Leoric",
    "Li Li","Li-Ming","Lt. Morales","Lunara","Lúcio","Maiev","Mal'Ganis",
    "Malfurion","Malthael","Medivh","Mei","Mephisto","Muradin","Murky",
    "Nazeebo","Nova","Orphea","Probius","Qhira","Ragnaros","Raynor",
    "Rehgar","Rexxar","Samuro","Sgt. Hammer","Sonya","Stitches","Stukov",
    "Sylvanas","Tassadar","The Butcher","The Lost Vikings","Thrall","Tracer",
    "Tychus","Tyrael","Tyrande","Uther","Valeera","Valla","Varian",
    "Whitemane","Xul","Yrel","Zagara","Zarya","Zeratul","Zul'jin",
]
for i, h in enumerate(HEROES):
    HERO_TO_IDX[h] = i

MAPS = [
    "Alterac Pass", "Battlefield of Eternity", "Blackheart's Bay",
    "Braxis Holdout", "Cursed Hollow", "Dragon Shire",
    "Garden of Terror", "Hanamura Temple", "Infernal Shrines",
    "Sky Temple", "Tomb of the Spider Queen", "Towers of Doom",
    "Volskaya Foundry", "Warhead Junction",
]
MAP_TO_IDX = {m: i for i, m in enumerate(MAPS)}


class TransformerWPDataset(Dataset):
    """Per-hero token encoding for transformer model."""
    def __init__(self, data):
        self.samples = []
        for d in data:
            t0_idx = [HERO_TO_IDX[h] for h in d["team0_heroes"] if h in HERO_TO_IDX]
            t1_idx = [HERO_TO_IDX[h] for h in d["team1_heroes"] if h in HERO_TO_IDX]
            map_idx = MAP_TO_IDX.get(d["game_map"], 0)
            tier_idx = ["low", "mid", "high"].index(d["skill_tier"]) if d["skill_tier"] in ["low", "mid", "high"] else 1
            y = float(d["winner"] == 0)
            self.samples.append((t0_idx, t1_idx, map_idx, tier_idx, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        t0, t1, map_idx, tier_idx, y = self.samples[idx]
        # Tokens: [CLS] [TEAM0_1..5] [SEP] [TEAM1_1..5] [MAP] [TIER]
        # Hero IDs: 0-89, SEP=90, CLS=91, MAP offset=92, TIER offset=92+14=106
        tokens = [91]  # CLS
        tokens.extend(t0)  # team 0 heroes
        tokens.append(90)  # SEP
        tokens.extend(t1)  # team 1 heroes
        tokens.append(92 + map_idx)  # map token
        tokens.append(106 + tier_idx)  # tier token
        # Segment IDs: 0=CLS, 1=team0, 2=SEP, 3=team1, 4=map, 5=tier
        segments = [0]
        segments.extend([1] * len(t0))
        segments.append(2)
        segments.extend([3] * len(t1))
        segments.append(4)
        segments.append(5)
        return (torch.tensor(tokens, dtype=torch.long),
                torch.tensor(segments, dtype=torch.long),
                torch.tensor(y, dtype=torch.float32))


def collate_transformer(batch):
    tokens_list, segs_list, ys = zip(*batch)
    max_len = max(t.size(0) for t in tokens_list)
    padded_tokens = torch.zeros(len(batch), max_len, dtype=torch.long)
    padded_segs = torch.zeros(len(batch), max_len, dtype=torch.long)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, (t, s) in enumerate(zip(tokens_list, segs_list)):
        padded_tokens[i, :t.size(0)] = t
        padded_segs[i, :s.size(0)] = s
        mask[i, :t.size(0)] = True
    return padded_tokens, padded_segs, mask, torch.stack(ys)


# ── Models ──

class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(x))
        out = self.dropout(self.fc1(out))
        out = F.relu(self.bn2(out))
        out = self.fc2(out)
        return out + residual


class WPResNet(nn.Module):
    def __init__(self, width, num_blocks, compress_to=128, dropout=0.2):
        super().__init__()
        self.input_fc = nn.Linear(INPUT_DIM, width)
        self.input_bn = nn.BatchNorm1d(width)
        self.blocks = nn.Sequential(*[ResBlock(width, dropout) for _ in range(num_blocks)])
        self.compress = nn.Linear(width, compress_to)
        self.compress_bn = nn.BatchNorm1d(compress_to)
        self.output = nn.Linear(compress_to, 1)

    def forward(self, x):
        h = F.relu(self.input_bn(self.input_fc(x)))
        h = self.blocks(h)
        h = F.relu(self.compress_bn(self.compress(h)))
        return torch.sigmoid(self.output(h)).squeeze(-1)


class WPTransformer(nn.Module):
    """
    Transformer WP model. Each hero is a token with a learned embedding.
    Uses segment embeddings for team identity, CLS token for prediction.
    """
    def __init__(self, d_model=128, nhead=8, num_layers=6, dropout=0.2):
        super().__init__()
        # 90 heroes + SEP(90) + CLS(91) + 14 maps(92-105) + 3 tiers(106-108) = 109 tokens
        self.token_embed = nn.Embedding(109, d_model)
        self.segment_embed = nn.Embedding(6, d_model)  # 0=CLS,1=team0,2=SEP,3=team1,4=map,5=tier
        self.pos_embed = nn.Embedding(16, d_model)  # max 14 tokens

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, tokens, segments, mask):
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device).unsqueeze(0).expand(B, T)
        h = self.token_embed(tokens) + self.segment_embed(segments) + self.pos_embed(pos)
        # Transformer expects key_padding_mask=True for padding positions
        h = self.encoder(h, src_key_padding_mask=~mask)
        h = self.norm(h)
        # Use CLS token (position 0) for prediction
        cls = h[:, 0]
        return torch.sigmoid(self.head(cls)).squeeze(-1)


# ── Training ──

def train_model(name, model, train_dl, test_dl, device, is_transformer=False):
    params = sum(p.numel() for p in model.parameters())
    print(f"{name}: {params:,} params")
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
    criterion = nn.BCELoss()

    best_test_loss = float('inf')
    best_test_acc = 0.0
    best_epoch = 0
    patience = 25
    patience_counter = 0

    for epoch in range(200):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch in train_dl:
            if is_transformer:
                tokens, segs, mask, y = batch
                tokens, segs, mask, y = tokens.to(device), segs.to(device), mask.to(device), y.to(device)
                pred = model(tokens, segs, mask)
            else:
                X, y = batch
                X, y = X.to(device), y.to(device)
                pred = model(X)

            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(y)
            train_correct += ((pred > 0.5).float() == y).sum().item()
            train_total += len(y)

        scheduler.step()

        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for batch in test_dl:
                if is_transformer:
                    tokens, segs, mask, y = batch
                    tokens, segs, mask, y = tokens.to(device), segs.to(device), mask.to(device), y.to(device)
                    pred = model(tokens, segs, mask)
                else:
                    X, y = batch
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
            gap = train_acc - test_acc
            lr = optimizer.param_groups[0]['lr']
            print(f"  {name} ep {epoch+1:3d}: train={train_acc:.1f}% test={test_acc:.1f}% "
                  f"gap={gap:.1f}% loss={avg_test_loss:.6f} lr={lr:.6f}")

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            best_test_acc = test_acc
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  {name} early stop at epoch {epoch+1}")
                break

    print(f"  {name} RESULT: best_acc={best_test_acc:.2f}% epoch={best_epoch} params={params:,}")
    return best_test_acc, best_epoch


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "resnet_4x"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}, Variant: {variant}")

    data = load_replay_data()
    train_data, test_data = split_data(data)
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")

    if variant == "transformer":
        train_ds = TransformerWPDataset(train_data)
        test_ds = TransformerWPDataset(test_data)
        train_dl = DataLoader(train_ds, batch_size=512, shuffle=True, collate_fn=collate_transformer)
        test_dl = DataLoader(test_ds, batch_size=512, collate_fn=collate_transformer)

        # Test multiple transformer sizes
        for name, kwargs in [
            ("xfmr_small",  dict(d_model=128, nhead=8,  num_layers=6,  dropout=0.2)),
            ("xfmr_medium", dict(d_model=256, nhead=8,  num_layers=8,  dropout=0.2)),
            ("xfmr_large",  dict(d_model=384, nhead=8,  num_layers=12, dropout=0.25)),
        ]:
            model = WPTransformer(**kwargs)
            train_model(name, model, train_dl, test_dl, device, is_transformer=True)

    else:
        train_ds = WinProbDataset(train_data)
        test_ds = WinProbDataset(test_data)
        train_dl = DataLoader(train_ds, batch_size=1024, shuffle=True)
        test_dl = DataLoader(test_ds, batch_size=1024)

        configs = {
            "resnet_4x":  (512,  8, 128, 0.2),
            "resnet_18x": (768, 16, 256, 0.25),
            "resnet_48x": (1024, 24, 256, 0.3),
        }
        width, blocks, compress, dropout = configs[variant]
        model = WPResNet(width, blocks, compress, dropout)
        train_model(variant, model, train_dl, test_dl, device)


if __name__ == "__main__":
    main()
