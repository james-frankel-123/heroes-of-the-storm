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
import json
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
    load_replay_data, split_data, embed_onnx_weights,
    optimize_onnx, quantize_onnx, verify_quantized_model,
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


class CompactDraftDataset(Dataset):
    """DraftDataset streamed to disk and memory-mapped, for corpora whose
    float32 sample arrays do not fit in RAM (2.4M replays -> ~30M samples:
    35 GB of states + 11 GB of masks as float32).

    Same samples, same tensors. States are stored as uint8: every entry is
    0/1 except step_num = pick_number / 15, stored as pick_number and decoded
    exactly as replay_to_training_samples computes it. The valid mask is not
    stored: it is 1 - (team0 | team1 | bans), which is how the sample builder
    defines it. Files: <cache_dir>/{states.u8,targets.i16,meta.json}; meta is
    written last, so an interrupted build is redone and a finished one reused.
    """
    STEP_COL = NUM_HEROES * 3 + NUM_MAPS + NUM_TIERS

    def __init__(self, data, cache_dir, chunk=200_000):
        meta_p = os.path.join(cache_dir, "meta.json")
        if not os.path.exists(meta_p):
            self._build(data, cache_dir, chunk)
        meta = json.load(open(meta_p))
        assert meta["dim"] == INPUT_DIM, f"cache dim {meta['dim']} != {INPUT_DIM}"
        self.n = meta["n"]
        self.X = np.memmap(os.path.join(cache_dir, "states.u8"), dtype=np.uint8,
                           mode="r", shape=(self.n, INPUT_DIM))
        self.y = np.memmap(os.path.join(cache_dir, "targets.i16"), dtype=np.int16,
                           mode="r", shape=(self.n,))

    def _build(self, data, cache_dir, chunk):
        os.makedirs(cache_dir, exist_ok=True)
        n = 0
        bx, by = [], []
        with open(os.path.join(cache_dir, "states.u8"), "wb") as fx, \
                open(os.path.join(cache_dir, "targets.i16"), "wb") as fy:
            def flush():
                if bx:
                    np.stack(bx).tofile(fx)
                    np.asarray(by, dtype=np.int16).tofile(fy)
                    bx.clear()
                    by.clear()
            for d in data:
                for x, target, _ in replay_to_training_samples(d):
                    u = x.astype(np.uint8)
                    u[self.STEP_COL] = int(round(x[self.STEP_COL] * 15))
                    bx.append(u)
                    by.append(target)
                    n += 1
                if len(bx) >= chunk:
                    flush()
            flush()
        json.dump({"n": n, "dim": INPUT_DIM, "heroes": NUM_HEROES, "maps": NUM_MAPS},
                  open(os.path.join(cache_dir, "meta.json"), "w"))

    def decode(self, rows):
        x = rows.astype(np.float32)
        x[:, self.STEP_COL] = (rows[:, self.STEP_COL].astype(np.float64) / 15.0).astype(np.float32)
        taken = np.minimum(x[:, :NUM_HEROES] + x[:, NUM_HEROES:2 * NUM_HEROES]
                           + x[:, 2 * NUM_HEROES:3 * NUM_HEROES], 1.0)
        return x, (1.0 - taken).astype(np.float32)

    def __len__(self):
        return self.n

    def __getitems__(self, idx):
        # batched fetch (DataLoader calls this with a batch's indices)
        order = np.asarray(idx)
        x, mask = self.decode(np.asarray(self.X[order]))
        y = np.asarray(self.y[order], dtype=np.int64)
        X, Y, M = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask)
        return [(X[i], Y[i], M[i]) for i in range(len(order))]

    def __getitem__(self, idx):
        return self.__getitems__([idx])[0]


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

    # Opt-in epoch-level resume (GD_RESUME_PATH): after every epoch the model,
    # optimizer, early-stopping state, data-order generator and RNG states are
    # written atomically; a rerun of the same command continues from the last
    # finished epoch. Unset = original behaviour.
    resume_path = os.environ.get("GD_RESUME_PATH")
    start_epoch = 0
    st = None
    if resume_path and os.path.exists(resume_path):
        st = torch.load(resume_path, weights_only=False, map_location="cpu")
        if st.get("loss_kind") != "legal":   # saved before the legal-target fix
            print("  ignoring resume state saved with the old loss; starting fresh")
            st = None
    if st is not None:
        model.load_state_dict(st["model"])
        optimizer.load_state_dict(st["optimizer"])
        best_test_loss, patience_counter = st["best_test_loss"], st["patience_counter"]
        start_epoch = st["epoch"] + 1
        train_dl.generator.set_state(st["gen_state"])
        torch.set_rng_state(st["torch_rng"])
        if torch.cuda.is_available() and st.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(st["cuda_rng"])
        np.random.set_state(st["np_rng"])
        print(f"  resumed after epoch {start_epoch} (best test loss {best_test_loss:.4f})")
        if st.get("stopped"):
            start_epoch = 200

    # A few replays record a pick of a hero that is already picked or banned
    # (about 3 in 100,000 samples). Their target is masked out of the legal
    # set, so the masked cross-entropy is ~1e9 per row: harmless for the
    # gradient (softmax minus one-hot, bounded) but it swamps the summed eval
    # loss and made checkpoint selection and early stopping track 1e9 x the
    # count of such rows. Rows whose target is not a legal action are dropped
    # from training and evaluation; selection and stopping use the
    # cross-entropy over legal-target rows.
    def _legal(y, mask):
        return mask.gather(1, y.unsqueeze(1)).squeeze(1) > 0.5

    skipped = {"train": 0, "test": 0}
    for epoch in range(start_epoch, 200):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        for X, y, mask in train_dl:
            X, y, mask = X.to(device), y.to(device), mask.to(device)
            ok = _legal(y, mask)
            if not bool(ok.all()):
                skipped["train"] += int((~ok).sum())
                X, y, mask = X[ok], y[ok], mask[ok]
                if len(y) == 0:
                    continue
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
                ok = _legal(y, mask)
                if not bool(ok.all()):
                    skipped["test"] += int((~ok).sum())
                    X, y, mask = X[ok], y[ok], mask[ok]
                    if len(y) == 0:
                        continue
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

        print(f"  Epoch {epoch+1:3d}: train_acc={train_acc:.1f}% "
              f"test_acc(top1)={test_acc:.2f}% test_top5={test_top5_acc:.1f}% "
              f"test_loss={avg_test_loss:.5f} (legal-target rows; skipped "
              f"train {skipped['train']} test {skipped['test']})", flush=True)
        skipped = {"train": 0, "test": 0}

        stop = False
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            patience_counter = 0
            torch.save(model.state_dict(), pt_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                stop = True
        if resume_path:
            tmp = resume_path + ".tmp"
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "best_test_loss": best_test_loss, "patience_counter": patience_counter,
                        "epoch": epoch, "gen_state": train_dl.generator.get_state(),
                        "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                        "np_rng": np.random.get_state(), "stopped": stop,
                        "loss_kind": "legal"}, tmp)
            os.replace(tmp, resume_path)
        if stop:
            break

    # Export to ONNX (on CPU to avoid device mismatch)
    model.load_state_dict(torch.load(pt_path, weights_only=True, map_location="cpu"))
    model.cpu().eval()
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
    embed_onnx_weights(onnx_path)
    print(f"  Exported ONNX: {onnx_path} ({os.path.getsize(onnx_path) / 1024:.1f} KB)")

    # Optimize + quantize (only variant 0 — used in browser)
    if variant_idx == 0:
        optimize_onnx(onnx_path)
        calib_data = load_replay_data(limit=2000)
        quant_path = quantize_onnx(onnx_path, calib_data, model_type="gd")
        verify_quantized_model(onnx_path, quant_path, calib_data, model_type="gd")

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
