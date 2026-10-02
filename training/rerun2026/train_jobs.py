"""
Phase 1 job runner: each subcommand trains ONE model, reusing the historical
training functions by import and writing checkpoints to rerun2026/models/
(never the historical training/*.pt files).

Where a historical script hardcodes an output directory (RESULTS_DIR,
SCRIPT_DIR, __file__-relative saves), we monkeypatch the module attribute
before calling its training function instead of editing the script.

Invoked by phase1_models.py as a subprocess with CUDA_VISIBLE_DEVICES set.
Every subcommand also honors --dry-run (print config, exit).
"""
import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rerun2026 import common
from rerun2026.common import (
    MODELS_DIR, CACHE_DIR, FULL_TRAIN_NPZ, FULL_TEST_NPZ, STEP_NPZ,
    CQL_NAIVE_TRAIN, CQL_NAIVE_TEST, CQL_ENR_TRAIN, CQL_ENR_TEST,
    GD_TRAIN, GD_TEST,
)

common.setup()

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# Enriched feature groups of the deployed WP models (9 groups, 86 dims)
from experiment_synthetic_augmentation import ENRICHED_GROUPS

WP_GROUP_PRESETS = {
    "naive": [],
    "true_base": [],  # same 197d features as naive; separate training run
    "herostrength": ["hero_wr", "team_avg_wr"],          # 197 + 12 = 209d
    "enriched": ENRICHED_GROUPS,                          # 197 + 86 = 283d
    # MANIFEST "Phase 4 addition", corrected: TRAINED-WITHOUT ablation models
    # (replaces the invalid inference-time-zeroed M/N attempt). Order is
    # ENRICHED_GROUPS-relative so the kept enriched columns match the kernel's
    # 86-dim layout (KERNEL_ENRICHED_LAYOUT) for phase4 wiring.
    # relational: drop ABSOLUTE groups -> 197 + 76 = 273d
    "relational": [g for g in ENRICHED_GROUPS
                   if g not in ("meta_strength", "comp_wr", "team_avg_wr")],
    # absolute: drop RELATIONAL groups -> 197 + 32 = 229d
    "absolute": [g for g in ENRICHED_GROUPS
                 if g not in ("counter_detail", "pairwise_counters",
                              "pairwise_synergies")],
}


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _cols_for(groups):
    from sweep_enriched_wp import compute_group_indices
    gi = compute_group_indices()
    cols = []
    for g in groups:
        s, e = gi[g]
        cols.extend(range(s, e))
    return cols


def _load_full_cache(path):
    cached = np.load(path)
    return (torch.tensor(cached["bases"]), torch.tensor(cached["enricheds"]),
            torch.tensor(cached["labels"]))


def _full_tensors(groups, device):
    """train_X, test_X, train_y, test_y for a WP feature-group subset."""
    cols = _cols_for(groups)
    tb, te, tl = _load_full_cache(FULL_TRAIN_NPZ)
    vb, ve, vl = _load_full_cache(FULL_TEST_NPZ)
    if cols:
        train_X = torch.cat([tb, te[:, cols]], dim=1).to(device)
        test_X = torch.cat([vb, ve[:, cols]], dim=1).to(device)
    else:
        train_X, test_X = tb.to(device), vb.to(device)
    return train_X, test_X, tl.to(device), vl.to(device), cols


# ── Memmap-backed datasets (drop-in for CQLDataset / DraftDataset) ──

class MemmapCQLDataset(Dataset):
    """Drop-in replacement for experiment_cql_draft.CQLDataset that takes a
    phase0 cache directory instead of a transition list. Valid masks are
    derived from the state (cols 0:270 = team0 + team1 + bans), which is
    exactly the mask the original transitions stored."""

    def __init__(self, cache_dir):
        if not isinstance(cache_dir, str):
            raise TypeError("MemmapCQLDataset expects a phase0 cache dir path")
        self._cache_dir = cache_dir
        self._open()

    def _open(self):
        mm = common.memmap_open(self._cache_dir)
        self.states = mm["states"]
        self.actions = mm["actions"]
        self.outcomes = mm["outcomes"]

    # Python 3.14 defaults to forkserver: DataLoader workers receive the
    # dataset by pickle, and np.memmap pickles as a full in-RAM ndarray
    # (58 GB per worker -> host OOM). Pickle only the path; reopen lazily.
    def __getstate__(self):
        return {"_cache_dir": self._cache_dir}

    def __setstate__(self, state):
        self._cache_dir = state["_cache_dir"]
        self._open()

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        state = np.array(self.states[idx], dtype=np.float32)
        occupied = state[:270].reshape(3, 90).sum(axis=0)
        mask = 1.0 - np.clip(occupied, 0.0, 1.0)
        return (torch.from_numpy(state),
                torch.tensor(int(self.actions[idx]), dtype=torch.long),
                torch.tensor(float(self.outcomes[idx]), dtype=torch.float32),
                torch.from_numpy(mask.astype(np.float32)))


class MemmapGDDataset(Dataset):
    """Drop-in for train_generic_draft.DraftDataset backed by phase0 memmaps."""

    def __init__(self, cache_dir):
        self._cache_dir = cache_dir
        self._open()

    def _open(self):
        mm = common.memmap_open(self._cache_dir)
        self.X = mm["states"]
        self.y = mm["actions"]

    def __getstate__(self):
        return {"_cache_dir": self._cache_dir}

    def __setstate__(self, state):
        self._cache_dir = state["_cache_dir"]
        self._open()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = np.array(self.X[idx], dtype=np.float32)
        occupied = x[:270].reshape(3, 90).sum(axis=0)
        mask = (1.0 - np.clip(occupied, 0.0, 1.0)).astype(np.float32)
        return (torch.from_numpy(x),
                torch.tensor(int(self.y[idx]), dtype=torch.long),
                torch.from_numpy(mask))


# ── Subcommands ──

def cmd_wp(args):
    """Full-draft WP variant (Table I / MCTS value functions).
    Reuses retrain_frozen_stats.train_wp_model (May-2026 rerun convention).
    --seeds > 1 keeps the best seed, matching the Table I protocol
    (archive/experiment_value_function_quality.py trained 3 seeds)."""
    from sweep_enriched_wp import WinProbEnrichedModel
    from retrain_frozen_stats import train_wp_model

    groups = WP_GROUP_PRESETS[args.groups]
    arch = [int(x) for x in args.arch.split(",")]
    device = _device()
    train_X, test_X, train_y, test_y, cols = _full_tensors(groups, device)
    dim = 197 + len(cols)
    print(f"wp {args.name}: dim={dim} arch={arch} seeds={args.seeds} "
          f"train={len(train_X):,} test={len(test_X):,}")

    best = {"acc": -1.0}
    accs = []
    for seed in [42, 123, 777][:args.seeds]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = WinProbEnrichedModel(dim, arch, dropout=0.3)
        model, acc = train_wp_model(model, train_X, test_X, train_y, test_y,
                                    f"{args.name}-s{seed}", device)
        accs.append(acc)
        if acc > best["acc"]:
            best = {"acc": acc, "seed": seed,
                    "state": {k: v.cpu().clone() for k, v in model.state_dict().items()}}

    out = os.path.join(MODELS_DIR, f"{args.name}.pt")
    torch.save(best["state"], out)
    common.write_meta(args.name, {
        "kind": "wp", "groups": groups, "arch": arch, "input_dim": dim,
        "best_acc": best["acc"], "best_seed": best["seed"], "all_accs": accs,
        "out": out,
    })
    print(f"{args.name}: best_acc={best['acc']:.2f}% -> {out}")


def cmd_wp_aug(args):
    """Augmented WP model: real training features (cached) + freshly extracted
    synthetic records. Mirrors retrain_frozen_stats.py sections 1-2 /
    rerun_wr_sweep.py training (arch + generator parameterized)."""
    import random
    from sweep_enriched_wp import WinProbEnrichedModel, FEATURE_GROUPS, extract_features
    from experiment_synthetic_augmentation import (
        generate_synthetic_data, generate_synthetic_data_v2)
    from retrain_frozen_stats import train_wp_model

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    arch = [int(x) for x in args.arch.split(",")]
    device = _device()
    stats = common.stats_cache()
    train_X, test_X, train_y, test_y, cols = _full_tensors(ENRICHED_GROUPS, device)
    dim = 197 + len(cols)
    all_mask = [True] * len(FEATURE_GROUPS)

    # Synthetic generation needs the raw train replays as the opponent pool
    train_data, _ = common.load_split()
    comp_path = os.path.join(common.TRAINING_DIR, "..", "src", "lib", "data",
                             "compositions.json")
    comp_data = json.load(open(comp_path))

    if args.generator == "v2":
        synthetic, syn_stats = generate_synthetic_data_v2(
            train_data, comp_data, stats,
            unseen_wr=args.wr, unseen_volume=args.volume, scope=args.scope)
    else:
        synthetic, syn_stats = generate_synthetic_data(
            train_data, comp_data,
            unseen_wr=args.wr, unseen_volume=args.volume, scope=args.scope)
    print(f"synthetic records: {syn_stats.get('total', len(synthetic))}")

    sb, se, sl = [], [], []
    for d in synthetic:
        try:
            b, e = extract_features(d, stats, all_mask)
            sb.append(b)
            se.append(e[cols])
            sl.append(float(d["winner"] == 0))
        except Exception:
            continue
    sX = torch.cat([torch.tensor(np.array(sb, dtype=np.float32)),
                    torch.tensor(np.array(se, dtype=np.float32))], dim=1).to(device)
    sy = torch.tensor(np.array(sl, dtype=np.float32)).to(device)
    aX = torch.cat([train_X, sX])
    ay = torch.cat([train_y, sy])
    pm = torch.randperm(len(aX))
    aX, ay = aX[pm], ay[pm]
    print(f"training data: {len(train_X):,} real + {len(sX):,} synthetic")

    model, acc = train_wp_model(WinProbEnrichedModel(dim, arch, dropout=0.3),
                                aX, test_X, ay, test_y, args.name, device)
    out = os.path.join(MODELS_DIR, f"{args.name}.pt")
    torch.save(model.cpu().state_dict(), out)
    common.write_meta(args.name, {
        "kind": "wp_aug", "arch": arch, "input_dim": dim, "generator": args.generator,
        "wr": args.wr, "volume": args.volume, "scope": args.scope, "seed": seed,
        "n_synthetic": len(sX), "best_acc": acc, "out": out,
    })
    print(f"{args.name}: acc={acc:.2f}% -> {out}")


def cmd_partial(args):
    """Step-conditioned WP model (PartialStateWP) from phase0 step caches.
    Default: replay-level split (the leakage fix). --sample-split reproduces
    the historical sample-level leakage bug (train_wp_512.py) so the paper's
    inflated-accuracy numbers (Section VI-C) can be regenerated for contrast."""
    from train_partial_wp import PartialStateWP

    torch.manual_seed(42)
    np.random.seed(42)
    device = _device()
    mode = args.mode
    input_dim = 283 if mode == "partial" else 197

    tr = np.load(STEP_NPZ[("train", mode)])
    te = np.load(STEP_NPZ[("test", mode)])
    if args.sample_split:
        # Reproduce the bug: pool everything, split 15% by SAMPLE
        F = np.concatenate([tr["features"], te["features"]])
        S = np.concatenate([tr["steps"], te["steps"]])
        L = np.concatenate([tr["labels"], te["labels"]])
        rng = np.random.RandomState(42)
        ix = rng.permutation(len(F))
        nt = int(len(F) * 0.15)
        te_ix, tr_ix = ix[:nt], ix[nt:]
        Xtr, Str_, Ytr = F[tr_ix], S[tr_ix], L[tr_ix]
        Xte, Ste, Yte = F[te_ix], S[te_ix], L[te_ix]
        split_kind = "sample-level (leakage-bug reproduction)"
    else:
        Xtr, Str_, Ytr = tr["features"], tr["steps"], tr["labels"]
        Xte, Ste, Yte = te["features"], te["steps"], te["labels"]
        split_kind = "replay-level"
    print(f"partial[{mode}] {split_kind}: train={len(Xtr):,} test={len(Xte):,} dim={input_dim}")

    Xtr = torch.tensor(Xtr).to(device)
    Str_ = torch.tensor(Str_).to(device)
    Ytr = torch.tensor(Ytr).to(device)
    Xte = torch.tensor(Xte).to(device)
    Ste = torch.tensor(Ste).to(device)
    Yte = torch.tensor(Yte).to(device)

    model = PartialStateWP(input_dim=input_dim, step_embed_dim=8,
                           hidden=(256, 128)).to(device)
    criterion = nn.BCELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)

    best_acc, best_state = 0.0, None
    n = len(Xtr)
    for ep in range(50):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, 4096):
            idx = perm[i:i + 4096]
            loss = criterion(model(Xtr[idx], Str_[idx]), Ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sch.step()
        if (ep + 1) % 5 == 0 or ep == 0:
            model.eval()
            correct = 0
            with torch.no_grad():
                for i in range(0, len(Xte), 65536):
                    p = model(Xte[i:i + 65536], Ste[i:i + 65536])
                    correct += ((p > 0.5).float() == Yte[i:i + 65536]).sum().item()
            acc = correct / len(Xte) * 100
            print(f"  ep{ep+1}: test_acc={acc:.2f}% (best {best_acc:.2f}%)")
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    # Per-step accuracy breakdown (paper's per-step table)
    model.to(device).eval()
    per_step = {}
    with torch.no_grad():
        preds = []
        for i in range(0, len(Xte), 65536):
            preds.append(model(Xte[i:i + 65536], Ste[i:i + 65536]).cpu())
        preds = torch.cat(preds)
    steps_np = Ste.cpu().numpy()
    yte_np = Yte.cpu().numpy()
    for s in np.unique(steps_np):
        m = steps_np == s
        per_step[int(s)] = float((((preds.numpy() > 0.5) == yte_np) & m).sum() / m.sum())

    out = os.path.join(MODELS_DIR, f"{args.name}.pt")
    torch.save({"model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "input_dim": input_dim, "step_embed_dim": 8, "hidden": [256, 128],
                "best_test_acc": best_acc / 100.0,
                "wp_groups": ENRICHED_GROUPS if mode == "partial" else []}, out)
    common.write_meta(args.name, {
        "kind": "partial_wp", "mode": mode, "split": split_kind,
        "input_dim": input_dim, "best_acc": best_acc, "per_step_acc": per_step,
        "out": out,
    })
    print(f"{args.name}: acc={best_acc:.2f}% -> {out}")


def cmd_gd(args):
    """Generic Draft variant. Reuses train_generic_draft.train_single_model
    unchanged; __file__ is monkeypatched so all saves (pt/onnx/quantized)
    land in rerun2026/models/."""
    import train_generic_draft as tgd

    tgd.__file__ = os.path.join(MODELS_DIR, "train_generic_draft.py")
    variant = tgd.MODEL_VARIANTS[args.variant]
    device = _device()
    train_ds = MemmapGDDataset(GD_TRAIN)
    test_ds = MemmapGDDataset(GD_TEST)
    print(f"gd variant {args.variant}: {len(train_ds):,} train / {len(test_ds):,} test samples")
    loss = tgd.train_single_model(args.variant, variant, train_ds, test_ds, device)
    common.write_meta(f"gd_{args.variant}", {
        "kind": "gd", "variant": variant, "best_test_loss": loss,
        "out": os.path.join(MODELS_DIR, f"generic_draft_{args.variant}.pt"),
    })


def cmd_gourdeau(args):
    """Gourdeau WP reimplementation (train_gourdeau_baseline). SCRIPT_DIR and
    the data loader are monkeypatched (output redirect + 2.55 filter)."""
    import train_gourdeau_baseline as tgb
    tgb.SCRIPT_DIR = MODELS_DIR
    tgb.load_replay_data = lambda *a, **kw: common.load_data()
    tgb.train()
    common.write_meta("gourdeau_wp", {
        "kind": "gourdeau", "out": os.path.join(MODELS_DIR, "gourdeau_wp.pt")})


def cmd_siamese(args):
    """Gourdeau independent-baseline Siamese network
    (archive/experiment_independent_baseline.train_siamese_model)."""
    import shutil
    from archive import experiment_independent_baseline as eib

    device = _device()
    train_data, test_data = common.load_split()
    result = eib.train_siamese_model(train_data, test_data, device)
    model = result[0] if isinstance(result, tuple) else result
    out = os.path.join(MODELS_DIR, "wp_independent_siamese.pt")
    torch.save(model.cpu().state_dict(), out)
    # train_siamese_model checkpoints into archive/ during training; tidy it up
    stray = os.path.join(os.path.dirname(eib.__file__), "wp_independent_siamese.pt")
    if os.path.exists(stray):
        os.remove(stray)
    meta = {"kind": "siamese", "out": out}
    if isinstance(result, tuple) and len(result) > 1:
        meta["best_acc"] = result[1]
    common.write_meta("wp_independent_siamese", meta)


def _cql_out_dir(sub):
    d = os.path.join(MODELS_DIR, sub)
    os.makedirs(d, exist_ok=True)
    return d


def cmd_cql_naive(args):
    import experiment_cql_draft as ecd
    ecd.RESULTS_DIR = _cql_out_dir("cql")
    ecd.CQLDataset = MemmapCQLDataset
    device = _device()
    model, metrics = ecd.train_cql(CQL_NAIVE_TRAIN, CQL_NAIVE_TEST,
                                   alpha=args.alpha, lr=3e-4, epochs=args.epochs,
                                   batch_size=2048, device=device)
    common.write_meta(f"cql_naive_a{args.alpha}", {
        "kind": "cql_naive", "alpha": args.alpha, "metrics": metrics,
        "out": os.path.join(ecd.RESULTS_DIR, f"_cql_temp_a{args.alpha}.pt")})


def cmd_cql_enr(args):
    import experiment_cql_enriched as ece
    ece.RESULTS_DIR = _cql_out_dir("cql")
    ece.EnrichedCQLDataset = MemmapCQLDataset
    device = _device()
    input_dim = common.memmap_meta(CQL_ENR_TRAIN)["state_dim"]
    model, best_loss = ece.train_enriched_cql(CQL_ENR_TRAIN, CQL_ENR_TEST,
                                              input_dim, alpha=args.alpha,
                                              epochs=args.epochs, device=device)
    common.write_meta(f"cql_enriched_a{args.alpha}", {
        "kind": "cql_enriched", "alpha": args.alpha, "input_dim": input_dim,
        "best_test_loss": best_loss,
        "out": os.path.join(ece.RESULTS_DIR, f"_cql_enriched_a{args.alpha}.pt")})


def cmd_cql_hp(args):
    import experiment_cql_hyperparams as ech
    ech.RESULTS_DIR = _cql_out_dir("cql_hyperparams")
    ech.CQLDataset = MemmapCQLDataset
    device = _device()
    hidden = tuple(int(x) for x in args.arch.split(","))
    model, metrics = ech.train_cql_flex(CQL_NAIVE_TRAIN, CQL_NAIVE_TEST,
                                        alpha=args.alpha, tau=args.tau,
                                        hidden_dims=hidden, lr=3e-4,
                                        epochs=args.epochs, batch_size=2048,
                                        device=device)
    arch_str = "x".join(str(d) for d in hidden)
    common.write_meta(f"cql_hp_a{args.alpha}_t{args.tau}_{arch_str}", {
        "kind": "cql_hp", "alpha": args.alpha, "tau": args.tau,
        "hidden_dims": list(hidden), "metrics": metrics,
        "out": os.path.join(ech.RESULTS_DIR,
                            f"_cql_temp_a{args.alpha}_t{args.tau}_{arch_str}.pt")})


def cmd_mcq(args):
    import experiment_mcq_draft as emd
    emd.RESULTS_DIR = _cql_out_dir("mcq")
    emd.CQLDataset = MemmapCQLDataset
    device = _device()
    model, metrics = emd.train_mcq(CQL_NAIVE_TRAIN, CQL_NAIVE_TEST,
                                   threshold=args.threshold, alpha=1.0,
                                   epochs=args.epochs, device=device)
    common.write_meta(f"mcq_t{args.threshold}", {
        "kind": "mcq", "threshold": args.threshold, "metrics": metrics,
        "out": os.path.join(emd.RESULTS_DIR, f"_mcq_temp_t{args.threshold}.pt")})


def cmd_bccql(args):
    import experiment_mcq_draft as emd
    emd.RESULTS_DIR = _cql_out_dir("mcq")
    emd.CQLDataset = MemmapCQLDataset
    device = _device()
    model, metrics = emd.train_bc_cql(CQL_NAIVE_TRAIN, CQL_NAIVE_TEST,
                                      alpha=1.0, bc_weight=args.bc_weight,
                                      epochs=args.epochs, device=device)
    common.write_meta(f"bccql_b{args.bc_weight}", {
        "kind": "bc_cql", "bc_weight": args.bc_weight, "metrics": metrics,
        "out": os.path.join(emd.RESULTS_DIR, f"_bc_cql_temp_bc{args.bc_weight}.pt")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("wp")
    p.add_argument("--name", required=True)
    p.add_argument("--groups", required=True, choices=list(WP_GROUP_PRESETS))
    p.add_argument("--arch", default="256,128")
    p.add_argument("--seeds", type=int, default=1)
    p.set_defaults(fn=cmd_wp)

    p = sub.add_parser("wp_aug")
    p.add_argument("--name", required=True)
    p.add_argument("--wr", type=float, required=True)
    p.add_argument("--volume", type=int, default=100)
    p.add_argument("--scope", default="tier2_only")
    p.add_argument("--arch", default="512,256,128")
    p.add_argument("--generator", default="flat", choices=["flat", "v2"])
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(fn=cmd_wp_aug)

    p = sub.add_parser("partial")
    p.add_argument("--name", required=True)
    p.add_argument("--mode", required=True, choices=["partial", "base"])
    p.add_argument("--sample-split", action="store_true")
    p.set_defaults(fn=cmd_partial)

    p = sub.add_parser("gd")
    p.add_argument("--variant", type=int, required=True)
    p.set_defaults(fn=cmd_gd)

    p = sub.add_parser("gourdeau")
    p.set_defaults(fn=cmd_gourdeau)

    p = sub.add_parser("siamese")
    p.set_defaults(fn=cmd_siamese)

    p = sub.add_parser("cql_naive")
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.set_defaults(fn=cmd_cql_naive)

    p = sub.add_parser("cql_enr")
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.set_defaults(fn=cmd_cql_enr)

    p = sub.add_parser("cql_hp")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--tau", type=float, required=True)
    p.add_argument("--arch", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.set_defaults(fn=cmd_cql_hp)

    p = sub.add_parser("mcq")
    p.add_argument("--threshold", type=float, required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.set_defaults(fn=cmd_mcq)

    p = sub.add_parser("bccql")
    p.add_argument("--bc-weight", type=float, required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.set_defaults(fn=cmd_bccql)

    args = parser.parse_args()
    if args.dry_run:
        print(f"dry run: {args.cmd} {vars(args)}")
        return
    t0 = time.time()
    args.fn(args)
    print(f"job done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
