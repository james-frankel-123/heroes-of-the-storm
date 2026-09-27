"""
PHASE D2 job runner: trains ONE enriched WP model under a drift regime and
evaluates it on the strictly-future test builds. Invoked by phase_d2.py as a
subprocess with CUDA_VISIBLE_DEVICES set.

Protocol = paper-1 wp_enriched_256 (WinProbEnrichedModel 283d -> 256 -> 128,
dropout 0.3, AdamW lr 5e-4 wd 5e-3, cosine T_max=100, batch 4096, <=200
epochs, patience 25) with two deliberate departures, both drift-specific:
  1. TEMPORAL split (train = builds <= cutoff, test = builds > cutoff)
     instead of a random 2% split; early stopping uses a replay-level 2%
     VALIDATION slice of the TRAIN period, never the future test set.
  2. Optional per-sample weights (time-decay regime) and a learned
     minor-build embedding (patch-embedding regime).

Regimes (--regime):
  all      train on every build <= cutoff
  window   train on the last --window builds ending at the cutoff
  decay    all-history with sample weight 0.5^(age_days / --half-life)
  embed    all-history + learned per-build embedding (test rows use the
           cutoff build's embedding — a deployed model's only causal choice)

Features (--features) select the stats sourcing pass of
build_drift_features.py: cutoff | local | cumulative_prev | frozen.
D2(b) uses cutoff for all regimes; D2(c) compares feature passes at
regime=all.

Outputs: models/<name>.pt, results/d2/<name>.json (accuracies overall /
per-test-build / sanity suite).
"""
import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drift2026 import common

common.setup()

import numpy as np
import torch
import torch.nn as nn

from sweep_enriched_wp import WinProbEnrichedModel, compute_group_indices
from experiment_synthetic_augmentation import ENRICHED_GROUPS

VAL_FRAC = 0.02


def enriched_cols():
    gi = compute_group_indices()
    cols = []
    for g in ENRICHED_GROUPS:
        s, e = gi[g]
        cols.extend(range(s, e))
    return cols


class WPWithPatchEmbed(nn.Module):
    """wp_enriched_256 trunk + learned embedding over minor-build index."""

    def __init__(self, input_dim, n_patches, embed_dim=8,
                 arch=(256, 128), dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(n_patches, embed_dim)
        self.net = WinProbEnrichedModel(input_dim + embed_dim, list(arch), dropout)

    def forward(self, x, p):
        return self.net(torch.cat([x, self.embed(p)], dim=1))


def train_model(model, Xtr, ytr, Xva, yva, device, wtr=None, ptr=None, pva=None,
                name="", lr=5e-4, max_epochs=200):
    """train_wp_model protocol + optional weights / patch indices."""
    model.to(device)
    criterion = nn.BCELoss(reduction="none")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100,
                                                     eta_min=min(1e-5, lr))
    best_loss, best_acc, best_state, pat = float("inf"), 0.0, None, 0
    n = len(Xtr)
    for ep in range(max_epochs):
        model.train()
        pm = torch.randperm(n, device=device)
        for i in range(0, n, 4096):
            idx = pm[i:i + 4096]
            pred = (model(Xtr[idx], ptr[idx]) if ptr is not None
                    else model(Xtr[idx]))
            losses = criterion(pred, ytr[idx])
            if wtr is not None:
                losses = losses * wtr[idx]
                loss = losses.sum() / wtr[idx].sum()
            else:
                loss = losses.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sch.step()
        model.eval()
        with torch.no_grad():
            preds = []
            for i in range(0, len(Xva), 65536):
                preds.append(model(Xva[i:i + 65536], pva[i:i + 65536])
                             if pva is not None else model(Xva[i:i + 65536]))
            vp = torch.cat(preds)
            vl = criterion(vp, yva).mean().item()
            va = ((vp > 0.5).float() == yva).float().mean().item() * 100
        if vl < best_loss:
            best_loss, best_acc = vl, va
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= 25:
                break
        if (ep + 1) % 20 == 0:
            print(f"  {name} ep{ep+1}: val_acc={va:.2f}% best={best_acc:.2f}%",
                  flush=True)
    model.load_state_dict(best_state)
    model.eval()
    return model, best_acc, ep + 1


@torch.no_grad()
def batch_acc(model, X, y, p=None):
    preds = []
    for i in range(0, len(X), 65536):
        preds.append(model(X[i:i + 65536], p[i:i + 65536]) if p is not None
                     else model(X[i:i + 65536]))
    pred = torch.cat(preds)
    return ((pred > 0.5).float() == y).float().mean().item() * 100


def sanity_suite(eval_fn):
    """phase3_benchmarks._sanity_for, inlined (21/28 suites + degen scores)."""
    from test_wp_sanity import TESTS, run_tests
    from experiment_synthetic_augmentation import DEGEN_COMPS, STANDARD
    passed, total, results_list = run_tests(eval_fn, verbose=False)
    passed_21 = sum(1 for p in results_list[:21] if p)
    by_cat = {}
    for t, p in zip(TESTS, results_list):
        c = t.get("category", "")
        by_cat.setdefault(c, [0, 0])
        by_cat[c][1] += 1
        if p:
            by_cat[c][0] += 1
    degen = {name: eval_fn(comp, STANDARD, "Cursed Hollow", "mid")
             for name, comp in DEGEN_COMPS.items()}
    return {"passed_28": passed, "passed_21": passed_21,
            "by_category": {c: f"{a}/{b}" for c, (a, b) in by_cat.items()},
            "degen_scores": degen}


def deploy_stats(features):
    """Stats a deployed model of this arm would use at test time (for the
    sanity suite's feature extraction)."""
    idx = common.load_patch_index()
    builds_255 = [b for b in idx["builds"] if b.startswith("2.55")]
    if features == "cutoff" or features.startswith("causal"):
        # causal_* arms deploy with the cutoff prior (their day-0-of-build
        # state; within-build local counts start empty at deployment).
        return common.load_patch_stats("cumulative", common.TRAIN_CUTOFF_BUILD)
    if features.startswith("cutoff_"):
        return common.load_patch_stats("cumulative", features[len("cutoff_"):])
    if features == "hybrid":
        # W3(d): the hybrid file of the LAST build = window + cumulative
        # through the last completed build (cumprev deploy convention).
        return common.load_patch_stats("hybrid", builds_255[-1])
    if features == "local":
        return common.load_patch_stats("per_patch", builds_255[-1])
    if features == "cumulative_prev":
        return common.load_patch_stats("cumulative", builds_255[-2])
    if features.startswith("decayed") and features.endswith("_prev"):
        # Q7 decayed-aggregate arms: same prev-build deploy convention
        return common.load_patch_stats(features[:-len("_prev")], builds_255[-2])
    from sweep_enriched_wp import StatsCache
    return StatsCache()   # frozen


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    ap.add_argument("--features", required=True,
                    help="feature pass name: any feature_cache/features_<X>.npz "
                         "(cutoff | local | cumulative_prev | frozen | causal_* "
                         "| cutoff_<build>)")
    ap.add_argument("--regime", required=True,
                    choices=["all", "window", "decay", "embed"])
    ap.add_argument("--window", type=int, default=0, help="builds (regime=window)")
    ap.add_argument("--half-life", type=float, default=0.0,
                    help="days (regime=decay)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--embed-dim", type=int, default=8)
    ap.add_argument("--cutoff-build", default=common.TRAIN_CUTOFF_BUILD,
                    help="train on builds <= this build (W2c timeline replay)")
    ap.add_argument("--skip-sanity", action="store_true",
                    help="skip the sanity suite (W2c timeline jobs)")
    ap.add_argument("--results-subdir", default="d2")
    ap.add_argument("--init-from", default=None,
                    help="warm-start: load this checkpoint's state_dict "
                         "before training (Q8 finetune chain)")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--max-train-rows", type=int, default=0,
                    help="W8a volume-matched control: seeded random subsample "
                         "of the (post-val-split) training rows to this count")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.dry_run:
        print(f"dry run: {vars(args)}")
        return

    t_start = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    npz_path = os.path.join(common.CACHE_DIR, f"features_{args.features}.npz")
    z = np.load(npz_path)
    bases, enr = z["bases"], z["enricheds"]
    labels = z["labels"]
    build_idx = z["build_idx"].astype(np.int64)
    date_days = z["date_days"]
    rids = z["replay_ids"]

    builds = common.load_patch_index()["builds"]
    present = np.unique(build_idx)
    pos_of = {int(b): i for i, b in enumerate(present)}
    cpos = pos_of[builds.index(args.cutoff_build)]
    pos = np.vectorize(pos_of.get, otypes=[np.int64])(build_idx)

    cols = enriched_cols()
    X = np.concatenate([bases, enr[:, cols]], axis=1)
    del bases, enr
    print(f"{args.name}: X={X.shape} cutoff_pos={cpos}/{len(present)-1}")

    train_mask = pos <= cpos
    if args.regime == "window":
        assert args.window > 0
        train_mask &= pos > cpos - args.window
    test_mask = pos > cpos

    # replay-level validation split within the train period (fixed seed 42,
    # independent of --seed, matching rerun2026's fixed-split convention)
    tr_rids = np.unique(rids[train_mask])
    rng = np.random.RandomState(common.SEED)
    val_ids = set(tr_rids[rng.permutation(len(tr_rids))
                          [:max(1, int(len(tr_rids) * VAL_FRAC))]].tolist())
    is_val = np.fromiter((int(r) in val_ids for r in rids), bool, len(rids))
    va_mask = train_mask & is_val
    tr_mask = train_mask & ~is_val

    if args.max_train_rows and int(tr_mask.sum()) > args.max_train_rows:
        tr_idx = np.flatnonzero(tr_mask)
        keep = np.random.RandomState(args.seed).choice(
            tr_idx, size=args.max_train_rows, replace=False)
        tr_mask = np.zeros_like(tr_mask)
        tr_mask[keep] = True
        print(f"  volume match: subsampled train rows "
              f"{len(tr_idx):,} -> {args.max_train_rows:,} (seed {args.seed})")

    def T(a, dtype=torch.float32):
        return torch.tensor(a, dtype=dtype, device=device)

    Xtr, ytr = T(X[tr_mask]), T(labels[tr_mask])
    Xva, yva = T(X[va_mask]), T(labels[va_mask])
    Xte, yte = T(X[test_mask]), T(labels[test_mask])
    print(f"  train={len(Xtr):,} val={len(Xva):,} test={len(Xte):,} rows")

    wtr = None
    if args.regime == "decay":
        assert args.half_life > 0
        age = date_days[tr_mask].max() - date_days[tr_mask]
        w = np.power(0.5, age / args.half_life).astype(np.float32)
        wtr = T(w)
        print(f"  decay hl={args.half_life}d: sum_w={w.sum():,.0f} "
              f"(effective fraction {w.mean():.3f})")

    ptr = pva = pte = None
    model_cfg = {"input_dim": X.shape[1], "arch": [256, 128], "dropout": 0.3}
    if args.regime == "embed":
        n_patches = cpos + 1
        model = WPWithPatchEmbed(X.shape[1], n_patches, args.embed_dim)
        ptr = T(np.minimum(pos[tr_mask], cpos), torch.long)
        pva = T(np.minimum(pos[va_mask], cpos), torch.long)
        pte = T(np.minimum(pos[test_mask], cpos), torch.long)  # clamped: causal
        model_cfg.update({"embed": True, "n_patches": n_patches,
                          "embed_dim": args.embed_dim})
    else:
        model = WinProbEnrichedModel(X.shape[1], [256, 128], dropout=0.3)

    if args.init_from:
        ck = torch.load(args.init_from, map_location="cpu")
        model.load_state_dict(ck["state_dict"])
        print(f"  warm-start from {args.init_from}")

    model, val_acc, epochs = train_model(model, Xtr, ytr, Xva, yva, device,
                                         wtr=wtr, ptr=ptr, pva=pva,
                                         name=args.name, lr=args.lr,
                                         max_epochs=args.max_epochs)

    test_acc = batch_acc(model, Xte, yte, pte)
    per_build, sizable_accs = {}, []
    te_pos = pos[test_mask]
    for p in np.unique(te_pos):
        m = te_pos == p
        b = builds[int(present[p])]
        acc = batch_acc(model, Xte[m], yte[m],
                        pte[m] if pte is not None else None)
        per_build[b] = {"n_rows": int(m.sum()), "acc": round(acc, 3)}
        if b in common.SIZABLE_TEST_BUILDS:
            sizable_accs.append(acc)
    m_sizable = np.isin(te_pos,
                        [pos_of[builds.index(b)] for b in common.SIZABLE_TEST_BUILDS])
    sizable_acc = (batch_acc(model, Xte[m_sizable], yte[m_sizable],
                             pte[m_sizable] if pte is not None else None)
                   if m_sizable.any() else None)

    if args.skip_sanity:
        sanity = None
    else:
        # sanity suite with deployment-time stats for this arm
        from experiment_synthetic_augmentation import make_eval_fn
        stats = deploy_stats(args.features)
        if args.regime == "embed":
            class _Fixed(nn.Module):
                def __init__(self, m, p):
                    super().__init__()
                    self.m, self.p = m, p

                def forward(self, x):
                    return self.m(x, torch.full((len(x),), self.p, dtype=torch.long,
                                                device=x.device))
            eval_model = _Fixed(model, cpos).to(device).eval()
        else:
            eval_model = model
        sanity = sanity_suite(make_eval_fn(eval_model, cols, stats, device))

    out_pt = os.path.join(common.MODELS_DIR, f"{args.name}.pt")
    torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                **model_cfg}, out_pt)
    meta = {
        "name": args.name, "features": args.features, "regime": args.regime,
        "window": args.window or None, "half_life": args.half_life or None,
        "seed": args.seed, "cutoff_build": args.cutoff_build,
        "init_from": args.init_from, "lr": args.lr,
        "max_epochs": args.max_epochs,
        "max_train_rows": args.max_train_rows or None,
        "n_train": len(Xtr), "n_val": len(Xva), "n_test": len(Xte),
        "epochs": epochs, "val_acc": round(val_acc, 3),
        "test_acc_future": round(test_acc, 3),
        "test_acc_sizable": round(sizable_acc, 3) if sizable_acc is not None else None,
        "test_acc_per_build": per_build,
        "sanity": sanity,
        "minutes": round((time.time() - t_start) / 60, 1),
        "out": out_pt,
    }
    common.write_json(os.path.join(common.RESULTS_DIR, args.results_subdir,
                                   f"{args.name}.json"), meta)
    print(f"{args.name}: val={val_acc:.2f}% future={test_acc:.2f}% "
          f"sanity={sanity['passed_28'] if sanity else '-'}/28 "
          f"({meta['minutes']} min)")


if __name__ == "__main__":
    main()
