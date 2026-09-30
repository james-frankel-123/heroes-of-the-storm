"""
R5b — train the era-matched opponent model (GenericDraftModel, variant 0)
with the same algorithm as train_generic_draft.train_single_model (batch 512,
Adam lr 1e-3 wd 1e-5, cross-entropy, per-epoch test loss, patience 10, keep
the best-test-loss weights, at most 200 epochs), but with the memmap cache
held on the GPU as uint8 (every state value is a multiple of 1/15, so the
encoding is exact). The stock DataLoader loop needed ~25 min per epoch on the
shared machine; this one needs well under a minute. Only the data pipeline
differs; the shuffling order differs from the DataLoader's.

The checkpoint is written to a temporary name and renamed only when training
ends, so no job can pick up a partially trained opponent.

Usage: CUDA_VISIBLE_DEVICES=1 python3 drift_rebuild/r5b_fast_gd.py --cutoff 2.55.4.91418
"""
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rb_common as rb  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402


def load_u8(cache_dir, device):
    from rerun2026 import common as rc
    mm = rc.memmap_open(cache_dir)
    X, y = mm["states"], mm["actions"]
    n = len(y)
    out = torch.empty((n, X.shape[1]), dtype=torch.uint8, device=device)
    for i in range(0, n, 1_000_000):
        blk = np.rint(np.asarray(X[i:i + 1_000_000], np.float32) * 15).astype(np.uint8)
        out[i:i + len(blk)] = torch.from_numpy(blk).to(device)
    return out, torch.from_numpy(np.asarray(y, np.int64)).to(device)


def _batch(X, idx):
    x = X[idx].float() / 15.0
    occ = x[:, :270].reshape(len(x), 3, 90).sum(1)
    return x, 1.0 - occ.clamp(0, 1)


def train_gd(Xtr, ytr, Xte, yte, variant_idx, out_path, t0=None, verbose=True):
    """train_single_model's algorithm on GPU-resident uint8 data; writes the
    best-test-loss state_dict to out_path (via a .partial file)."""
    from train_generic_draft import GenericDraftModel, MODEL_VARIANTS
    t0 = t0 or time.time()
    dev = Xtr.device
    v = MODEL_VARIANTS[variant_idx]
    torch.manual_seed(v["seed"])
    model = GenericDraftModel(dropout1=v["dropout1"], dropout2=v["dropout2"]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=v["lr"], weight_decay=1e-5)
    crit = nn.CrossEntropyLoss()
    tmp = out_path + ".partial"
    best, pat = float("inf"), 0
    g = torch.Generator(device=dev)
    g.manual_seed(v["seed"])
    for ep in range(200):
        model.train()
        perm = torch.randperm(len(ytr), device=dev, generator=g)
        for i in range(0, len(perm), 512):
            idx = perm[i:i + 512]
            x, m = _batch(Xtr, idx)
            loss = crit(model(x, m), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        tl, tc = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(yte), 8192):
                idx = torch.arange(i, min(i + 8192, len(yte)), device=dev)
                x, m = _batch(Xte, idx)
                lg = model(x, m)
                tl += crit(lg, yte[idx]).item() * len(idx)
                tc += (lg.argmax(1) == yte[idx]).sum().item()
        tl /= len(yte)
        if verbose:
            print(f"  epoch {ep+1}: test_loss={tl:.4f} top1={100*tc/len(yte):.2f}% "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if tl < best:
            best, pat = tl, 0
            torch.save(model.state_dict(), tmp)
        else:
            pat += 1
            if pat >= 10:
                break
    os.replace(tmp, out_path)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", required=True)
    args = ap.parse_args()
    from train_generic_draft import GenericDraftModel, MODEL_VARIANTS
    dev = torch.device("cuda")
    v = MODEL_VARIANTS[0]
    torch.manual_seed(v["seed"])
    base = os.path.join(rb.CACHE_DIR, f"gd_era_{args.cutoff}")
    t0 = time.time()
    Xtr, ytr = load_u8(base + "_train", dev)
    Xte, yte = load_u8(base + "_val", dev)
    print(f"loaded {len(ytr):,} train / {len(yte):,} val samples "
          f"({time.time()-t0:.0f}s)", flush=True)

    def batch(X, idx):
        x = X[idx].float() / 15.0
        occ = x[:, :270].reshape(len(x), 3, 90).sum(1)
        return x, 1.0 - occ.clamp(0, 1)

    model = GenericDraftModel(dropout1=v["dropout1"], dropout2=v["dropout2"]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=v["lr"], weight_decay=1e-5)
    crit = nn.CrossEntropyLoss()
    out_dir = os.path.join(rb.MODELS_DIR, f"gd_era_{args.cutoff}")
    os.makedirs(out_dir, exist_ok=True)
    tmp = os.path.join(out_dir, "generic_draft_0.pt.partial")
    best, pat = float("inf"), 0
    g = torch.Generator(device=dev)
    g.manual_seed(v["seed"])
    for ep in range(200):
        model.train()
        perm = torch.randperm(len(ytr), device=dev, generator=g)
        for i in range(0, len(perm), 512):
            idx = perm[i:i + 512]
            x, m = batch(Xtr, idx)
            loss = crit(model(x, m), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        tl, tc, t5 = 0.0, 0, 0
        with torch.no_grad():
            for i in range(0, len(yte), 8192):
                idx = torch.arange(i, min(i + 8192, len(yte)), device=dev)
                x, m = batch(Xte, idx)
                lg = model(x, m)
                tl += crit(lg, yte[idx]).item() * len(idx)
                tc += (lg.argmax(1) == yte[idx]).sum().item()
                t5 += (lg.topk(5, 1).indices == yte[idx, None]).any(1).sum().item()
        tl /= len(yte)
        print(f"  epoch {ep+1}: test_loss={tl:.4f} top1={100*tc/len(yte):.2f}% "
              f"top5={100*t5/len(yte):.2f}% ({time.time()-t0:.0f}s)", flush=True)
        if tl < best:
            best, pat = tl, 0
            torch.save(model.state_dict(), tmp)
        else:
            pat += 1
            if pat >= 10:
                print(f"  early stop at epoch {ep+1}")
                break
    os.replace(tmp, os.path.join(out_dir, "generic_draft_0.pt"))
    print(f"gd_era_{args.cutoff}: best test loss {best:.4f} ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
