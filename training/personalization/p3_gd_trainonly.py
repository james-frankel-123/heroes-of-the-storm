"""
P3 drafter models trained on the training window only (review 6.8).

The paper-1 GD pool (rerun2026/models/generic_draft_{0..4}.pt) and the BC
prior distilled from it (overfit2026/models/bc_prior.pt) were trained on a
random 98% of the whole 2026-05-22 snapshot, so they have seen the V1 and V2
games every P3 drafter result is scored on. They serve as rollout policy,
opponent model, imitation feature and MCTS prior. This script retrains both
on games that ended before 2026-02-10 (the WP training cutoff, the end of the
P3 estimation window E): same architecture, same five hyperparameter
variants, early stopping on a 2% game holdout of that window.

Stages (from training/; pause-safe: finished variants are skipped, a variant
in progress resumes from its last finished epoch):
  data     compact memmapped samples of the train window
  gd       the five GD variants -> cache/gd_trainonly/generic_draft_{i}.pt
  bc       BC prior distilled from variant 0 on train-window states
           -> cache/gd_trainonly/bc_prior.pt
  eval     V2 picks and bans (games on or after 2026-03-29): top-1/top-5 and
           log loss, train-only vs paper-1 pool -> results p3_gd_trainonly.json
Usage: python3 personalization/p3_gd_trainonly.py {data,gd,bc,eval,all}
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING = os.path.dirname(HERE)
sys.path.insert(0, TRAINING)
sys.path.insert(0, HERE)
import numpy as np
import torch
import torch.nn.functional as F

import p3_hs_core as C

OUT = os.path.join(C.CACHE, "gd_trainonly")
TRAIN_END = "2026-02-10"   # games strictly before this date
V2_START = "2026-03-29"
SNAP_END = "2026-05-23"


def _games(lo=None, hi=None):
    from shared import load_replay_data
    rows = load_replay_data()
    out = [r for r in rows if (lo is None or str(r["game_date"])[:10] >= lo)
           and (hi is None or str(r["game_date"])[:10] < hi)]
    print(f"games in [{lo}, {hi}): {len(out):,} of {len(rows):,}", flush=True)
    return out


def data():
    from shared import split_data
    from train_generic_draft import CompactDraftDataset
    for name in ("train", "hold"):
        if os.path.exists(os.path.join(OUT, name, "meta.json")):
            print(f"{name}: cached")
    if all(os.path.exists(os.path.join(OUT, n, "meta.json")) for n in ("train", "hold")):
        return
    rows = _games(hi=TRAIN_END)
    tr, ho = split_data(rows, 0.02, seed=42)
    del rows
    for name, part in (("train", tr), ("hold", ho)):
        ds = CompactDraftDataset(part, os.path.join(OUT, name))
        print(f"{name}: {len(ds):,} samples", flush=True)


def gd():
    from train_generic_draft import CompactDraftDataset, GenericDraftModel, MODEL_VARIANTS
    from torch.utils.data import DataLoader
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr = CompactDraftDataset(None, os.path.join(OUT, "train"))
    ho = CompactDraftDataset(None, os.path.join(OUT, "hold"))
    for i, var in enumerate(MODEL_VARIANTS):
        pt = os.path.join(OUT, f"generic_draft_{i}.pt")
        done = os.path.join(OUT, f"generic_draft_{i}.done")
        if os.path.exists(done):
            print(f"variant {i}: done")
            continue
        torch.manual_seed(var["seed"])
        np.random.seed(var["seed"])
        gen = torch.Generator().manual_seed(var["seed"])
        tdl = DataLoader(tr, batch_size=512, shuffle=True, generator=gen, num_workers=0)
        hdl = DataLoader(ho, batch_size=4096)
        m = GenericDraftModel(dropout1=var["dropout1"], dropout2=var["dropout2"]).to(dev)
        opt = torch.optim.Adam(m.parameters(), lr=var["lr"], weight_decay=1e-5)
        best, bad, ep0 = float("inf"), 0, 0
        rs = os.path.join(OUT, f"resume_{i}.pt")
        if os.path.exists(rs):
            st = torch.load(rs, weights_only=False, map_location="cpu")
            m.load_state_dict(st["model"])
            opt.load_state_dict(st["opt"])
            best, bad, ep0 = st["best"], st["bad"], st["epoch"] + 1
            gen.set_state(st["gen"])
            torch.set_rng_state(st["rng"])
            print(f"variant {i}: resumed after epoch {ep0}", flush=True)
        for ep in range(ep0, 200):
            m.train()
            t0 = time.time()
            for X, y, mk in tdl:
                X, y, mk = X.to(dev), y.to(dev), mk.to(dev)
                ok = mk.gather(1, y[:, None])[:, 0] > 0.5
                X, y, mk = X[ok], y[ok], mk[ok]
                loss = F.cross_entropy(m(X, mk), y)
                opt.zero_grad()
                loss.backward()
                opt.step()
            m.eval()
            tot, n = 0.0, 0
            with torch.no_grad():
                for X, y, mk in hdl:
                    X, y, mk = X.to(dev), y.to(dev), mk.to(dev)
                    ok = mk.gather(1, y[:, None])[:, 0] > 0.5
                    tot += F.cross_entropy(m(X[ok], mk[ok]), y[ok], reduction="sum").item()
                    n += int(ok.sum())
            hl = tot / n
            print(f"variant {i} epoch {ep + 1}: holdout loss {hl:.5f} ({time.time() - t0:.0f}s)", flush=True)
            if hl < best:
                best, bad = hl, 0
                torch.save({k: v.cpu() for k, v in m.state_dict().items()}, pt + ".tmp")
                os.replace(pt + ".tmp", pt)
            else:
                bad += 1
            torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "best": best, "bad": bad,
                        "epoch": ep, "gen": gen.get_state(), "rng": torch.get_rng_state()}, rs + ".tmp")
            os.replace(rs + ".tmp", rs)
            if bad >= 10:
                break
        with open(done, "w") as f:
            json.dump({"holdout_loss": best}, f)
        os.remove(rs)


def bc(n_samples=4_000_000, epochs=3, bs=4096):
    from train_generic_draft import CompactDraftDataset, GenericDraftModel
    from train_draft_policy import AlphaZeroDraftNet
    path = os.path.join(OUT, "bc_prior.pt")
    if os.path.exists(path):
        print("bc: done")
        return
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr = CompactDraftDataset(None, os.path.join(OUT, "train"))
    rng = np.random.RandomState(0)
    idx = np.sort(rng.choice(len(tr), min(n_samples, len(tr)), replace=False))
    x, mk = tr.decode(np.asarray(tr.X[idx]))
    S, M = torch.from_numpy(x), torch.from_numpy(mk)
    gd = GenericDraftModel()
    gd.load_state_dict(torch.load(os.path.join(OUT, "generic_draft_0.pt"), weights_only=True, map_location="cpu"))
    gd.eval().to(dev)
    torch.manual_seed(0)
    net = AlphaZeroDraftNet(size="base", policy_head_type="linear").to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    n = len(S)
    hold = slice(n - 50000, n)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n - 50000)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            xb, mb = S[b].to(dev), M[b].to(dev)
            with torch.no_grad():
                tgt = F.softmax(gd(xb, mb), -1)
            lg, _ = net(torch.cat([xb, torch.zeros(len(xb), 1, device=dev)], 1), mb)
            loss = -(tgt * F.log_softmax(lg, -1)).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            xb, mb = S[hold].to(dev), M[hold].to(dev)
            tgt = F.softmax(gd(xb, mb), -1)
            lg, _ = net(torch.cat([xb, torch.zeros(len(xb), 1, device=dev)], 1), mb)
            ce = -(tgt * F.log_softmax(lg, -1)).sum(-1).mean().item()
            ent = -(tgt * torch.log(tgt.clamp_min(1e-12))).sum(-1).mean().item()
        print(f"bc epoch {ep + 1}: KL {ce - ent:.4f}", flush=True)
    torch.save({k: v.cpu() for k, v in net.state_dict().items()}, path + ".tmp")
    os.replace(path + ".tmp", path)


def _pool(dirname, names):
    from train_generic_draft import GenericDraftModel
    ms = []
    for i in range(5):
        m = GenericDraftModel()
        m.load_state_dict(torch.load(os.path.join(dirname, names.format(i)), weights_only=True, map_location="cpu"))
        ms.append(m.eval())
    return ms


def evaluate():
    from train_generic_draft import replay_to_training_samples
    rows = _games(lo=V2_START, hi=SNAP_END)
    X, Y, Mk, T = [], [], [], []
    for r in rows:
        for x, y, mk in replay_to_training_samples(r):
            X.append(x)
            Y.append(y)
            Mk.append(mk)
            T.append(x[-1])
    X, Y, Mk, T = (torch.tensor(np.array(a)) for a in (X, Y, Mk, T))
    ok = Mk[torch.arange(len(Y)), Y] > 0.5
    X, Y, Mk, T = X[ok], Y[ok], Mk[ok], T[ok]
    out = {"v2_games": len(rows)}
    for nm, ms in (("train-only", _pool(OUT, "generic_draft_{}.pt")),
                   ("paper-1 (98% random)", _pool(os.path.join(TRAINING, "rerun2026", "models"),
                                                  "generic_draft_{}.pt"))):
        with torch.no_grad():
            p = sum(torch.softmax(m(X, Mk), 1) for m in ms) / 5
        lp = torch.log(p.clamp_min(1e-12))
        res = {}
        for kind, sel in (("picks", T == 1), ("bans", T == 0)):
            yy, ll = Y[sel], lp[sel]
            top5 = ll.topk(5, 1).indices
            res[kind] = {"n": int(sel.sum()), "log_loss": float(-ll[torch.arange(len(yy)), yy].mean()),
                         "top1": float((ll.argmax(1) == yy).float().mean()),
                         "top5": float((top5 == yy[:, None]).any(1).float().mean())}
        out[nm] = res
        print(nm, json.dumps(res), flush=True)
    with open(os.path.join(C.RESULTS, "p3_gd_trainonly.json"), "w") as f:
        json.dump(out, f, indent=1)


def main():
    os.makedirs(OUT, exist_ok=True)
    what = sys.argv[1]
    for st, fn in (("data", data), ("gd", gd), ("bc", bc), ("eval", evaluate)):
        if what in (st, "all"):
            fn()


if __name__ == "__main__":
    main()
