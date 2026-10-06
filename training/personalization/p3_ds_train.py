"""
P3 distillation, stage 2: fit the player-conditioned prior to MCTS root
visit distributions.

Train: personalized MCTS searches at real states of V1 lobbies
(cache/ds_targets_s400.pkl.gz), 90% of lobbies; validation: the other 10%.
Test: the personalized MCTS searches at real states of the 5,724 V2 lobbies
(cache/mcts_real_s400.pkl.gz), never used in training.
Loss: cross-entropy of the prior to the root visit distribution over the
kernel's valid mask. Compared: BC prior alone; alpha * BC (fitted alpha);
alpha * BC + g (the distilled prior); g alone (no draft state).

Run (from training/):
  OMP_NUM_THREADS=4 nice -n 19 taskset -c 48-63 python3 personalization/p3_ds_train.py
Outputs: cache/ds_prior.pt, results/p3_ds_train.json
"""
import os
import sys
import gzip
import json
import time
import pickle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from p3_heroes import NUM_HEROES
import p3_hs_core as C
import p3_mcts_core as M
import p3_ds_common as DS
from p3_mcts_drafter import Setup


def build(S, R, pers_only=True):
    """Examples from a searches pickle: states, masks, targets, features."""
    lobs = R["lobbies"]
    items = []
    for dec in R["decisions"]:
        if len(dec) == 4:
            li, k, pol, q = dec
        else:
            li, k, pers, pol, q = dec
            if pers_only and pers != 1:
                continue
        items.append((li, k, pol))
    rows = np.array([int(lobs[li]["rows"][lobs[li]["slot_of"][k]]) for li, k, _ in items])
    ur, inv = np.unique(rows, return_inverse=True)
    Fr = DS.player_features(S, ur)                      # my order
    F = np.zeros((len(items), NUM_HEROES, len(DS.FEATS)), np.float32)
    X = np.zeros((len(items), 290), np.float32)
    Mk = np.zeros((len(items), NUM_HEROES), bool)
    Y = np.zeros((len(items), NUM_HEROES), np.float32)
    A = np.zeros(len(items), np.int64)
    G = np.zeros(len(items), np.int64)
    for i, (li, k, pol) in enumerate(items):
        lob = dict(lobs[li])
        if "map" not in lob:
            lb = S.lobby(lob["gi"])
            lob["map"], lob["tier"] = lb["map"], lb["tier"]
        row = rows[i]
        X[i], Mk[i] = DS.state_and_mask(lob, k, S.T["pool"][S.T["pos"][int(row)]], S.to_sh)
        F[i][S.to_sh] = Fr[inv[i]]
        y = np.where(Mk[i], pol, 0)
        Y[i] = y / max(y.sum(), 1e-9)
        A[i] = int(lob["acts_real"][k])
        G[i] = lob["gi"]
    return X, Mk, Y, F, A, G


def metrics(u, Y, A, Mk):
    u = np.where(Mk, u, -1e9)
    lse = np.log(np.exp(u - u.max(1, keepdims=True)).sum(1)) + u.max(1)
    logq = u - lse[:, None]
    ce = -(Y * np.where(Mk, logq, 0)).sum(1).mean()
    ent = -(Y * np.log(np.maximum(Y, 1e-12))).sum(1).mean()
    top = u.argmax(1)
    return {"cross_entropy": float(ce), "kl_to_mcts": float(ce - ent),
            "top1_equals_mcts_argmax": float((top == Y.argmax(1)).mean()),
            "top1_equals_real_pick": float((top == A).mean()),
            "loglik_real_pick": float(logq[np.arange(len(A)), A].mean())}


def main():
    t0 = time.time()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    S = Setup()
    import search as SR
    net = SR.policy_net("path:" + __import__("p3_dr_core").bc_prior_path())
    tr = pickle.load(gzip.open(os.path.join(C.CACHE, "ds_targets_s400.pkl.gz")))
    te = pickle.load(gzip.open(os.path.join(C.CACHE, "mcts_real_s400.pkl.gz")))
    Xa, Ma, Ya, Fa, Aa, Ga = build(S, tr)
    Xt, Mt, Yt, Ft, At, Gt = build(S, te)
    print(f"examples: train+val {len(Xa):,}, test {len(Xt):,} ({time.time() - t0:.0f}s)", flush=True)
    Ba = DS.bc_logp(net, Xa, Ma)
    Bt = DS.bc_logp(net, Xt, Mt)
    rng = np.random.RandomState(0)
    ug = np.unique(Ga)
    val_g = set(rng.choice(ug, len(ug) // 10, replace=False).tolist())
    va = np.array([g in val_g for g in Ga])
    trm = ~va
    out = {"n_train": int(trm.sum()), "n_val": int(va.sum()), "n_test": int(len(Xt)), "features": DS.FEATS}

    def fit(kind, epochs=30):
        m = DS.DistillPrior()
        with torch.no_grad():
            f = torch.tensor(Fa[trm]).reshape(-1, len(DS.FEATS))
            m.mu.copy_(f.mean(0))
            m.sd.copy_(f.std(0) + 1e-6)
        if kind == "alpha only":
            for p in m.g.parameters():
                p.requires_grad_(False)
                p.data.zero_()
        if kind == "g only":
            m.alpha.data.zero_()
            m.alpha.requires_grad_(False)
        opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=3e-3)
        idx = np.flatnonzero(trm)
        B = torch.tensor(Ba)
        F = torch.tensor(Fa)
        Mk = torch.tensor(Ma)
        Y = torch.tensor(Ya)
        best, best_state = 1e9, None
        for ep in range(epochs):
            rng.shuffle(idx)
            for s in range(0, len(idx), 1024):
                b = idx[s:s + 1024]
                u = m(B[b], F[b], Mk[b])
                loss = -(Y[b] * torch.log_softmax(u, -1).masked_fill(~Mk[b], 0)).sum(1).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                uv = m(B[va], F[va], Mk[va])
                lv = float(-(Y[va] * torch.log_softmax(uv, -1).masked_fill(~Mk[va], 0)).sum(1).mean())
            if lv < best - 1e-5:
                best, best_state = lv, {k: v.clone() for k, v in m.state_dict().items()}
        m.load_state_dict(best_state)
        m.eval()
        return m, best

    res = {}
    res["BC prior alone"] = {"test": metrics(Bt, Yt, At, Mt)}
    for kind in ("alpha only", "g only", "distilled (alpha BC + g)"):
        m, bv = fit(kind)
        with torch.no_grad():
            ut = m(torch.tensor(Bt), torch.tensor(Ft), torch.tensor(Mt)).numpy()
        res[kind] = {"val_ce": bv, "alpha": float(m.alpha), "test": metrics(ut, Yt, At, Mt)}
        print(kind, json.dumps(res[kind]), f"({time.time() - t0:.0f}s)", flush=True)
        if kind.startswith("distilled"):
            torch.save({"state_dict": m.state_dict(), "features": DS.FEATS, "alpha": float(m.alpha)}, DS.MODEL)
    print("BC prior alone", json.dumps(res["BC prior alone"]), flush=True)
    out["models"] = res
    with open(os.path.join(C.RESULTS, "p3_ds_train.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
