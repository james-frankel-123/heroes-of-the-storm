"""
P3 exploration study, value side: does a value net that sees all ten
players' skill vectors and the hero-to-player structure beat the linear
personal sum? Offline, on real games with real outcomes.

Data: pre_lobbies (snapshot-era games 2026-02-11 to 05-22 with full drafts,
all ten players known), as-of player states (pre_state), and the slot
table's causal WP (paper 2's d2c_cumprev, trained through 2026-02-10, so out
of sample here). Each player is on the hero he actually played.
Validation: the last 15% of days; 3 seeds per net; log-loss gain over the
linear model x1000 with SE by game.

Models (all: logit = b1 logit WP + h(team 0) - h(team 1), antisymmetric):
  L    linear: b2 S + b3 O (the drafters' leaf)
  A    the pvnet leaf: MLP on [S, O, sorted assigned s, sorted log1p n] (12)
  P    per-player set model: each player i -> phi(x_i), x_i = [s_i(played),
       sd_i(played), log1p n_i(played), off_i, the player's best s, mean s
       over his pool, his pool size, s_i(played) - best], h = sum_i phi
  F    P plus each player's full skill vector over all 90 heroes through a
       learned 90 -> 16 hero embedding (shared), and the played hero's
       embedding (interaction by dot product)
Output: $P3_RESULTS/p3t/vstudy.json
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("P1_TIERS", "site")
import numpy as np
import torch
import torch.nn as nn

import p3_hs_core as C
import p3t_data as TD


def load():
    L = dict(np.load(os.path.join(TD.OUT, "pre_lobbies.npz")))
    S = np.load(os.path.join(TD.OUT, "pre_state.npz"))
    G = len(L["replay_id"])
    s = S["s"].astype(np.float32).reshape(G, 10, 90)
    sd = S["sd"].astype(np.float32).reshape(G, 10, 90)
    n = S["n"].astype(np.float32).reshape(G, 10, 90)
    off = S["off"].reshape(G, 10, 90).astype(np.float32)
    st = L["steps"]
    hero = st[:, :, 0][st[:, :, 1] == 1].reshape(G, 10)
    team = st[:, :, 2][st[:, :, 1] == 1].reshape(G, 10)
    ar = np.arange(G)[:, None]
    jj = np.arange(10)[None, :]
    sp, sdp, np_, op = s[ar, jj, hero], sd[ar, jj, hero], n[ar, jj, hero], off[ar, jj, hero]
    pool = n > 0
    best = np.where(pool, s, -9).max(-1)
    mean_pool = (s * pool).sum(-1) / np.maximum(pool.sum(-1), 1)
    X = np.stack([sp, sdp, np.log1p(np_), op, best, mean_pool, np.log1p(pool.sum(-1)), sp - best], -1)  # (G,10,8)
    y = (L["winner"] == 0).astype(np.float32)
    wp = np.clip(L["wp"], 1e-4, 1 - 1e-4)
    lo = np.log(wp / (1 - wp)).astype(np.float32)
    sign = np.where(team == 0, 1.0, -1.0).astype(np.float32)
    return dict(X=X.astype(np.float32), s=s, hero=hero, sign=sign, y=y, lo=lo, day=L["day"], G=G)


class Net(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.b1 = nn.Parameter(torch.tensor(1.0))
        if kind == "L":
            self.w = nn.Linear(2, 1, bias=False)
        elif kind == "A":
            self.h = nn.Sequential(nn.Linear(12, 16), nn.ReLU(), nn.Linear(16, 1))
        else:
            din = 8 + (17 if kind == "F" else 0)
            self.phi = nn.Sequential(nn.Linear(din, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 32), nn.ReLU(),
                                     nn.Linear(32, 1))
            if kind == "F":
                self.emb = nn.Linear(90, 16, bias=False)
                self.hemb = nn.Embedding(90, 16)

    def team_h(self, B, t):
        sg = B["sign"]
        m = (sg > 0) if t == 0 else (sg < 0)                      # (b, 10) players of team t
        if self.kind == "L":
            S = (B["X"][..., 0] * m).sum(1)
            O = (B["X"][..., 3] * m).sum(1)
            return self.w(torch.stack([S, O], -1)).squeeze(-1)
        if self.kind == "A":
            x = B["X"]
            S = (x[..., 0] * m).sum(1, keepdim=True)
            O = (x[..., 3] * m).sum(1, keepdim=True)
            sp = torch.where(m, x[..., 0], torch.full_like(x[..., 0], -9)).sort(1, descending=True).values[:, :5]
            nn_ = torch.where(m, x[..., 2], torch.full_like(x[..., 2], -9)).sort(1, descending=True).values[:, :5]
            return self.h(torch.cat([S, O, sp, nn_], 1)).squeeze(-1)
        x = B["X"]
        if self.kind == "F":
            e = self.emb(B["s"])                                   # (b, 10, 16)
            he = self.hemb(B["hero"])                              # (b, 10, 16)
            x = torch.cat([x, e, (e * he).sum(-1, keepdim=True)], -1)
        return (self.phi(x).squeeze(-1) * m).sum(1)

    def forward(self, B):
        return self.b1 * B["lo"] + self.team_h(B, 0) - self.team_h(B, 1)


def main():
    import p3_x_common as XC
    t0 = time.time()
    D = load()
    cut = np.quantile(D["day"], 0.85)
    tr, va = np.flatnonzero(D["day"] < cut), np.flatnonzero(D["day"] >= cut)
    # the inner validation for early stopping: the last 15% of the training days
    cut2 = np.quantile(D["day"][tr], 0.85)
    tr_in, va_in = tr[D["day"][tr] < cut2], tr[D["day"][tr] >= cut2]
    out = {"games": int(D["G"]), "train": int(len(tr_in)), "early_stop": int(len(va_in)), "test": int(len(va))}
    y = D["y"][va]
    preds = {}
    for kind in ("L", "A", "P", "F"):
        ps = []
        for seed in (0, 1, 2) if kind != "L" else (0,):
            # early stop on the inner split, report on the held-out last days
            net_p = fit_eval(D, kind, tr_in, va_in, va, seed)
            ps.append(net_p)
        preds[kind] = np.mean(ps, 0)
        print(f"{kind} done ({time.time() - t0:.0f}s)", flush=True)
    base = XC.logloss(preds["L"], y)
    for kind, p in preds.items():
        ll = XC.logloss(p, y)
        d = base - ll
        out[kind] = {"test_ll": float(ll.mean()), "gain_over_linear_x1000": float(1000 * d.mean()),
                     "se_x1000": float(1000 * d.std(ddof=1) / np.sqrt(len(d))),
                     "acc": float(((p > 0.5) == (y == 1)).mean())}
    wp = 1 / (1 + np.exp(-D["lo"][va]))
    out["wp_only_test_ll"] = float(XC.logloss(wp, y).mean())
    json.dump(out, open(os.path.join(C.RESULTS, "p3t", "vstudy.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


def fit_eval(D, kind, tr, va_stop, te, seed):
    """fit with early stopping on va_stop, predict te."""
    torch.manual_seed(seed)
    net = Net(kind)
    T = {k: torch.from_numpy(D[k]) for k in ("X", "s", "sign", "y", "lo")}
    T["hero"] = torch.from_numpy(D["hero"]).long()
    sub = lambda idx: {k: v[idx] for k, v in T.items()}
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    best, bad = (1e9, None), 0
    trI, vaI, teI = torch.from_numpy(tr), torch.from_numpy(va_stop), torch.from_numpy(te)
    for ep in range(80):
        net.train()
        perm = trI[torch.randperm(len(trI))]
        for b0 in range(0, len(perm), 4096):
            B = sub(perm[b0:b0 + 4096])
            opt.zero_grad()
            l = nn.functional.binary_cross_entropy_with_logits(net(B), B["y"])
            l.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            B = sub(vaI)
            lv = nn.functional.binary_cross_entropy_with_logits(net(B), B["y"]).item()
        if lv < best[0] - 1e-6:
            best, bad = (lv, {k: v.clone() for k, v in net.state_dict().items()}), 0
        else:
            bad += 1
            if bad >= 8:
                break
    net.load_state_dict(best[1])
    net.eval()
    with torch.no_grad():
        return torch.sigmoid(net(sub(teI))).numpy()


if __name__ == "__main__":
    main()
