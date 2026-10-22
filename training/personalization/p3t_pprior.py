"""
P3 tournament: a search prior trained WITH player features (one step of
expert iteration), against the player-blind F prior plus a personal leaf.

Prior at a pick state x for team t (kernel labels, team mode):
  logit(h) = alpha * log p_F(h | x) + log mean_{i in team t} exp g(f(p_i, h))
  p_F   paper-1 F policy (seed of the draft), the blind-trained prior
  f     per (player, hero) features from the drafter's own state (cur):
          b2 * s(p, h), off-role, log(1 + games on h), never played h,
          share of p's games on h, log share of p's games in h's fine role
  g     MLP 6 -> 32 -> 32 -> 1 shared across heroes; the team mixture
        (log-mean-exp over its five players) because the team picks.
Targets: root visits of the player-aware search over the behavioral prior
(aMCTS-bc, 1,600 sims) at the real pick states of tournament-window lobbies
that are neither tournament nor probe lobbies. No outcome enters the
targets or the features. Fit: cross-entropy to the visit distribution over
the legal heroes; 10% of lobbies held out.

Systems using it: aMCTS-pp (prior kernel cuda_prior, cfg[56] = 1, the bias
above, alpha in coefs[5]; same leaf, sims and seeds as aMCTS-cur) and pPol-pp
(argmax of the prior, no search).

Stages: targets (GPU, 3090) | train (CPU) ; output $P3_CACHE/p3t/pprior.pt,
$P3_RESULTS/p3t/pprior.json
"""
import argparse
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

PICKS = [4, 5, 6, 7, 8, 11, 12, 13, 14, 15]
MODEL = os.path.join(TD.OUT, "pprior.pt")
NF = 6


class PPrior(nn.Module):
    def __init__(self, h=32):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.g = nn.Sequential(nn.Linear(NF, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
        self.register_buffer("mu", torch.zeros(NF))
        self.register_buffer("sd", torch.ones(NF))

    def slot_bias(self, F):
        """F (..., 10, 90, NF) -> per-slot team-mixture bias (..., 10, 90)."""
        gg = self.g((F - self.mu) / self.sd).squeeze(-1)
        out = []
        for t in (0, 1):
            blk = gg[..., 5 * t:5 * t + 5, :]
            m = torch.logsumexp(blk, dim=-2) - np.log(5.0)
            out.append(m.unsqueeze(-2).expand(*m.shape[:-1], 5, m.shape[-1]))
        return torch.cat(out, -2)


def features(s, off, n, b2):
    """(B, 10, 90) state arrays -> (B, 10, 90, NF) float32."""
    from shared import HEROES, HERO_ROLE_FINE, FINE_ROLE_NAMES
    fine = np.array([FINE_ROLE_NAMES.index(HERO_ROLE_FINE[h]) for h in HEROES])
    n = n.astype(np.float32)
    tot = n.sum(-1, keepdims=True)
    fc = np.zeros(n.shape[:-1] + (fine.max() + 1,), np.float32)
    for f in range(fine.max() + 1):
        fc[..., f] = n[..., fine == f].sum(-1)
    fshare = fc[..., fine] / np.maximum(tot, 1)
    F = np.stack([b2 * s, off.astype(np.float32), np.log1p(n), (n == 0).astype(np.float32),
                  n / np.maximum(tot, 1), np.log(fshare + 1e-3)], -1)
    return F.astype(np.float32)


def lobby_n(gi, src):
    z = np.load(os.path.join(TD.OUT, "state_cur.npz"))["n"]
    rows = (gi * 10)[:, None] + src
    return z[rows.ravel()].reshape(len(gi), 10, 90)


def targets(n_lobbies, device=0):
    import p3t_backtest as BT
    import p3t_probe as PR
    import p3_mcts_core as M
    import p3t_systems as SY
    L = TD.load_lobbies()
    nall = np.load(os.path.join(TD.OUT, "state_cur.npz"))["n"]
    from shared import MAPS
    rich = (nall.astype(np.float64).sum(1) >= 50).reshape(-1, 10).all(1) & np.isin(L["map"], MAPS)
    excl = TD.eval_lobbies()                 # tournament, tune and party-study lobbies
    excl |= set(np.load(os.path.join(TD.OUT, "probe_lobbies.npz"))["gi"].tolist())
    cand = np.array([g for g in np.flatnonzero(rich & (L["day"] <= TD.LOBBY_END)) if g not in excl])
    gi = np.sort(np.random.RandomState(88).choice(cand, n_lobbies, replace=False))
    K = BT.kernel_lobbies(gi)
    SY.SIMS = 1600
    v = SY.vintage("cur_bc")
    kmod = SY.kernel_module()
    vis = np.zeros((len(gi), 16, 90), np.float32)
    t0 = time.time()
    for k in PICKS:
        ctx = {"acts": K["acts"], "k": k, "map": K["map"], "tier": K["tier"], "team_pool": K["team_pool"],
               "pool_bits": K["pool_bits"], "slot_of": K["slot_of"], "s": K["s"], "off": K["off"],
               "seed_i": np.arange(len(gi)), "kseed": 8800 + k}
        vis[:, k] = PR._search(SY, M, kmod, v, ctx, 1, device)["visits"]
        print(f"  step {k} ({time.time() - t0:.0f}s)", flush=True)
    np.savez(os.path.join(TD.OUT, "pprior_targets.npz"), gi=gi, vis=vis, n=lobby_n(gi, K["src"]), **K)
    print(f"targets: {len(gi)} lobbies x {len(PICKS)} picks ({time.time() - t0:.0f}s)")


def f_logp(acts, k, mp, tr, team_pool, seed_i):
    """log p_F over legal heroes at step k (B, 90), F seed = seed_i % 5."""
    import p3t_systems as SY
    v = SY.vintage("cur")
    X = SY.features(acts, k, mp, tr, SY.DRAFT_TEAM[k])
    m = SY.legal_mask(acts, k, team_pool)
    out = np.full((len(X), 90), -30.0, np.float32)
    for i in range(v.n_seeds):
        sel = np.flatnonzero(seed_i % v.n_seeds == i)
        if not len(sel):
            continue
        with torch.no_grad():
            lg, _ = v.policy_net(i)(torch.from_numpy(X[sel]), torch.from_numpy(m[sel].astype(np.float32)))
        lg = lg.numpy()
        lg = np.where(m[sel], lg, -1e9)
        lg = lg - lg.max(1, keepdims=True)
        lp = lg - np.log(np.exp(lg).sum(1, keepdims=True))
        out[sel] = np.where(m[sel], lp, -30.0)
    return out, m


def fit_prior(F, LP, MS, TG, TT, II, G, init=None, seed=0):
    """Fit alpha and g by cross-entropy to visit targets. F (G, 10, 90, NF)
    roster features; per state: LP (F log-probs), MS (legal), TG (targets),
    TT (acting team), II (roster index). 10% of rosters held out."""
    torch.manual_seed(seed)
    val = (np.random.RandomState(3).rand(G) < 0.1)[II]
    model = PPrior()
    Ft = torch.from_numpy(F)
    if init is not None:
        model.load_state_dict(init)
    else:
        model.mu.copy_(Ft.reshape(-1, NF).mean(0))
        model.sd.copy_(Ft.reshape(-1, NF).std(0) + 1e-6)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    LPt, MSt, TGt = torch.from_numpy(LP), torch.from_numpy(MS), torch.from_numpy(TG)
    TTt, IIt = torch.from_numpy(TT), torch.from_numpy(II)

    def loss_on(sel):
        bias = model.slot_bias(Ft[IIt[sel]])                     # (b, 10, 90)
        bt = bias[torch.arange(len(sel)), 5 * TTt[sel]]           # team's mixture (same for its 5 slots)
        u = model.alpha * LPt[sel] + bt
        u = u.masked_fill(~MSt[sel], -1e9)
        lq = torch.log_softmax(u, 1)
        return -(TGt[sel] * lq.clamp(min=-1e4)).sum(1).mean(), lq

    tr = np.flatnonzero(~val)
    va = np.flatnonzero(val)
    with torch.no_grad():
        best = (loss_on(torch.from_numpy(va))[0].item(), {k: v.clone() for k, v in model.state_dict().items()})
    bad = 0
    for ep in range(60):
        perm = np.random.permutation(tr)
        for b0 in range(0, len(perm), 2048):
            sel = torch.from_numpy(perm[b0:b0 + 2048])
            opt.zero_grad()
            l, _ = loss_on(sel)
            l.backward()
            opt.step()
        with torch.no_grad():
            lv, _ = loss_on(torch.from_numpy(va))
        if lv.item() < best[0] - 1e-5:
            best = (lv.item(), {k: v.clone() for k, v in model.state_dict().items()})
            bad = 0
        else:
            bad += 1
            if bad >= 6:
                break
    model.load_state_dict(best[1])
    with torch.no_grad():
        sel = torch.from_numpy(va)
        _, lq = loss_on(sel)
        tg = TGt[sel]
        ent = -(tg * torch.log(tg.clamp(min=1e-12))).sum(1)
        kl_pp = (-(tg * lq.clamp(min=-1e4)).sum(1) - ent).mean().item()
        lF = torch.log_softmax(LPt[sel].masked_fill(~MSt[sel], -1e9), 1)
        kl_f = (-(tg * lF.clamp(min=-1e4)).sum(1) - ent).mean().item()
        top_pp = (lq.argmax(1) == tg.argmax(1)).float().mean().item()
        top_f = (lF.argmax(1) == tg.argmax(1)).float().mean().item()
    return model, {"states": int(len(TG)), "val_states": int(val.sum()), "alpha": float(model.alpha),
                   "kl_to_targets_F_prior": kl_f, "kl_to_targets_fit": kl_pp,
                   "top1_matches_target_F": top_f, "top1_matches_target_fit": top_pp}


def train():
    import p3t_systems as SY
    z = dict(np.load(os.path.join(TD.OUT, "pprior_targets.npz")))
    b2 = float(SY.vintage("cur").spec["combiner"][2])
    F = features(z["s"], z["off"], z["n"], b2)
    G = len(z["gi"])
    LP, MS, TG, TT, II = [], [], [], [], []
    for k in PICKS:
        lp, m = f_logp(z["acts"], k, z["map"], z["tier"], z["team_pool"], np.arange(G))
        tg = z["vis"][:, k] * m
        tg = tg / np.maximum(tg.sum(1, keepdims=True), 1e-9)
        LP.append(lp)
        MS.append(m)
        TG.append(tg)
        TT.append(np.full(G, SY.DRAFT_TEAM[k]))
        II.append(np.arange(G))
    LP, MS, TG, TT, II = (np.concatenate(x) for x in (LP, MS, TG, TT, II))
    model, out = fit_prior(F, LP, MS, TG, TT, II, G)
    torch.save(model.state_dict(), MODEL)
    out["b2"] = b2
    os.makedirs(os.path.join(C.RESULTS, "p3t"), exist_ok=True)
    json.dump(out, open(os.path.join(C.RESULTS, "p3t", "pprior.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


_PP = {}


def load_model(path=MODEL):
    if path not in _PP:
        m = PPrior()
        m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        m.eval()
        _PP[path] = m
    return _PP[path]


def bias_for(ctx_s, ctx_off, ctx_n, b2, path=MODEL):
    m = load_model(path)
    with torch.no_grad():
        return m.slot_bias(torch.from_numpy(features(ctx_s, ctx_off, ctx_n, b2))).numpy().astype(np.float32), float(m.alpha)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["targets", "train"])
    ap.add_argument("--lobbies", type=int, default=3000)
    a = ap.parse_args()
    if a.stage == "targets":
        targets(a.lobbies)
    else:
        train()


if __name__ == "__main__":
    main()
