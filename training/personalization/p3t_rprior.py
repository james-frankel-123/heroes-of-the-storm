"""
P3 exploration study: a richer, learned draft prior that sees all ten players.

For the acting team at a draft state, a shared MLP scores every hero from
hero-level features (32), then softmax over the legal heroes:
  population   log p_F(h | x) (paper 1's F policy, seed by draft), log p_GD(h | x)
  own team     max_i b2 s_ih, log-mean-exp_i b2 s_ih, sd at the argmax player,
               max_i and mean_i log1p n_ih, the assignment gain (b2 x best
               hero-to-player assignment with h minus without), comfort
               (opponent-model form: log mean_i exp(w1 log1p n_ih + w2 [n_ih = 0]))
  opponents    max_j b2 s_jh, max_j log1p n_jh, their assignment gain if they
               got h (denial), their comfort
  composition  h's Blizzard role (6, one-hot) and the acting team's role counts (6)
  step         the team's pick index (5, one-hot), ban flag, second phase flag
Bans use the same network (the ban flag and the opponent features let it
learn denial).

Training signals (one model each; stage train --signal X):
  bcow    behavioral cloning of real actions in pre-tournament games
          (pre_lobbies, 2026-02-11 to 05-22), weighted by the acting team's
          realized advantage exp(2 (y - WP)) and restricted to the high tier
  onestep distillation of the player-aware one-step drafter's choices
          (soft targets softmax(V / 0.02) over its candidates) at the real
          pick states of 3,000 pre-tournament games
  search  distillation of the player-aware search's root visits over the
          behavioral prior (the pprior targets: tournament-window lobbies
          that are neither tournament, probe nor tune lobbies; no outcomes)
Used as: aPol-rich-X (argmax, structural mask at picks) and aMCTS-rich-X
(the kernel prior gets a static bias equal to rich logits minus alpha log p_F
at the decision state, alpha = 1; picks, and bans through cuda_explore's
ban rows when available).

Stages: data_bcow | data_onestep | train --signal X  (3090 CPU)
"""
import argparse
import itertools
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

NF = 32
H = 90
DT = [0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1]
IP = [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1]
PERM = {m: np.array(list(itertools.permutations(range(5), m)), np.int64) for m in range(6)}


def assign_gain(s5, own):
    """(B, 90): best assignment value with h added minus without; s5 (B, 5, 90),
    own (B, m) the team's heroes so far (same m for the batch)."""
    B = len(s5)
    m = own.shape[1]
    ar = np.arange(B)[:, None]
    if m:
        P = PERM[m]
        v0 = np.zeros((B, len(P)))
        for i in range(m):
            v0 += s5[ar, P[None, :, i], own[:, i:i + 1]]
        best0 = v0.max(1)
    else:
        best0 = np.zeros(B)
    P = PERM[m + 1]
    base = np.zeros((B, len(P)))
    for i in range(m):
        base += s5[ar, P[None, :, i], own[:, i:i + 1]]
    last = s5[:, P[:, m], :]                                           # (B, |P|, 90)
    return (base[:, :, None] + last).max(1) - best0[:, None]


def role_idx():
    from shared import HEROES, HERO_ROLE_FINE, FINE_TO_BLIZZ_ROLE
    R = ("Tank", "Bruiser", "Healer", "Ranged Assassin", "Melee Assassin", "Support")
    return np.array([R.index(FINE_TO_BLIZZ_ROLE.get(HERO_ROLE_FINE.get(h, ""), "Ranged Assassin")) for h in HEROES])


ROLE = role_idx()


def features(acts, k, mp, tr, s, sd, n, seed_i, b2, w):
    """acts (B,16) kernel order (entries >= k ignored), states (B,10,90) in kernel slots.
    Returns F (B, 90, NF) float32 and the legal mask (B, 90)."""
    import p3t_pprior as PP
    import p3t_systems as SY
    B = len(acts)
    t = DT[k]
    pool = n > 0
    team_pool = np.stack([pool[:, :5].any(1), pool[:, 5:].any(1)], 1)
    lpF, m = PP.f_logp(acts, k, mp, tr, team_pool if IP[k] else np.ones_like(team_pool), seed_i)
    X = SY.features(acts, k, mp, tr, t)
    lpG = np.log(np.maximum(SY.gd_probs(SY.vintage("cur"), X[:, :289], m), 1e-9))
    own, opp = slice(5 * t, 5 * t + 5), slice(5 * (1 - t), 5 * (1 - t) + 5)
    so, sp_ = b2 * s[:, own], b2 * s[:, opp]
    no, np_ = np.log1p(n[:, own]), np.log1p(n[:, opp])
    am = so.argmax(1)
    sd_at = np.take_along_axis(sd[:, own], am[:, None, :], 1)[:, 0]
    lme = lambda u: u.max(1) + np.log(np.exp(u - u.max(1, keepdims=True)).mean(1))
    comf_o = lme(w["w1"] * no + w["w2"] * (n[:, own] == 0))
    comf_p = lme(w["w1"] * np_ + w["w2"] * (n[:, opp] == 0))
    own_h = np.array([[acts[i, j] for j in range(k) if IP[j] and DT[j] == t] for i in range(B)], np.int64).reshape(B, -1)
    opp_h = np.array([[acts[i, j] for j in range(k) if IP[j] and DT[j] == 1 - t] for i in range(B)], np.int64).reshape(B, -1)
    ga_o = b2 * assign_gain(s[:, own], own_h) if own_h.shape[1] < 5 else np.zeros((B, H))
    ga_p = b2 * assign_gain(s[:, opp], opp_h) if opp_h.shape[1] < 5 else np.zeros((B, H))
    rc = np.zeros((B, 6), np.float32)
    for i in range(B):
        for h in own_h[i]:
            rc[i, ROLE[h]] += 1
    roh = np.eye(6, dtype=np.float32)[ROLE]                              # (90, 6)
    pick_i = sum(1 for j in range(k) if IP[j] and DT[j] == t)
    stp = np.zeros(7, np.float32)
    if IP[k]:
        stp[min(pick_i, 4)] = 1
    else:
        stp[5] = 1
    stp[6] = float(k >= 9)
    F = np.concatenate([
        lpF[..., None], lpG[..., None],
        so.max(1)[..., None], lme(so)[..., None], sd_at[..., None], no.max(1)[..., None], no.mean(1)[..., None],
        ga_o[..., None], comf_o[..., None],
        sp_.max(1)[..., None], np_.max(1)[..., None], ga_p[..., None], comf_p[..., None],
        np.broadcast_to(roh[None], (B, H, 6)), np.broadcast_to(rc[:, None, :], (B, H, 6)),
        np.broadcast_to(stp[None, None, :], (B, H, 7))], -1).astype(np.float32)
    assert F.shape[-1] == NF, F.shape
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=-30.0), m


class RichPrior(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(NF, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
        self.register_buffer("mu", torch.zeros(NF))
        self.register_buffer("sd", torch.ones(NF))

    def forward(self, F, mask):
        u = self.net((F - self.mu) / self.sd).squeeze(-1)
        return u.masked_fill(~mask, -1e9)


def _kernel_state(L, S, gi):
    """kernel-labelled acts and per-slot states for pre_lobbies rows gi."""
    G = len(gi)
    st = L["steps"][gi]
    first = L["first_team"][gi].astype(np.int64)
    acts = st[:, :, 0].astype(np.int64)
    src = np.zeros((G, 10), np.int64)
    for i in range(G):
        cnt, j = [0, 0], 0
        for k in range(16):
            if st[i, k, 1] == 1:
                kt = st[i, k, 2] ^ first[i]
                src[i, 5 * kt + cnt[kt]] = j
                cnt[kt] += 1
                j += 1
    rows = (gi * 10)[:, None] + src
    g = lambda a: a[rows.ravel()].astype(np.float32).reshape(G, 10, 90)
    from shared import MAPS, SKILL_TIERS
    mp = np.array([MAPS.index(str(x)) for x in L["map"][gi]], np.int32)
    tr = np.array([SKILL_TIERS.index(str(x)) for x in L["tier"][gi]], np.int32)
    # outcome from the side of kernel team 0 (the first-pick team)
    y0 = np.where(first == 0, L["winner"][gi] == 0, L["winner"][gi] == 1).astype(np.float32)
    wp0 = np.where(first == 0, L["wp"][gi], 1 - L["wp"][gi])
    return acts, mp, tr, g(S["s"]), g(S["sd"]), g(S["n"]), y0, wp0


def _w():
    return json.load(open(os.path.join(C.RESULTS, "p3t", "opp_model.json")))


def data_bcow(n_games=40000):
    import p3t_systems as SY
    t0 = time.time()
    L = dict(np.load(os.path.join(TD.OUT, "pre_lobbies.npz")))
    S = {k: np.load(os.path.join(TD.OUT, "pre_state.npz"))[k] for k in ("s", "sd", "n")}
    hi = np.flatnonzero((L["tier"] == "high") & ((S["n"].reshape(-1, 10, 90) > 0).sum(-1) > 0).all(1))
    gi = np.sort(np.random.RandomState(5).choice(hi, min(n_games, len(hi)), replace=False))
    acts, mp, tr, s, sd, n, y0, wp0 = _kernel_state(L, S, gi)
    b2 = float(SY.vintage("cur").spec["combiner"][2])
    w = _w()
    Fs, Ms, Ys, Ws, Gs = [], [], [], [], []
    for k in range(16):
        for c0 in range(0, len(gi), 4000):
            sl = slice(c0, c0 + 4000)
            F, m = features(acts[sl], k, mp[sl], tr[sl], s[sl], sd[sl], n[sl], np.arange(len(gi))[sl], b2, w)
            t = DT[k]
            yt = y0[sl] if t == 0 else 1 - y0[sl]
            wt = wp0[sl] if t == 0 else 1 - wp0[sl]
            Fs.append(F.astype(np.float16))
            Ms.append(m)
            Ys.append(acts[sl, k])
            Ws.append(np.exp(2.0 * (yt - wt)))
            Gs.append(np.arange(len(gi))[sl])
        print(f"  step {k} ({time.time() - t0:.0f}s)", flush=True)
    np.savez(os.path.join(TD.OUT, "rprior_bcow.npz"), F=np.concatenate(Fs), M=np.concatenate(Ms),
             Y=np.concatenate(Ys), W=np.concatenate(Ws).astype(np.float32), G=np.concatenate(Gs), gi=gi)
    print(f"bcow data: {len(gi):,} high-tier games ({time.time() - t0:.0f}s)")


def data_onestep(n_games=3000, procs=12):
    """the aware one-step drafter's soft choice at every real pick state."""
    import p3t_systems as SY
    t0 = time.time()
    L = dict(np.load(os.path.join(TD.OUT, "pre_lobbies.npz")))
    S = {k: np.load(os.path.join(TD.OUT, "pre_state.npz"))[k] for k in ("s", "sd", "n", "off")}
    G0 = len(L["replay_id"])
    rich = (S["n"].reshape(G0, 10, 90).sum(-1) >= 50).all(1)
    gi = np.sort(np.random.RandomState(6).choice(np.flatnonzero(rich), n_games, replace=False))
    acts, mp, tr, s, sd, n, y0, wp0 = _kernel_state(L, S, gi)
    G = len(gi)
    st = L["steps"][gi]
    first = L["first_team"][gi].astype(np.int64)
    src = np.zeros((G, 10), np.int64)
    for i in range(G):
        cnt, j = [0, 0], 0
        for k in range(16):
            if st[i, k, 1] == 1:
                kt = st[i, k, 2] ^ first[i]
                src[i, 5 * kt + cnt[kt]] = j
                cnt[kt] += 1
                j += 1
    off = S["off"][((gi * 10)[:, None] + src).ravel()].reshape(G, 10, 90)
    v = SY.vintage("cur")
    b = v.spec["combiner"]
    pool = n > 0
    os_ = SY.OneStep("cur", procs)
    b2 = float(b[2])
    w = _w()
    Fs, Ms, Ts, Gs = [], [], [], []
    for k in [4, 5, 6, 7, 8, 11, 12, 13, 14, 15]:
        lobs = [SY.dr_lobby({"map": mp[i], "tier": tr[i]}, s[i], off[i], pool[i], b) for i in range(G)]
        os_.start()
        tasks = [(lobs[i], acts[i], k, 900 + 31 * k + i) for i in range(G)]
        res = os_.pool.map(_soft_task, tasks, chunksize=4)
        T = np.zeros((G, 90), np.float32)
        for i, (cand, V) in enumerate(res):
            z = np.asarray(V) / 0.02
            z = np.exp(z - z.max())
            T[i, cand] = z / z.sum()
        F, m = features(acts, k, mp, tr, s, sd, n, np.arange(G), b2, w)
        Fs.append(F.astype(np.float16))
        Ms.append(m)
        Ts.append(T)
        Gs.append(np.arange(G))
        print(f"  step {k} ({time.time() - t0:.0f}s)", flush=True)
    os_.close()
    np.savez(os.path.join(TD.OUT, "rprior_onestep.npz"), F=np.concatenate(Fs), M=np.concatenate(Ms),
             T=np.concatenate(Ts), G=np.concatenate(Gs), gi=gi)
    print(f"onestep data: {G:,} games ({time.time() - t0:.0f}s)")


def _soft_task(task):
    import p3t_systems as SY
    lob, acts, k, seed = task
    DR = SY._OS["DR"]
    DR._W["b"] = np.asarray(lob["b"], float)
    st = {"t0": {}, "t1": {}, "bans": set(), "taken": np.zeros(H, bool)}
    for j in range(k):
        st = DR._apply_ban(st, int(acts[j])) if not IP[j] else DR._apply(st, j, int(acts[j]), lob)
    dd = DR._decide(st, k, lob, np.random.RandomState(seed), "gd")
    return dd["cand"], dd["V"]


def data_search():
    """the pprior targets (aware search over the BC prior, 1,600 sims) as soft targets."""
    import p3t_systems as SY
    z = dict(np.load(os.path.join(TD.OUT, "pprior_targets.npz")))
    G = len(z["gi"])
    b2 = float(SY.vintage("cur").spec["combiner"][2])
    w = _w()
    sd = np.load(os.path.join(TD.OUT, "state_cur.npz"))["sd"]
    rows = (z["gi"] * 10)[:, None] + z["src"]
    sdk = sd[rows.ravel()].astype(np.float32).reshape(G, 10, 90)
    Fs, Ms, Ts, Gs = [], [], [], []
    for k in [4, 5, 6, 7, 8, 11, 12, 13, 14, 15]:
        F, m = features(z["acts"], k, z["map"], z["tier"], z["s"].astype(np.float32), sdk, z["n"].astype(np.float32),
                        np.arange(G), b2, w)
        T = z["vis"][:, k] * m
        T = T / np.maximum(T.sum(1, keepdims=True), 1e-9)
        Fs.append(F.astype(np.float16))
        Ms.append(m)
        Ts.append(T.astype(np.float32))
        Gs.append(np.arange(G))
    np.savez(os.path.join(TD.OUT, "rprior_search.npz"), F=np.concatenate(Fs), M=np.concatenate(Ms),
             T=np.concatenate(Ts), G=np.concatenate(Gs), gi=z["gi"])
    print(f"search data: {G:,} lobbies")


def train(signal):
    torch.manual_seed(0)
    z = dict(np.load(os.path.join(TD.OUT, f"rprior_{signal}.npz")))
    F = torch.from_numpy(z["F"].astype(np.float32))
    M = torch.from_numpy(z["M"])
    if "T" in z:
        T = torch.from_numpy(z["T"])
        W = torch.ones(len(T))
    else:
        T = torch.zeros(len(z["Y"]), H)
        T[torch.arange(len(T)), torch.from_numpy(z["Y"]).long()] = 1
        W = torch.from_numpy(z["W"])
    ok = (T * M).sum(1) > 0
    Gg = z["G"]
    val = torch.from_numpy((np.random.RandomState(4).rand(int(Gg.max()) + 1) < 0.1)[Gg]) & ok
    tr = (~val) & ok
    m = RichPrior()
    m.mu.copy_(F[tr].reshape(-1, NF).mean(0))
    m.sd.copy_(F[tr].reshape(-1, NF).std(0) + 1e-6)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-5)
    tri, vai = torch.nonzero(tr).squeeze(1), torch.nonzero(val).squeeze(1)

    def loss(idx):
        lq = torch.log_softmax(m(F[idx], M[idx]), 1)
        l = -(T[idx] * lq.clamp(min=-1e4)).sum(1)
        return (l * W[idx]).sum() / W[idx].sum(), lq
    best, bad = (1e9, None), 0
    for ep in range(40):
        perm = tri[torch.randperm(len(tri))]
        for b0 in range(0, len(perm), 2048):
            opt.zero_grad()
            l, _ = loss(perm[b0:b0 + 2048])
            l.backward()
            opt.step()
        with torch.no_grad():
            lv, _ = loss(vai)
        if lv.item() < best[0] - 1e-5:
            best, bad = (lv.item(), {k: v.clone() for k, v in m.state_dict().items()}), 0
        else:
            bad += 1
            if bad >= 5:
                break
    m.load_state_dict(best[1])
    with torch.no_grad():
        _, lq = loss(vai)
        top = (lq.argmax(1) == T[vai].argmax(1)).float().mean().item()
        lF = torch.log_softmax(F[vai][..., 0].masked_fill(~M[vai], -1e9), 1)
        topF = (lF.argmax(1) == T[vai].argmax(1)).float().mean().item()
    torch.save(m.state_dict(), os.path.join(TD.OUT, f"rprior_{signal}.pt"))
    out = {"signal": signal, "states": int(ok.sum()), "val_states": int(val.sum()), "val_loss": best[0],
           "val_top1_vs_target": top, "F_prior_top1_vs_target": topF}
    p = os.path.join(C.RESULTS, "p3t", "rprior.json")
    old = json.load(open(p)) if os.path.exists(p) else {}
    old[signal] = out
    json.dump(old, open(p, "w"), indent=1)
    print(json.dumps(out))


_RP = {}


def logits(signal, acts, k, mp, tr, s, sd, n, seed_i, b2):
    """Rich prior logits (B, 90) with the legal mask applied."""
    if signal not in _RP:
        m = RichPrior()
        m.load_state_dict(torch.load(os.path.join(TD.OUT, f"rprior_{signal}.pt"), map_location="cpu", weights_only=True))
        m.eval()
        _RP[signal] = m
    F, msk = features(acts, k, mp, tr, s, sd, n, seed_i, b2, _w())
    with torch.no_grad():
        u = _RP[signal](torch.from_numpy(F), torch.from_numpy(msk)).numpy()
    return u, F[..., 0], msk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["data_bcow", "data_onestep", "data_search", "train"])
    ap.add_argument("--signal", default="bcow")
    a = ap.parse_args()
    if a.stage == "data_bcow":
        data_bcow()
    elif a.stage == "data_onestep":
        data_onestep()
    elif a.stage == "data_search":
        data_search()
    else:
        train(a.signal)


if __name__ == "__main__":
    main()
