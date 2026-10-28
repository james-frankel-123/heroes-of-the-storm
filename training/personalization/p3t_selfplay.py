"""
P3 tournament, option (d): self-play with sampled real rosters, so the
policy learns to exploit personal edges.

Policy (the search prior, per acting team, cuda_prior kernel):
  logit(h) = alpha * log p_F(h | x) + log mean_{i in team} exp g(f(p_i, h))
with p_F paper 1's F policy (blind-trained, frozen) and g the player-feature
MLP of p3t_pprior (same features). Expert iteration:
  iteration 0: g = 0, alpha = 1 (the F prior), i.e. aMCTS-cur in self-play
  each iteration: R rosters (p3t_data rosters: games 2026-03-29 to 05-22, all
    ten players with 50+ earlier games, state as of the day before; no
    outcome and no draft order used), full self-play drafts in which every
    step is searched from the acting team's side with the player-aware leaf
    (800 sims, root Dirichlet noise eps 0.25, root temperature 1), the visit
    distribution at every pick step becomes a target; alpha and g are refit
    on the last two iterations' targets (warm start).
Output: $P3_CACHE/p3t/selfplay_prior.pt (+ per-iteration copies),
$P3_RESULTS/p3t/selfplay.json (per iteration: rosters, fit, GPU seconds).
Run (GPU host with the rosters file): python personalization/p3t_selfplay.py --iters 4 --rosters 4096
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

import p3_hs_core as C
import p3t_data as TD
import p3t_pprior as PP

OUTM = os.path.join(TD.OUT, "selfplay_prior.pt")
PICKS = [4, 5, 6, 7, 8, 11, 12, 13, 14, 15]


def selfplay_round(R, model, sims, seed, device=0):
    """One batch of full self-play drafts; returns acts (n, 16) and visits (n, 16, 90)."""
    import p3_mcts_core as M
    import p3t_systems as SY
    kmod = M.load_module("prior_kernel", os.path.join(HERE, "cuda_prior"))
    v = SY.vintage("cur")
    n = len(R["map"])
    b = v.spec["combiner"]
    cfg = np.full((n, M.CFG_LEN), -1, np.int32)
    cfg[:, 0], cfg[:, 1], cfg[:, 2] = R["map"], R["tier"], 0
    cfg[:, 3], cfg[:, 4], cfg[:, 5], cfg[:, 6], cfg[:, 7] = 1, -1, 1, 0, 1
    slot_of, cnt = [], [0, 0]
    for k in range(16):
        t = SY.DRAFT_TEAM[k]
        if SY.IS_PICK[k]:
            slot_of.append(5 * t + cnt[t])
            cnt[t] += 1
        else:
            slot_of.append(-1)
    cfg[:, 24:40] = np.array(slot_of, np.int32)[None]
    cfg[:, 40:56] = -1
    cfg[:, 56:M.CFG_LEN] = 0
    cfg[:, 56] = 1
    cfg[:, M.CFG_ASSIGN] = 1
    s = R["s"].astype(np.float32)
    off = R["off"].astype(np.float32)
    pool = np.stack([np.stack([M.pool_bits(R["n"][i, q] > 0) for q in range(10)]) for i in range(n)])
    if model is None:
        bias, alpha = np.zeros((n, 10, 90), np.float32), 1.0
    else:
        with torch.no_grad():
            bias = model.slot_bias(torch.from_numpy(PP.features(s, R["off"], R["n"], float(b[2])))).numpy().astype(np.float32)
        alpha = float(model.alpha)
    coefs = np.array(list(b) + [0.0, alpha], np.float32)
    wf, wo, lut = v.wcfg()
    acts = np.zeros((n, 16), np.int64)
    vis = np.zeros((n, 16, 90), np.float32)
    seeds = np.arange(n) % v.n_seeds
    for i in range(v.n_seeds):
        sel = np.flatnonzero(seeds == i)
        pf, po = v.policy_flat(i)
        gf, go = v.gd_flat(i)
        for b0 in range(0, len(sel), 256):
            ii = sel[b0:b0 + 256]
            eng = (getattr(kmod, "PersonalEngine", None) or kmod.PriorEngine)(pf, gf, wf, po, go, wo, lut, max_concurrent=len(ii), device_id=device)
            r = eng.run(np.ascontiguousarray(cfg[ii]), np.ascontiguousarray(s[ii]), np.ascontiguousarray(off[ii]),
                        np.zeros((len(ii), 10, 90), np.float32), np.ascontiguousarray(pool[ii]), coefs,
                        np.ascontiguousarray(bias[ii]), sims, 2.0, int(seed + b0 + 7919 * i), 1.0, 0.3, 0.25, 1,
                        search_mode=1)
            del eng
            acts[ii] = r[3]
            turns = r[4]
            for j, e in enumerate(ii):
                for t_ in range(int(turns[j])):
                    st = int(r[7][j, t_])
                    if 0 <= st < 16:
                        vis[e, st] = r[5][j, t_]
    return acts, vis


def targets_from(R, acts, vis):
    import p3t_systems as SY
    n = len(acts)
    team_pool = np.stack([(R["n"][:, :5] > 0).any(1), (R["n"][:, 5:] > 0).any(1)], 1)
    LP, MS, TG, TT, II = [], [], [], [], []
    for k in PICKS:
        lp, m = PP.f_logp(acts, k, R["map"], R["tier"], team_pool, np.arange(n))
        tg = vis[:, k] * m
        ok = tg.sum(1) > 0
        tg = tg / np.maximum(tg.sum(1, keepdims=True), 1e-9)
        LP.append(lp[ok])
        MS.append(m[ok])
        TG.append(tg[ok])
        TT.append(np.full(ok.sum(), SY.DRAFT_TEAM[k]))
        II.append(np.flatnonzero(ok))
    return [np.concatenate(x) for x in (LP, MS, TG, TT, II)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--rosters", type=int, default=4096)
    ap.add_argument("--sims", type=int, default=800)
    a = ap.parse_args()
    import p3t_systems as SY
    torch.set_num_threads(4)
    Z = dict(np.load(os.path.join(TD.OUT, "rosters.npz")))
    NR = len(Z["map"])
    b2 = float(SY.vintage("cur").spec["combiner"][2])
    rng = np.random.RandomState(2026)
    log = {"sims": a.sims, "rosters_per_iter": a.rosters, "roster_pool": NR, "iters": []}
    model, prev = None, None
    for it in range(a.iters):
        idx = rng.choice(NR, a.rosters, replace=False)
        R = {k: Z[k][idx] for k in ("s", "off", "n", "map", "tier")}
        t0 = time.time()
        acts, vis = selfplay_round(R, model, a.sims, seed=100000 * (it + 1))
        gsec = time.time() - t0
        cur = (R, targets_from(R, acts, vis))
        # fit on this and the previous iteration's targets
        sets = [cur] + ([prev] if prev is not None else [])
        F = np.concatenate([PP.features(S_["s"], S_["off"], S_["n"], b2) for S_, _ in sets])
        off_i = np.cumsum([0] + [len(S_["map"]) for S_, _ in sets])
        LP, MS, TG, TT, II = [np.concatenate([t[j] for _, t in sets]) for j in range(4)] + \
            [np.concatenate([t[4] + o for (_, t), o in zip(sets, off_i)])]
        model, fit = PP.fit_prior(F, LP, MS, TG, TT, II, len(F),
                                  init=None if model is None else model.state_dict(), seed=it)
        torch.save(model.state_dict(), OUTM.replace(".pt", f"_it{it}.pt"))
        torch.save(model.state_dict(), OUTM)
        # how far the self-play drafts moved from the F prior: share of picks equal to the F argmax
        fa = []
        team_pool = np.stack([(R["n"][:, :5] > 0).any(1), (R["n"][:, 5:] > 0).any(1)], 1)
        for k in PICKS:
            lp, _ = PP.f_logp(acts, k, R["map"], R["tier"], team_pool, np.arange(len(acts)))
            fa.append((lp.argmax(1) == acts[:, k]).mean())
        ent = vis[:, PICKS] / np.maximum(vis[:, PICKS].sum(-1, keepdims=True), 1e-9)
        ent = float((-(ent * np.log(np.maximum(ent, 1e-12))).sum(-1)).mean())
        rec = {"iter": it, "gpu_seconds": gsec, "picks_equal_F_argmax": float(np.mean(fa)),
               "root_visit_entropy": ent, **fit}
        log["iters"].append(rec)
        print(json.dumps(rec), flush=True)
        prev = cur
        os.makedirs(os.path.join(C.RESULTS, "p3t"), exist_ok=True)
        json.dump(log, open(os.path.join(C.RESULTS, "p3t", "selfplay.json"), "w"), indent=1)
    print("SELFPLAY DONE")


if __name__ == "__main__":
    main()
