"""
P3 tournament: the opponent model of the exploration study (all ten
players in the search). At a real pick state the acting team picks hero h
with probability proportional to
  exp(w0 log p_GD(h | x) + log mean_{i in team} exp(w1 log1p n_ih + w2 [n_ih == 0]))
(n_ih: player i's earlier games on h; the team mixture because who picks is
unknown). Fit by maximum likelihood on the real picks of the pprior target
lobbies (tournament window, neither tournament nor probe lobbies; no
outcome used). Output: results/p3t/opp_model.json (w0, w1, w2, top-1 and
log-likelihood against GD alone).
Run (3090, run_rcf_full env): python personalization/p3t_oppmodel.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("P1_TIERS", "site")
import numpy as np
import torch

import p3_hs_core as C
import p3t_data as TD

PICKS = [4, 5, 6, 7, 8, 11, 12, 13, 14, 15]


def main():
    import p3t_systems as SY
    z = dict(np.load(os.path.join(TD.OUT, "pprior_targets.npz")))
    v = SY.vintage("cur")
    acts, n = z["acts"], z["n"].astype(np.float32)
    G = len(acts)
    LG, F1, F0, Y, M = [], [], [], [], []
    for k in PICKS:
        t = SY.DRAFT_TEAM[k]
        X = SY.features(acts, k, z["map"], z["tier"], t)
        m = SY.legal_mask(acts, k, np.ones((G, 2, 90), bool))
        p = SY.gd_probs(v, X[:, :289], m)
        LG.append(np.log(np.maximum(p, 1e-9)))
        n5 = n[:, 5 * t:5 * t + 5]
        F1.append(np.log1p(n5))
        F0.append((n5 == 0).astype(np.float32))
        Y.append(acts[:, k])
        M.append(m)
    LG, F1, F0, Y, M = (torch.from_numpy(np.concatenate(a)) for a in (LG, F1, F0, Y, M))
    Y = Y.long()
    w = torch.tensor([1.0, 0.5, -1.0], requires_grad=True)
    tr = torch.from_numpy(np.random.RandomState(1).rand(len(Y)) < 0.9)

    def nll(sel, w):
        u = w[1] * F1[sel] + w[2] * F0[sel]                       # (b, 5, 90)
        mix = torch.logsumexp(u, 1) - np.log(5.0)
        z_ = w[0] * LG[sel].float() + mix
        z_ = z_.masked_fill(~M[sel], -1e9)
        return -(torch.log_softmax(z_, 1).gather(1, Y[sel][:, None])).mean(), z_
    opt = torch.optim.LBFGS([w], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        l, _ = nll(tr, w)
        l.backward()
        return l
    opt.step(closure)
    with torch.no_grad():
        lv, z_ = nll(~tr, w)
        l0, z0 = nll(~tr, torch.tensor([1.0, 0.0, 0.0]))
        top = (z_.argmax(1) == Y[~tr]).float().mean().item()
        top0 = (z0.argmax(1) == Y[~tr]).float().mean().item()
    out = {"w0": float(w[0]), "w1": float(w[1]), "w2": float(w[2]), "picks": int(len(Y)),
           "val_nll": float(lv), "val_nll_gd_only": float(l0), "val_top1": top, "val_top1_gd_only": top0}
    os.makedirs(os.path.join(C.RESULTS, "p3t"), exist_ok=True)
    json.dump(out, open(os.path.join(C.RESULTS, "p3t", "opp_model.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
