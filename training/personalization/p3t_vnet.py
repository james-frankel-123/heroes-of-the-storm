"""
P3 tournament, option (c): a value net with per-lobby player features.

Leaf (cuda_pvnet, cfg[57] = 1):
  V = sigmoid(b1 logit WP_pop + h(F_ours) - h(F_theirs))
  F_t = [S_t, O_t, the five assigned s sorted descending, the five assigned
         log1p(earlier games on the hero) sorted descending]
  assignment: each team's best hero -> player permutation by b2 S + b3 O
  (the linear combiner's b2, b3; the same rule as the linear leaf)
  h: MLP 12 -> 16 -> 1 (ReLU); antisymmetric by construction.
Data (stage data, 3090): real games the paper-1 WP never trained on (its
test and validation splits, plus snapshot-era games uploaded after the
snapshot), 2024-04-01 to 2026-05-22, all ten slots known; skill state as of
the day before (offset + posterior + layer), WP_pop = the cur vintage WP.
Fit (stage train, CPU): log loss, b1 and h; validation = the last 15% of
days; reports validation log loss against the linear combiner.
Output: $P3_CACHE/p3t/vnet_data.npz, $P3_RESULTS/p3t/vnet.json (weights + metrics);
the weights are copied into p3t_vintages.json (cur_bc_vn).
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

DATA = os.path.join(TD.OUT, "vnet_data.npz")
PERMS = np.array(list(__import__("itertools").permutations(range(5))), np.int64)


def data():
    import p3t_value as PV
    import p3t_systems as SY
    from shared import HEROES
    t0 = time.time()
    par, d, aug, mets, mw = TD.load_model()
    lay = TD.Layer(d, aug, TD.layer_meta())
    meta = PV.game_meta()
    test, is_val = PV.heldout_ids(None)
    lo_d, hi_d = C.day_of("2024-04-01"), C.day_of("2026-05-22")
    cand = np.flatnonzero((d["day"] >= lo_d) & (d["day"] <= hi_d))
    keep = np.array([(int(r) in test) or is_val(int(r)) or bool(p) for r, p in zip(d["replay_id"][cand], d["post"][cand])])
    q = cand[keep]
    q = q[np.isin(d["replay_id"][q], np.fromiter(meta.keys(), np.int64))]
    cnt = np.bincount(d["g"][q], minlength=int(d["g"].max()) + 1)
    q = q[cnt[d["g"][q]] == 10]
    o = np.lexsort((d["team"][q], d["g"][q]))
    q = q[o].reshape(-1, 10)
    assert (d["team"][q[:, :5]] == 0).all()
    flat = q.ravel()
    res = TD.walk(aug, par, mets, mw, d["pid"][flat].astype(np.int64), d["day"][flat].astype(np.int64))
    s = TD.offset(aug["table"], res["n_slot"]) + res["mean_all"] + lay.at(
        d["pid"][flat].astype(np.int64), d["day"][flat].astype(np.int64), res["n_all"])
    off = TD.off_role(d, res["n_all"])
    G = len(q)
    hero = d["hero"][q]                                     # (G, 10) d index
    # per team: (5 players x 5 team heroes) matrices
    S5 = np.zeros((G, 2, 5, 5), np.float32)
    O5 = np.zeros((G, 2, 5, 5), np.float32)
    N5 = np.zeros((G, 2, 5, 5), np.float32)
    sr = s.reshape(G, 10, -1)
    orr = off.reshape(G, 10, -1)
    nr = res["n_all"].reshape(G, 10, -1)
    for t in (0, 1):
        hs = hero[:, 5 * t:5 * t + 5]
        for i in range(5):
            S5[:, t, i] = np.take_along_axis(sr[:, 5 * t + i], hs, 1)
            O5[:, t, i] = np.take_along_axis(orr[:, 5 * t + i], hs, 1)
            N5[:, t, i] = np.log1p(np.take_along_axis(nr[:, 5 * t + i], hs, 1))
    y = np.where(d["team"][q[:, 0]] == 0, d["y"][q[:, 0]], 1 - d["y"][q[:, 0]])
    rows = []
    sh = TD.to_shared(d["hero_names"])
    for g in range(G):
        tier, gm = meta[int(d["replay_id"][q[g, 0]])]
        rows.append(([HEROES[sh[h]] for h in hero[g, :5]], [HEROES[sh[h]] for h in hero[g, 5:]], gm, tier))
    v = SY.vintage("cur")
    wp = np.concatenate([v.wp_drafts(rows[i:i + 5000]) for i in range(0, G, 5000)])
    np.savez(DATA, S5=S5, O5=O5, N5=N5, y=y, wp=wp, day=d["day"][q[:, 0]])
    print(f"vnet data: {G:,} games ({time.time() - t0:.0f}s)", flush=True)


class H(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(12, 16)
        self.l2 = nn.Linear(16, 1)
        self.b1 = nn.Parameter(torch.tensor(1.0))

    def h(self, F):
        return self.l2(torch.relu(self.l1(F))).squeeze(-1)


def team_feats(S5, O5, N5, b2, b3):
    """best assignment by b2 S + b3 O; F (G, 2, 12)."""
    i = np.arange(5)
    sp = S5[:, :, i[None, :], PERMS].sum(-1)            # (G, 2, 120)
    op = O5[:, :, i[None, :], PERMS].sum(-1)
    best = np.argmax(b2 * sp + b3 * op, -1)               # (G, 2)
    perm = PERMS[best]                                    # (G, 2, 5): player i takes hero perm[i]
    s_as = np.take_along_axis(S5, perm[..., None], -1)[..., 0]
    n_as = np.take_along_axis(N5, perm[..., None], -1)[..., 0]
    S = s_as.sum(-1)
    O = np.take_along_axis(O5, perm[..., None], -1)[..., 0].sum(-1)
    return np.concatenate([S[..., None], O[..., None], -np.sort(-s_as, -1), -np.sort(-n_as, -1)], -1).astype(np.float32)


def train():
    import p3t_systems as SY
    import p3_x_common as XC
    z = dict(np.load(DATA))
    b = SY.vintage("cur").spec["combiner"]
    F = team_feats(z["S5"], z["O5"], z["N5"], b[2], b[3])
    lo = np.log(np.clip(z["wp"], 1e-6, 1 - 1e-6) / np.clip(1 - z["wp"], 1e-6, 1 - 1e-6)).astype(np.float32)
    y = z["y"].astype(np.float32)
    cut = np.quantile(z["day"], 0.85)
    tr, va = z["day"] < cut, z["day"] >= cut
    torch.manual_seed(0)
    m = H()
    with torch.no_grad():
        # start at the linear combiner: h(F) = b2 S + b3 O
        m.l1.weight.zero_()
        m.l1.bias.zero_()
        m.l1.weight[0, 0], m.l1.weight[0, 1] = 1.0, 0.0
        m.l1.bias[0] = 5.0                                 # keeps the unit in its linear region
        m.l2.weight.zero_()
        m.l2.weight[0, 0] = b[2]
        m.l1.weight[1, 1] = 1.0
        m.l1.bias[1] = 5.0
        m.l2.weight[0, 1] = b[3]
        m.b1.fill_(b[1])
    Ft, lot, yt = torch.from_numpy(F), torch.from_numpy(lo), torch.from_numpy(y)

    def ll(sel):
        z_ = m.b1 * lot[sel] + m.h(Ft[sel, 0]) - m.h(Ft[sel, 1])
        return nn.functional.binary_cross_entropy_with_logits(z_, yt[sel])
    tri, vai = torch.from_numpy(np.flatnonzero(tr)), torch.from_numpy(np.flatnonzero(va))
    with torch.no_grad():
        l0 = ll(vai).item()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)
    best, bad = (l0, {k: v.clone() for k, v in m.state_dict().items()}), 0
    for ep in range(200):
        perm = tri[torch.randperm(len(tri))]
        for b0 in range(0, len(perm), 4096):
            opt.zero_grad()
            l = ll(perm[b0:b0 + 4096])
            l.backward()
            opt.step()
        with torch.no_grad():
            lv = ll(vai).item()
        if lv < best[0] - 1e-6:
            best, bad = (lv, {k: v.clone() for k, v in m.state_dict().items()}), 0
        else:
            bad += 1
            if bad >= 10:
                break
    m.load_state_dict(best[1])
    # linear combiner on the same validation games (refit on train for a fair comparison)
    X = np.column_stack([lo, F[:, 0, 0] - F[:, 1, 0], F[:, 0, 1] - F[:, 1, 1]])
    w = C.fit_logistic(X[tr], y[tr])
    pl = C.predict(w, X[va])
    with torch.no_grad():
        pn = torch.sigmoid(m.b1 * lot[vai] + m.h(Ft[vai, 0]) - m.h(Ft[vai, 1])).numpy()
    llin = float(XC.logloss(pl, y[va]).mean())
    lnet = float(XC.logloss(pn, y[va]).mean())
    d_ = XC.logloss(pl, y[va]) - XC.logloss(pn, y[va])
    vnet = np.concatenate([m.l1.weight.detach().numpy().ravel(), m.l1.bias.detach().numpy(),
                           m.l2.weight.detach().numpy().ravel(), m.l2.bias.detach().numpy()]).astype(np.float32)
    assert len(vnet) == 225
    out = {"games": int(len(y)), "train": int(tr.sum()), "val": int(va.sum()), "val_from_day": float(cut),
           "b1": float(m.b1), "val_ll_linear": llin, "val_ll_net": lnet,
           "net_minus_linear_gain_x1000": float(1000 * d_.mean()),
           "se_x1000": float(1000 * d_.std(ddof=1) / np.sqrt(len(d_))), "vnet": vnet.tolist()}
    json.dump(out, open(os.path.join(C.RESULTS, "p3t", "vnet.json"), "w"), indent=1)
    vs = json.load(open(SY.VFILE))
    spec = dict(vs["cur_bc"])
    spec["note"] = "cur_bc with the value-net leaf (p3t_vnet; cuda_pvnet cfg[57]=1)"
    spec["vnet"] = vnet.tolist()
    spec["vnet_b1"] = float(m.b1)
    vs["cur_bc_vn"] = spec
    json.dump(vs, open(SY.VFILE, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "vnet"}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["data", "train"])
    a = ap.parse_args()
    data() if a.stage == "data" else train()


if __name__ == "__main__":
    main()
