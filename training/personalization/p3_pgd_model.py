"""
P3 personalized GD, stage 3: the personal model for picks and bans.

  pick:  u(h) = alpha_p log p_metaGD(h | state) + f_p(context of the acting player, h)
  ban:   u(h) = alpha_b log p_metaGD(h | state) + f_b(context of both teams' players
                                                  still to pick, h)
f_p: MLP 12 -> 64 -> 64 -> 1 on the acting player's per-hero features
(p3_pgd_feats.FEATS: shrunken share toward the player's preference cluster,
cluster share, embedding affinity, games, never played, EWMA shares, recency,
role share, main x main share, MAWP, volume).
f_b: MLP on per-hero aggregates over the opponents still to pick (max and
sum of shrunken share, max main x main share, max share-weighted MAWP, max
EWMA-20 share, number with 10+ games on the hero, max affinity) and over the
banning team's own players still to pick (max shrunken share, max main x
main share, sum shrunken share), plus the number of opponents left.
Both personal terms are fixed within a draft phase, so the MCTS kernel can
add them as per-slot (picks) and per-team-and-phase (bans) vectors.

Data: train 50,000 games from 2025-04-01 .. 2026-02-09 (before the cutoff),
validation 5,000 other pre-cutoff games; test 25,000 V2 games and 25,000
post-snapshot games. Baselines: paper-1 GD (in sample on V2), causal meta GD,
the imitation model (p3_dr_imitation weights; picks only). Ablation: the same
model without clusters and embedding (shrinkage toward the population share).
Breakdowns: picks by the acting player's games before the game (0, 1-20,
21-100, 100+) and type (one-trick / specialist / flexible, among 30+ games);
bans by the remaining opponents' mean history and the most concentrated
remaining opponent.

Run (from training/): OMP_NUM_THREADS=4 nice -n 19 taskset -c 48-63 python3 personalization/p3_pgd_model.py
Outputs: cache/pgd_personal.pt, results/p3_pgd_model.json
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn as nn

from p3_heroes import NUM_HEROES
import p3_hs_core as C
import p3_x_common as X
import p3_pgd_common as P
import p3_pgd_feats as PF

OUT = os.path.join(C.CACHE, "pgd_personal.pt")
BAN_FEATS = ["opp_max_share", "opp_sum_share", "opp_max_main", "opp_max_mawp_x_share", "opp_max_e20",
             "opp_n_with_10_games", "opp_max_affinity", "own_max_share", "own_max_main", "own_sum_share",
             "n_opp_left"]


class Head(nn.Module):
    def __init__(self, nf, h=64):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.f = nn.Sequential(nn.Linear(nf, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
        self.register_buffer("mu", torch.zeros(nf))
        self.register_buffer("sd", torch.ones(nf))

    def bias(self, F):
        return self.f((F - self.mu) / self.sd).squeeze(-1)

    def forward(self, lp, F, M):
        return (self.alpha * lp + self.bias(F)).masked_fill(~M, -1e9)


def remaining_slots(g, gi, k):
    """For ban step k: (opponent slot steps, own slot steps) still to pick."""
    T = g["T"][gi]
    t_ban = T[k]
    later = [s for s in P.PICK_STEPS if s > k]
    opp = [s for s in later if T[s] != t_ban]
    own = [s for s in later if T[s] == t_ban]
    return opp, own


def build(g, d, games, hist_rows, Fall, aux, lp_meta_fn, lp_paper_fn, kind):
    """Samples for picks or bans of the given games."""
    rowpos = hist_rows
    if kind == "pick":
        gi = np.repeat(games, 10)
        k = np.tile(P.PICK_STEPS, len(games))
    else:
        gi = np.repeat(games, 6)
        k = np.tile(P.BAN_STEPS, len(games))
    Xs, M = P.states(g, gi, k)
    y = g["H"][gi, k]
    ok = M[np.arange(len(y)), y]
    gi, k, Xs, M, y = gi[ok], k[ok], Xs[ok], M[ok], y[ok]
    lpm = lp_meta_fn(Xs, M)
    lpp = lp_paper_fn(Xs, M)
    if kind == "pick":
        rows = g["R"][gi, k]
        q = rowpos[rows]
        F = Fall[q]
        meta = {"tot": aux["tot"][q], "main_share": aux["main_share"][q],
                "main": aux["is_main"][q].argmax(1), "q": q}
        return dict(gi=gi, k=k, M=M, y=y, lpm=lpm, lpp=lpp, F=F, meta=meta)
    G = np.zeros((len(gi), NUM_HEROES, len(BAN_FEATS)), np.float32)
    opp_tot = np.zeros(len(gi))
    opp_type = np.zeros(len(gi), np.int64)
    for i, (gg, kk) in enumerate(zip(gi, k)):
        opp, own = remaining_slots(g, gg, kk)
        qo = rowpos[g["R"][gg, opp]]
        qw = rowpos[g["R"][gg, own]] if own else np.zeros(0, np.int64)
        so = aux["sshare"][qo]
        mo = aux["is_main"][qo] * aux["main_share"][qo][:, None]
        G[i, :, 0] = so.max(0)
        G[i, :, 1] = so.sum(0)
        G[i, :, 2] = mo.max(0)
        G[i, :, 3] = ((Fall[qo, :, 10]) * so).max(0)
        G[i, :, 4] = Fall[qo, :, 5].max(0)
        G[i, :, 5] = (np.expm1(Fall[qo, :, 3]) >= 10).sum(0) / 5.0
        G[i, :, 6] = Fall[qo, :, 2].max(0)
        if len(qw):
            G[i, :, 7] = aux["sshare"][qw].max(0)
            G[i, :, 8] = (aux["is_main"][qw] * aux["main_share"][qw][:, None]).max(0)
            G[i, :, 9] = aux["sshare"][qw].sum(0)
        G[i, :, 10] = len(opp) / 5.0
        opp_tot[i] = aux["tot"][qo].mean()
        ms = np.where(aux["tot"][qo] >= 30, aux["main_share"][qo], 0)
        opp_type[i] = 0 if (ms >= 0.5).any() else (1 if (ms >= 0.25).any() else (2 if (aux["tot"][qo] >= 30).any() else 3))
    return dict(gi=gi, k=k, M=M, y=y, lpm=lpm, lpp=lpp, F=G, meta={"opp_tot": opp_tot, "opp_type": opp_type})


def train_head(tr, va, nf, epochs=12, seed=0):
    torch.manual_seed(seed)
    m = Head(nf)
    with torch.no_grad():
        f = torch.from_numpy(tr["F"]).reshape(-1, nf)
        m.mu.copy_(f.mean(0))
        m.sd.copy_(f.std(0) + 1e-6)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    T = {k: torch.from_numpy(tr[k]) for k in ("lpm", "F", "M", "y")}
    V = {k: torch.from_numpy(va[k]) for k in ("lpm", "F", "M", "y")}
    best, state = 1e9, None
    rng = np.random.RandomState(seed)
    for ep in range(epochs):
        idx = rng.permutation(len(tr["y"]))
        for s in range(0, len(idx), 2048):
            b = torch.from_numpy(idx[s:s + 2048])
            loss = nn.functional.cross_entropy(m(T["lpm"][b], T["F"][b], T["M"][b]), T["y"][b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            lv = 0.0
            for s in range(0, len(va["y"]), 8192):
                sl = slice(s, s + 8192)
                lv += float(nn.functional.cross_entropy(m(V["lpm"][sl], V["F"][sl], V["M"][sl]), V["y"][sl],
                                                        reduction="sum"))
            lv /= len(va["y"])
        if lv < best - 1e-5:
            best, state = lv, {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(state)
    m.eval()
    return m, best


def logp_head(m, S):
    out = []
    with torch.no_grad():
        for s in range(0, len(S["y"]), 8192):
            sl = slice(s, s + 8192)
            out.append(torch.log_softmax(m(torch.from_numpy(S["lpm"][sl]), torch.from_numpy(S["F"][sl]),
                                           torch.from_numpy(S["M"][sl])), -1).numpy())
    return np.concatenate(out)


def imitation_logp(S, hist_q, fine, nfine):
    """p3_dr_imitation weights on equivalent features (picks)."""
    w = np.load(os.path.join(C.CACHE, "dr_imitation.npz"))["w"]
    n = hist_q["n"]
    tot = n.sum(1)
    fc = np.stack([n[:, fine == f].sum(1) for f in range(nfine)], 1)
    fshare = (fc[:, fine] + 1) / (tot[:, None] + nfine)
    days = hist_q["days"]
    Xf = np.stack([S["lpp"], np.log1p(n), (n == 0), hist_q["e20"], hist_q["e100"],
                   np.log1p(np.maximum(days, 0)), (days < 0), np.log(fshare)], -1).astype(np.float32)
    u = np.where(S["M"], (Xf * w).sum(-1), -1e9)
    return u - np.log(np.exp(u - u.max(1, keepdims=True)).sum(1, keepdims=True)) - u.max(1, keepdims=True)


def main():
    t0 = time.time()
    torch.set_num_threads(4)
    rng = np.random.RandomState(0)
    g = P.load_games()
    d = X.load_ext()
    meta = C.hero_meta(d["hero_names"])
    tops, n_fit, sizes = PF.fit_pref(d, P.CUTOFF)
    print(f"preference clusters from {n_fit:,} players: sizes {sizes.tolist()}", flush=True)
    w = g["window"]
    pre = np.flatnonzero((w == 0) & (g["day"] >= C.day_of("2025-04-01")))
    sel = rng.choice(pre, 55000, replace=False)
    sets = {"train": sel[:50000], "val": sel[50000:],
            "V2": rng.choice(np.flatnonzero(w == 2), 25000, replace=False),
            "post": rng.choice(np.flatnonzero(w == 3), 25000, replace=False)}
    allg = np.concatenate(list(sets.values()))
    qrows = np.unique(g["R"][allg][:, P.PICK_STEPS].ravel())
    hist = PF.walk(d, qrows)
    rowpos = np.full(len(d["pid"]), -1, np.int64)
    rowpos[qrows] = np.arange(len(qrows))
    print(f"walker: {len(qrows):,} player-game rows ({time.time() - t0:.0f}s)", flush=True)
    import p3_dr_core as D
    gd = D.GDPolicy(d["hero_names"])
    mg = P.load_metagd()
    lp_meta = lambda Xs, M: P.logp_metagd(mg, Xs, M)
    lp_paper = lambda Xs, M: P.logp_paper_gd(gd, Xs, M)
    out = {"cluster_top_heroes": tops, "cluster_sizes": sizes.tolist(), "pick_feats": PF.FEATS,
           "ban_feats": BAN_FEATS, "sets_games": {k: int(len(v)) for k, v in sets.items()}}
    variants = {}
    for vname, use_cl in (("personal GD", True), ("personal GD without clusters", False)):
        Fall, aux = PF.context(hist, meta["fine"], use_cluster=use_cl)
        S = {}
        for nm, gs in sets.items():
            S[nm] = {"pick": build(g, d, gs, rowpos, Fall, aux, lp_meta, lp_paper, "pick"),
                     "ban": build(g, d, gs, rowpos, Fall, aux, lp_meta, lp_paper, "ban")}
        heads = {}
        for kind, nf in (("pick", len(PF.FEATS)), ("ban", len(BAN_FEATS))):
            heads[kind], bv = train_head(S["train"][kind], S["val"][kind], nf)
            print(f"{vname} {kind}: val log loss {bv:.4f}, alpha {float(heads[kind].alpha):.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        variants[vname] = (S, heads)
        if use_cl:
            torch.save({"pick": heads["pick"].state_dict(), "ban": heads["ban"].state_dict(),
                        "pick_feats": PF.FEATS, "ban_feats": BAN_FEATS}, OUT)
            del Fall
    S, heads = variants["personal GD"]
    S0, heads0 = variants["personal GD without clusters"]
    hq_all = {k: v for k, v in hist.items()}
    res = {}
    for split in ("V2", "post"):
        res[split] = {}
        for kind in ("pick", "ban"):
            s, s0 = S[split][kind], S0[split][kind]
            ks = (1, 5) if kind == "pick" else (1, 3)
            lps = {"paper-1 GD": s["lpp"], "meta GD (causal)": s["lpm"],
                   "personal GD": logp_head(heads[kind], s),
                   "personal GD without clusters": logp_head(heads0[kind], s0)}
            if kind == "pick":
                hq = {k: v[s["meta"]["q"]] for k, v in hq_all.items()}
                lps["imitation model"] = imitation_logp(s, hq, meta["fine"], len(meta["fine_names"]))
                tot, ms = s["meta"]["tot"], s["meta"]["main_share"]
                groups = {"all": np.ones(len(tot), bool),
                          "history 0": tot == 0, "history 1-20": (tot >= 1) & (tot <= 20),
                          "history 21-100": (tot > 20) & (tot <= 100), "history 100+": tot > 100,
                          "one-trick (30+ games, main >= 50%)": (tot >= 30) & (ms >= 0.5),
                          "specialist (25-50%)": (tot >= 30) & (ms >= 0.25) & (ms < 0.5),
                          "flexible (< 25%)": (tot >= 30) & (ms < 0.25),
                          "under 30 games": tot < 30}
            else:
                ot, ty = s["meta"]["opp_tot"], s["meta"]["opp_type"]
                groups = {"all": np.ones(len(ot), bool),
                          "opponents' mean history 0-20": ot <= 20, "21-100": (ot > 20) & (ot <= 100),
                          "100+": ot > 100,
                          "a one-trick opponent still to pick": ty == 0,
                          "specialist (no one-trick)": ty == 1, "flexible opponents only": ty == 2,
                          "low-history opponents only": ty == 3}
            tab = {}
            for gname, msk in groups.items():
                if msk.sum() < 200:
                    continue
                tab[gname] = {m: P.scores(lp[msk], s["y"][msk], ks) for m, lp in lps.items()}
            # one-trick main calibration (picks): model probability of the main vs real
            if kind == "pick":
                msk = groups["one-trick (30+ games, main >= 50%)"]
                mains = s["meta"]["main"][msk]
                av = s["M"][msk][np.arange(msk.sum()), mains]
                real = (s["y"][msk] == mains)[av].mean()
                tab["one-trick main pick, main available"] = {
                    "real frequency": float(real),
                    **{m: float(np.exp(lp[msk][np.arange(msk.sum()), mains][av]).mean()) for m, lp in lps.items()}}
            res[split][kind] = tab
            print(split, kind, json.dumps({g_: {m: (v["top1"], round(v["log_loss"], 3)) for m, v in t.items()}
                                           if "real frequency" not in t else t for g_, t in tab.items()}), flush=True)
    out["results"] = res
    with open(os.path.join(C.RESULTS, "p3_pgd_model.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
