"""
Are hero counters (and pair synergies) big enough for search to exploit? Two checks.

real   Real outcomes, no learned value function. Logistic main-effects model fit on the
       training split: hero strength by tier and by map (team 0 minus team 1). Then
         (a) pair effects from the residuals: counter(a, b) = mean residual of the side with a
             in games where a faces b (both seatings); synergy(a, b) = same for a and b on one
             team. SE from the binomial variance; empirical Bayes: the SD of the TRUE pair
             effects = sqrt(var(estimates) - mean(SE^2)), shrunk estimates, top pairs.
         (b) predictive test: refit with a counter matrix and a synergy matrix (L2, weight
             chosen on val) and report the held-out (test split) log-loss gain over main
             effects alone. If counters are real and usable, they must predict new games.
         (c) per hero: the largest shrunk counter against it (what an opponent could do
             to that first pick with one answer pick), in win-probability points.
model  The tournament value function (enriched leak-free WP, search_actor.LeafWP) on held-out
       human drafts: for each game, replace one random hero of the opponent with every free
       hero b; the value function's counter(a, b) is how much more b helps against teams with
       a than against teams without a. Its spread and per-hero worst case, and the correlation
       with the real shrunk counters.

Usage (from training/, p1site env):  python3 paper1_revision/counter_checks.py real|model
Output: site/results/counter_checks/{real,model}.json
"""
import os
import sys
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)
from paper1_revision import core

OUT = os.path.join(core.RESULTS, "counter_checks")
H = 90


def arrays(games):
    from shared import HERO_TO_IDX
    T0 = np.array([[HERO_TO_IDX[h] for h in g[3]] for g in games], np.int64)
    T1 = np.array([[HERO_TO_IDX[h] for h in g[4]] for g in games], np.int64)
    y = np.array([1.0 if g[6] == 0 else 0.0 for g in games])
    tiers = sorted({g[1] for g in games})
    maps = sorted({g[2] for g in games})
    return T0, T1, y, [g[1] for g in games], [g[2] for g in games]


def encode(tier, gmap, tiers, maps):
    ti = np.array([tiers.index(t) if t in tiers else -1 for t in tier])
    mi = np.array([maps.index(m) if m in maps else -1 for m in gmap])
    return ti, mi


def fit(T0, T1, ti, mi, y, nt, nm, pairs=False, l2_pair=0.0, l2=1e-6, iters=60, init=None):
    """Full-batch LBFGS logistic regression in torch. Main effects: hero, hero x tier, hero x map.
    pairs=True adds counter C[a,b] (a on team 0 vs b on team 1, antisymmetrized) and synergy S."""
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    t0, t1 = torch.tensor(T0), torch.tensor(T1)
    tt, mm = torch.tensor(np.maximum(ti, 0)), torch.tensor(np.maximum(mi, 0))
    tok, mok = torch.tensor(ti >= 0).double(), torch.tensor(mi >= 0).double()
    Y = torch.tensor(y)
    p = {"b": torch.zeros(1, dtype=torch.float64), "h": torch.zeros(H, dtype=torch.float64),
         "ht": torch.zeros(nt, H, dtype=torch.float64), "hm": torch.zeros(nm, H, dtype=torch.float64)}
    if pairs:
        p["C"] = torch.zeros(H, H, dtype=torch.float64)
        p["S"] = torch.zeros(H, H, dtype=torch.float64)
    if init:
        for k, v in init.items():
            p[k] = torch.tensor(v, dtype=torch.float64)
    for v in p.values():
        v.requires_grad_(True)
    ia, ib = np.triu_indices(5, 1)

    def logit():
        def side(t):
            return (p["h"][t] + p["ht"][tt[:, None], t] * tok[:, None] + p["hm"][mm[:, None], t] * mok[:, None]).sum(1)
        z = p["b"] + side(t0) - side(t1)
        if pairs:
            C = p["C"] - p["C"].T
            z = z + C[t0[:, :, None], t1[:, None, :]].sum((1, 2))
            S = p["S"] + p["S"].T
            z = z + S[t0[:, ia], t0[:, ib]].sum(1) - S[t1[:, ia], t1[:, ib]].sum(1)
        return z

    opt = torch.optim.LBFGS(list(p.values()), lr=1, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = logit()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, Y)
        loss = loss + l2 * sum((v ** 2).sum() for k, v in p.items() if k in ("h", "ht", "hm"))
        if pairs:
            loss = loss + l2_pair * ((p["C"] ** 2).sum() + (p["S"] ** 2).sum())
        loss.backward()
        return loss
    opt.step(closure)
    return {k: v.detach().numpy().copy() for k, v in p.items()}, logit


def predict(par, T0, T1, ti, mi):
    def side(t):
        return (par["h"][t] + par["ht"][np.maximum(ti, 0)[:, None], t] * (ti >= 0)[:, None]
                + par["hm"][np.maximum(mi, 0)[:, None], t] * (mi >= 0)[:, None]).sum(1)
    z = par["b"][0] + side(T0) - side(T1)
    if "C" in par:
        C = par["C"] - par["C"].T
        z = z + C[T0[:, :, None], T1[:, None, :]].sum((1, 2))
        S = par["S"] + par["S"].T
        ia, ib = np.triu_indices(5, 1)
        z = z + S[T0[:, ia], T0[:, ib]].sum(1) - S[T1[:, ia], T1[:, ib]].sum(1)
    return 1 / (1 + np.exp(-z))


def ll(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def pair_resid(T0, T1, r, w):
    """counter[a,b]: mean residual of a's side when a faces b; synergy[a,b]: a and b together.
    r = y - p (team 0 view); w = p (1 - p) for the SE."""
    sc, nc, vc = np.zeros((H, H)), np.zeros((H, H)), np.zeros((H, H))
    ss, ns, vs = np.zeros((H, H)), np.zeros((H, H)), np.zeros((H, H))
    for i in range(5):
        for j in range(5):
            np.add.at(sc, (T0[:, i], T1[:, j]), r); np.add.at(nc, (T0[:, i], T1[:, j]), 1); np.add.at(vc, (T0[:, i], T1[:, j]), w)
            np.add.at(sc, (T1[:, j], T0[:, i]), -r); np.add.at(nc, (T1[:, j], T0[:, i]), 1); np.add.at(vc, (T1[:, j], T0[:, i]), w)
    for i in range(5):
        for j in range(5):
            if i == j:
                continue
            for T, sgn in ((T0, 1), (T1, -1)):
                np.add.at(ss, (T[:, i], T[:, j]), sgn * r); np.add.at(ns, (T[:, i], T[:, j]), 1); np.add.at(vs, (T[:, i], T[:, j]), w)
    out = {}
    for name, s, n, v in (("counter", sc, nc, vc), ("synergy", ss, ns, vs)):
        est = np.where(n > 0, s / np.maximum(n, 1), np.nan)
        se = np.where(n > 0, np.sqrt(v) / np.maximum(n, 1), np.nan)
        out[name] = (est, se, n)
    return out


def eb(est, se, n, min_n=200):
    """Empirical-Bayes SD of the true effects and shrunk estimates (normal-normal)."""
    ok = (n >= min_n) & np.isfinite(est)
    e, s = est[ok], se[ok]
    tau2 = max(float(np.var(e) - np.mean(s ** 2)), 0.0)
    shrink = np.full(est.shape, np.nan)
    shrink[ok] = e * tau2 / (tau2 + s ** 2)
    return {"pairs": int(ok.sum()), "sd_estimates_pts": 100 * float(np.std(e)), "mean_se_pts": 100 * float(np.sqrt(np.mean(s ** 2))),
            "sd_true_pts": 100 * float(np.sqrt(tau2))}, shrink


def real():
    from shared import HEROES
    G = core.split_games()
    tr, va, te = G["train"], G["val"], G["test"]
    tiers = sorted({g[1] for g in tr})
    maps = sorted({g[2] for g in tr})
    A = {k: arrays(v) for k, v in (("train", tr), ("val", va), ("test", te))}
    E = {k: encode(A[k][3], A[k][4], tiers, maps) for k in A}
    T0, T1, y = A["train"][:3]
    ti, mi = E["train"]
    print("fit main effects", flush=True)
    par0, _ = fit(T0, T1, ti, mi, y, len(tiers), len(maps))
    p = predict(par0, T0, T1, ti, mi)
    res = {"games": {k: int(len(A[k][2])) for k in A}, "main": {}}
    for k in ("val", "test"):
        res["main"][f"{k}_ll"] = ll(predict(par0, A[k][0], A[k][1], *E[k]), A[k][2])
    res["main"]["test_ll_coinflip"] = ll(np.full(len(A["test"][2]), A["train"][2].mean()), A["test"][2])
    print(res["main"], flush=True)
    # (a) residual pair effects + empirical Bayes
    pr = pair_resid(T0, T1, y - p, p * (1 - p))
    res["pairs"] = {}
    shr = {}
    for name in ("counter", "synergy"):
        est, se, n = pr[name]
        est = est.copy()
        est[np.eye(H, dtype=bool)] = np.nan
        summ, sh = eb(est, se, n)
        shr[name] = sh
        o = np.argsort(np.nan_to_num(sh, nan=0.0), axis=None)
        top = []
        for idx in (o[:12] if name == "counter" else o[::-1][:12]):
            a, b = divmod(int(idx), H)
            top.append({"a": HEROES[a], "b": HEROES[b], "n": int(n[a, b]), "raw_pts": 100 * float(est[a, b]),
                        "se_pts": 100 * float(se[a, b]), "shrunk_pts": 100 * float(sh[a, b])})
        summ["top"] = top
        res["pairs"][name] = summ
        print(name, {k: v for k, v in summ.items() if k != "top"}, flush=True)
    # (c) per hero: the worst shrunk counter against it (a's side loses by this much vs b)
    worst = np.nanmin(shr["counter"], axis=1)
    res["worst_counter_per_hero_pts"] = {"median": 100 * float(np.nanmedian(worst)), "p10": 100 * float(np.nanpercentile(worst, 10)),
                                         "min": 100 * float(np.nanmin(worst)),
                                         "by_hero": {HEROES[a]: 100 * float(worst[a]) for a in range(H)}}
    # (b) predictive test with pair matrices, L2 on val
    tv, mv = E["val"]
    best = None
    for l2p in (1e-4, 1e-5, 1e-6):
        print("fit pairs l2", l2p, flush=True)
        par, _ = fit(T0, T1, ti, mi, y, len(tiers), len(maps), pairs=True, l2_pair=l2p,
                     init={k: v for k, v in par0.items()})
        v = ll(predict(par, A["val"][0], A["val"][1], tv, mv), A["val"][2])
        print("  val ll", v, flush=True)
        if best is None or v < best[0]:
            best = (v, l2p, par)
    v, l2p, par = best
    te_ll = ll(predict(par, A["test"][0], A["test"][1], *E["test"]), A["test"][2])
    res["pair_model"] = {"l2_pair": l2p, "val_ll": v, "test_ll": te_ll,
                         "test_gain_x1000": 1000 * (res["main"]["test_ll"] - te_ll),
                         "main_gain_over_coinflip_x1000": 1000 * (res["main"]["test_ll_coinflip"] - res["main"]["test_ll"])}
    print(res["pair_model"], flush=True)
    os.makedirs(OUT, exist_ok=True)
    np.save(os.path.join(OUT, "real_counter_shrunk.npy"), shr["counter"])
    json.dump(res, open(os.path.join(OUT, "real.json"), "w"), indent=1)
    print("wrote real.json")


def model(n_games=1500, seed=0):
    from shared import HEROES, HERO_TO_IDX
    from paper1_revision import search_actor as sa
    G = core.split_games()
    rng = np.random.default_rng(seed)
    va = [G["val"][i] for i in rng.choice(len(G["val"]), n_games, replace=False)]
    leaf = sa.LeafWP("enriched")
    D = np.full((n_games, H), np.nan)   # WP of team 0 with team-1 slot replaced by b
    has = np.zeros((n_games, H), bool)  # a on team 0
    for gi, g in enumerate(va):
        t0 = [HERO_TO_IDX[h] for h in g[3]]
        t1 = [HERO_TO_IDX[h] for h in g[4]]
        bans = [HERO_TO_IDX[h] for h in g[5] if h in HERO_TO_IDX]
        has[gi, t0] = True
        j = int(rng.integers(5))
        rest = [h for k, h in enumerate(t1) if k != j]
        cand = [b for b in range(H) if b not in t0 and b not in rest and b not in bans]
        keys = [(tuple(sorted(t0)), tuple(sorted(rest + [b])), g[2], g[1]) for b in cand]
        vals = leaf.value_batch(keys)
        for b, (mu, _) in zip(cand, vals):
            D[gi, b] = mu
        if gi % 100 == 0:
            print(gi, flush=True)
    # centre per game: how much b hurts team 0 relative to the average replacement
    Dc = D - np.nanmean(D, 1, keepdims=True)
    cm = np.full((H, H), np.nan)
    for a in range(H):
        w, wo = has[:, a], ~has[:, a]
        if w.sum() >= 20:
            cm[a] = np.nanmean(Dc[w], 0) - np.nanmean(Dc[wo], 0)   # >0: b helps team 0 (with a) more
    # counter of b against a from a's side: cm < 0 means b hurts a's team more than average (b counters a)
    cnt = cm
    cnt[np.eye(H, dtype=bool)] = np.nan
    worst = np.nanmin(cnt, 1)
    res = {"games": n_games, "sd_pair_effect_pts": 100 * float(np.nanstd(cnt)),
           "worst_counter_per_hero_pts": {"median": 100 * float(np.nanmedian(worst)),
                                          "p10": 100 * float(np.nanpercentile(worst, 10)), "min": 100 * float(np.nanmin(worst))},
           "spread_of_replacement_pts": {"median_range": 100 * float(np.nanmedian(np.nanmax(D, 1) - np.nanmin(D, 1))),
                                         "median_sd": 100 * float(np.nanmedian(np.nanstd(D, 1)))}}
    rp = os.path.join(OUT, "real_counter_shrunk.npy")
    if os.path.exists(rp):
        rs = np.load(rp)
        ok = np.isfinite(rs) & np.isfinite(cnt)
        res["corr_with_real_shrunk"] = float(np.corrcoef(rs[ok], cnt[ok])[0, 1])
        res["pairs_compared"] = int(ok.sum())
    os.makedirs(OUT, exist_ok=True)
    np.save(os.path.join(OUT, "model_counter.npy"), cnt)
    json.dump(res, open(os.path.join(OUT, "model.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    {"real": real, "model": model}[sys.argv[1]]()
