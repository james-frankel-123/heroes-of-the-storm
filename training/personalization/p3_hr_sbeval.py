"""
B1 of P3_HERO_RECS_AND_REPLAYS: how much the scoreboard matters, measured
prospectively with a frozen state.

Design. State frozen at t = 2026-02-10 (end of E) from every history row
before t. Targets: every corpus slot game after t (V1, V2, OOT windows,
2026-02-10 to 2026-09-27) of every player. Each slot's baseline is the
draft WP plus the experience offset at the slot's lag-1 counts (counts
only, no results). A skill score s for the player's hero (as of t) enters
as p = WP + offset + beta * s. beta (and any learned model) is fit on the
fit-half players' slots; everything is scored on the held-out half's slots
(player split, the same RandomState(7) halves as the model fits).
Gain = slot log-loss improvement over WP + offset, x1000, SE by player.

Arms:
  linear scores: main (full_ms), main without scoreboard evidence, imp2,
    today's win-only kernel, and every per-metric ablation refit
    (results/hr/ablate/abl_*.npz, from p3_hr_ablate.py on the 5090);
  learned on the metric profile (torch MLP / ridge, fit on fit-half slots):
    features = cell and player metric means, counts, win records, play
    share, hero embedding; with and without the main model's estimate.

Leak check: the state is built from rows with day < t only; the script
asserts that rewriting every row on or after t (results, residuals,
metrics) leaves the features unchanged.
"""
import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import p3_hr_common as HC  # noqa: E402
import p3_hs_core as C  # noqa: E402

H = 90
T0 = "2026-02-10"
ABL = os.path.join(HC.HR_RES, "ablate")


def setup():
    import p3_rcf as R
    import p3_c_merged as M
    from p3_heroes import HKEY
    a = HC.load()
    t = C.day_of(T0)
    post = a["s_win"] > 0
    sp = a["s_pid"].astype(np.int64)
    players = np.unique(sp[post])
    st = HC.state_at(a, t, players)
    # leak check: rows on or after t never enter the state
    b = dict(a)
    late = a["day"] >= t
    rng = np.random.RandomState(9)
    for k in ("r_adj", "r"):
        b[k] = a[k].copy()
        b[k][late] = rng.randn(late.sum()).astype(np.float32)
    b["Z"] = a["Z"].copy()
    b["Z"][late] = np.float16(3.0)
    sub = players[:: max(1, len(players) // 3000)]
    s1, s2 = HC.state_at(a, t, sub), HC.state_at(b, t, sub)
    assert all(np.array_equal(s1[k], s2[k]) for k in s1), "post-t rows leak into the state"
    del b
    np_s = M.lag1_key_sums(a["s_day"], sp).astype(np.int64)
    nph_s = M.lag1_key_sums(a["s_day"], sp * HKEY + a["s_hero"].astype(np.int64)).astype(np.int64)
    q = np.flatnonzero(post)
    off = a["table"][C.exp_bins(np_s[q], nph_s[q])]
    pos = np.full(int(a["n_players"]), -1, np.int64)
    pos[players] = np.arange(len(players))
    fit = R.fit_half(int(a["n_players"]))
    S = {"q": q, "pi": pos[sp[q]], "h": a["s_hero"][q].astype(np.int64), "wp": a["s_wp"][q].astype(np.float64),
         "y": a["s_y"][q].astype(np.float64), "off": off, "win": a["s_win"][q], "fit": fit[sp[q]], "pid": sp[q]}
    print(f"[setup] players {len(players):,}, post slots {len(q):,} (fit half {S['fit'].mean():.3f}); leak check ok",
          flush=True)
    return a, st, players, S


def ll(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def score_arm(S, s, name, boot):
    """beta fit on fit-half slots by 1-d Newton on log loss; gain on held-out."""
    base = S["wp"] + S["off"]
    f = S["fit"]
    b = 1.0
    for _ in range(30):
        p = np.clip(base[f] + b * s[f], 1e-4, 1 - 1e-4)
        g = ((p - S["y"][f]) / (p * (1 - p)) * s[f]).sum()
        hh = ((1 / (p * (1 - p))) * s[f] ** 2).sum()
        step = g / hh
        b -= step
        if abs(step) < 1e-6:
            break
    h = ~f
    d = ll(base[h], S["y"][h]) - ll(base[h] + b * s[h], S["y"][h])
    out = {"beta": float(b)}
    pid = S["pid"][h]
    up, inv = np.unique(pid, return_inverse=True)
    for wn, m in (("all", np.ones(h.sum(), bool)), ("V1", S["win"][h] == 1), ("V2", S["win"][h] == 2), ("OOT", S["win"][h] == 3)):
        per = np.bincount(inv[m], d[m], len(up))
        cnt = np.bincount(inv[m], None, len(up))
        est = 1000 * per.sum() / cnt.sum()
        bs = [1000 * (per * w).sum() / max((cnt * w).sum(), 1) for w in boot[:, : len(up)]]
        out[wn] = {"gain": float(est), "se": float(np.std(bs, ddof=1)), "slots": int(m.sum())}
    print(f"  {name:32s} beta {b:.3f}  held-out gain x1000 {out['all']['gain']:+.3f} ({out['all']['se']:.3f})  "
          f"V2 {out['V2']['gain']:+.3f} OOT {out['OOT']['gain']:+.3f}", flush=True)
    return out, d


def lookup(m, S):
    return m[S["pi"], S["h"]].astype(np.float64)


def features(st, main_m):
    X, Cm = st["X"].astype(np.float32), st["Cm"].astype(np.float32)
    n = X.shape[0]
    cell = X / (Cm + 2.0)
    prof = X.sum(1) / (Cm.sum(1) + 10.0)
    N = st["N"].astype(np.float32)
    tot = N.sum(1, keepdims=True)
    S, P = st["S"], st["P"]
    cell_rec = (S / (P + 150.0)).astype(np.float32)
    pl_rec = (S.sum(1) / (P.sum(1) + 150.0)).astype(np.float32)
    return {"cell": cell, "prof": prof, "logN": np.log1p(N), "share": N / np.maximum(tot, 1), "logtot": np.log1p(tot[:, 0]),
            "cell_rec": cell_rec, "pl_rec": pl_rec, "logC": np.log1p(Cm.sum(2)), "main": main_m.astype(np.float32)}


def slot_matrix(F, S, idx, with_main, with_metrics=True):
    pi, h = S["pi"][idx], S["h"][idx]
    cols = [F["logN"][pi, h][:, None], F["share"][pi, h][:, None], F["logtot"][pi][:, None],
            F["cell_rec"][pi, h][:, None] * 10, F["pl_rec"][pi][:, None] * 10]
    if with_metrics:
        cols += [F["cell"][pi, h], F["prof"][pi], F["logC"][pi, h][:, None]]
    if with_main:
        cols.append(F["main"][pi, h][:, None] * 10)
    return np.nan_to_num(np.concatenate(cols, 1).astype(np.float32)), h


def train_mlp(Xtr, htr, ytr, Xva, hva, yva, linear=False, seed=0, hidden=64, epochs=12):
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    D = Xtr.shape[1]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(H, 8)
            if linear:
                self.f = torch.nn.Linear(D + 8, 1)
            else:
                self.f = torch.nn.Sequential(torch.nn.Linear(D + 8, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden),
                                             torch.nn.ReLU(), torch.nn.Linear(hidden, 1))
            self.hb = torch.nn.Embedding(H, 1)
            torch.nn.init.zeros_(self.hb.weight)

        def forward(self, x, h):
            return self.f(torch.cat([x, self.emb(h)], 1))[:, 0] * 0.05 + self.hb(h)[:, 0]
    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-5)
    T = lambda x: torch.from_numpy(np.ascontiguousarray(x))
    xt, ht, yt = T((Xtr - mu) / sd), T(htr), T(ytr.astype(np.float32))
    xv, hv, yv = T((Xva - mu) / sd), T(hva), T(yva.astype(np.float32))
    best, best_state, bad = None, None, 0
    for ep in range(epochs):
        perm = torch.randperm(len(yt))
        net.train()
        for i in range(0, len(perm), 4096):
            j = perm[i:i + 4096]
            loss = ((net(xt[j], ht[j]) - yt[j]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(((net(xv, hv) - yv) ** 2).mean())
        if best is None or vl < best - 1e-7:
            best, best_state, bad = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 2:
                break
    net.load_state_dict(best_state)

    def pred(X, h):
        with torch.no_grad():
            return net(T((X - mu) / sd), T(h)).numpy().astype(np.float64)
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-mlp", action="store_true")
    a_ = ap.parse_args()
    t0 = time.time()
    a, st, players, S = setup()
    up = np.unique(S["pid"][~S["fit"]])
    boot = np.random.RandomState(2).poisson(1.0, (200, len(up))).astype(np.float64)
    res = {"t": T0, "slots_heldout": int((~S["fit"]).sum()), "players_heldout": int(len(up)), "arms": {}}
    base_gain = ll(S["wp"][~S["fit"]], S["y"][~S["fit"]]) - ll(S["wp"][~S["fit"]] + S["off"][~S["fit"]], S["y"][~S["fit"]])
    res["offset_vs_wp_x1000"] = float(1000 * base_gain.mean())
    mains = {}
    D = {}
    arms = [("main", "full_ms", {}), ("main_no_scoreboard", "full_ms", {"use_metrics": False}), ("imp2", "full_imp2", {}),
            ("impmet", "impmet", {}), ("today_win_only", "today", {})]
    for nm, tag, kw in arms:
        m, _ = HC.post_all(HC.params(tag), st, **kw)
        mains[nm] = m
        res["arms"][nm], D[nm] = score_arm(S, lookup(m, S), nm, boot)
    import p3_rcf as R
    for p in sorted(glob.glob(os.path.join(ABL, "abl_*.npz"))):
        if p.endswith("_heldout.npz") or not os.path.exists(p[:-4] + ".json"):
            continue
        nm = os.path.basename(p)[:-4]
        par = R.Params.load(p)
        m, _ = HC.post_all(par, st)
        r_, D[nm] = score_arm(S, lookup(m, S), nm, boot)
        r_["heldout_ll_fit"] = par.meta.get("heldout_ll")
        res["arms"][nm] = r_
    HC.save_json("sbeval.json", res)
    if not a_.skip_mlp:
        F = features(st, mains["main"])
        target = S["y"] - S["wp"] - S["off"]
        fi = np.flatnonzero(S["fit"])
        rng = np.random.RandomState(1)
        pf = np.unique(S["pid"][fi])
        va_p = set(rng.choice(pf, len(pf) // 10, replace=False).tolist())
        va = np.array([p in va_p for p in S["pid"][fi]])
        tr_i, va_i = fi[~va], fi[va]
        ho = np.flatnonzero(~S["fit"])
        for nm, wm, lin, wmet in (("ridge_profile", False, True, True), ("mlp_profile", False, False, True),
                                  ("mlp_no_metrics", False, False, False),
                                  ("mlp_profile_plus_main", True, False, True), ("ridge_profile_plus_main", True, True, True),
                                  ("ridge_no_metrics_plus_main", True, True, False), ("mlp_no_metrics_plus_main", True, False, False)):
            Xtr, htr = slot_matrix(F, S, tr_i, wm, wmet)
            Xva, hva = slot_matrix(F, S, va_i, wm, wmet)
            pred = train_mlp(Xtr, htr, target[tr_i], Xva, hva, target[va_i], linear=lin)
            s = np.zeros(len(S["y"]))
            for idx in (fi, ho):
                Xx, hx = slot_matrix(F, S, idx, wm, wmet)
                s[idx] = pred(Xx, hx)
            res["arms"][nm], D[nm] = score_arm(S, s, nm, boot)
    pid = S["pid"][~S["fit"]]
    upp, inv = np.unique(pid, return_inverse=True)
    cnt = np.bincount(inv, None, len(upp))
    pairs = [("main", "main_no_scoreboard"), ("main", "imp2"), ("impmet", "main"), ("impmet", "imp2"), ("main", "today_win_only"),
             ("mlp_profile", "mlp_no_metrics"), ("mlp_profile", "ridge_profile"),
             ("mlp_profile_plus_main", "ridge_profile_plus_main"), ("ridge_profile_plus_main", "main"),
             ("ridge_profile_plus_main", "ridge_no_metrics_plus_main"), ("mlp_profile_plus_main", "mlp_no_metrics_plus_main"),
             ("ridge_no_metrics_plus_main", "main"),
             ("mlp_profile_plus_main", "main")]
    pairs += [(k, "abl_all") for k in D if k.startswith("abl_") and k != "abl_all"]
    res["pairs"] = {}
    for x, z in pairs:
        if x not in D or z not in D:
            continue
        per = np.bincount(inv, D[x] - D[z], len(upp))
        est = 1000 * per.sum() / cnt.sum()
        bs = [1000 * (per * w).sum() / (cnt * w).sum() for w in boot[:, : len(upp)]]
        res["pairs"][f"{x} - {z}"] = {"diff": float(est), "se": float(np.std(bs, ddof=1))}
        print(f"  pair {x} - {z}: {est:+.3f} ({np.std(bs, ddof=1):.3f})", flush=True)
    HC.save_json("sbeval.json", res)
    print(f"[sbeval] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
