"""
Out-of-fold gN / gN_naive predictions on the no-drift post-snapshot games (N),
so the v2 structural correction for those judges can be fitted on N itself
(no balance drift) without scoring any game a judge trained on.

Two hash halves of N (data.half(rid, salt=5), the same halves as the RN
cross-fit). For half k: statistics and model training use only half k with
the gN recipe unchanged (split.py: 2-fold out-of-fold statistics for training
rows, deploy statistics over all of half k, 2% early-stopping subset,
256x128 MLP, dropout 0.3, AdamW 5e-4, wd 5e-3, patience 25; enriched seeds
200-202 and a hero-identity model, seed 200), except that the composition
features use the half's own role-composition table (out of fold for training
rows), as the rebuilt gN does (comp_gn_rebuild.py). The half-k models then score the
other half with half-k deploy statistics. The deployed gN judges are NOT
replaced; these fold models exist only to estimate how the judge family errs
on real teams it has not seen.

Usage (from training/): CUDA_VISIBLE_DEVICES=<gpu> nice -n 19 taskset -c 48-53 \
    python3 overfit2026/comp_gn_folds.py
Output: overfit2026/cache/comp_gn_oof_N.npz (rid, gN, gN_naive, y)
"""
import os
import sys
import json
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)

import numpy as np

from overfit2026 import data, feats, split

DIR = os.path.join(HERE, "cache", "comp_folds_owncomp")
OUT = os.path.join(HERE, "cache", "comp_gn_oof_N.npz")
SPEC = dict(arch=[256, 128], dropout=0.3, wd=5e-3, epochs=200)


def stats(games, name):
    """Statistics of `games` with their OWN role-composition table (cells
    admitted at >= 50 games, paper1_revision.core convention), written beside
    the stats JSON so feats.stats_from_json picks it up. No external table."""
    from paper1_revision import core
    p = os.path.join(DIR, f"{name}.json")
    if not os.path.exists(p):
        js = core.stats_json(split.counts_for(games), {"subset": name, "games_used": len(games)})
        comps = js.pop("compositions")
        json.dump(js, open(p, "w"))
        json.dump(comps, open(p[:-5] + "_compositions.json", "w"))
    return feats.stats_from_json(p)


def rows(games):
    return [(g[3], g[4], g[2], g[1]) for g in games]


def train(X, Y, V, dim, seed, device):
    import torch
    import torch.nn as nn
    from sweep_enriched_wp import WinProbEnrichedModel
    tX, tY = torch.tensor(X[~V], device=device), torch.tensor(Y[~V], device=device)
    vX, vY = torch.tensor(X[V], device=device), torch.tensor(Y[V], device=device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = WinProbEnrichedModel(dim, SPEC["arch"], dropout=SPEC["dropout"]).to(device)
    crit = nn.BCELoss()
    opt = torch.optim.AdamW(m.parameters(), lr=5e-4, weight_decay=SPEC["wd"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=1e-5)
    best, state, pat, n = float("inf"), None, 0, len(tX)
    for ep in range(SPEC["epochs"]):
        m.train()
        pm = torch.randperm(n, device=device)
        for i in range(0, n, 4096):
            idx = pm[i:i + 4096]
            loss = crit(m(tX[idx]).view(-1), tY[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sch.step()
        m.eval()
        with torch.no_grad():
            vl = crit(m(vX).view(-1), vY).item()
        if vl < best:
            best, pat = vl, 0
            state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        else:
            pat += 1
            if pat >= 25:
                break
    m.load_state_dict(state)
    m.eval()
    return m.cpu(), ep + 1, best


def main():
    import torch
    from drift2026 import common as dcommon
    dcommon._bind_statscache_methods()
    torch.set_num_threads(4)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(DIR, exist_ok=True)
    games = split.games_for("N")
    h = np.array([data.half(g[0], salt=5) for g in games])
    pred = {"gN": np.zeros(len(games)), "gN_naive": np.zeros(len(games))}
    for k in (0, 1):
        t0 = time.time()
        G = [g for g, hh in zip(games, h) if hh == k]
        H = [g for g, hh in zip(games, h) if hh != k]
        fold = np.array([split.fold_of(g[0]) for g in G])
        Xf = np.zeros((len(G), 283), np.float32)
        Xs = np.zeros_like(Xf)
        for f in (0, 1):
            st = stats([g for g, ff in zip(G, fold) if ff == 1 - f], f"N_h{k}f{1 - f}")
            idx = np.where(fold == f)[0]
            a, b = feats.featurize(rows([G[i] for i in idx]), st, nproc=6)
            Xf[idx], Xs[idx] = a, b
        y = np.array([1.0 if g[6] == 0 else 0.0 for g in G], np.float32)
        val = np.array([split.is_val(g[0]) for g in G])
        X = np.concatenate([Xf, Xs])
        Y = np.concatenate([y, 1 - y])
        V = np.concatenate([val, val])
        hf, hs = feats.featurize(rows(H), stats(G, f"N_h{k}"), nproc=6)
        print(f"half {k}: train {len(G):,} score {len(H):,} feats {time.time() - t0:.0f}s", flush=True)
        enr = []
        for seed in (200, 201, 202):
            m, ep, bl = train(X, Y, V, 283, seed, dev)
            enr.append(feats.predict_sym(m, hf, hs, "cpu"))
            print(f"  enriched s{seed}: {ep} epochs, val loss {bl:.5f} ({time.time() - t0:.0f}s)", flush=True)
        m, ep, bl = train(X[:, :197], Y, V, 197, 200, dev)
        sel = h != k
        pred["gN"][sel] = np.mean(enr, 0)
        pred["gN_naive"][sel] = feats.predict_sym(m, hf[:, :197], hs[:, :197], "cpu")
        print(f"  naive: {ep} epochs, val loss {bl:.5f} ({time.time() - t0:.0f}s)", flush=True)
    np.savez(OUT, rid=np.array([g[0] for g in games]), y=np.array([1.0 if g[6] == 0 else 0.0 for g in games]),
             **pred)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
