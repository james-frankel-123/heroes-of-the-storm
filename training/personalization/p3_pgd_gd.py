"""
P3 personalized GD, stage 2: a causal, meta-aware generic-draft model.

The paper-1 GD was trained on a random 98% of the whole snapshot, so it is in
sample for V2. This model has the paper-1 GD architecture (MLP 256-128) on the
same draft-state input plus causal meta rates (28-day per-tier pick and ban
rates per hero), and is trained only on games before the WP training cutoff
(2026-02-10), all 16 steps (picks and bans). Early stopping on a 2% game
holdout of the training window.

Evaluated on V2 and post-snapshot games: picks top-1/3/5 and log loss; bans
top-1/3 and log loss; against the paper-1 GD.

Run (from training/): OMP_NUM_THREADS=4 nice -n 19 taskset -c 48-63 python3 personalization/p3_pgd_gd.py
Outputs: cache/pgd_metagd.pt, results/p3_pgd_gd.json
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

import p3_hs_core as C
import p3_pgd_common as P


def batches(g, games, rng, bs):
    gi = np.repeat(games, 16)
    k = np.tile(np.arange(16), len(games))
    o = rng.permutation(len(gi))
    for s in range(0, len(o), bs):
        b = o[s:s + bs]
        yield gi[b], k[b]


def evaluate_split(g, model, gd, games, rng, n=20000):
    games = rng.choice(games, min(n, len(games)), replace=False)
    res = {}
    for kind, steps in (("picks", P.PICK_STEPS), ("bans", P.BAN_STEPS)):
        gi = np.repeat(games, len(steps))
        k = np.tile(steps, len(games))
        X, M = P.states(g, gi, k)
        y = g["H"][gi, k]
        ok = M[np.arange(len(y)), y]
        X, M, y = X[ok], M[ok], y[ok]
        res[kind] = {"meta GD (causal)": P.scores(P.logp_metagd(model, X, M), y),
                     "paper-1 GD": P.scores(P.logp_paper_gd(gd, X, M), y)}
    return res


def main():
    t0 = time.time()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    rng = np.random.RandomState(0)
    g = P.load_games()
    tr_all = np.flatnonzero(g["window"] == 0)
    hold = rng.rand(len(tr_all)) < 0.02
    tr, va = tr_all[~hold], tr_all[hold]
    print(f"train games {len(tr):,}, holdout {len(va):,} ({time.time() - t0:.0f}s)", flush=True)
    m = P.MetaGD()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    Xv, Mv = P.states(g, np.repeat(va, 16), np.tile(np.arange(16), len(va)))
    yv = g["H"][np.repeat(va, 16), np.tile(np.arange(16), len(va))]
    okv = Mv[np.arange(len(yv)), yv]
    Xv, Mv, yv = torch.from_numpy(Xv[okv]), torch.from_numpy(Mv[okv]), torch.from_numpy(yv[okv])
    best, best_state, bad = 1e9, None, 0
    for ep in range(8):
        m.train()
        for i, (gi, k) in enumerate(batches(g, tr, rng, 4096)):
            X, M = P.states(g, gi, k)
            y = g["H"][gi, k]
            ok = M[np.arange(len(y)), y]
            lg = m(torch.from_numpy(X[ok]), torch.from_numpy(M[ok]))
            loss = torch.nn.functional.cross_entropy(lg, torch.from_numpy(y[ok]))
            opt.zero_grad()
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            lv = float(torch.nn.functional.cross_entropy(m(Xv, Mv), yv))
        print(f"epoch {ep}: holdout log loss {lv:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if lv < best - 1e-4:
            best, best_state, bad = lv, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 2:
                break
        for pg in opt.param_groups:
            pg["lr"] *= 0.6
    m.load_state_dict(best_state)
    m.eval()
    torch.save(m.state_dict(), os.path.join(C.CACHE, "pgd_metagd.pt"))
    import p3_dr_core as D
    d_names = np.load(C.SLOTS)["hero_names"]
    gd = D.GDPolicy(d_names)
    out = {"train_games": int(len(tr)), "holdout_log_loss": best}
    for nm, w in (("V2", 2), ("post", 3)):
        out[nm] = evaluate_split(g, m, gd, np.flatnonzero(g["window"] == w), np.random.RandomState(1))
        print(nm, json.dumps(out[nm]), flush=True)
    with open(os.path.join(C.RESULTS, "p3_pgd_gd.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
