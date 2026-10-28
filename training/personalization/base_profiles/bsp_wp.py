"""
BSP: paper 1's base draft WP on every evaluation game (5090, CPU only).

Model: paper 1's leak-free site WP "enriched" (283 inputs: heroes, map, tier
plus the enriched statistic groups; trained on out-of-fold statistics), all
three seeds, swap-symmetrized p = (p_fwd + 1 - p_swap) / 2 exactly as
paper1_revision/train_wp.evaluate. The selected seed (42) is the base; the
seed mean is kept for a check.

Evaluation games:
  p1val, p1test   paper 1's own validation and test games (cached deploy-stat
                  features, site/cache/feats/{val,test}.npz)
  post            every Storm League game dated 2026-05-23 to 2026-09-27
                  (after the snapshot's last day), featurized here with the
                  deploy statistics (frozen at the snapshot, as deployed)

Env: P1_TIERS=site, PYTHONPATH=<repo>/training. Output: $BSP/data/wp_base.npz
"""
import datetime
import os
import sys
import time

os.environ.setdefault("P1_TIERS", "site")
import numpy as np

BSP = os.environ.get("BSP", os.path.expanduser("~/hots/base_profiles"))
POST0 = (datetime.date(2026, 5, 23) - datetime.date(1970, 1, 1)).days
SEEDS = (42, 123, 777)
TIERS = ["low", "mid", "high"]


def predict(models, Xf, Xs):
    import torch
    out = []
    with torch.no_grad():
        for m, cols in models:
            pf = m(torch.tensor(Xf[:, cols])).view(-1).numpy()
            ps = m(torch.tensor(Xs[:, cols])).view(-1).numpy()
            out.append(np.clip(0.5 * (pf + 1 - ps), 1e-6, 1 - 1e-6))
    return np.stack(out)


def main():
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    from paper1_revision import core, train_wp
    models = [train_wp.load("enriched", seed=s) for s in SEEDS]
    res = {}
    for name in ("val", "test"):
        Xf, Xs, y, rid, tier = core.load_features(name)
        P = predict(models, Xf, Xs)
        res["p1" + name] = (rid.astype(np.int64), y, P)
        ll = -np.mean(y * np.log(P[0]) + (1 - y) * np.log(1 - P[0]))
        print(f"p1{name}: n {len(y):,} ll {ll:.4f} acc {np.mean((P[0] > .5) == (y > .5)):.4f}", flush=True)
        del Xf, Xs

    z = np.load(os.path.join(BSP, "data", "games.npz"))
    g = {k: z[k] for k in ("replay_id", "end_ts", "map", "tier", "t0", "t1", "winner", "heroes", "maps")}
    H, M = [str(h) for h in g["heroes"]], [str(m) for m in g["maps"]]
    day = g["end_ts"] // 86400
    ok = ((day >= POST0) & (g["map"] >= 0) & (g["tier"] >= 0) & (g["t0"] >= 0).all(1)
          & (g["t1"] >= 0).all(1) & (g["winner"] >= 0))
    idx = np.where(ok)[0]
    print(f"post games: {len(idx):,}", flush=True)
    st = core.load_stats("deploy")
    t0s, t1s, ms, ts = g["t0"][idx], g["t1"][idx], g["map"][idx], g["tier"][idx]
    rows = [([H[k] for k in t0s[i]], [H[k] for k in t1s[i]], M[ms[i]], TIERS[ts[i]]) for i in range(len(idx))]
    Ps = []
    t0 = time.time()
    CH = 20000
    for a in range(0, len(rows), CH):
        Xf, Xs = core.featurize(rows[a:a + CH], st, nproc=int(os.environ.get("BSP_NPROC", "4")))
        Ps.append(predict(models, Xf, Xs))
        print(f"  {a + len(Xf):,} ({time.time() - t0:.0f}s)", flush=True)
    P = np.concatenate(Ps, 1)
    y = (g["winner"][idx] == 0).astype(np.float32)
    res["post"] = (g["replay_id"][idx].astype(np.int64), y, P)
    ll = -np.mean(y * np.log(P[0]) + (1 - y) * np.log(1 - P[0]))
    print(f"post: ll {ll:.4f} acc {np.mean((P[0] > .5) == (y > .5)):.4f}", flush=True)
    out = {}
    for k, (r, yy, PP) in res.items():
        out[f"{k}_rid"], out[f"{k}_y"], out[f"{k}_p"] = r, yy, PP.astype(np.float32)
    np.savez(os.path.join(BSP, "data", "wp_base.npz"), **out)


if __name__ == "__main__":
    main()
