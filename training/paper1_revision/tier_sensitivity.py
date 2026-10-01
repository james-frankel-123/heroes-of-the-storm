"""
Tier-label sensitivity check (2026-09-30 tier bug).

The replay daemon stored listing league_tier one above the tier names
(1 = Wood, 2 = Bronze, ..., 6 = Diamond; NULL with an avg_mmr = Master) and
mapped stored <=2 -> low, 3-4 -> mid, else high, NULL -> mid. So the pinned
snapshot's research tiers are really:
  low  = Wood + Bronze
  mid  = Silver + Gold + Master (+ the few rows with neither tier nor MMR)
  high = Platinum + Diamond
The per-replay league_tier and avg_mmr come from the pre-relabel backup
(backups/replay_draft_skill_tier_20260930.csv.gz), which also covers the
post-snapshot reference games.

Schemes retrained (leak-free enriched and naive WP, same protocol as
train_wp.py, seed 42 first):
  stated  the tiers the submission described: Bronze-Gold / Platinum-Diamond /
          Master (Wood -> low)
  site    the site's scheme: Bronze+Silver / Gold+Platinum / Diamond+Master
Rows with neither tier nor MMR stay in 'mid' (their count is reported).
Statistics (deploy + 5 out-of-fold) and features are rebuilt per scheme,
because every statistic is per tier.

Usage: python3 paper1_revision/tier_sensitivity.py crosstab|build|train|report
Output: results/tier_sensitivity.json
"""
import os
import sys
import json
import gzip
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)
from paper1_revision import core

import numpy as np

BACKUP = os.path.join(TRAINING_DIR, "..", "backups", "replay_draft_skill_tier_20260930.csv.gz")
OUT = os.path.join(core.RESULTS, "tier_sensitivity.json")
NPROC = int(os.environ.get("P1R_NPROC", "4"))
NAMES = {1: "Wood", 2: "Bronze", 3: "Silver", 4: "Gold", 5: "Platinum", 6: "Diamond"}


def real_tier(lt, mmr):
    if lt is None:
        return "Master" if mmr is not None else "unknown"
    return NAMES.get(lt, f"id{lt}")


SCHEMES = {
    "stated": {"Wood": "low", "Bronze": "low", "Silver": "low", "Gold": "low",
               "Platinum": "mid", "Diamond": "mid", "Master": "high", "unknown": "mid"},
    "site": {"Wood": "low", "Bronze": "low", "Silver": "low", "Gold": "mid",
             "Platinum": "mid", "Diamond": "high", "Master": "high", "unknown": "mid"},
}

_REAL = None


def real_tiers():
    """replay_id -> real tier name, from the pre-relabel backup."""
    global _REAL
    if _REAL is None:
        _REAL = {}
        with gzip.open(BACKUP, "rt") as f:
            next(f)
            for line in f:
                rid, st, lt, mmr = line.rstrip("\n").split(",")
                _REAL[int(rid)] = (st, int(lt) if lt else None, float(mmr) if mmr else None)
    return _REAL


def stats_dir(scheme):
    d = os.path.join(core.CACHE, f"stats_{scheme}")
    os.makedirs(d, exist_ok=True)
    return d


def feat_path(scheme, name):
    return os.path.join(core.FEAT_DIR, f"{scheme}_{name}.npz")


def relabel(games, scheme):
    R = real_tiers()
    out, missing = [], 0
    for g in games:
        r = R.get(g[0])
        if r is None:
            missing += 1
            out.append(g)
            continue
        t = SCHEMES[scheme][real_tier(r[1], r[2])]
        out.append((g[0], t) + tuple(g[2:]))
    return out, missing


def cmd_crosstab():
    R = real_tiers()
    sp = core.split_games()
    snap = sp["train"] + sp["val"] + sp["test"]
    ct, miss = Counter(), 0
    for g in snap:
        r = R.get(g[0])
        if r is None:
            miss += 1
            continue
        ct[(g[1], real_tier(r[1], r[2]))] += 1
    post = core.gold_games("NODRIFT")
    ctp, missp = Counter(), 0
    for g in post:
        r = R.get(g[0])
        if r is None:
            missp += 1
            continue
        ctp[(g[1], real_tier(r[1], r[2]))] += 1
    res = {"snapshot": {f"{a}|{b}": n for (a, b), n in sorted(ct.items())}, "snapshot_missing": miss,
           "NODRIFT": {f"{a}|{b}": n for (a, b), n in sorted(ctp.items())}, "NODRIFT_missing": missp,
           "n_snapshot": len(snap), "n_NODRIFT": len(post)}
    for k in ("snapshot", "NODRIFT"):
        print(k, res[k], res[f"{k}_missing"])
    prev = json.load(open(OUT)) if os.path.exists(OUT) else {}
    prev["crosstab"] = res
    json.dump(prev, open(OUT, "w"), indent=1)


def cmd_build(scheme):
    from overfit2026.split import counts_for
    sp = core.split_games()
    tr, _ = relabel(sp["train"], scheme)
    folds = np.array([core.fold_of(g[0]) for g in tr])
    sd = stats_dir(scheme)
    jobs = {"deploy": tr}
    for k in range(core.N_FOLDS):
        jobs[f"oof{k}"] = [g for g, f in zip(tr, folds) if f != k]
    stats = {}
    for name, games in jobs.items():
        p, cp = os.path.join(sd, f"{name}.json"), os.path.join(sd, f"{name}_compositions.json")
        if not os.path.exists(p):
            js = core.stats_json(counts_for(games), {"subset": name, "scheme": scheme})
            comps = js.pop("compositions")
            json.dump(js, open(p, "w"))
            json.dump(comps, open(cp, "w"))
            print(f"stats {scheme}/{name}", flush=True)
        from overfit2026 import feats
        st = feats.stats_from_json(p, compositions=False)
        st.comp_data = core.comps_from_json(cp)
        stats[name] = st
    sets = {"train_oof": tr, "val": relabel(sp["val"], scheme)[0], "test": relabel(sp["test"], scheme)[0],
            "NODRIFT": relabel(core.gold_games("NODRIFT"), scheme)[0]}
    for name, games in sets.items():
        path = feat_path(scheme, name)
        if os.path.exists(path):
            continue
        t0 = time.time()
        if name == "train_oof":
            Xf = np.zeros((len(games), 357), np.float32)
            Xs = np.zeros_like(Xf)
            for k in range(core.N_FOLDS):
                idx = np.where(folds == k)[0]
                a, b = core.featurize(core.rows_of([games[i] for i in idx]), stats[f"oof{k}"], nproc=NPROC)
                Xf[idx], Xs[idx] = a, b
        else:
            Xf, Xs = core.featurize(core.rows_of(games), stats["deploy"], nproc=NPROC)
        np.savez(path, Xf=Xf, Xs=Xs, y=core.labels(games), rid=np.array([g[0] for g in games]),
                 tier=np.array([g[1] for g in games]))
        print(f"features {scheme}/{name}: {len(games):,} ({time.time() - t0:.0f}s)", flush=True)


def _load(scheme, name):
    z = np.load(feat_path(scheme, name) if scheme != "snapshot" else core.feat_path(name))
    return z["Xf"], z["Xs"], z["y"], z["tier"]


def _metrics(model, cols, scheme, name, device):
    import torch
    from paper1_revision.train_wp import fit_slope
    Xf, Xs, y, tier = _load(scheme, name)
    ps = []
    with torch.no_grad():
        for X in (Xf, Xs):
            out = []
            for i in range(0, len(X), 65536):
                out.append(model(torch.tensor(X[i:i + 65536][:, cols], device=device)).float().view(-1).cpu().numpy())
            ps.append(np.concatenate(out))
    pf, pw = ps
    acc = float(0.5 * (np.mean((pf > .5) == (y > .5)) + np.mean((pw > .5) == (y < .5))))
    p = np.clip(0.5 * (pf + 1 - pw), 1e-6, 1 - 1e-6)
    res = {"acc": acc, "acc_sym": float(np.mean((p > .5) == (y > .5))),
           "ll": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), "slope": fit_slope(p, y),
           "per_tier": {}}
    for t in ("low", "mid", "high"):
        m = tier == t
        if m.sum():
            res["per_tier"][t] = {"n": int(m.sum()), "acc_sym": float(np.mean((p[m] > .5) == (y[m] > .5)))}
    return res


def cmd_train(scheme, seeds=(42,)):
    import torch
    from paper1_revision import train_wp
    dev = torch.device("cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
    torch.set_num_threads(NPROC)
    prev = json.load(open(OUT)) if os.path.exists(OUT) else {}
    E = train_wp.enriched_groups()
    for mname, groups in (("enriched", E), ("naive", [])):
        cols = core.group_cols(groups)
        for seed in seeds:
            key = f"{scheme}|{mname}|s{seed}"
            if key in prev.get("runs", {}):
                continue
            t0 = time.time()
            if scheme == "snapshot":
                model, _ = train_wp.load(mname, seed=seed, device=dev)
                vl = None
            else:
                Xf, Xs, y, _ = _load(scheme, "train_oof")
                tX = torch.tensor(np.concatenate([Xf[:, cols], Xs[:, cols]]), device=dev)
                tY = torch.tensor(np.concatenate([y, 1 - y]), device=dev)
                del Xf, Xs
                Xf, Xs, y, _ = _load(scheme, "val")
                vX = torch.tensor(np.concatenate([Xf[:, cols], Xs[:, cols]]), device=dev)
                vY = torch.tensor(np.concatenate([y, 1 - y]), device=dev)
                model, vl, hist = train_wp.train_one(key, {"arch": [256, 128]}, seed, dev, (tX, tY, vX, vY))
                torch.save(model.state_dict(), os.path.join(core.MODEL_DIR, f"tier_{scheme}_{mname}_s{seed}.pt"))
                del tX, tY
            r = {"val_loss": vl, "test": _metrics(model, cols, scheme, "test", dev),
                 "NODRIFT": _metrics(model, cols, scheme, "NODRIFT", dev), "secs": time.time() - t0}
            prev = json.load(open(OUT)) if os.path.exists(OUT) else {}
            prev.setdefault("runs", {})[key] = r
            json.dump(prev, open(OUT, "w"), indent=1)
            print(key, {k: round(r[k]["acc"], 4) for k in ("test", "NODRIFT")},
                  {k: round(r[k]["slope"], 3) for k in ("test", "NODRIFT")}, f"{r['secs']:.0f}s", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "crosstab":
        cmd_crosstab()
    elif cmd == "build":
        cmd_build(sys.argv[2])
    elif cmd == "train":
        cmd_train(sys.argv[2], tuple(int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "42").split(",")))
