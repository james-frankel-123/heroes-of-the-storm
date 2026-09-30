"""
Stage C: does the proxy mispredict real outcomes exactly where the search
concentrates? (the direct optimizer's-curse test on realized results)

1. Score every unseen real game (gold sets BF, T97, T17; none in any paper-1
   model's training data) with the paper proxy (wp_enriched_256 + frozen
   stats, symmetrized) and the 20-member ensemble (models/ens_unc).
2. Calibration on those games: overall, by proxy decile.
3. Search-likeness weighting: for each saved agent configuration c, build the
   frequency table f_c of k-hero subsets (k = 1..5) of its own teams. Each
   real team T gets weight w_c(T) = mean over k-subsets s of T of f_c(s).
   The config's "selected-set calibration gap" is the weighted mean of
   (proxy prediction for T) - (realized result of T). Compared with the
   unweighted gap it measures how much more the proxy overrates the kinds of
   teams config c picks. That excess is the Goodhart gap.

Output: results/stage_c.json, cache/gold_games_proxy.npz
"""
import os
import sys
import json
import glob
import itertools

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)

import numpy as np
import torch

from overfit2026 import data, feats

RR = os.path.join(TRAINING_DIR, "rerun2026")
OUT = os.path.join(HERE, "results", "stage_c.json")
CACHE = os.path.join(HERE, "cache", "gold_games_proxy.npz")
SETS = ["BF", "T97", "T17"]


def ens_members():
    from experiment_synthetic_augmentation import ENRICHED_GROUPS
    from sweep_enriched_wp import FEATURE_GROUP_DIMS
    sys.argv = sys.argv[:1]
    span, off = {}, 197
    for g in ENRICHED_GROUPS:
        span[g] = list(range(off, off + FEATURE_GROUP_DIMS[g]))
        off += FEATURE_GROUP_DIMS[g]
    from rerun2026.ensemble_uncertainty import ROSTER, preset_groups, model_path
    out = []
    for name, preset, arch_s, half, seed in ROSTER:
        cols = list(range(197)) + [c for g in preset_groups(preset) for c in span[g]]
        arch = [int(x) for x in arch_s.split(",")]
        m = feats.load_wp(model_path(name), dim=len(cols), arch=arch)
        out.append((name, np.array(cols), m))
    return out


def build_cache():
    fut, bf = data.load_future(), data.load_backfill()
    sets = {"BF": [g for v in bf.values() for g in v],
            "T97": fut["2.55.16.97039"],
            "T17": [g for b, v in fut.items() if b.startswith("2.55.17.") for g in v]}
    st = feats.paper_stats()
    proxy = feats.load_wp(os.path.join(RR, "models", "wp_enriched_256.pt"))
    members = ens_members()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {}
    for s in SETS:
        games = sets[s]
        rows = [(g[3], g[4], g[2], g[1]) for g in games]
        Xf, Xs = feats.featurize(rows, st)
        out[f"{s}_proxy"] = feats.predict_sym(proxy, Xf, Xs, dev)
        E = np.stack([feats.predict_sym(m, Xf, Xs, dev, cols=c) for _, c, m in members])
        out[f"{s}_ens_mean"] = E.mean(0)
        out[f"{s}_ens_std"] = E.std(0)
        out[f"{s}_y"] = np.array([1.0 if g[6] == 0 else 0.0 for g in games])
        out[f"{s}_rid"] = np.array([g[0] for g in games])
        print(s, len(games), "proxy mean", out[f"{s}_proxy"].mean(), "y", out[f"{s}_y"].mean(),
              flush=True)
    np.savez_compressed(CACHE, **out)


def team_keys(team, k):
    return [tuple(sorted(c)) for c in itertools.combinations(team, k)]


def config_tables():
    """Per config: subset frequency tables from the saved diversity dumps."""
    groups = {}
    for p in glob.glob(os.path.join(RR, "results", "diversity", "*__*.json")):
        d = json.load(open(p))
        if "drafts" not in d:
            continue
        tag = d["variant"]["tag"]
        key = d["experiment"] if (tag == "base" and d["seed"] is not None) else \
            os.path.basename(p)[:-5]
        groups.setdefault(key, []).extend(x["our"] for x in d["drafts"])
    tabs = {}
    for key, teams in groups.items():
        t = {}
        for k in range(1, 6):
            c = {}
            for team in teams:
                for s in team_keys(team, k):
                    c[s] = c.get(s, 0) + 1
            n = len(teams)
            t[k] = {s: v / n for s, v in c.items()}
        tabs[key] = (t, len(teams))
    return tabs


def weighted_gap(pred, y, w):
    m = w > 0
    if m.sum() == 0:
        return None
    w, d = w[m], (pred - y)[m]
    W = w.sum()
    g = float((w * d).sum() / W)
    neff = float(W ** 2 / (w ** 2).sum())
    # sandwich SE of a weighted mean
    se = float(np.sqrt((w ** 2 * (d - g) ** 2).sum()) / W)
    return {"gap": g, "se": se, "neff": neff, "pred": float((w * pred[m]).sum() / W),
            "real": float((w * y[m]).sum() / W), "n_nonzero": int(m.sum())}


def analyze():
    z = np.load(CACHE)
    fut, bf = data.load_future(), data.load_backfill()
    games = {"BF": [g for v in bf.values() for g in v],
             "T97": fut["2.55.16.97039"],
             "T17": [g for b, v in fut.items() if b.startswith("2.55.17.") for g in v]}
    res = {"calibration": {}, "selected_gap": {}}
    for s in SETS:
        p, y, sd = z[f"{s}_proxy"], z[f"{s}_y"], z[f"{s}_ens_std"]
        ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        dec = []
        q = np.quantile(p, np.linspace(0, 1, 11))
        for i in range(10):
            m = (p >= q[i]) & (p <= q[i + 1])
            dec.append({"pred": float(p[m].mean()), "real": float(y[m].mean()), "n": int(m.sum())})
        # team-level view: both teams of every game
        res["calibration"][s] = {"n": len(p), "mean_pred": float(p.mean()),
                                 "mean_y": float(y.mean()), "acc": float(np.mean((p > .5) == (y > .5))),
                                 "logloss": ll, "deciles": dec,
                                 "ens_std_mean": float(sd.mean())}
    tabs = config_tables()
    for s in SETS:
        g = games[s]
        p, y = z[f"{s}_proxy"], z[f"{s}_y"]
        # team-level arrays: team0 then team1
        teams = [x[3] for x in g] + [x[4] for x in g]
        tp = np.concatenate([p, 1 - p])
        ty = np.concatenate([y, 1 - y])
        tstd = np.concatenate([z[f"{s}_ens_std"]] * 2)
        base = {"gap": float((tp - ty).mean())}
        res["selected_gap"].setdefault(s, {"_uniform": base})
        keys_by_k = {k: [team_keys(t, k) for t in teams] for k in range(1, 6)}
        for cfg, (tab, n) in sorted(tabs.items()):
            row = {"n_teams": n}
            for k in range(1, 6):
                f = tab[k]
                w = np.array([np.mean([f.get(sk, 0.0) for sk in ks]) for ks in keys_by_k[k]])
                wg = weighted_gap(tp, ty, w)
                if wg is not None:
                    wg["excess"] = wg["gap"] - base["gap"]
                    wg["ens_std"] = float((w * tstd).sum() / w.sum())
                row[k] = wg
            res["selected_gap"][s][cfg] = row
        print(s, "done", flush=True)
    json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    if not os.path.exists(CACHE):
        build_cache()
    analyze()
