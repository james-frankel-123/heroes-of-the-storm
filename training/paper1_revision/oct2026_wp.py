"""
oct2026 refresh: the tournament's four win-probability models (evaluators and
greedy value functions), leak-free, written under the names the unchanged
rerun2026 scripts load from the namespace models dir:

  wp_naive          197d, 256x128
  wp_herostrength   209d (hero_wr, team_avg_wr), 256x128
  wp_enriched_256   283d (ENRICHED_GROUPS), 256x128   (also the MCTS leaf WP)
  wp_aug_v2_512     283d + synthetic unseen-composition records (v2 generator,
                    WR 10%, volume 100, unseen scope, own composition table),
                    512x256x128

Training rows carry out-of-fold features (oct2026_data.py). Early stopping and
seed selection use a 2% hash validation subset of the training rows (also
out-of-fold), never the test rows; 3 seeds (42, 123, 777), lowest validation
loss kept. Optimizer as train_wp.py. Test accuracy and calibration slope
(deploy-statistics features) are recorded in models/meta/<name>.json.

Env: RERUN_NS=oct2026, REPLAY_SNAPSHOT_PATH, WP_STATS_PATH (deploy stats),
P1R_COMP_PATH, CUDA_VISIBLE_DEVICES.
Usage: python3 paper1_revision/oct2026_wp.py [name ...]
"""
import os
import sys
import json
import random

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)

import numpy as np

SEEDS = (42, 123, 777)


def specs():
    from experiment_synthetic_augmentation import ENRICHED_GROUPS
    E = list(ENRICHED_GROUPS)
    return {"wp_naive": ([], [256, 128], False),
            "wp_herostrength": (["hero_wr", "team_avg_wr"], [256, 128], False),
            "wp_enriched_256": (E, [256, 128], False),
            "wp_aug_v2_512": (E, [512, 256, 128], True)}


def cols_for(groups):
    from sweep_enriched_wp import compute_group_indices
    gi = compute_group_indices()
    c = []
    for g in groups:
        s, e = gi[g]
        c.extend(range(s, e))
    return c


def synthetic(cols):
    """v2 synthetic records -> (X, y) with out-of-fold features (round-robin folds)."""
    from rerun2026 import common
    from experiment_synthetic_augmentation import generate_synthetic_data_v2
    from paper1_revision.oct2026_data import load_stats, stats_dir, _extract, N_FOLDS
    cache = os.path.join(common.CACHE_DIR, "synthetic_v2_wr10.npz")
    if not os.path.exists(cache):
        random.seed(42)
        np.random.seed(42)
        train, _ = common.load_split()
        comp_raw = json.load(open(os.path.join(stats_dir(), "deploy_compositions.json")))
        syn, _ = generate_synthetic_data_v2(train, comp_raw, load_stats("deploy"), unseen_wr=10.0,
                                            unseen_volume=100, scope="tier2_only")
        for i, d in enumerate(syn):
            d["replay_id"] = -(i + 1)
        parts = []
        for k in range(N_FOLDS):
            parts.append(_extract([d for i, d in enumerate(syn) if i % N_FOLDS == k], load_stats(f"oof{k}")))
        b, e, l, r = [np.concatenate([p[i] for p in parts]) for i in range(4)]
        np.savez(cache, bases=b, enricheds=e, labels=l)
        print(f"synthetic: {len(syn):,} records -> {len(l):,} rows", flush=True)
    z = np.load(cache)
    return np.concatenate([z["bases"], z["enricheds"][:, cols]], 1), z["labels"]


def main():
    import torch
    from rerun2026 import common
    from paper1_revision import core, train_wp
    common.setup()
    dev = torch.device("cuda")
    S = specs()
    names = sys.argv[1:] or list(S)
    tr = np.load(common.FULL_TRAIN_NPZ)
    te = np.load(common.FULL_TEST_NPZ)
    val = np.array([core.is_val(r) for r in tr["rids"]])
    for name in names:
        out = os.path.join(common.MODELS_DIR, f"{name}.pt")
        if os.path.exists(out):
            print("skip", name)
            continue
        groups, arch, aug = S[name]
        cols = cols_for(groups)
        X = np.concatenate([tr["bases"], tr["enricheds"][:, cols]], 1) if cols else tr["bases"]
        y = tr["labels"]
        Xt, yt = X[~val], y[~val]
        if aug:
            sX, sy = synthetic(cols)
            Xt, yt = np.concatenate([Xt, sX]), np.concatenate([yt, sy])
        tX, tY = torch.tensor(Xt, device=dev), torch.tensor(yt, device=dev)
        vX, vY = torch.tensor(X[val], device=dev), torch.tensor(y[val], device=dev)
        Xte = np.concatenate([te["bases"], te["enricheds"][:, cols]], 1) if cols else te["bases"]
        res = {"name": name, "groups": groups, "arch": arch, "input_dim": int(X.shape[1]),
               "n_train_rows": int(len(Xt)), "n_val_rows": int(val.sum()), "seeds": {}}
        best = None
        for seed in SEEDS:
            model, vl, hist = train_wp.train_one(name, {"arch": arch}, seed, dev, (tX, tY, vX, vY))
            with torch.no_grad():
                p = model(torch.tensor(Xte, device=dev)).view(-1).cpu().numpy()
            yy = te["labels"]
            pf, ps = p[0::2], p[1::2]                     # adjacent team orders
            psym = np.clip(0.5 * (pf + 1 - ps), 1e-6, 1 - 1e-6)
            r = {"val_loss": vl, "epochs": len(hist),
                 "test_acc": float(np.mean((p > .5) == (yy > .5))),
                 "test_slope_sym": train_wp.fit_slope(psym, yy[0::2])}
            res["seeds"][str(seed)] = r
            print(name, seed, {k: round(v, 4) for k, v in r.items()}, flush=True)
            if best is None or vl < best[0]:
                best = (vl, seed, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        torch.save(best[2], out)
        res["selected_seed"] = best[1]
        res["out"] = out
        os.makedirs(common.META_DIR, exist_ok=True)
        json.dump(res, open(os.path.join(common.META_DIR, f"{name}.json"), "w"), indent=1)
        del tX, tY, vX, vY
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
