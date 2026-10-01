"""
Rebuild the gN judges with their own role-composition table (audit P1-A5).

The original gN judges (gN8_oof_s0..2) read the external Heroes Profile
composition table for 4 of their 283 inputs (the empirical win rate of each
team's role multiset). That table aggregates games that overlap the snapshot
the agents trained on, so gN was not fully independent of the submitted
agents. gN8o_oof_s0..2 use the same recipe and the same 291,837 post-snapshot
no-drift games, with the composition table built from those games only:
out of fold for training rows (2 folds, split.fold_of), and over all of them
(stats "No8") for scoring. Cells are admitted at >= 50 games
(paper1_revision.core convention); others get the 33% default.
gN_naive (hero identity, 197 inputs) reads no statistics; it is trained here
too when absent (a tier-scheme rebuild needs it; the legacy one is kept).

Usage (from training/): CUDA_VISIBLE_DEVICES=<gpu> nice -n 19 taskset -c 48-53 \
    python3 overfit2026/comp_gn_rebuild.py
Outputs: overfit2026/cache/split_stats/No8{,f0,f1}.json (+ _compositions.json),
         overfit2026/models/gN8o_oof_s{0,1,2}.{pt,json}
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

from overfit2026 import feats, split
from overfit2026.comp_gn_folds import train, SPEC

NAMES = ["gN8o_oof_s0", "gN8o_oof_s1", "gN8o_oof_s2"]


def write_stats(games, name):
    from paper1_revision import core
    p = split.stats_path(name)
    if not os.path.exists(p):
        js = core.stats_json(split.counts_for(games), {"subset": name, "games_used": len(games)})
        comps = js.pop("compositions")
        json.dump(js, open(p, "w"))
        json.dump(comps, open(p[:-5] + "_compositions.json", "w"))
    return split.load_stats(name)


def main():
    import torch
    from drift2026 import common as dcommon
    dcommon._bind_statscache_methods()
    torch.set_num_threads(4)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    games = split.games_for("N")
    fold = np.array([split.fold_of(g[0]) for g in games])
    write_stats(games, "No8")
    Xf = np.zeros((len(games), 283), np.float32)
    Xs = np.zeros_like(Xf)
    for f in (0, 1):
        st = write_stats([g for g, ff in zip(games, fold) if ff == 1 - f], f"No8f{1 - f}")
        idx = np.where(fold == f)[0]
        a, b = feats.featurize([(games[i][3], games[i][4], games[i][2], games[i][1]) for i in idx],
                               st, nproc=6)
        Xf[idx], Xs[idx] = a, b
    y = np.array([1.0 if g[6] == 0 else 0.0 for g in games], np.float32)
    val = np.array([split.is_val(g[0]) for g in games])
    X = np.concatenate([Xf, Xs])
    Y = np.concatenate([y, 1 - y])
    V = np.concatenate([val, val])
    print(f"features {time.time() - t0:.0f}s", flush=True)
    for name, seed in zip(NAMES, (200, 201, 202)):
        if os.path.exists(split.model_path(name)):
            continue
        m, ep, bl = train(X, Y, V, 283, seed, dev)
        torch.save(m.state_dict(), split.model_path(name))
        meta = dict(side="No", e=8, mode="oof", arch=SPEC["arch"], dropout=SPEC["dropout"],
                    wd=SPEC["wd"], epochs=SPEC["epochs"], es=True, seed=seed, inp="enriched",
                    name=name, input_dim=283, n_train_rows=int((~V).sum()), best_val_loss=bl,
                    epochs_run=ep, compositions="own (post-snapshot no-drift games, OOF)")
        json.dump(meta, open(split.model_path(name)[:-3] + ".json", "w"))
        print(f"{name}: {ep} epochs, val loss {bl:.5f} ({time.time() - t0:.0f}s)", flush=True)
    # hero-identity judge (gN_naive): same games and split, the 197
    # statistics-free inputs only (original recipe: seed 200)
    name = "gN8_naive"
    if not os.path.exists(split.model_path(name)):
        m, ep, bl = train(X[:, :197], Y, V, 197, 200, dev)
        torch.save(m.state_dict(), split.model_path(name))
        meta = dict(side="No", e=8, mode="naive", arch=SPEC["arch"], dropout=SPEC["dropout"],
                    wd=SPEC["wd"], epochs=SPEC["epochs"], es=True, seed=200, inp="naive",
                    name=name, input_dim=197, n_train_rows=int((~V).sum()), best_val_loss=bl,
                    epochs_run=ep)
        json.dump(meta, open(split.model_path(name)[:-3] + ".json", "w"))
        print(f"{name}: {ep} epochs, val loss {bl:.5f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
