"""
Game arrays for the hidden-worlds module (shared by papers 1 and 2).

Every set is the paper-1 site corpus (P1_TIERS=site; patch 2.55 only, nothing from the sealed
2.57 build is ever read):
  train  paper1_revision core.split_games()["train"]: the games every paper-1 model and the
         support corpus (min3, FLOOR = 25) are built from; the world model is fit here
  val    core val split (hyperparameters of the world model)
  test   the paper's test split
  NODRIFT post-snapshot games on the snapshot's builds (certify, step-1 slope)
  T17    2.55.17 builds (balance patch 2026-07-20): the drift family's measured movement

Arrays (cached in site/cache/data_<name>.npz): T0, T1 (n, 5) sorted hero indices, m (map
index, len(MAPS) = unknown), t (tier index), y (1 = team 0 won), rid.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
if TRAINING_DIR not in sys.path:
    sys.path.insert(0, TRAINING_DIR)
os.environ.setdefault("P1_TIERS", "site")

from overfit2026 import data as odata  # noqa: E402

if odata.TIER_SCHEME != "site":
    raise SystemExit("worlds needs P1_TIERS=site")
H = 90
CACHE = odata.art(HERE, "cache")
RESULTS = odata.art(HERE, "results")
LOGS = odata.art(HERE, "logs")
SETS = ("train", "val", "test", "NODRIFT", "T17")
_META = os.path.join(CACHE, "data_meta.npz")


def _meta(train=None):
    if not os.path.exists(_META):
        maps = sorted({g[2] for g in train})
        tiers = sorted({g[1] for g in train})
        np.savez(_META, maps=np.array(maps), tiers=np.array(tiers))
    z = np.load(_META)
    return z["maps"].tolist(), z["tiers"].tolist()


def _arrays(games, maps, tiers):
    from shared import HERO_TO_IDX
    mi = {m: i for i, m in enumerate(maps)}
    ti = {t: i for i, t in enumerate(tiers)}
    T0 = np.array([sorted(HERO_TO_IDX[h] for h in g[3]) for g in games], np.int64)
    T1 = np.array([sorted(HERO_TO_IDX[h] for h in g[4]) for g in games], np.int64)
    m = np.array([mi.get(g[2], len(maps)) for g in games], np.int64)
    t = np.array([ti.get(g[1], len(tiers)) for g in games], np.int64)
    y = np.array([1.0 if g[6] == 0 else 0.0 for g in games])
    rid = np.array([g[0] for g in games], np.int64)
    return dict(T0=T0, T1=T1, m=m, t=t, y=y, rid=rid)


def build():
    from paper1_revision import core
    G = core.split_games()
    maps, tiers = _meta(G["train"])
    for k in ("train", "val", "test"):
        np.savez(os.path.join(CACHE, f"data_{k}.npz"), **_arrays(G[k], maps, tiers))
    for k in ("NODRIFT", "T17"):
        np.savez(os.path.join(CACHE, f"data_{k}.npz"), **_arrays(core.gold_games(k), maps, tiers))
    for k in SETS:
        z = np.load(os.path.join(CACHE, f"data_{k}.npz"))
        print(k, len(z["y"]), "games", flush=True)


def load(name):
    p = os.path.join(CACHE, f"data_{name}.npz")
    if not os.path.exists(p):
        build()
    z = np.load(p)
    return {k: z[k] for k in z.files}


def meta():
    """(maps, tiers) of the training split."""
    if not os.path.exists(_META):
        build()
    return _meta()


def subset(D, sel):
    return {k: v[sel] for k, v in D.items()}


def from_rows(rows):
    """[(team0 names, team1 names, map, tier)] -> arrays (no y) on the training split's indexing."""
    from shared import HERO_TO_IDX
    maps, tiers = meta()
    mi = {m: i for i, m in enumerate(maps)}
    ti = {t: i for i, t in enumerate(tiers)}
    return dict(T0=np.array([sorted(HERO_TO_IDX[h] for h in r[0]) for r in rows], np.int64),
                T1=np.array([sorted(HERO_TO_IDX[h] for h in r[1]) for r in rows], np.int64),
                m=np.array([mi.get(r[2], len(maps)) for r in rows], np.int64),
                t=np.array([ti.get(r[3], len(tiers)) for r in rows], np.int64))


if __name__ == "__main__":
    build()
