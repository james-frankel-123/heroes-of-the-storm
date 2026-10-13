"""Loaders for the broker's private snapshot. Proposers never import this (no mount)."""
import json

import numpy as np

from training.scilearn import config as C
from training.scilearn import tranches as T


def latest_snapshot():
    snaps = sorted(p.name for p in C.SNAPSHOTS.glob("snap_*") if (p / "manifest.json").exists())
    return snaps[-1]


def load(snap_id=None, tranches=None, labels=False, players=False, root=None):
    """Return (games, meta, y0 or None, players or None) restricted to `tranches`.
    The pool and sealed rows are never in these files."""
    d = (root or C.SNAPSHOTS) / (snap_id or latest_snapshot())
    meta = json.loads((d / "manifest.json").read_text())
    g = dict(np.load(d / "games.npz"))
    if tranches is None:
        m = np.ones(len(g["replay_id"]), bool)
    else:
        m = np.isin(g["tranche"], [T.CODES[t] for t in tranches])
    g = {k: v[m] for k, v in g.items()}
    y = None
    if labels:
        lab = np.load(d / "labels.npz")
        assert np.array_equal(lab["replay_id"][m], g["replay_id"])
        y = lab["y0"][m].astype(np.int8)
    pl = None
    if players:
        p = np.load(d / "players.npz")
        pl = {"player_key": p["player_key"][m], "party": p["party"][m]}
    meta["snap_dir"] = str(d)
    return g, meta, y, pl


def load_kit(meta, path=None):
    """Kit table aligned to meta['heroes'] -> (array (H, K), axes)."""
    p = path or (C.CODE / "kit" / "kit_2.57.json")
    if not p.exists():
        return None, []
    k = json.loads(p.read_text())
    axes = k["axes"] if isinstance(k["axes"][0], str) else [a["name"] for a in k["axes"]]
    vals = k["heroes"]
    arr = np.array([[float(vals[h]["values"][a]) if isinstance(vals[h], dict) and "values" in vals[h]
                     else float(vals[h][a]) for a in axes] for h in meta["heroes"]])
    return arr, axes
