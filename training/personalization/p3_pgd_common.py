"""
P3 personalized GD: shared pieces.

Draft-state encoding (hero index = the players-cache order used by all p3
code; teams = real team labels, as in the paper-1 GD):
  [0:90] team-0 picks, [90:180] team-1 picks, [180:270] bans, [270:284] map,
  [284:287] tier, [287] step / 15, [288] step type (1 pick, 0 ban)
Meta inputs (causal): per tier, each hero's pick rate and ban rate per game
over the 28 days ending the day before the game (from cache/pgd_games.npz),
appended as [289:379] pick rates and [379:469] ban rates.

Windows (by game day): train = before the WP training cutoff (2026-02-10);
V1 = 2026-02-10 .. 2026-03-31; V2 = 2026-04-01 .. 2026-05-22 (snapshot end);
post = post-snapshot games (2026-05-23 .. 2026-09-27, builds 2.55.16.97039
and 2.55.17.*).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn as nn

import p3_hs_core as C

GAMES = os.path.join(C.CACHE, "pgd_games.npz")
TYPES = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1])
PICK_STEPS = np.flatnonzero(TYPES == 1)
BAN_STEPS = np.flatnonzero(TYPES == 0)
CUTOFF = C.day_of("2026-02-10")
V2_START = C.day_of("2026-04-01")
SNAP_END = C.day_of("2026-05-22")
DIM = 469
META_DAYS = 28


def load_games():
    z = np.load(GAMES)
    g = {k: z[k] for k in z.files}
    # causal trailing meta rates per (day, tier)
    pc, bc, gc = g["pick_counts"], g["ban_counts"], g["game_counts"]
    cp = np.concatenate([np.zeros((1,) + pc.shape[1:], np.float64), np.cumsum(pc, 0, dtype=np.float64)])
    cb = np.concatenate([np.zeros((1,) + bc.shape[1:], np.float64), np.cumsum(bc, 0, dtype=np.float64)])
    cg = np.concatenate([np.zeros((1, 3)), np.cumsum(gc, 0, dtype=np.float64)])
    nd = pc.shape[0]
    hi = np.arange(nd)                    # window ends the day before
    lo = np.maximum(hi - META_DAYS, 0)
    games_w = np.maximum(cg[hi] - cg[lo], 1.0)
    g["meta_pick"] = ((cp[hi] - cp[lo]) / games_w[:, :, None]).astype(np.float32)   # (nd, 3, 90)
    g["meta_ban"] = ((cb[hi] - cb[lo]) / games_w[:, :, None]).astype(np.float32)
    g["window"] = np.where(g["post"], 3, np.where(g["day"] < CUTOFF, 0, np.where(g["day"] < V2_START, 1, 2)))
    return g


def states(g, gi, k):
    """X (B, 469) and valid mask (B, 90) for games gi at steps k (arrays)."""
    B = len(gi)
    X = np.zeros((B, DIM), np.float32)
    H, T = g["H"][gi], g["T"][gi]
    taken = np.zeros((B, 90), bool)
    ar = np.arange(B)
    for j in range(16):
        sel = k > j
        if not sel.any():
            continue
        h = H[sel, j]
        if TYPES[j] == 1:
            X[ar[sel], np.where(T[sel, j] == 0, 0, 90) + h] = 1
        else:
            X[ar[sel], 180 + h] = 1
        taken[ar[sel], h] = True
    X[ar, 270 + g["map"][gi]] = 1
    X[ar, 284 + g["tier"][gi]] = 1
    X[:, 287] = k / 15.0
    X[:, 288] = TYPES[k]
    di = g["day"][gi] - int(g["day0"])
    X[:, 289:379] = g["meta_pick"][di, g["tier"][gi]]
    X[:, 379:469] = g["meta_ban"][di, g["tier"][gi]]
    return X, ~taken


class MetaGD(nn.Module):
    def __init__(self, d_in=DIM, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 256), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 90))

    def forward(self, x, mask=None):
        lg = self.net(x)
        return lg if mask is None else lg.masked_fill(~mask, -1e9)


def load_metagd():
    m = MetaGD()
    m.load_state_dict(torch.load(os.path.join(C.CACHE, "pgd_metagd.pt"), weights_only=True))
    m.eval()
    return m


def logp_metagd(m, X, M, bs=16384):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(torch.log_softmax(m(torch.from_numpy(X[i:i + bs]), torch.from_numpy(M[i:i + bs])), -1).numpy())
    return np.concatenate(out)


def logp_paper_gd(gd, X, M):
    """Paper-1 GD (5-model average, p3_dr_core.GDPolicy) on the same states."""
    return gd.logprobs(X[:, 0:90], X[:, 90:180], X[:, 180:270], X[:, 270:284], X[:, 284:287], X[:, 287] * 15.0,
                       X[:, 288], M)


def scores(logp, y, ks=(1, 3, 5)):
    lp = np.where(np.isfinite(logp), logp, -1e9)
    top = np.argsort(-lp, axis=1)
    out = {f"top{k}": float((top[:, :k] == y[:, None]).any(1).mean()) for k in ks}
    ll = lp[np.arange(len(y)), y]
    out["log_loss"] = float(-np.mean(np.maximum(ll, np.log(1e-9))))
    out["n"] = int(len(y))
    return out
