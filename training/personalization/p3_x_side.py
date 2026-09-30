"""
P3 extensions: per-slot side information aligned to cache/hs_slots.npz
(scoreboard style and talent conformity), and causal per-player running
means of it. Used by tasks 2 and 4.

Per slot:
  z_raw   20 scoreboard numbers per minute, standardized within hero
          (hero mean and sd from E-window rows)
  z_neu   z_raw minus its E-window mean for (hero, won): "how the player
          plays" with the part explained by winning or losing removed
  conf    talent conformity: share of the 7 talent tiers where the player
          took the hero's most common talent (mode from E-window rows)
Running means use only the player's rows from EARLIER DAYS (same lag-1-day
rule as the skill model).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import p3_hs_core as C

SIDE_P = os.path.join(C.CACHE, "x_side_players.npz")
SIDE_G = os.path.join(C.CACHE, "x_side_games.npz")
GAMETIME = os.path.join(C.CACHE, "gametime_2024q2.npz")
OUT = os.path.join(C.CACHE, "x_side_slots.npz")


def align_players(d, side):
    """Row of side (replay_id, blizz_id) for every slot of d (-1 if none)."""
    blizz = d["player_keys"][d["pid"]] % (1 << 40)
    ka = d["replay_id"].astype(np.int64) * (1 << 40) + blizz
    kb = side["replay_ids"].astype(np.int64) * (1 << 40) + side["blizz_ids"]
    o = np.argsort(kb)
    kbs = kb[o]
    j = np.searchsorted(kbs, ka)
    ok = (j < len(kbs)) & (kbs[np.minimum(j, len(kbs) - 1)] == ka)
    out = np.full(len(ka), -1, np.int64)
    out[ok] = o[j[ok]]
    return out


def game_lookup(d, rid_src):
    o = np.argsort(rid_src)
    s = rid_src[o]
    j = np.searchsorted(s, d["replay_id"])
    ok = (j < len(s)) & (s[np.minimum(j, len(s) - 1)] == d["replay_id"])
    out = np.full(len(d["replay_id"]), -1, np.int64)
    out[ok] = o[j[ok]]
    return out


def build(d=None):
    if d is None:
        d = C.load_slots()
    sp = np.load(SIDE_P)
    side = {k: sp[k] for k in sp.files}
    ai = align_players(d, side)
    print(f"slots with side rows: {(ai >= 0).mean():.4f}", flush=True)
    gt = np.load(GAMETIME)
    gi = game_lookup(d, gt["replay_ids"])
    glen = np.where(gi >= 0, gt["game_length"][np.maximum(gi, 0)], -1).astype(np.float64)
    glen = np.where(glen > 60, glen, np.nan) / 60.0
    sb = np.where(ai[:, None] >= 0, side["scoreboard"][np.maximum(ai, 0)], np.nan).astype(np.float64)
    sb = np.nan_to_num(sb, nan=0.0) / glen[:, None]
    valid = (ai >= 0) & np.isfinite(glen)
    e_mask = d["in_sample"] & (d["day"] >= C.day_of(C.E_START)) & valid
    H = len(d["hero_names"])
    nf = sb.shape[1]
    z = np.zeros_like(sb, dtype=np.float32)
    zn = np.zeros_like(sb, dtype=np.float32)
    won = d["y"] > 0.5
    for h in range(H):
        mh = d["hero"] == h
        me = mh & e_mask
        if me.sum() < 50:
            continue
        mu = sb[me].mean(0)
        sd = sb[me].std(0) + 1e-9
        zz = (sb[mh] - mu) / sd
        z[mh] = zz
        zh = z[me]
        wm = [zh[won[me] == w].mean(0) for w in (False, True)]
        zn[mh] = zz - np.where(won[mh][:, None], wm[1], wm[0])
    z[~valid] = 0
    zn[~valid] = 0
    # talent conformity
    T = np.where(ai[:, None] >= 0, side["talents"][np.maximum(ai, 0)], -1)
    conf = np.full(len(T), np.nan, np.float32)
    for h in range(H):
        mh = np.flatnonzero(d["hero"] == h)
        me = mh[e_mask[mh]]
        modes = np.full(7, -2)
        for t in range(7):
            v = T[me, t]
            v = v[v >= 0]
            if len(v):
                modes[t] = np.bincount(v).argmax()
        Th = T[mh]
        have = (Th >= 0).sum(1)
        match = (Th == modes[None, :]).sum(1)
        conf[mh] = np.where(have >= 4, match / np.maximum(have, 1), np.nan)
    np.savez(OUT, z=z, zn=zn, conf=conf, valid=valid, sb_names=side["sb_names"])
    print(f"wrote {OUT}")


def load():
    z = np.load(OUT)
    return {k: z[k] for k in z.files}


def running_mean_prior_days(d, X, w=None):
    """Per row: mean of X over the SAME player's rows from earlier days
    (and the count). X (n, k); rows with w == 0 are ignored."""
    n = len(d["pid"])
    if w is None:
        w = np.ones(n)
    o = np.lexsort((d["replay_id"], d["day"], d["pid"]))
    pid, day = d["pid"][o], d["day"][o].astype(np.int64)
    Xo = np.where(w[o][:, None] > 0, X[o], 0.0)
    cs = np.vstack([np.zeros((1, X.shape[1])), np.cumsum(Xo, axis=0)])
    cw = np.r_[0.0, np.cumsum(w[o] > 0)]
    comp = pid * 100000 + day
    first_of_day = np.searchsorted(comp, comp, side="left")
    first_of_player = np.searchsorted(pid, pid, side="left")
    sums = cs[first_of_day] - cs[first_of_player]
    cnt = cw[first_of_day] - cw[first_of_player]
    mean = sums / np.maximum(cnt, 1)[:, None]
    out = np.empty_like(mean)
    out[o] = mean
    c = np.empty(n)
    c[o] = cnt
    return out, c


if __name__ == "__main__":
    build()
