"""
P3 personalized GD: per-player context at a game (causal).

Walker over the extended slot table in play-time order (game end time;
snapshot times from cache/gametime_2024q2.npz, post-snapshot times from
cache/x_post_games.json.gz). For each query row (a player at a game), per
hero, from the player's games that ended before this game started:
  n     games on the hero
  e20, e100  EWMA pick share (half-life 20 / 100 of the player's games)
  days  days since the player last played the hero (-1 never)
  mawp  Max's MAWP (src/lib/mawp.ts; p3_x_ban_feat._mawp, exact)

Preference structure, fit on games before the cutoff (2026-02-10):
  hero embedding  rank-12 SVD of row-centered log(1 + games) of players with
                  >= 100 games (the P3_PREF_SIMILARITY definition)
  clusters        k-means (k = 12) on the players' normalized embeddings;
                  cluster profile = mean hero-share vector of its members
                  (5% population mix)
Per query, from the player's counts n (any history length):
  P(cluster | n)  multinomial posterior with prior = cluster sizes; with no
                  games it is the prior, so the cluster share below is the
                  population share
  cluster share   sum_c P(c | n) profile_c(h)
  shrunk share    (n_h + 10 cluster_share_h) / (n + 10)
  affinity        rank-12 reconstruction of the player's centered log counts
                  at hero h, damped by n / (n + 20)
  role share      shrunk share summed over h's fine role
  main            most-played hero (n >= 10), main share = n_main / n
"""
import os
import sys
import gzip
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from numba import njit, prange

import p3_hs_core as C
import p3_x_ban_feat as B

PREF = os.path.join(C.CACHE, "pgd_pref.npz")
K_CLUSTERS = 12
RANK = 12
KAPPA = 10.0


def play_end_times(d):
    z = np.load(os.path.join(C.CACHE, "gametime_2024q2.npz"))
    o = np.argsort(z["replay_ids"])
    zs = z["replay_ids"][o]
    t_end = np.full(len(d["pid"]), np.nan)
    gl = np.full(len(d["pid"]), 1200.0)
    j = np.minimum(np.searchsorted(zs, d["replay_id"]), len(zs) - 1)
    ok = zs[j] == d["replay_id"]
    t_end[ok] = z["ts"][o][j[ok]]
    gl[ok] = np.where(z["game_length"][o][j[ok]] > 0, z["game_length"][o][j[ok]], 1200.0)
    pg = json.load(gzip.open(os.path.join(C.CACHE, "x_post_games.json.gz"), "rt"))
    pm = {g["replay_id"]: (g["ts"], g["game_length"] or 1200) for g in pg}
    miss = np.flatnonzero(~ok)
    for i in miss:
        v = pm.get(int(d["replay_id"][i]))
        if v is not None:
            t_end[i], gl[i] = v
    bad = np.isnan(t_end)
    t_end[bad] = d["day"][bad] * 86400.0 + 43200.0
    return t_end, t_end - gl


@njit(parallel=False, cache=True)
def _walk(starts, ends, hero, y, t_end, t_start, qslot, q_n, q_e20, q_e100, q_days, q_mawp):
    a20 = 1 - np.exp(-np.log(2) / 20.0)
    a100 = 1 - np.exp(-np.log(2) / 100.0)
    bt = np.zeros((90, 400))
    by = np.zeros((90, 400))
    for p in range(starts.shape[0]):
        cnt = np.zeros(90)
        e20 = np.zeros(90)
        e100 = np.zeros(90)
        w20 = 0.0
        w100 = 0.0
        last = np.full(90, -1.0)
        head = np.zeros(90, np.int64)
        for i in range(starts[p], ends[p]):
            q = qslot[i]
            if q >= 0:
                now = t_start[i]
                for h in range(90):
                    q_n[q, h] = cnt[h]
                    q_e20[q, h] = e20[h] / w20 if w20 > 0 else 0.0
                    q_e100[q, h] = e100[h] / w100 if w100 > 0 else 0.0
                    q_days[q, h] = (now - last[h]) / 86400.0 if last[h] >= 0 else -1.0
                    q_mawp[q, h] = B._mawp(bt[h], by[h], int(cnt[h]), head[h], now)
            h0 = hero[i]
            for h in range(90):
                e20[h] *= 1 - a20
                e100[h] *= 1 - a100
            e20[h0] += a20
            e100[h0] += a100
            w20 = (1 - a20) * w20 + a20
            w100 = (1 - a100) * w100 + a100
            cnt[h0] += 1
            last[h0] = t_end[i]
            bt[h0, head[h0] % 400] = t_end[i]
            by[h0, head[h0] % 400] = y[i]
            head[h0] += 1


def walk(d, qrows):
    """Per-hero history arrays for qrows (my hero order)."""
    t_end, t_start = play_end_times(d)
    n = len(d["pid"])
    srt = np.lexsort((d["replay_id"], t_end, d["pid"]))
    pid = d["pid"][srt]
    brk = np.flatnonzero(np.r_[True, pid[1:] != pid[:-1]])
    ends = np.r_[brk[1:], n]
    qpos = np.full(n, -1, np.int64)
    qpos[qrows] = np.arange(len(qrows))
    nq = len(qrows)
    arrs = [np.zeros((nq, 90), np.float32) for _ in range(5)]
    _walk(brk.astype(np.int64), ends.astype(np.int64), d["hero"][srt].astype(np.int64),
          d["y"][srt].astype(np.float64), t_end[srt], t_start[srt], qpos[srt], *arrs)
    return dict(zip(["n", "e20", "e100", "days", "mawp"], arrs))


def fit_pref(d, cutoff_day):
    """Hero embedding and clusters from pre-cutoff counts."""
    m = d["day"] < cutoff_day
    npl = int(d["n_players"])
    cnt = np.bincount(d["pid"][m] * 90 + d["hero"][m], minlength=npl * 90).reshape(npl, 90).astype(np.float64)
    tot = cnt.sum(1)
    keep = tot >= 100
    Xl = np.log1p(cnt[keep])
    Xl -= Xl.mean(1, keepdims=True)
    mu = Xl.mean(0)
    U, S, Vt = np.linalg.svd(Xl - mu, full_matrices=False)
    V = Vt[:RANK].T                                  # (90, r) orthonormal
    emb = (Xl - mu) @ V
    embn = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)
    rng = np.random.RandomState(0)
    cen = embn[rng.choice(len(embn), K_CLUSTERS, replace=False)]
    for _ in range(100):
        lab = np.argmax(embn @ cen.T, 1)
        new = np.stack([embn[lab == c].mean(0) if (lab == c).any() else cen[c] for c in range(K_CLUSTERS)])
        new /= np.maximum(np.linalg.norm(new, axis=1, keepdims=True), 1e-9)
        if np.allclose(new, cen):
            break
        cen = new
    lab = np.argmax(embn @ cen.T, 1)
    share = cnt[keep] / tot[keep, None]
    pop = cnt.sum(0) / cnt.sum()
    prof = np.stack([share[lab == c].mean(0) for c in range(K_CLUSTERS)])
    prof = 0.95 * prof + 0.05 * pop
    pi = np.bincount(lab, minlength=K_CLUSTERS) / len(lab)
    names = [str(h) for h in d["hero_names"]]
    tops = {int(c): [names[h] for h in np.argsort(-prof[c])[:6]] for c in range(K_CLUSTERS)}
    np.savez(PREF, V=V, mu=mu, cen=cen, prof=prof, pi=pi, pop=pop)
    return tops, int(keep.sum()), np.bincount(lab, minlength=K_CLUSTERS)


def context(hist, fine, pref=None, use_cluster=True):
    """Feature arrays (nq, 90, nf) for the personal model from walker output."""
    z = np.load(PREF) if pref is None else pref
    n = hist["n"].astype(np.float64)
    tot = n.sum(1)
    if use_cluster:
        logpost = np.log(z["pi"])[None, :] + n @ np.log(z["prof"]).T      # (nq, K)
        logpost -= logpost.max(1, keepdims=True)
        post = np.exp(logpost)
        post /= post.sum(1, keepdims=True)
        cshare = post @ z["prof"]                                          # (nq, 90)
        xl = np.log1p(n)
        xl = xl - xl.mean(1, keepdims=True)
        rec = ((xl - z["mu"]) @ z["V"]) @ z["V"].T + z["mu"]
        aff = rec * (tot / (tot + 20.0))[:, None]
    else:
        cshare = np.repeat(z["pop"][None, :], len(n), 0)
        aff = np.zeros_like(n)
    sshare = (n + KAPPA * cshare) / (tot[:, None] + KAPPA)
    nf = int(fine.max()) + 1
    role = np.zeros_like(sshare)
    for f in range(nf):
        role[:, fine == f] = sshare[:, fine == f].sum(1, keepdims=True)
    main = n.argmax(1)
    mshare = np.where(tot >= 10, n[np.arange(len(n)), main] / np.maximum(tot, 1), 0.0)
    is_main = np.zeros_like(n)
    is_main[np.arange(len(n)), main] = (tot >= 10)
    days = hist["days"]
    F = np.stack([np.log(sshare), np.log(cshare), aff, np.log1p(n), (n == 0).astype(np.float64),
                  hist["e20"], hist["e100"], np.where(days >= 0, np.log1p(np.maximum(days, 0)), np.log1p(1000.0)),
                  np.log(role), is_main * mshare[:, None], hist["mawp"] - 0.5,
                  np.repeat(np.log1p(tot)[:, None], 90, 1)], -1).astype(np.float32)
    return F, {"tot": tot, "main_share": mshare, "sshare": sshare, "is_main": is_main}


FEATS = ["log_shrunk_share", "log_cluster_share", "embedding_affinity", "log1p_games_on_hero", "never_played",
         "ewma20_share", "ewma100_share", "log1p_days_since", "log_role_share", "main_x_main_share",
         "mawp_minus_half", "log1p_total_games"]
