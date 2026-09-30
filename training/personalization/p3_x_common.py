"""
P3 extensions: shared helpers for the p3_x_* scripts.

The extended slot table (cache/x_slots_ext.npz) is the phase-1 slot table
(cache/hs_slots.npz, snapshot games 2024-04-01 .. 2026-05-22) with the
post-snapshot Storm League games appended (build 2.55.16.97039 after the
snapshot and 2.55.17.*, games dated before 2026-09-28). Post-snapshot rows
carry the causal WP from p3_x_wp_post.py. Player ids of snapshot rows are
unchanged; new accounts get new ids after the old ones. Rows are sorted by
(day, replay_id) as in hs_slots.

Everything that reads shared code (p3_hs_core) goes through here, so an
audit fix in p3_hs_core or a rebuilt wp_drift.npz only needs
`p3_x_common.py ext` and the downstream p3_x_* scripts rerun.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import p3_hs_core as C

CACHE = C.CACHE
RESULTS = C.RESULTS
EXT = os.path.join(CACHE, "x_slots_ext.npz")
POST_P = os.path.join(CACHE, "x_post_players.npz")
POST_WP = os.path.join(CACHE, "x_wp_post.npz")
SNAP_LAST_DAY = C.day_of("2026-05-22")
THREADS_ENV = {"OMP_NUM_THREADS": "4", "NUMBA_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}


def build_ext():
    s = np.load(C.SLOTS)
    d = {k: s[k] for k in s.files}
    n_snap_games = int(d["g"].max()) + 1
    wp = np.load(POST_WP)
    new = ~wp["in_snapshot"]
    g_rid = wp["replay_ids"][new]
    o = np.argsort(g_rid)
    g_rid = g_rid[o]
    g_y = wp["y"][new][o]
    g_wp = wp["wp0"][new][o].astype(np.float64)
    g_day = wp["date_days"][new][o]
    g_ver = wp["version"][new][o]
    p = np.load(POST_P)
    assert list(p["hero_names"]) == list(d["hero_names"])
    gi = np.searchsorted(g_rid, p["replay_ids"])
    ok = (gi < len(g_rid)) & (g_rid[np.minimum(gi, len(g_rid) - 1)] == p["replay_ids"]) \
        & (p["hero"] >= 0)
    # keep only games with exactly ten usable player rows
    cnt = np.bincount(gi[ok], minlength=len(g_rid))
    ok &= cnt[np.minimum(gi, len(g_rid) - 1)] == 10
    print(f"post player rows: {len(ok):,}, kept {ok.sum():,} "
          f"({(cnt == 10).sum():,} of {len(g_rid):,} games complete)")
    gi = gi[ok]
    team = p["team"][ok].astype(np.int8)
    y = np.where(team == 0, g_y[gi], 1 - g_y[gi]).astype(np.float32)
    wpt = np.where(team == 0, g_wp[gi], 1 - g_wp[gi])
    key = p["region"][ok].astype(np.int64) * (1 << 40) + p["blizz_ids"][ok]
    old_keys = d["player_keys"]
    j = np.searchsorted(old_keys, key)
    known = (j < len(old_keys)) & (old_keys[np.minimum(j, len(old_keys) - 1)] == key)
    nk, ninv = np.unique(key[~known], return_inverse=True)
    pid = np.empty(len(key), np.int64)
    pid[known] = j[known]
    pid[~known] = len(old_keys) + ninv
    add = dict(replay_id=p["replay_ids"][ok], g=(n_snap_games + gi).astype(d["g"].dtype),
               pid=pid, hero=p["hero"][ok].astype(np.int64), team=team,
               party=p["party"][ok], y=y, wp=wpt, day=g_day[gi].astype(np.int32),
               in_sample=np.zeros(len(pid), bool), player_mmr=p["player_mmr"][ok],
               hero_mmr=p["hero_mmr"][ok], role_mmr=p["role_mmr"][ok],
               hero_level=p["hero_level"][ok])
    out = {}
    for k, v in add.items():
        out[k] = np.concatenate([d[k], v.astype(d[k].dtype)])
    out["post"] = np.r_[np.zeros(len(d["pid"]), bool), np.ones(len(pid), bool)]
    out["version"] = np.r_[np.full(len(d["pid"]), -1, np.int16), g_ver[gi].astype(np.int16)]
    srt = np.lexsort((out["replay_id"], out["day"]))
    out = {k: v[srt] for k, v in out.items()}
    out["n_players"] = np.int64(len(old_keys) + len(nk))
    out["hero_names"] = d["hero_names"]
    out["player_keys"] = np.r_[old_keys, nk]
    out["versions"] = wp["versions"]
    out["n_snap_games"] = np.int64(n_snap_games)
    np.savez(EXT, **out)
    print(f"wrote {EXT}: {len(srt):,} slots ({len(pid):,} post), "
          f"{int(out['n_players']):,} players ({len(nk):,} new)")


def load_ext():
    z = np.load(EXT, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["r"] = d["y"] - d["wp"]
    d["v"] = d["wp"] * (1 - d["wp"])
    return d


def kernels():
    """Phase-1 kernels (fit on E) and the experience table."""
    from p3_hs_fit import KERNELS, KOUT
    kz = np.load(KOUT)
    bases = {k[6:]: kz[k] for k in kz.files if k.startswith("basis_")}
    Ks = {name: C.kernel_from([bases[b] for b in bl], kz["theta_" + name])
          for name, bl in KERNELS.items()}
    return Ks, kz["experience_table"]


def online_predict(d, add_mask, query_mask, K, r_adj, lag=1, freeze_day=None,
                   extra=None):
    """Causal per-query GP posterior (mean, var) at the query's own hero, plus
    the query's lagged counts (games on hero, games overall), without storing
    the (nq, 90) state for all queries. State holds add_mask rows with
    day <= query day - lag (and day <= freeze_day if given).
    extra: optional list of kernels to evaluate at the same state.
    Returns qidx, mean, var, Nh, Np, [extra means]."""
    H = K.shape[0]
    day = d["day"]
    npl = int(d["n_players"])
    Sst = np.zeros((npl, H))
    Pst = np.zeros((npl, H))
    Nst = np.zeros((npl, H), np.float32)
    qidx = np.flatnonzero(query_mask)
    nq = len(qidx)
    mean = np.empty(nq)
    var = np.empty(nq)
    Nh = np.empty(nq, np.float32)
    Np = np.empty(nq, np.float32)
    ex = [np.empty(nq) for _ in (extra or [])]
    add_idx = np.flatnonzero(add_mask)
    add_day = day[add_idx]
    ap = 0
    ud, qs = np.unique(day[qidx], return_index=True)
    qe = np.r_[qs[1:], nq]
    for t, a, b in zip(ud, qs, qe):
        lim = t - lag if freeze_day is None else min(t - lag, freeze_day)
        e = np.searchsorted(add_day, lim, side="right")
        if e > ap:
            rr = add_idx[ap:e]
            pi, hi = d["pid"][rr], d["hero"][rr]
            np.add.at(Sst, (pi, hi), r_adj[rr] / d["v"][rr])
            np.add.at(Pst, (pi, hi), 1.0 / d["v"][rr])
            np.add.at(Nst, (pi, hi), 1.0)
            ap = e
        q = qidx[a:b]
        pq, hq = d["pid"][q], d["hero"][q]
        Sq = np.ascontiguousarray(Sst[pq])
        Pq = np.ascontiguousarray(Pst[pq])
        mean[a:b], var[a:b] = C.gp_query(Sq, Pq, hq, K)
        for k, K2 in enumerate(extra or []):
            ex[k][a:b] = C.gp_query(Sq, Pq, hq, K2)[0]
        Nh[a:b] = Nst[pq, hq]
        Np[a:b] = Nst[pq].sum(axis=1)
    return qidx, mean, var, Nh, Np, ex


def team_diff(values, g, team, n_games):
    sign = np.where(team == 0, 1.0, -1.0)
    return np.bincount(g, weights=sign * values, minlength=n_games)


def game_arrays(d):
    n_games = int(d["g"].max()) + 1
    y = np.zeros(n_games)
    wp0 = np.full(n_games, 0.5)
    t0 = d["team"] == 0
    y[d["g"][t0]] = d["y"][t0]
    wp0[d["g"][t0]] = d["wp"][t0]
    cnt = np.bincount(d["g"], minlength=n_games)
    gday = np.zeros(n_games, np.int64)
    gday[d["g"]] = d["day"]
    return n_games, y, wp0, cnt, gday


def boot_gain(ll0, ll1, idx, n=200, seed=0):
    rng = np.random.RandomState(seed)
    diffs = [float((ll0[b] - ll1[b]).mean()) for b in
             (rng.choice(idx, len(idx)) for _ in range(n))]
    return float((ll0[idx] - ll1[idx]).mean()), [float(np.percentile(diffs, 2.5)),
                                                   float(np.percentile(diffs, 97.5))]


def logloss(p, y, eps=1e-7):
    return -(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))


if __name__ == "__main__":
    if "ext" in sys.argv:
        build_ext()
