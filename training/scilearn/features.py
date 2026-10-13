"""As-of feature store (FRAMING 6.3 rule 1, 11.5).

Contract: any statistic or state used for game g comes only from games that our ingestion
first saw before g started (first_seen_ts < game_ts[g]). Historical first-seen is the game
date plus one day (retrospective); new rows get true first-seen times from the daily diff.
The fixtures in fixtures.py check every builder here.
"""
import numpy as np


def _asof_sum(keys_ev, t_ev, ts_ev, keys_q, t_q, val_ev, tau):
    """For each query (key, t_q): sum over events with the same key and t_ev < t_q of
    val_ev * exp(-(t_q - ts_ev) / tau) (tau = inf means no decay)."""
    out = np.zeros(len(keys_q))
    ne, nq = len(keys_ev), len(keys_q)
    # merge: events sort before queries at equal time (strict inequality handled by order key)
    k = np.concatenate([keys_ev, keys_q])
    t = np.concatenate([t_ev, t_q])
    is_q = np.concatenate([np.zeros(ne, np.int8), np.ones(nq, np.int8)])
    # at equal (key, time) put queries first, so events at exactly t_q do not count
    order = np.lexsort((-is_q, t, k))
    k, t, is_q = k[order], t[order], is_q[order]
    ts_all = np.concatenate([ts_ev, t_q])[order]
    v_all = np.concatenate([val_ev, np.zeros(nq)])[order]
    qidx = np.concatenate([np.full(ne, -1), np.arange(nq)])[order]
    decay = np.isfinite(tau)
    cur_key, s, t_last = None, 0.0, 0
    import math
    CH = 2_000_000
    for c0 in range(0, len(k), CH):
        kl, tl, ql = k[c0:c0 + CH].tolist(), t[c0:c0 + CH].tolist(), is_q[c0:c0 + CH].tolist()
        tsl, vl, il = ts_all[c0:c0 + CH].tolist(), v_all[c0:c0 + CH].tolist(), qidx[c0:c0 + CH].tolist()
        for i in range(len(kl)):
            if kl[i] != cur_key:
                cur_key, s, t_last = kl[i], 0.0, tl[i]
            if decay:
                s *= math.exp(-(tl[i] - t_last) / tau)
                t_last = tl[i]
            if ql[i]:
                out[il[i]] = s
            else:
                # event enters at its first-seen time, decayed from its game time
                s += vl[i] * (math.exp(-(tl[i] - tsl[i]) / tau) if decay else 1.0)
    return out


def player_hero_experience(ctx, half_life_days):
    """log1p of the decayed count of the player's earlier games on this hero, per slot (n, 10).
    Private or unknown players (key -1) get 0."""
    if ctx.players is None:
        raise ValueError("player histories are not loaded")
    pk = ctx.players["player_key"]
    n = ctx.n
    key = np.where(pk >= 0, pk * 128 + ctx.heroes, -1).ravel()
    fs = np.repeat(ctx.first_seen_ts, 10)
    gt = np.repeat(ctx.game_ts, 10)
    m = key >= 0
    tau = half_life_days * 86400 / np.log(2) if half_life_days and half_life_days > 0 else np.inf
    v = np.zeros(n * 10)
    v[m] = _asof_sum(key[m], fs[m], gt[m], key[m], gt[m], np.ones(m.sum()), tau)
    return np.log1p(v).reshape(n, 10)


def asof_hero_winrate(ctx, y0, prior_n=200.0):
    """Shrunk as-of win rate of each slot's hero, from games first seen before this game
    started. Legitimate builder; it must pass every fixture."""
    H = ctx.H
    n = ctx.n
    hero = ctx.heroes.ravel()
    win = np.concatenate([y0[:, None].repeat(5, 1), 1 - y0[:, None].repeat(5, 1)], 1).ravel().astype(float)
    fs = np.repeat(ctx.first_seen_ts, 10)
    gt = np.repeat(ctx.game_ts, 10)
    wins = _asof_sum(hero, fs, gt, hero, gt, win, np.inf)
    cnt = _asof_sum(hero, fs, gt, hero, gt, np.ones_like(win), np.inf)
    return ((wins + 0.5 * prior_n) / (cnt + prior_n)).reshape(n, 10)
