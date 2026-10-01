"""
Composition audit, step 2 data: per-game player-skill and pick-position
controls for the "is the degenerate-team penalty causal?" test.

Skill = the P3 causal per-player, per-hero strength (personalization/
p3_hs_core, "+CF rank 2" kernel and experience offsets fit on the E window,
frozen; online state from every game on days <= game day - 1). Two state
variants:
  all    the P3 product as published (every earlier game enters the state)
  clean  slots on degenerate teams are left out of the state, so a player's
         skill is never estimated from the composition penalty itself
Queries: snapshot games from 2025-06-01 on (a year of history before them)
and every post-snapshot game in the extended slot table (2.55.16.97039 after
the snapshot and 2.55.17.*, whitelisted builds only).

Per game (team0 orientation): y, structure of both teams, team skill sums
under both variants, skill by within-team pick position (1st..5th pick of
that team), which team had first pick, party sizes, experience counts.

Usage (from training/):
  nice -n 19 taskset -c 48-53 python3 overfit2026/comp_causal_data.py
Output: overfit2026/cache/comp_causal_games.npz
"""
import os
import sys
import time

for k in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "4")
HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)
sys.path.insert(0, os.path.join(TRAINING_DIR, "personalization"))

import numpy as np

import p3_hs_core as C
import p3_x_common as X
from overfit2026.structure import struct_vec

OUT = os.path.join(HERE, "cache", "comp_causal_games.npz")
Q_START = C.day_of("2025-06-01")
BEST = "+CF rank 2"


def pick_ranks(d, names):
    """pick rank (0..9) per slot from the snapshot and post pick-order caches."""
    rank = np.full(len(d["pid"]), -1, np.int16)
    key_slot = d["replay_id"] * 128 + d["hero"]
    o = np.argsort(key_slot)
    ks = key_slot[o]
    hidx = {h: i for i, h in enumerate(names)}
    for f in (os.path.join(TRAINING_DIR, "personalization", "cache", "pickorder_2024q2.npz"),
              os.path.join(HERE, "cache", "comp_pickorder_post.npz")):
        z = np.load(f)
        hi = np.array([hidx.get(h, -1) for h in z["hero"]])
        ok = hi >= 0
        kq = z["replay_ids"][ok] * 128 + hi[ok]
        j = np.searchsorted(ks, kq)
        hit = (j < len(ks)) & (ks[np.minimum(j, len(ks) - 1)] == kq)
        rank[o[j[hit]]] = z["pick_rank"][ok][hit]
    return rank


def main():
    t0 = time.time()
    d = X.load_ext()
    names = list(d["hero_names"])
    Ks, table = X.kernels()
    n_p, n_ph = C.experience_counts(d)
    r_adj = d["r"] - table[C.exp_bins(n_p, n_ph)]
    n_games, y, wp0, cnt, gday = X.game_arrays(d)
    g, team = d["g"], d["team"]
    print(f"loaded {time.time() - t0:.0f}s: {len(g):,} slots, {n_games:,} games", flush=True)

    # team structure per slot
    o = np.lexsort((team, g))
    gs, ts, hs = g[o], team[o], d["hero"][o]
    brk = np.flatnonzero(np.r_[True, (gs[1:] != gs[:-1]) | (ts[1:] != ts[:-1]), True])
    degen_slot = np.zeros(len(g), bool)
    S_team = np.zeros((n_games, 2, 3))
    full = np.zeros((n_games, 2), bool)
    for a, b in zip(brk[:-1], brk[1:]):
        if b - a != 5:
            continue
        s = struct_vec([names[h] for h in hs[a:b]])
        S_team[gs[a], ts[a]] = s
        full[gs[a], ts[a]] = True
        if max(s) > 0:
            degen_slot[o[a:b]] = True
    print(f"structure {time.time() - t0:.0f}s, degenerate slots {degen_slot.mean():.4f}", flush=True)

    query = (cnt[g] == 10) & ((d["post"] & (d["day"] > X.SNAP_LAST_DAY))
                              | (~d["post"] & (d["day"] >= Q_START)))
    skill = {}
    for var, add in (("all", np.ones(len(g), bool)), ("clean", ~degen_slot)):
        qi, m, v, Nh, Np, _ = X.online_predict(d, add, query, Ks[BEST], r_adj)
        mu = table[C.exp_bins(Np.astype(np.int64), Nh.astype(np.int64))]
        a = np.full(len(g), np.nan)
        a[qi] = mu + m
        skill[var] = a
        if var == "all":
            nh = np.full(len(g), np.nan)
            nh[qi] = Nh
            npl = np.full(len(g), np.nan)
            npl[qi] = Np
        print(f"skill {var} {time.time() - t0:.0f}s", flush=True)

    rank = pick_ranks(d, names)
    q = np.flatnonzero(query)
    qg = g[q]
    games = np.unique(qg)
    gi = np.searchsorted(games, qg)
    ng = len(games)
    out = {"g": games, "day": gday[games], "y": y[games], "wp0": wp0[games]}
    rid = np.zeros(ng, np.int64)
    rid[gi] = d["replay_id"][q]
    post = np.zeros(ng, bool)
    post[gi] = d["post"][q]
    out.update(rid=rid, post=post, s0=S_team[games, 0], s1=S_team[games, 1],
               full=full[games].all(1))
    for var in skill:
        T = np.zeros((ng, 2))
        np.add.at(T, (gi, team[q]), skill[var][q])
        out[f"skill_{var}"] = T
    # skill by within-team pick position, first-pick team, parties, experience
    pos = np.full((ng, 2, 5), np.nan)
    rk = rank[q].astype(np.int64)
    ok_rank = np.zeros(ng, bool)
    have = np.bincount(gi[rk >= 0], minlength=ng) == 10
    ok_rank[:] = have
    ordr = np.lexsort((rk, team[q], gi))
    qq, gg, tt = q[ordr], gi[ordr], team[q][ordr]
    within = np.zeros(len(qq), np.int64)
    first = np.r_[True, (gg[1:] != gg[:-1]) | (tt[1:] != tt[:-1])]
    st = np.flatnonzero(first)
    within = np.arange(len(qq)) - np.repeat(st, np.diff(np.r_[st, len(qq)]))
    sel = within < 5
    pos[gg[sel], tt[sel], within[sel]] = skill["all"][qq][sel]
    fp = np.full(ng, -1, np.int8)
    is0 = rk == 0
    fp[gi[is0]] = team[q][is0]
    out.update(skill_pos=pos, first_pick_team=fp, rank_ok=ok_rank)
    party = d["party"][q]
    psz = np.zeros((ng, 2))
    np.add.at(psz, (gi, team[q]), (party != 0).astype(float))
    out["partied"] = psz
    nhq = np.zeros((ng, 2))
    np.add.at(nhq, (gi, team[q]), (nh[q] == 0).astype(float))
    out["never_played_slots"] = nhq
    np.savez(OUT, **out)
    print(f"wrote {OUT}: {ng:,} games ({post.sum():,} post), rank ok {ok_rank.mean():.3f}, "
          f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
