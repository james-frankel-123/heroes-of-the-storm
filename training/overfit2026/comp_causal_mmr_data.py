"""
Composition audit, step 2 data, MMR version: the controls of comp_causal_data.py with the
per-player, per-hero skill replaced by the causal player MMR (personalization/cache/mmr_at_game.npz:
the latest Heroes Profile player rating parsed before the game started). No per-hero quantity
enters: the never-played-hero counts are dropped (set to zero) and so is the hero MMR.

Same query games and keys as comp_causal_games.npz (read from it); kept are games where all ten
players have a causal MMR. Per game (team0 orientation, same as comp_causal_games): team MMR sums
(stored as skill_all / skill_clean, both the same, per 1000 MMR), MMR by within-team pick position,
first-pick team, rank_ok, party counts, never_played_slots = 0.

Usage (from training/): nice -n 19 taskset -c 44-47 /usr/bin/python3 overfit2026/comp_causal_mmr_data.py
Output: overfit2026/cache/comp_causal_games_mmr.npz
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)
sys.path.insert(0, os.path.join(TRAINING_DIR, "personalization"))
from overfit2026 import data as _odata  # noqa: E402
from overfit2026.comp_causal_data import pick_ranks  # noqa: E402

PCACHE = os.path.join(TRAINING_DIR, "personalization", "cache")
SRC = _odata.art(HERE, "cache", "comp_causal_games.npz")
OUT = _odata.art(HERE, "cache", "comp_causal_games_mmr.npz")


def main():
    t0 = time.time()
    c = dict(np.load(SRC))
    e = np.load(os.path.join(PCACHE, "x_slots_ext.npz"))
    d = {k: e[k] for k in ("replay_id", "hero", "team", "party", "g")}
    names = list(e["hero_names"])
    mz = np.load(os.path.join(PCACHE, "mmr_at_game.npz"))
    assert list(mz["hero_names"]) == names
    print(f"loaded {time.time() - t0:.0f}s", flush=True)

    # slots of the query games, oriented by comp_causal's game index (its "g" is the ext game id)
    games = c["g"]
    sel = np.isin(d["g"], games)
    q = np.flatnonzero(sel)
    gi = np.searchsorted(games, d["g"][q])
    assert (games[gi] == d["g"][q]).all()
    team = d["team"][q].astype(np.int64)

    # causal player MMR per slot, joined on (replay_id, hero)
    kq = d["replay_id"][q] * 128 + d["hero"][q]
    km = mz["replay_ids"] * 128 + mz["hero"].astype(np.int64)
    om = np.argsort(km)
    ks = km[om]
    j = np.searchsorted(ks, kq)
    hit = (j < len(ks)) & (ks[np.minimum(j, len(ks) - 1)] == kq)
    mmr = np.full(len(q), np.nan)
    mmr[hit] = mz["player_mmr_causal"][om[j[hit]]]
    ng = len(games)
    nfin = np.bincount(gi[np.isfinite(mmr)], minlength=ng)
    nslot = np.bincount(gi, minlength=ng)
    keep = (nfin == 10) & (nslot == 10)
    print(f"slots {len(q):,}, MMR hit {np.isfinite(mmr).mean():.3f}; games with 10 MMRs {keep.sum():,} of {ng:,}",
          flush=True)

    m1k = np.nan_to_num(mmr) / 1000.0
    T = np.zeros((ng, 2))
    np.add.at(T, (gi, team), m1k)

    # MMR by within-team pick position (ranks from the same caches as comp_causal_data)
    full_d = {"replay_id": d["replay_id"], "hero": d["hero"], "pid": d["g"]}
    rank = pick_ranks(full_d, names)
    rk = rank[q].astype(np.int64)
    pos = np.full((ng, 2, 5), np.nan)
    have = np.bincount(gi[rk >= 0], minlength=ng) == 10
    ordr = np.lexsort((rk, team, gi))
    gg, tt, mm = gi[ordr], team[ordr], m1k[ordr]
    first = np.r_[True, (gg[1:] != gg[:-1]) | (tt[1:] != tt[:-1])]
    st = np.flatnonzero(first)
    within = np.arange(len(gg)) - np.repeat(st, np.diff(np.r_[st, len(gg)]))
    s5 = within < 5
    pos[gg[s5], tt[s5], within[s5]] = mm[s5]
    fp = np.full(ng, -1, np.int8)
    is0 = rk == 0
    fp[gi[is0]] = team[is0]

    party = d["party"][q]
    psz = np.zeros((ng, 2))
    np.add.at(psz, (gi, team), (party != 0).astype(float))

    out = {k: c[k][keep] for k in ("g", "day", "y", "wp0", "rid", "post", "s0", "s1", "full")}
    out.update(skill_all=T[keep], skill_clean=T[keep], skill_pos=pos[keep], first_pick_team=fp[keep],
               rank_ok=have[keep], partied=psz[keep], never_played_slots=np.zeros((int(keep.sum()), 2)))
    # sanity: the rebuilt party counts match comp_causal's on the kept games
    assert np.allclose(out["partied"], c["partied"][keep])
    np.savez(OUT, **out)
    print(f"wrote {OUT}: {keep.sum():,} games ({out['post'].sum():,} post), rank ok {have[keep].mean():.3f}, "
          f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
