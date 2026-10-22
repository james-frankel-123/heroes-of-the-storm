"""
At-game MMR features of a slot (shared by the carry-forward diagnostic
p2f_carry_mmr.py and the rich-CF MMR-adjusted expectation in p3_rcf.py).

MMR is the causal player rating (cache/mmr_at_game.npz, P3_MMR_AT_GAME): the
latest Heroes Profile stamp parsed before the game started. For a slot:
  dteam   own team's mean at-game player MMR minus the opponents' (per 100)
  own     the player's at-game MMR minus own team mean (per 100)
  hrel    the player's at-game hero MMR minus own player MMR (per 100), NaN
          where hero MMR is unknown
  full    both teams have all five players' MMR
These are the definitions of p2f_carry_mmr.py (section 11 of
P2_NARRATIVE_FACTS); tests/test_mmr_feats.py checks the two agree.
"""
import os

import numpy as np


def load_stamps(cache):
    Z = np.load(os.path.join(cache, "mmr_at_game.npz"))
    zk = (Z["region"].astype(np.int64) << 40) | Z["blizz_ids"]
    comp = Z["replay_ids"].astype(np.int64) * (1 << 44) + zk
    o = np.argsort(comp)
    return {"comp": comp[o], "pm": Z["player_mmr_causal"][o], "hm": Z["hero_mmr_causal"][o]}


def slot_features(rid, key, team, stamps):
    """rid, key ((region << 40) | blizz_id), team per slot; slots of a game
    may be in any order. Returns dict of dteam, own, hrel, full, pm."""
    rid = np.asarray(rid, np.int64)
    key = np.asarray(key, np.int64)
    team = np.asarray(team, np.int64)
    comp = stamps["comp"]
    qc = rid * (1 << 44) + key
    j = np.minimum(np.searchsorted(comp, qc), len(comp) - 1)
    ok = comp[j] == qc
    pm = np.where(ok, stamps["pm"][j], np.nan).astype(float)
    hm = np.where(ok, stamps["hm"][j], np.nan).astype(float)
    gk = rid * 2 + team
    ug, gi = np.unique(gk, return_inverse=True)
    good = np.isfinite(pm)
    s = np.bincount(gi, weights=np.where(good, pm, 0), minlength=len(ug))
    c = np.bincount(gi, weights=good.astype(float), minlength=len(ug))
    tm = s / np.maximum(c, 1)
    opp = np.searchsorted(ug, rid * 2 + (1 - team))
    opp_ok = (opp < len(ug)) & (ug[np.minimum(opp, len(ug) - 1)] == rid * 2 + (1 - team))
    opp = np.minimum(opp, len(ug) - 1)
    full = (c[gi] == 5) & (c[opp] == 5) & opp_ok
    dteam = (tm[gi] - tm[opp]) / 100
    own = (pm - tm[gi]) / 100
    hrel = (hm - pm) / 100
    return {"dteam": dteam, "own": own, "hrel": hrel, "full": full, "pm": pm}


def design(f):
    """Baseline covariates, zero where unknown: dteam and own (full games),
    hrel (where known), and indicators."""
    full = f["full"]
    hk = full & np.isfinite(f["hrel"])
    return np.column_stack([np.where(full, f["dteam"], 0.0), np.where(full, f["own"], 0.0),
                            np.where(hk, f["hrel"], 0.0), hk.astype(float)])
