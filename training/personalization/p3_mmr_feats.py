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


def _pair(rid, key, ukeys):
    """Collision-free (game, player) key: rid * 2^22 + dense player index.
    (The former rid * 2^44 + key overflowed int64 for replay ids >= 2^19 and
    kept only the low 20 bits of the replay id, so two games of the same player
    1,048,576 ids apart shared a key.) Players not in ukeys get -1."""
    rid = np.asarray(rid, np.int64)
    key = np.asarray(key, np.int64)
    j = np.minimum(np.searchsorted(ukeys, key), len(ukeys) - 1)
    ok = ukeys[j] == key
    assert len(ukeys) < (1 << 22) and (rid.max(initial=0) < (1 << 40))
    return np.where(ok, rid * (1 << 22) + j, -1)


def load_stamps(cache):
    if os.environ.get("P3_MMR_KEY_FIX") != "1":
        return _load_stamps_legacy(cache)
    Z = np.load(os.path.join(cache, "mmr_at_game.npz"))
    zk = (Z["region"].astype(np.int64) << 40) | Z["blizz_ids"]
    ukeys = np.unique(zk)
    comp = _pair(Z["replay_ids"], zk, ukeys)
    # one value per (game, player); a genuine duplicate row keeps its value when
    # the copies agree and becomes unknown (NaN) when they conflict
    o = np.argsort(comp, kind="stable")
    comp = comp[o]
    st = np.flatnonzero(np.r_[True, comp[1:] != comp[:-1]])
    size = np.diff(np.r_[st, len(comp)])
    out = {"comp": comp[st], "ukeys": ukeys, "n_dup_keys": int((size > 1).sum())}
    for name, col in (("pm", "player_mmr_causal"), ("hm", "hero_mmr_causal")):
        x = Z[col][o].astype(np.float64)
        nan = np.isnan(x)
        lo = np.minimum.reduceat(np.where(nan, np.inf, x), st)
        hi = np.maximum.reduceat(np.where(nan, -np.inf, x), st)
        n_nan = np.add.reduceat(nan.astype(np.int64), st)
        agree = (n_nan == size) | ((n_nan == 0) & (lo == hi))
        v = x[st].copy()
        v[~agree] = np.nan
        out[name] = v.astype(Z[col].dtype)
    return out


def _load_stamps_legacy(cache):
    """The def1027 behaviour (default until the next model version): packed key
    rid * 2^44 + key, which overflows int64 and collides for games of one
    player 2^20 replay ids apart; the unstable sort picks one colliding stamp
    depending on the numpy build. P3_MMR_KEY_FIX=1 selects load_stamps' fix."""
    Z = np.load(os.path.join(cache, "mmr_at_game.npz"))
    zk = (Z["region"].astype(np.int64) << 40) | Z["blizz_ids"]
    comp = Z["replay_ids"].astype(np.int64) * (1 << 44) + zk
    o = np.argsort(comp)
    return {"comp": comp[o], "pm": Z["player_mmr_causal"][o], "hm": Z["hero_mmr_causal"][o], "legacy": True}


def stamps_at(cache, rid, key):
    """Causal player and hero MMR per (game, player) with the collision-free key, whatever
    P3_MMR_KEY_FIX says (for scripts that used the packed rid * 2^44 + key inline): (pm, hm),
    NaN where unknown or where duplicate stamps of one (game, player) conflict."""
    prev = os.environ.get("P3_MMR_KEY_FIX")
    os.environ["P3_MMR_KEY_FIX"] = "1"
    try:
        st = load_stamps(cache)
    finally:
        if prev is None:
            os.environ.pop("P3_MMR_KEY_FIX")
        else:
            os.environ["P3_MMR_KEY_FIX"] = prev
    qc = _pair(rid, key, st["ukeys"])
    j = np.minimum(np.searchsorted(st["comp"], qc), len(st["comp"]) - 1)
    ok = (qc >= 0) & (st["comp"][j] == qc)
    return np.where(ok, st["pm"][j], np.nan).astype(float), np.where(ok, st["hm"][j], np.nan).astype(float)


def slot_features(rid, key, team, stamps):
    """rid, key ((region << 40) | blizz_id), team per slot; slots of a game
    may be in any order. Returns dict of dteam, own, hrel, full, pm."""
    rid = np.asarray(rid, np.int64)
    key = np.asarray(key, np.int64)
    team = np.asarray(team, np.int64)
    comp = stamps["comp"]
    if stamps.get("legacy"):
        qc = rid * (1 << 44) + key
        j = np.minimum(np.searchsorted(comp, qc), len(comp) - 1)
        ok = comp[j] == qc
    else:
        qc = _pair(rid, key, stamps["ukeys"])
        j = np.minimum(np.searchsorted(comp, qc), len(comp) - 1)
        ok = (qc >= 0) & (comp[j] == qc)
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
