"""p3_mmr_feats agrees with the inline computation of p2f_carry_mmr.py on
real V2 / OOT slots (the shared MMR implementation), and only uses stamps of
the scored game itself (as-of by construction of mmr_at_game.npz).
Run: python3 personalization/tests/test_mmr_feats.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

import p3_mmr_feats as MF

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def p2f_inline(rid, key, team, Z):
    zk = (Z["region"].astype(np.int64) << 40) | Z["blizz_ids"]
    comp = Z["replay_ids"].astype(np.int64) * (1 << 44) + zk
    o = np.argsort(comp)
    comp = comp[o]
    qc = rid.astype(np.int64) * (1 << 44) + key
    j = np.minimum(np.searchsorted(comp, qc), len(comp) - 1)
    ok = comp[j] == qc
    pm = np.where(ok, Z["player_mmr_causal"][o][j], np.nan).astype(float)
    hm = np.where(ok, Z["hero_mmr_causal"][o][j], np.nan).astype(float)
    gk = rid * 2 + team
    ug, gi = np.unique(gk, return_inverse=True)
    good = np.isfinite(pm)
    s = np.bincount(gi, weights=np.where(good, pm, 0), minlength=len(ug))
    c = np.bincount(gi, weights=good.astype(float), minlength=len(ug))
    tm = s / np.maximum(c, 1)
    opp = np.searchsorted(ug, rid * 2 + (1 - team))
    full = (c[gi] == 5) & (c[opp] == 5)
    return (tm[gi] - tm[opp]) / 100, (pm - tm[gi]) / 100, (hm - pm) / 100, full


def test_agree():
    cache = os.environ.get("P3_CACHE") or os.path.join(HERE, "cache")
    z = np.load(os.path.join(cache, "x_slots_ext.npz"))
    rng = np.random.default_rng(0)
    g = np.unique(z["g"][z["day"] >= z["day"].max() - 60])
    g = rng.choice(g, min(20000, len(g)), replace=False)
    m = np.isin(z["g"], g)
    rid = z["replay_id"][m].astype(np.int64)
    key = z["player_keys"][z["pid"][m]].astype(np.int64)
    team = z["team"][m].astype(np.int64)
    Z = np.load(os.path.join(cache, "mmr_at_game.npz"))
    a = p2f_inline(rid, key, team, Z)
    f = MF.slot_features(rid, key, team, MF.load_stamps(cache))
    for nm, x, y in zip(("dteam", "own", "hrel"), a[:3], (f["dteam"], f["own"], f["hrel"])):
        assert np.allclose(x, y, equal_nan=True), nm
    assert np.array_equal(a[3], f["full"])
    print(f"  {len(rid):,} slots, full share {f['full'].mean():.3f}")


if __name__ == "__main__":
    test_agree()
    print("test_agree: pass")
