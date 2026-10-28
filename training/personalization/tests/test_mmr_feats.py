"""p3_mmr_feats agrees with an exact tuple-keyed reference on real recent
slots (the shared MMR implementation), only uses stamps of the scored game
itself (as-of by construction of mmr_at_game.npz), and has no key collisions.
Run: python3 personalization/tests/test_mmr_feats.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

os.environ["P3_MMR_KEY_FIX"] = "1"      # the tests cover the fixed loader
import p3_mmr_feats as MF

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ref_inline(rid, key, team, Z):
    """Reference by exact (game, player) tuple lookup (no packed integer key);
    a key present twice with conflicting stamps is unknown."""
    zr = np.asarray(Z["replay_ids"])
    sel = np.flatnonzero(np.isin(zr, np.unique(rid)))
    zr = zr[sel].tolist()
    zk = ((np.asarray(Z["region"])[sel].astype(np.int64) << 40) | np.asarray(Z["blizz_ids"])[sel]).tolist()
    zp = np.asarray(Z["player_mmr_causal"])[sel].tolist()
    zh = np.asarray(Z["hero_mmr_causal"])[sel].tolist()
    want = set(zip(rid.tolist(), key.tolist()))
    tab = {}
    for r_, k_, p_, h_ in zip(zr, zk, zp, zh):
        if (r_, k_) in want:
            tab.setdefault((r_, k_), []).append((p_, h_))

    def one(vals, j):
        xs = [v[j] for v in vals]
        if all(np.isnan(x) for x in xs):
            return np.nan
        if any(np.isnan(x) for x in xs) or len(set(xs)) > 1:
            return np.nan
        return xs[0]
    pm = np.array([one(tab[(r, k)], 0) if (r, k) in tab else np.nan for r, k in zip(rid.tolist(), key.tolist())])
    hm = np.array([one(tab[(r, k)], 1) if (r, k) in tab else np.nan for r, k in zip(rid.tolist(), key.tolist())])
    gk = rid * 2 + team
    ug, gi = np.unique(gk, return_inverse=True)
    good = np.isfinite(pm)
    s = np.bincount(gi, weights=np.where(good, pm, 0), minlength=len(ug))
    c = np.bincount(gi, weights=good.astype(float), minlength=len(ug))
    tm = s / np.maximum(c, 1)
    opp = np.searchsorted(ug, rid * 2 + (1 - team))
    full = (c[gi] == 5) & (c[opp] == 5)
    return (tm[gi] - tm[opp]) / 100, (pm - tm[gi]) / 100, (hm - pm) / 100, full


def test_collision(tmp=None):
    """Two games of one player 2^20 replay ids apart (the old packed key's
    collision) each get their own stamp; a conflicting duplicate is unknown."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        rid = np.array([62413700, 62413700 + (1 << 20), 70000000, 70000000], np.int64)
        np.savez(os.path.join(td, "mmr_at_game.npz"), replay_ids=rid, region=np.array([1, 1, 2, 2]),
                 blizz_ids=np.array([989591, 989591, 5, 5]),
                 player_mmr_causal=np.array([2000.0, 2500.0, 1800.0, 1900.0], np.float32),
                 hero_mmr_causal=np.array([1.0, 2.0, 3.0, 3.0], np.float32))
        st = MF.load_stamps(td)
        key1 = (1 << 40) | 989591
        f = MF.slot_features(rid[:2], np.array([key1, key1]), np.array([0, 0]), st)
        assert f["pm"].tolist() == [2000.0, 2500.0], f["pm"]
        f = MF.slot_features(rid[2:3], np.array([(2 << 40) | 5]), np.array([0]), st)
        assert np.isnan(f["pm"][0]), "conflicting duplicate must be unknown"
        assert st["n_dup_keys"] == 1


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
    a = ref_inline(rid, key, team, Z)
    f = MF.slot_features(rid, key, team, MF.load_stamps(cache))
    for nm, x, y in zip(("dteam", "own", "hrel"), a[:3], (f["dteam"], f["own"], f["hrel"])):
        assert np.allclose(x, y, equal_nan=True), nm
    assert np.array_equal(a[3], f["full"])
    print(f"  {len(rid):,} slots, full share {f['full'].mean():.3f}")


if __name__ == "__main__":
    test_collision()
    print("test_collision: pass")
    test_agree()
    print("test_agree: pass")
