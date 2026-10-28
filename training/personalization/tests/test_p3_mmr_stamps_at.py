"""p3_mmr_feats.stamps_at: collision-free (game, player) lookup of causal MMR.
  S1 two games of one player 2^20 replay ids apart (which collided under rid * 2^44 + key) return
     their own values; a missing (game, player) is NaN; conflicting duplicate stamps are NaN
  S2 the legacy packed key does collide on the same data (the bug this replaces)
Run: python3 personalization/tests/test_p3_mmr_stamps_at.py"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

import p3_mmr_feats as MF


def main():
    d = tempfile.mkdtemp(prefix="mmr_stamps_")
    r0 = 63_000_000
    rid = np.array([r0, r0 + (1 << 20), r0 + 5, r0 + 7, r0 + 7], np.int64)
    blz = np.array([42, 42, 42, 9, 9], np.int64)
    reg = np.array([1, 1, 2, 1, 1], np.int16)
    pm = np.array([2000.0, 2500.0, 1800.0, 1500.0, 1600.0], np.float32)
    np.savez(os.path.join(d, "mmr_at_game.npz"), replay_ids=rid, blizz_ids=blz, region=reg,
             player_mmr_causal=pm, hero_mmr_causal=pm + 1)
    key = (reg.astype(np.int64) << 40) | blz
    q_rid = np.array([r0, r0 + (1 << 20), r0 + 5, r0 + 7, r0 + 1], np.int64)
    q_key = np.array([key[0], key[1], key[2], key[3], key[0]])
    a, h = MF.stamps_at(d, q_rid, q_key)
    assert a[0] == 2000 and a[1] == 2500 and a[2] == 1800, a
    assert np.isnan(a[3]) and np.isnan(a[4]), a          # conflicting duplicate; absent pair
    assert h[0] == 2001
    print("S1 ok")
    comp = rid * (1 << 44) + key
    assert comp[0] == comp[1], "legacy key no longer collides?"
    print("S2 ok")
    print("ALL OK")


if __name__ == "__main__":
    main()
