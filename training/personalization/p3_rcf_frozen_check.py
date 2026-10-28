"""
Regression check of the frozen-coefficient mode (P3_RCF_FROZEN): an arm scored
with every V1-fitted piece loaded from a saved arm must reproduce that arm.

  python3 personalization/p3_rcf_frozen_check.py <reference arm npz> <frozen arm npz> [--windows V1,V2,OOT]
      [--dup-stamps <mmr_at_game.npz>]

Compares, on the games and slots both files hold (by replay id and player key),
the slot predictions s_pred, the combiner inputs g_D and the game
probabilities g_p, and the saved coefficients. Exits 1 on any difference above
1e-6 (the two runs may use different BLAS builds: ~1e-8 rounding). With
--dup-stamps, games holding a (game, player) key that appears more than once
in the MMR stamps table are reported separately: the definitive build picked
one of the duplicates through an unstable sort (numpy-build dependent); the
stamps loader now treats conflicting duplicates as unknown.
"""
import json
import sys

import numpy as np

WIN = {"V1": 1, "V2": 2, "OOT": 3, "PF": 4}


def main():
    ref, frz = np.load(sys.argv[1]), np.load(sys.argv[2])
    wins = sys.argv[sys.argv.index("--windows") + 1].split(",") if "--windows" in sys.argv else ["V1", "V2", "OOT"]
    ws = [WIN[w] for w in wins]
    bad = []
    # games
    gr = np.isin(ref["g_win"], ws)
    gf = np.isin(frz["g_win"], ws)
    rr, fr = ref["g_replay_id"][gr], frz["g_replay_id"][gf]
    common, ir, jf = np.intersect1d(rr, fr, return_indices=True)
    print(f"games: reference {len(rr):,}, frozen {len(fr):,}, common {len(common):,}")
    dupg = np.zeros(len(common), bool)
    if "--dup-stamps" in sys.argv:
        Z = np.load(sys.argv[sys.argv.index("--dup-stamps") + 1])
        comp = Z["replay_ids"].astype(np.int64) * (1 << 44) + ((Z["region"].astype(np.int64) << 40) | Z["blizz_ids"])
        u, c = np.unique(comp, return_counts=True)
        dup_rid = np.unique(u[c > 1] >> 44)
        dupg = np.isin(common, dup_rid)
        print(f"games with a duplicated stamps key: {dupg.sum():,} (reported separately)")
    if len(common) != len(rr):
        bad.append(f"{len(rr) - len(common)} reference games missing from the frozen arm")
    for k in ("g_y", "g_lo", "g_D", "g_p", "g_p0"):
        a = np.asarray(ref[k][gr][ir], np.float64)
        b = np.asarray(frz[k][gf][jf], np.float64)
        dd = np.abs(a - b)
        dm = float(dd[~dupg].max()) if (~dupg).any() else 0.0
        extra = f"; duplicate-key games max {dd[dupg].max():.3e}, {int((dd[dupg] > 1e-6).sum())} above 1e-6" if dupg.any() else ""
        print(f"  {k}: max |diff| {dm:.3e}{extra}")
        if dm > 1e-6:
            bad.append(f"{k} differs by {dm:.3e}")
    # slots
    sr = np.isin(ref["s_win"], ws)
    sf = np.isin(frz["s_win"], ws)
    key = lambda z, m: np.char.add(z["s_replay_id"][m].astype(str), np.char.add(":", z["s_key"][m].astype(str)))
    a_k, b_k = key(ref, sr), key(frz, sf)
    common_s, i2, j2 = np.intersect1d(a_k, b_k, return_indices=True)
    print(f"slots: reference {len(a_k):,}, common {len(common_s):,}")
    if len(common_s) != len(a_k):
        bad.append(f"{len(a_k) - len(common_s)} reference slots missing")
    for k in ("s_pred", "s_mean", "s_var", "s_mu"):
        a = np.asarray(ref[k][sr][i2], np.float64)
        b = np.asarray(frz[k][sf][j2], np.float64)
        dm = float(np.abs(a - b).max())
        print(f"  {k}: max |diff| {dm:.3e}")
        if dm > 1e-6:
            bad.append(f"{k} differs by {dm:.3e}")
    for k in ("coef", "coef0"):
        dm = float(np.abs(np.asarray(ref[k]) - np.asarray(frz[k])).max())
        print(f"  {k}: max |diff| {dm:.3e}")
        if dm > 0:
            bad.append(f"{k} differs")
    if bad:
        print("FROZEN CHECK FAILED: " + "; ".join(bad))
        sys.exit(1)
    print("frozen check passed")


if __name__ == "__main__":
    main()
