"""P3 tournament: p3t_data.load_model against the current p3_rcf API.
  L1 load_model reads the scoreboard card through p3_rcf.aug_metrics with the
     slot table d (no precomputed aug["bl"]): the call works and every aug row
     gets the metrics of its own (replay_id, blizz_id), rows without a card
     row stay NaN.
  L2 the player key used for the join is the blizz part of d["player_keys"]
     (region << 40 | blizz_id): a row whose region differs still joins.
  L3 (real data, when the slot table is present) p3_rcf.load_aug keeps every
     slot-table field p3t_data reads (REQUIRED below).
Guards the interface between the tournament (p3t_data) and the skill-model
code (p3_rcf): a signature change on either side fails here, not in a run.
Run (a host with the project env): python3 personalization/tests/test_p3t_load_model.py"""
import os
import sys
import tempfile
import types

TMP = tempfile.mkdtemp(prefix="p3t_lm_")
os.environ["P3_CACHE"] = os.path.join(TMP, "cache")
os.environ["P3_RESULTS"] = os.path.join(TMP, "results")
os.makedirs(os.environ["P3_CACHE"], exist_ok=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

import p3_rcf as R
import p3t_data as TD


def fixture():
    rng = np.random.default_rng(3)
    n_aug, M = 12, 4
    blizz = np.array([101, 202, 303], np.int64)
    region = np.array([1, 2, 2], np.int64)
    keys = (region << 40) | blizz
    from shared import HEROES
    d = {"player_keys": keys, "hero_names": np.array(HEROES[::-1])}
    aug = {"day": np.arange(n_aug, dtype=np.int32), "pid": rng.integers(0, 3, n_aug).astype(np.int32),
           "rid": (5000 + np.arange(n_aug)).astype(np.int64), "slot": np.where(np.arange(n_aug) < 6, np.arange(n_aug), -1)}
    # card rows for every second aug row, in shuffled order
    have = np.arange(0, n_aug, 2)
    Z = rng.standard_normal((len(have), M)).astype(np.float16)
    W = (0.5 + rng.random(len(have))).astype(np.float32)
    o = rng.permutation(len(have))
    card = {"replay_id": aug["rid"][have][o], "blizz_id": blizz[aug["pid"][have]][o], "Z": Z[o], "W": W[o]}
    return d, aug, card, have, Z, W


def main():
    d, aug, card, have, Z, W = fixture()
    os.makedirs(R.RCF_CACHE, exist_ok=True)
    np.savez(os.path.join(R.RCF_CACHE, "sb_fx.npz"), **card)
    par = types.SimpleNamespace(G=np.zeros((Z.shape[1], 90, 1)), meta={"sb": "fx"})
    R.get_params = lambda tag: par
    R.load_aug = lambda hist: (d, dict(aug))
    os.environ.pop("P3_RCF_WRITE_AUG", None)
    p, d2, aug2, mets, mw = TD.load_model()
    assert p is par and d2 is d
    assert mets.shape == (len(aug["day"]), Z.shape[1]) and mw.shape == (len(aug["day"]),)
    got = np.zeros(len(aug["day"]), bool)
    got[have] = True
    assert np.array_equal(mets[have], Z), "L1: card rows joined to the wrong aug rows"
    assert np.isnan(mets[~got]).all(), "L1: rows without a card row must stay NaN"
    assert np.allclose(mw[have], W) and (mw[~got] == 1).all()
    print("L1 ok: load_model -> aug_metrics(aug, card, d) joins on (replay_id, blizz_id)")
    # L2: region bits never enter the join
    assert (d["player_keys"] >> 40).max() > 1
    print("L2 ok: join uses the blizz part of player_keys")
    real_l3()
    print("ALL OK")


REQUIRED = ("pid", "player_keys", "hero", "day", "y", "wp", "r", "v", "post", "version", "versions", "hero_names",
            "in_sample", "g", "team", "replay_id", "n_players")


def real_l3():
    """Run only where the real slot table exists (the workers): load_aug on real data."""
    real_cache = os.environ.get("P3T_TEST_REAL_CACHE")
    if not real_cache:
        print("L3 skipped (set P3T_TEST_REAL_CACHE to a cache_full with x_slots_ext.npz and merged rows)")
        return
    import importlib
    os.environ["P3_CACHE"] = real_cache
    import p3_hs_core as Cm
    import p3_x_common as Xm
    importlib.reload(Cm)
    importlib.reload(Xm)
    importlib.reload(R)
    d = Xm.load_ext()
    missing = [k for k in REQUIRED if k not in d]
    assert not missing, f"L3: slot table lacks {missing}"
    d2, aug = R.load_aug("merged")
    missing = [k for k in REQUIRED if k not in d2]
    assert not missing, f"L3: p3_rcf.load_aug drops fields p3t_data reads: {missing}"
    for k in ("day", "pid", "hero", "slot", "r_adj", "v", "table", "rid", "n_players"):
        assert k in aug, f"L3: aug lacks {k}"
    print("L3 ok: load_aug keeps every field p3t_data reads")


if __name__ == "__main__":
    main()
