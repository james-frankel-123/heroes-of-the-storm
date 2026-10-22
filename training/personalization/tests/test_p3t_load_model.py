"""P3 tournament: p3t_data.load_model against the current p3_rcf API.
  L1 load_model reads the scoreboard card through p3_rcf.aug_metrics with the
     slot table d (no precomputed aug["bl"]): the call works and every aug row
     gets the metrics of its own (replay_id, blizz_id), rows without a card
     row stay NaN.
  L2 the player key used for the join is the blizz part of d["player_keys"]
     (region << 40 | blizz_id): a row whose region differs still joins.
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
    print("ALL OK")


if __name__ == "__main__":
    main()
