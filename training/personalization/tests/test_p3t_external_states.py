"""P3 tournament: external player states (p3t_data.walk record / replay).
  E1 replay returns the external s, sd^2 and n reordered from shared.HEROES
     order to the slot table's hero order, for want_all and for the played
     hero (want_all=False)
  E2 the queries are checked: a replay file whose echoed player keys or
     as-of days differ from this run's queries is refused
  E3 the query fields: as_of_day = query day - 1, the evidence filters
     (none / odd days / even days outside the tournament window), the fit
     half, and the hero in shared.HEROES order
Run (a host with the project env): python3 personalization/tests/test_p3t_external_states.py"""
import os
import sys
import tempfile

TMP = tempfile.mkdtemp(prefix="p3t_ext_")
os.environ["P3_CACHE"] = os.path.join(TMP, "cache")
os.environ["P3_RESULTS"] = os.path.join(TMP, "results")
os.environ["P3T_WALK_REPLAY"] = os.path.join(TMP, "ext")
os.makedirs(os.environ["P3_CACHE"], exist_ok=True)
os.makedirs(os.environ["P3T_WALK_REPLAY"], exist_ok=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

import p3t_data as TD

H = 90


def main():
    rng = np.random.default_rng(5)
    sh = rng.permutation(H)                         # d index -> shared index
    keys = (np.array([1, 2, 1], np.int64) << 40) | np.array([11, 22, 33], np.int64)
    TD._KEYS["keys"], TD._KEYS["sh"] = keys, sh
    par = object()
    TD._HALF[id(par)] = 1
    sys.argv = ["p3t_data.py", "states"]
    q_pid = np.array([0, 2, 1, 2])
    q_day = np.array([20600, 20600, 20601, 20700])
    q_hero = np.array([3, 4, 5, 6])
    qs = TD._walk_queries(par, q_pid, q_day, TD.J0, q_hero, TD.SPLIT_J)
    assert (qs["as_of_day"] == q_day - 1).all() and (qs["half"] == 1).all()
    assert (qs["ev_parity"] == 2).all() and (qs["exclude_lo"] == TD.OOT_START).all() and (qs["exclude_hi"] == TD.LOBBY_END).all()
    assert (qs["hero"] == sh[q_hero]).all() and (qs["player_key"] == keys[q_pid]).all()
    assert (TD._walk_queries(par, q_pid, q_day, None, None, TD.SPLIT_D)["ev_parity"] == 1).all()
    print("E3 ok")
    S = rng.standard_normal((4, H)).astype(np.float32)      # external, shared order
    SD = rng.random((4, H)).astype(np.float32)
    N = rng.integers(0, 50, (4, H)).astype(np.float32)
    TD._CALLS[0] = 0
    np.savez(os.path.join(os.environ["P3T_WALK_REPLAY"], "p3t_data_states__0.npz"), s=S, sd=SD, n=N,
             **TD._walk_queries(par, q_pid, q_day, None, None, None))
    np.savez(os.path.join(os.environ["P3T_WALK_REPLAY"], "p3t_data_states__1.npz"), s=S, sd=SD, n=N,
             **TD._walk_queries(par, q_pid, q_day, None, q_hero, None))
    TD._CALLS[0] = 0
    r = TD.walk(None, par, None, None, q_pid, q_day)
    for i in range(H):
        assert np.allclose(r["mean_all"][:, i], S[:, sh[i]]) and np.allclose(r["var_all"][:, i], SD[:, sh[i]] ** 2)
        assert np.allclose(r["n_all"][:, i], N[:, sh[i]])
    r = TD.walk(None, par, None, None, q_pid, q_day, q_hero=q_hero, want_all=False)
    assert np.allclose(r["mean"], S[np.arange(4), sh[q_hero]])
    print("E1 ok")
    np.savez(os.path.join(os.environ["P3T_WALK_REPLAY"], "p3t_data_states__2.npz"), s=S, sd=SD, n=N,
             **TD._walk_queries(par, q_pid, q_day + 1, None, None, None))
    try:
        TD.walk(None, par, None, None, q_pid, q_day)
        raise AssertionError("E2: mismatched queries accepted")
    except SystemExit:
        pass
    print("E2 ok")
    print("ALL OK")


if __name__ == "__main__":
    main()
