"""Assign-mode reference (p3_assign): brute force vs an independent search and
hand cases. Run: python3 personalization/tests/test_assign.py (or pytest)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from p3_assign import DRAFT_TEAM, IS_PICK, assignment_terms, best_assignment

NH = 90


def _dfs_best(c):
    """Independent optimum: depth-first over injective pick -> slot maps.
    Returns (value, slot of each pick)."""
    n = c.shape[0]
    best = [-np.inf, None]

    def go(j, used, acc, perm):
        if j == n:
            if acc > best[0] + 1e-12:
                best[0], best[1] = acc, tuple(perm)
            return
        for k in range(5):
            if not used & (1 << k):
                go(j + 1, used | (1 << k), acc + c[j, k], perm + [k])

    go(0, 0, 0.0, [])
    return best


def _cost(s5, off5, h, b2, b3):
    return np.array([[b2 * s5[k, x] + b3 * off5[k, x] for k in range(5)] for x in h])


def test_swap():
    s = np.zeros((5, NH))
    s[0, 11], s[1, 10] = 1.0, 1.0      # slot 0 plays hero 11 well, slot 1 hero 10
    S, O, perm = best_assignment(s, np.zeros((5, NH)), [10, 11, 12, 13, 14], 1.0, -0.5)
    assert perm[:2] == (1, 0) and S == 2.0 and O == 0.0   # slot 0 plays pick 1 (hero 11)


def test_off_role_penalized():
    s = np.zeros((5, NH))
    off = np.zeros((5, NH))
    off[0, 20] = 1.0                   # hero 20 is off-role for slot 0 only
    S, O, perm = best_assignment(s, off, [20, 21, 22, 23, 24], 0.3, -0.7)
    assert O == 0.0 and perm[0] != 0
    assert perm == (1, 0, 2, 3, 4)     # first such permutation: slot 0 plays pick 1


def test_tie_identity_first():
    z = np.zeros((5, NH))
    assert best_assignment(z, z, [1, 2, 3, 4, 5], 0.3, -0.7)[2] == (0, 1, 2, 3, 4)
    s = np.zeros((5, NH))
    s[2, 7] = s[3, 7] = 1.0            # pick 0 (hero 7) equally good in slots 2 and 3
    assert best_assignment(s, z, [7, 8, 9, 10, 11], 1.0, 0.0)[2] == (1, 2, 0, 3, 4)


def test_partial_team():
    rng = np.random.RandomState(1)
    s5, off5 = rng.randn(5, NH), (rng.rand(5, NH) < 0.3).astype(float)
    h = [3, 40, 77]
    S, O, perm = best_assignment(s5, off5, h, 0.4, -0.6)
    val, p = _dfs_best(_cost(s5, off5, h, 0.4, -0.6))
    assert abs(0.4 * S - 0.6 * O - val) < 1e-12 and tuple(perm.index(j) for j in range(3)) == p
    assert best_assignment(s5, off5, [], 0.4, -0.6) == (0.0, 0.0, (0, 1, 2, 3, 4))


def test_random_vs_independent():
    rng = np.random.RandomState(0)
    for _ in range(300):
        s5 = rng.randn(5, NH)
        off5 = (rng.rand(5, NH) < 0.4).astype(float)
        h = list(rng.choice(NH, 5, replace=False))
        b2, b3 = rng.rand() + 0.05, -rng.rand() - 0.05
        S, O, perm = best_assignment(s5, off5, h, b2, b3)
        val, p = _dfs_best(_cost(s5, off5, h, b2, b3))
        assert abs(b2 * S + b3 * O - val) < 1e-9 and tuple(perm.index(j) for j in range(5)) == p
        assert abs(S - sum(s5[i, h[perm[i]]] for i in range(5))) < 1e-12
        assert O == sum(off5[i, h[perm[i]]] for i in range(5))
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        return
    for _ in range(200):
        s5, off5 = rng.randn(5, NH), (rng.rand(5, NH) < 0.4).astype(float)
        h = list(rng.choice(NH, 5, replace=False))
        c = _cost(s5, off5, h, 0.5, -0.8)
        r, k = linear_sum_assignment(c, maximize=True)
        S, O, _ = best_assignment(s5, off5, h, 0.5, -0.8)
        assert abs(0.5 * S - 0.8 * O - c[r, k].sum()) < 1e-9


def test_draft_terms():
    rng = np.random.RandomState(2)
    s, off = rng.randn(10, NH), (rng.rand(10, NH) < 0.3).astype(float)
    acts = list(rng.choice(NH, 16, replace=False))
    S, O, slots = assignment_terms(s, off, acts, 0.5, -0.8)
    for t in (0, 1):
        h = [acts[k] for k in range(16) if IS_PICK[k] and DRAFT_TEAM[k] == t]
        assert sorted(slots[t]) == list(range(5 * t, 5 * t + 5))
        assert abs(S[t] - sum(s[sl, x] for sl, x in zip(slots[t], h))) < 1e-12
        assert abs(O[t] - sum(off[sl, x] for sl, x in zip(slots[t], h))) < 1e-12
        val, _ = _dfs_best(_cost(s[5 * t:5 * t + 5], off[5 * t:5 * t + 5], h, 0.5, -0.8))
        assert abs(0.5 * S[t] - 0.8 * O[t] - val) < 1e-9
        # never worse than the slot-mode (identity) assignment
        assert 0.5 * S[t] - 0.8 * O[t] >= sum(0.5 * s[5 * t + j, x] - 0.8 * off[5 * t + j, x]
                                               for j, x in enumerate(h)) - 1e-12


def test_matches_one_step_drafter():
    """Same objective and tie-break as p3_dr_drafter._assign_terms (copied
    rule: player i takes hero perm[i], first argmax over permutations)."""
    perms = np.array(list(__import__("itertools").permutations(range(5))), np.int64)
    rng = np.random.RandomState(3)
    for trial in range(200):
        s5 = np.round(rng.randn(5, NH), 1) if trial % 2 else np.zeros((5, NH))
        off5 = (rng.rand(5, NH) < 0.4).astype(float)
        h = np.array(rng.choice(NH, 5, replace=False))
        i = np.arange(5)[None, :]
        sp = s5[i, h[perms]].sum(-1)
        op = off5[i, h[perms]].sum(-1)
        k = int(np.argmax(0.5 * sp - 0.8 * op))
        S, O, perm = best_assignment(s5, off5, h, 0.5, -0.8)
        assert perm == tuple(perms[k]) and abs(S - sp[k]) < 1e-12 and O == op[k]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
