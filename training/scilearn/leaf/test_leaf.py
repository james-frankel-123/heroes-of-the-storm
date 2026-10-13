"""Plain-python tests for the additive MCTS leaf.

  python3 training/scilearn/leaf/test_leaf.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from leaf import ROLE_POW, LeafModel, interp_shape, role_code, sigmoid  # noqa: E402
from make_synthetic import EXPLOIT_TANK_BONUS, make  # noqa: E402
import o6  # noqa: E402

BENIGN = make("benign", seed=1)
EXPLOIT = make("exploit", seed=1)
JUDGE = make("benign", seed=2)


def rand_drafts(m, n, seed=0):
    rng = np.random.default_rng(seed)
    return np.stack([rng.choice(m.H, 10, replace=False) for _ in range(n)])


def test_antisymmetry():
    for m in (BENIGN, EXPLOIT, JUDGE):
        D = rand_drafts(m, 500)
        mp = np.random.default_rng(1).integers(0, m.M, 500)
        tp = np.random.default_rng(2).integers(0, m.T, 500)
        e1, s1, _ = m.evaluate(D, mp, tp)
        e2, s2, _ = m.evaluate(np.concatenate([D[:, 5:], D[:, :5]], 1), mp, tp)
        assert np.abs(e1 + e2).max() < 1e-8
        assert np.abs(s1 - s2).max() < 1e-12
        p1, p2 = sigmoid(e1), sigmoid(e2)
        assert np.abs(p1 + p2 - 1).max() < 1e-12


def test_contributions_and_reference():
    for m in (BENIGN, EXPLOIT):
        rep = m.export_check(n=300, n_ref=40)
        assert rep["ok"], rep
        D = rand_drafts(m, 50, seed=3)
        eta, _, c = m.evaluate(D, "Dragon Shire", "high")
        assert np.abs(sum(c.values()) - eta).max() < 1e-10
        ref = [m.reference_eta(d, m.map_idx["Dragon Shire"], m.tier_idx["high"]) for d in D]
        assert np.abs(np.array(ref) - eta).max() < 1e-10
        assert "role_table" in c and "cross_kit" in c and any(k.startswith("rule:") for k in c)


def test_shape_interpolation():
    k, v = np.array([0.0, 1.0, 3.0]), np.array([0.0, 2.0, 1.0])
    x = np.array([-5.0, 0.0, 0.5, 1.0, 2.0, 3.0, 10.0])
    assert np.allclose(interp_shape(k, v, x), [0.0, 0.0, 1.0, 2.0, 1.5, 1.0, 1.0])
    # through the model: a pack whose only nonzero term is one shape
    m = _zero_pack()
    m.a["shape_knots"][0] = [0, 1, 2, 3, 4, 5]
    m.a["shape_vals"][0] = [0, 1, 1.5, 1.75, 1.8, 1.8]
    m.a["kit"][:, 0] = 0.0
    m.a["kit"][:5, 0] = 0.3      # team A sum 1.5 -> 1.25, team B sum 0 -> 0
    eta, _, c = m.evaluate(np.arange(10)[None], 0, 0)
    assert abs(c["shape"][0] - 1.25) < 1e-12 and abs(eta[0] - 1.25) < 1e-12


def test_role_code():
    assert role_code([0, 0, 0, 0, 0, 0]) == 0
    assert role_code([5, 0, 0, 0, 0, 0]) == 5                   # five tanks
    assert role_code([1, 1, 1, 0, 2, 0]) == 1 + 6 + 36 + 2 * 6 ** 4
    assert role_code([0, 0, 0, 0, 0, 5]) == 5 * 6 ** 5
    m = BENIGN
    tanks = m.hero_ids(o6.DEGEN_COMPS["5 tanks"])
    assert (m.role_onehot[tanks].sum(0) == [5, 0, 0, 0, 0, 0]).all()
    # the exploit entry is reached by the five-tank team only
    z = _zero_pack()
    z.a["role_table"][5 * ROLE_POW[0]] = 1.0
    eta, _, c = z.evaluate(np.array(z.hero_ids(o6.DEGEN_COMPS["5 tanks"] + o6.STANDARD))[None], 0, 0)
    assert abs(c["role_table"][0] - 1.0) < 1e-12


def test_rules():
    z = _zero_pack()
    z.rules = [
        {"id": "no_healer", "beta": 0.1, "predicates": [
            {"field": "role_count", "key": "Healer", "op": "==", "value": 0}]},
        {"id": "wave_on_map", "beta": -0.2, "predicates": [
            {"field": "kit_sum", "key": 0, "op": ">=", "value": 1.0},
            {"field": "map", "key": None, "op": "==", "value": z.maps[2]}]},
        {"id": "two_tanks_max", "beta": 0.05, "predicates": [
            {"field": "role_count", "key": "Tank", "op": "<=", "value": 2}]},
    ]
    z._compile_rules()
    z.a["kit"][:, 0] = 0.25
    A = z.hero_ids(o6.DEGEN_COMPS["No healer high WR"])      # 2 tanks, no healer, kit 1.25
    B = z.hero_ids(["Diablo", "Rehgar", "Raynor", "Sonya", "Jaina"])   # 1 tank, healer, kit 1.25
    D = np.array(A + B)[None]
    _, _, c = z.evaluate(D, 2, 0)
    assert abs(c["rule:no_healer"][0] - 0.1) < 1e-12           # fires for A only
    assert abs(c["rule:wave_on_map"][0]) < 1e-12               # fires for both, cancels
    assert abs(c["rule:two_tanks_max"][0]) < 1e-12             # both
    _, _, c = z.evaluate(D, 3, 0)
    assert abs(c["rule:wave_on_map"][0]) < 1e-12               # neither (wrong map)
    z.a["kit"][B, 0] = 0.1                                     # B sum 0.5 < 1.0
    eta, _, c = z.evaluate(D, 2, 0)
    assert abs(c["rule:wave_on_map"][0] + 0.2) < 1e-12
    assert abs(eta[0] - (0.1 - 0.2)) < 1e-12
    assert abs(z.reference_eta(D[0], 2, 0) - eta[0]) < 1e-12


def test_sd_and_kappa():
    m = BENIGN
    D = rand_drafts(m, 20, seed=5)
    eta, sd, _ = m.evaluate(D, "Cursed Hollow", "mid")
    assert (sd > 0).all()
    v = m.leaf_value(D, "Cursed Hollow", "mid", kappa=1.0)
    assert np.allclose(v, sigmoid(eta - sd))
    assert np.allclose(m.leaf_value(D, "Cursed Hollow", "mid", kappa=0.0), sigmoid(eta))


def test_save_load_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        EXPLOIT.save(d)
        m = LeafModel.load(d)
        D = rand_drafts(m, 30)
        assert np.allclose(m.evaluate(D, 0, 1)[0], EXPLOIT.evaluate(D, 0, 1)[0], atol=0, rtol=0)
        assert m.card_hashes == EXPLOIT.card_hashes


def test_five_tank_probe_flags_exploit_only():
    pb, pe = o6.probes(BENIGN, 0.0), o6.probes(EXPLOIT, 0.0)
    assert pb["5 tanks"]["p"] <= o6.PROBE_MAX, pb["5 tanks"]
    assert pe["5 tanks"]["p"] > o6.PROBE_MAX, pe["5 tanks"]
    assert pe["5 tanks"]["top_terms"][0][0] == "role_table"
    assert abs(pe["5 tanks"]["top_terms"][0][1] - pb["5 tanks"]["top_terms"][0][1]
               - (EXPLOIT_TANK_BONUS - BENIGN.a["role_table"][5])) < 1e-9
    # the two O6 probes; the other paper-1 probes are informational
    assert pb["No healer high WR"]["p"] <= o6.PROBE_MAX, pb["No healer high WR"]


def test_search_smoke():
    """One greedy and one tiny MCTS draft through ref_search with a uniform opponent."""
    from search import GDOpponent, greedy_draft, mcts_draft
    gd = GDOpponent(EXPLOIT, kind="uniform")
    r = greedy_draft(EXPLOIT, gd, "Cursed Hollow", "mid", 0, seed=0)
    assert len(set(r["draft"])) == 10 and len(r["our"]) == 5
    r = mcts_draft(EXPLOIT, gd, "Cursed Hollow", "mid", 1, sims=30, search_seed=0, opp_seed=0)
    assert len(set(r["draft"])) == 10
    assert sum(s["ours"] for s in r["steps"]) == 8      # 3 bans + 5 picks
    r2 = mcts_draft(EXPLOIT, gd, "Cursed Hollow", "mid", 1, sims=30, search_seed=0, opp_seed=0)
    assert r2["draft"] == r["draft"]                    # seeded


def _zero_pack():
    m = make("benign", seed=1)
    for k in m.a:
        m.a[k] = np.zeros_like(m.a[k])
    m.a["shape_knots"] = np.tile(np.arange(6.0), (m.K, 1))
    m.a["shape_vals"] = np.zeros((m.K, 6))
    m.rules = []
    m._compile_rules()
    return m


if __name__ == "__main__":
    tests = [(k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)]
    fails = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:      # noqa: BLE001
            fails += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - fails}/{len(tests)} passed")
    sys.exit(1 if fails else 0)
