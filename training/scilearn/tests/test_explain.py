"""Fixtures for explanation (compression) cards, on synthetic tables with known structure.

  PYTHONPATH=. python -m training.scilearn.tests.test_explain

Planted truths must be recovered (accept, share near the planted share); nulls must fail:
an unrelated axis, a kit axis permuted across heroes, a kit axis permuted within role (the
harness canary), and a kit theory whose only real content is role-level (caught by the
within-role permutation check, while the honest role theory passes).
"""
import math
import sys

import numpy as np
from scipy.special import expit

from training.scilearn import explain as X
from training.scilearn import evaluator as E
from training.scilearn.terms import permuted_kit

ROLES = ["Tank", "Bruiser", "Healer", "Support", "Ranged Assassin", "Melee Assassin"]
AXES = ["hard_cc", "soft_cc_peel", "healing_throughput", "self_sustain_shields", "burst",
        "sustained_damage", "range", "mobility", "waveclear", "siege", "global_presence", "early_vs_late"]
MAPS = [f"map{i:02d}" for i in range(15)]
LANES2 = {"map00", "map01", "map02", "map03"}
H = 72


def world(seed=0, role_linked_waveclear=False):
    rng = np.random.default_rng(seed)
    roles = np.repeat(np.arange(6), H // 6)
    kit = rng.uniform(0, 4, size=(H, 12))
    if role_linked_waveclear:   # waveclear carries role information only: role mean + small noise
        kit[:, AXES.index("waveclear")] = np.array([0.5, 1.5, 0.5, 1.0, 3.5, 2.0])[roles] + rng.normal(0, 0.1, H)
    names = [f"hero{i:02d}" for i in range(H)]
    counts = rng.integers(5000, 60000, H)
    return rng, roles, kit, names, counts


def hero_map_table(rng, roles, kit, planted, tau=0.03):
    a, b = np.meshgrid(np.arange(H), np.arange(len(MAPS)), indexing="ij")
    a, b = a.ravel(), b.ravel()
    lanes3 = np.array([0.0 if m in LANES2 else 1.0 for m in MAPS])
    theta = rng.normal(0, 0.1, H)[a] + rng.normal(0, 0.05, len(MAPS))[b] + rng.normal(0, tau, len(a))
    theta += planted(a, b, lanes3)
    s = 1.0 / rng.uniform(300, 3000, len(a))
    y = theta + rng.normal(0, np.sqrt(s))
    return {"kind": "hero_ctx", "a": a, "b": b, "y": y, "s": s, "w": 1 / s, "lam": 50.0,
            "keep": np.ones(len(a), bool), "ctx": "map", "ctx_labels": MAPS, "n_ctx": len(MAPS)}, theta


def cross_table(rng, kit, coef, tau=0.03):
    a, b = np.triu_indices(H, 1)
    k, l = AXES.index("hard_cc"), AXES.index("mobility")
    c = rng.normal(0, 0.1, H)
    theta = c[a] - c[b] + coef * (kit[a, k] * kit[b, l] - kit[b, k] * kit[a, l]) + rng.normal(0, tau, len(a))
    s = 1.0 / rng.uniform(200, 2000, len(a))
    y = theta + rng.normal(0, np.sqrt(s))
    return {"kind": "cross", "a": a, "b": b, "y": y, "s": s, "w": 1 / s, "lam": 100.0,
            "keep": np.ones(len(a), bool)}, theta


def run_one(cells, theory, kit, roles, counts, names, n_perm=100, group_bits=0.0, onset=None):
    o3 = X.o3_heroes(names, counts, "fixture-o3")
    res = X.score_explain(cells, theory, kit, AXES, roles, ROLES, H, fold_key="fixture-fold",
                          perm_key="fixture-perm", o3=o3, lanes2=LANES2, n_perm=n_perm)
    if onset is not None:
        res["onset"] = onset
    ok, reasons, thr = X.gate(res, group_bits)
    return ok, reasons, res


def simulate_onset(rng, cells, theta, theory_prior, n_games=40000):
    """Games scored by an incumbent whose cross table is shrunk to 0, vs shrunk to the theory."""
    idx = {(int(x), int(y)): j for j, (x, y) in enumerate(zip(cells["a"], cells["b"]))}
    rows, cols, vals = [], [], []
    for g in range(n_games):
        h = rng.choice(H, 10, replace=False)
        A, B = h[:5], h[5:]
        for x in A:
            for y in B:
                lo, hi = min(x, y), max(x, y)
                rows.append(g); cols.append(idx[(lo, hi)]); vals.append(1.0 if x < y else -1.0)
    import scipy.sparse as sp
    Xt = sp.csr_matrix((vals, (rows, cols)), shape=(n_games, len(cells["a"])))
    eta_true = Xt @ theta
    yv = (rng.random(n_games) < expit(eta_true)).astype(float)
    beta_inc = X.shrink_to_theory(cells["y"], cells["s"], cells["lam"], np.zeros(len(theta)))
    eta0 = Xt @ beta_inc
    blocks = np.arange(n_games) // 50
    return X.onset_gain(Xt, beta_inc, cells, theory_prior, eta0, yv, blocks)


def run():
    res = []

    def check(name, cond, detail=""):
        res.append((name, bool(cond)))
        print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")

    # 1. hero x map: waveclear matters more on three-lane maps (planted share about 0.5).
    rng, roles, kit, names, counts = world(1)
    wc = kit[:, AXES.index("waveclear")]
    cells, theta = hero_map_table(rng, roles, kit, lambda a, b, l3: 0.03 * (wc[a] - 2) * (l3[b] - 0.73) * 2)
    t_true = {"features": [{"kit": "waveclear", "by": "lanes2"}]}
    ok, why, r = run_one(cells, t_true, kit, roles, counts, names)
    check("hero x map: planted waveclear x lanes recovered", ok,
          f"share {r['explain']['share']:.2f} z {r['explain']['z']:.1f} transfer z {r['transfer']['z']:.1f} "
          f"perm p {r['permutation']['p']:.3f} {why}")
    check("hero x map: recovered coefficient has the planted sign", r["coefs"][0] < 0,
          f"coef on lanes2 indicator {r['coefs'][0]:+.3f}")
    ok, why, r = run_one(cells, {"features": [{"kit": "siege", "by": "lanes2"}]}, kit, roles, counts, names)
    check("hero x map: unrelated axis (siege) fails", not ok, f"share {r['explain']['share']:.3f} {why[:1]}")
    kit_perm = kit.copy()
    kit_perm[:, AXES.index("waveclear")] = kit[np.random.default_rng(5).permutation(H), AXES.index("waveclear")]
    ok, why, r = run_one(cells, t_true, kit_perm, roles, counts, names)
    check("hero x map: waveclear permuted across heroes fails", not ok, f"share {r['explain']['share']:.3f}")
    ok, why, r = run_one(cells, t_true, permuted_kit(kit, roles, "canary:1"), roles, counts, names)
    check("hero x map: within-role permuted kit (canary) fails", not ok, f"share {r['explain']['share']:.3f}")

    # 2. Role-level truth: a kit axis that only carries role means must not get the credit.
    rng, roles, kit, names, counts = world(2, role_linked_waveclear=True)
    ra = (roles == ROLES.index("Ranged Assassin")).astype(float)
    cells, theta = hero_map_table(rng, roles, kit, lambda a, b, l3: 0.06 * ra[a] * (l3[b] - 0.73) * 2)
    ok_k, why_k, rk = run_one(cells, t_true, kit, roles, counts, names)
    check("role-level truth: kit theory fails the within-role permutation check",
          (not ok_k) and rk["permutation"]["p"] > X.GATE["p_perm"],
          f"share {rk['explain']['share']:.2f} perm p {rk['permutation']['p']:.3f}")
    ok_r, why_r, rr = run_one(cells, {"features": [{"role": "Ranged Assassin", "by": "lanes2"}]}, kit, roles,
                              counts, names)
    check("role-level truth: the role theory passes", ok_r, f"share {rr['explain']['share']:.2f} {why_r}")

    # 3. Counter table: hard CC beats mobility (planted share about 0.4).
    rng, roles, kit, names, counts = world(3)
    cells, theta = cross_table(rng, kit, 0.012)
    t_cc = {"features": [{"kit_pair": ["hard_cc", "mobility"], "sign": "+"}]}
    ok, why, r = run_one(cells, t_cc, kit, roles, counts, names)
    check("counters: planted hard_cc x mobility recovered", ok,
          f"share {r['explain']['share']:.2f} z {r['explain']['z']:.1f} transfer z {r['transfer']['z']:.1f} "
          f"perm p {r['permutation']['p']:.3f}")
    k_, l_ = AXES.index("hard_cc"), AXES.index("mobility")
    planted = 0.012 * (kit[cells["a"], k_] * kit[cells["b"], l_] - kit[cells["b"], k_] * kit[cells["a"], l_])
    P = X.Projector(X.nuisance_design(cells, H), cells["w"])
    true_share = float((cells["w"] * P(planted) ** 2).sum() / (cells["w"] * P(theta) ** 2).sum())
    check("counters: explained share within 0.15 of planted", abs(r["explain"]["share"] - true_share) < 0.15,
          f"estimated {r['explain']['share']:.2f}, planted {true_share:.2f}")
    ok, why, r = run_one(cells, {"features": [{"kit_pair": ["burst", "healing_throughput"]}]}, kit, roles, counts, names)
    check("counters: unrelated pair fails", not ok, f"share {r['explain']['share']:.3f}")
    ok, why, r = run_one(cells, t_cc, permuted_kit(kit, roles, "canary:2"), roles, counts, names)
    check("counters: within-role permuted kit (canary) fails", not ok, f"share {r['explain']['share']:.3f}")
    ok, why, r = run_one(cells, {"features": [{"kit_pair": ["hard_cc", "mobility"], "sign": "-"}]}, kit, roles,
                         counts, names)
    check("counters: wrong declared sign fails", not ok, f"share {r['explain']['share']:.3f}")

    # 4. Onset: shrinking the table toward the true theory helps; toward a null theory does not.
    #    Thin cells (20 to 200 games' worth of precision), as for rare pairs early in a patch.
    #    In the model, hero strength terms absorb hero-level effects, so the table holds the
    #    interaction part only: theta = projected bilinear + unexplained noise.
    rng, roles, kit, names, counts = world(6)
    cells, _ = cross_table(rng, kit, 0.0)
    cells["s"] = 1.0 / rng.uniform(20, 200, len(cells["a"]))
    cells["w"] = 1 / cells["s"]
    k_, l_ = AXES.index("hard_cc"), AXES.index("mobility")
    bil = 0.02 * (kit[cells["a"], k_] * kit[cells["b"], l_] - kit[cells["b"], k_] * kit[cells["a"], l_])
    theta = X.Projector(X.nuisance_design(cells, H), np.ones(len(bil)))(bil) + rng.normal(0, 0.03, len(bil))
    cells["y"] = theta + rng.normal(0, np.sqrt(cells["s"]))
    res_t = X.score_explain(cells, t_cc, kit, AXES, roles, ROLES, H, fold_key="f", perm_key="p", n_perm=0)
    prior_t = X.theory_prior_for(cells, t_cc, res_t["coefs"], kit, AXES, roles, ROLES, H)
    on_t = simulate_onset(np.random.default_rng(9), cells, theta, prior_t)
    pk = permuted_kit(kit, roles, "canary:3")
    res_n = X.score_explain(cells, t_cc, pk, AXES, roles, ROLES, H, fold_key="f", perm_key="p", n_perm=0)
    prior_n = X.theory_prior_for(cells, t_cc, res_n["coefs"], pk, AXES, roles, ROLES, H)
    on_n = simulate_onset(np.random.default_rng(9), cells, theta, prior_n)
    check("onset: true theory prior improves early-game log loss", on_t["z"] > 2, f"gain {on_t['gain']:.5f} z {on_t['z']:.1f}")
    check("onset: null theory prior does not", on_n["z"] < 2, f"gain {on_n['gain']:.5f} z {on_n['z']:.1f}")

    # 5. Null calibration: 40 random-axis theories on a table with no kit structure.
    rng, roles, kit, names, counts = world(4)
    cells, theta = hero_map_table(rng, roles, kit, lambda a, b, l3: 0.0)
    acc = 0
    for i in range(40):
        ax = AXES[i % 12]
        th = {"features": [{"kit": ax, "by": "lanes2" if i % 2 else "map_class"}]}
        ok, _, _ = run_one(cells, th, kit, roles, counts, names, n_perm=40)
        acc += ok
    check("null calibration: 0 of 40 single-axis theories accepted on a structureless table", acc == 0, f"{acc}/40")

    n = sum(ok for _, ok in res)
    print(f"{n}/{len(res)} passed")
    return n == len(res)


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
