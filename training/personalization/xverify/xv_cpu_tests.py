"""
CPU-only checks of the MCTS drafting stack (paper 2), independent of the GPU:
search semantics on toy drafts with exact optima, the personal value and its
symmetries, the opponent and prior transforms, the assignment rule, the
tournament harness (legality, structural mask, seats, seeds, leakage).

Usage (from training/): nice -n 19 taskset -c 48-51 python3 personalization/xverify/xv_cpu_tests.py
  (or pytest). Writes results/p3t_xverify_cpu.json next to the report.
"""
import itertools
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np

import xv_cases as XC
import xv_ref as R

TRAINING = R.TRAINING
sys.path.insert(0, os.path.join(TRAINING, "cuda_mcts"))
import ref_search as RS  # noqa: E402

REPORT = {}


# ------------------------------------------------------------------ search semantics on toy drafts

def test_toy_optimum_and_monotone():
    """Perfect leaf value (exact expectimax at every leaf) on small drafts with
    a computable optimum: the chosen action's exact value approaches the
    optimum as sims grow, and the mean regret does not increase with sims."""
    sims_list = (10, 40, 160, 640, 2560, 10240)
    reg = {s: [] for s in sims_list}
    for seed in range(40):
        g = RS.ToyDraft(n=7, turns=(0, 1, 1, 0, 1, 0, 0), seed=seed, exact_leaf=True)
        s0 = g.init()
        opt = RS.expectimax(g, s0)
        for sims in sims_list:
            for rs_seed in range(3):
                rs = RS.RefSearch(g, c_puct=2.0, pw_k=1.0, pw_alpha=0.5, seed=1000 * seed + rs_seed)
                nodes = rs.search(s0, sims)
                best = max((nodes[c] for c in nodes[0].child if c >= 0), key=lambda n: n.visits)
                reg[sims].append(opt - RS.expectimax(g, g.apply(s0, best.action)))
    m = {s: float(np.mean(v)) for s, v in reg.items()}
    se = {s: float(np.std(v, ddof=1) / math.sqrt(len(v))) for s, v in reg.items()}
    opt_frac = {s: float(np.mean(np.array(v) < 1e-12)) for s, v in reg.items()}
    for a, b in zip(sims_list, sims_list[1:]):
        assert m[b] <= m[a] + 2 * se[b] + 1e-12, (a, b, m)
    assert opt_frac[sims_list[-1]] >= 0.9 and m[sims_list[-1]] < 0.01, (opt_frac, m)
    REPORT["toy_monotone"] = {"mean_regret": m, "se": se, "optimal_frac": opt_frac, "drafts": 40, "seeds": 3}


def test_toy_chance_probabilities():
    """chance distributions of the reference sum to 1 and put no mass on taken heroes."""
    g = RS.ToyDraft(n=7, turns=(0, 1, 1, 0, 1, 0, 0), seed=3)
    s = g.apply(g.init(), 2)
    p = g.chance(s)
    assert abs(p.sum() - 1) < 1e-12 and p[2] == 0
    REPORT["toy_chance_probs"] = "ok"


# ------------------------------------------------------------------ personal value

def _rand_inputs(rng, flavor="personal", assign=1, cfg57=0, b=(0.03, 1.2, 4.5, -0.05, 0.0, 0.0), scale=0.3):
    cfg = np.full(59, -1)
    cfg[0:8] = [3, 1, 0, 0, -1, 1, 0, 1]
    cfg[24:40] = XC.slot_of_default()
    cfg[56:59] = [0, cfg57, assign]
    return R.P3Inputs(flavor, cfg, rng.normal(0, scale, (10, 90)), (rng.random((10, 90)) < 0.2), np.zeros((10, 90)),
                      np.ones((10, 90), bool), list(b), None, np.log1p(rng.poisson(4, (10, 90))),
                      rng.normal(0, 0.3, 225))


def _draft(acts, our=0):
    d = R.Draft(3, 1, our)
    for k in range(16):
        d.apply(int(acts[k]), R.TEAM[k], R.PICK[k])
    return d


def test_personal_value_symmetry():
    """V(our = 0) + V(our = 1) = 1; swapping the teams' heroes and tables gives
    logit V + logit V' = 2 b0 (b0 = side constant); with b0 = 0 exact
    antisymmetry. Slot, assign and pvnet leaves."""
    rng = np.random.default_rng(0)
    dev = {"flip": 0.0, "swap": 0.0}
    p0 = [k for k in range(16) if R.PICK[k] and R.TEAM[k] == 0]
    p1 = [k for k in range(16) if R.PICK[k] and R.TEAM[k] == 1]
    lg = lambda x: math.log(x / (1 - x))
    for trial in range(300):
        assign, pv = [(1, 0), (0, 0), (1, 1)][trial % 3]
        e = _rand_inputs(rng, "pvnet", assign, pv)
        acts = rng.permutation(90)[:16]
        w = float(rng.uniform(0.05, 0.95))
        d = _draft(acts)
        v0 = R.personal_value(w, d, 0, e)
        v1 = R.personal_value(1 - w, d, 1, e)
        dev["flip"] = max(dev["flip"], abs(v0 + v1 - 1))
        a2 = acts.copy()
        a2[p0], a2[p1] = acts[p1], acts[p0]
        e2 = _rand_inputs(rng, "pvnet", assign, pv)
        e2.s = np.concatenate([e.s[5:], e.s[:5]])
        e2.off = np.concatenate([e.off[5:], e.off[:5]])
        e2.feat = np.concatenate([e.feat[5:], e.feat[:5]])
        e2.vnet = e.vnet
        if assign == 0:
            # slot mode: a pick's player follows its pick step, so slots move with the heroes
            for x, y in zip(p0, p1):
                e2.cfg[24 + x] = e.cfg[24 + y] - 5
                e2.cfg[24 + y] = e.cfg[24 + x] + 5
        dsw = _draft(a2)
        vs = R.personal_value(1 - w, dsw, 0, e2)
        dev["swap"] = max(dev["swap"], abs(lg(v0) + lg(vs) - 2 * e.b0))
    assert dev["flip"] < 1e-12 and dev["swap"] < 1e-9, dev
    REPORT["personal_value_symmetry"] = dev


def test_personal_off_and_neutral():
    """cfg[7] = 0 returns the population value; tables of zeros with
    b = (0, 1, *, *) return it too (the kernel's early exit)."""
    rng = np.random.default_rng(1)
    for _ in range(50):
        e = _rand_inputs(rng)
        d = _draft(rng.permutation(90)[:16])
        w = float(rng.uniform(0.01, 0.99))
        e.cfg[7] = 0
        assert R.personal_value(w, d, 0, e) == w
        e.cfg[7] = 1
        e.s[:] = 0
        e.off[:] = 0
        e.b0, e.b1 = 0.0, 1.0
        assert R.personal_value(w, d, 0, e) == w
    REPORT["personal_off_neutral"] = "ok"


def test_assignment_rule():
    """kernel rule (first maximum over permutations in lexicographic order)
    == the host mirror p3_assign == the judge's best_assign, and it attains the
    brute-force maximum of b2 S + b3 O."""
    import p3_assign as PA
    sys.path.insert(0, os.path.join(TRAINING, "personalization"))
    rng = np.random.default_rng(2)
    worst = 0.0
    ties = 0
    for trial in range(400):
        n = int(rng.integers(0, 6))
        s5 = rng.normal(0, 1, (5, 90))
        o5 = (rng.random((5, 90)) < 0.3).astype(float)
        if trial % 5 == 0:
            s5[:] = s5[0]                       # identical players: many exact ties
        hs = list(rng.permutation(90)[:n])
        b2, b3 = 4.5, float(rng.choice([-0.05, 0.0, 0.3]))
        es = np.zeros((10, 90))
        eo = np.zeros((10, 90))
        es[:5], eo[:5] = s5, o5
        acts = [-1] * 16
        it = iter(hs)
        for k in range(16):
            if R.PICK[k] and R.TEAM[k] == 0:
                acts[k] = next(it, -1)
        step = max([k + 1 for k in range(16) if acts[k] >= 0] + [0])
        S, O, bp = R.best_assignment(0, acts, step, es, eo, b2, b3)
        S2, O2, bp2 = PA.best_assignment(s5, o5, hs, b2, b3)
        obj = lambda S_, O_: b2 * S_ + b3 * O_
        # the chosen permutation can differ only between exact ties of the objective
        # (summation order); S and O then agree too unless b3 trades off against b2
        assert abs(obj(S, O) - obj(S2, O2)) < 1e-9, (S, S2, bp, bp2)
        ties += tuple(bp) != tuple(bp2)
        brute = max(b2 * sum(s5[i, hs[p[i]]] for i in range(5) if p[i] < n)
                    + b3 * sum(o5[i, hs[p[i]]] for i in range(5) if p[i] < n) for p in itertools.permutations(range(5)))
        worst = max(worst, brute - (b2 * S + b3 * O))
        if n == 5:
            import p3t_judge as PJ
            Sj, Oj, _ = PJ.best_assign(s5, o5, hs, b2, b3)
            assert abs(obj(Sj, Oj) - obj(S, O)) < 1e-9
    assert worst < 1e-9, worst
    REPORT["assignment_rule"] = {"trials": 400, "max_gap_to_bruteforce": worst, "perm_differs_on_exact_ties": ties}


# ------------------------------------------------------------------ opponent and prior transforms

def test_opp_and_prior_transforms():
    """player-aware opponent: sums to 1, support = valid heroes with GD
    mass > 0, equals softmax(w0 log p + b) there; identity for w0 = 1 and b = 0;
    our own steps untouched. Prior bias: same algebra; alpha = 0 gives
    softmax(bias) over heroes whose prior is > 0 (a hero whose prior underflowed
    to exactly 0 cannot be revived by its bias)."""
    rng = np.random.default_rng(3)
    e = _rand_inputs(rng, "explore")
    e.cfg[6] = 2
    e.cfg[24:40] = np.where(np.array(R.PICK) == 1, XC.slot_of_default(), 7)
    e.imit = rng.normal(0, 1, (10, 90)).astype(np.float32)
    worst = 0.0
    for _ in range(200):
        p = rng.dirichlet(np.ones(90) * 0.3).astype(np.float32)
        p[rng.random(90) < 0.2] = 0
        mask = rng.random(90) < 0.9
        p = np.where(mask, p, 0)
        p /= p.sum()
        step = int(rng.integers(0, 16))
        our = 1 - R.TEAM[step]
        e.w0 = float(rng.choice([1.0, 0.7, 0.0]))
        q = R.opp_personalize(p, mask, step, our, e)
        assert abs(q.sum() - 1) < 1e-5
        assert ((q > 0) == ((p > 0) & mask)).all()
        row = int(e.cfg[24 + step])
        u = np.where((p > 0) & mask, e.w0 * np.log(np.where(p > 0, p, 1)) + e.imit[row], -np.inf)
        ref = np.exp(u - u.max())
        ref /= ref.sum()
        worst = max(worst, float(np.abs(ref - q).max()))
        assert R.opp_personalize(p, mask, step, 1 - our, e) is p
    e.w0 = 1.0
    e.imit[:] = 0
    p = rng.dirichlet(np.ones(90)).astype(np.float32)
    assert R.opp_personalize(p, np.ones(90, bool), 4, 1, e) is p
    # prior bias with alpha = 0 and a zero prior
    e2 = _rand_inputs(rng, "pvnet")
    e2.cfg[56] = 1
    e2.alpha = 0.0
    pri = rng.dirichlet(np.ones(90)).astype(np.float32)
    pri[5] = 0.0
    e2.bias[:] = 0
    e2.bias[int(e2.cfg[24 + 4]), 5] = 50.0
    out = R.prior_bias(pri, np.ones(90, bool), 4, e2)
    assert out[5] == 0.0 and abs(out.sum() - 1) < 1e-5
    REPORT["opp_prior_transforms"] = {"max_dev_vs_formula": worst,
                                      "alpha0_zero_prior_hero_stays_zero": True}


# ------------------------------------------------------------------ harness

def test_harness_masks():
    """tournament legal_mask == kernel pool rule (assign mode) on random
    states; struct_mask == brute-force feasibility over completions."""
    import p3t_systems as SY
    rng = np.random.default_rng(4)
    n_chk, n_struct = 0, 0
    for trial in range(300):
        pools = rng.random((10, 90)) < rng.uniform(0.02, 0.4)
        k = int(rng.integers(0, 16))
        acts = np.full(16, -1, np.int64)
        for j in range(k):
            acts[j] = int(rng.choice(np.flatnonzero(XC.legal_mask(acts, j, pools))))
        tp = np.stack([pools[:5].any(0), pools[5:].any(0)])
        m_h = SY.legal_mask(acts[None], k, tp[None])[0]
        cfg = np.full(59, -1)
        cfg[0:8] = [0, 0, R.TEAM[k], 0, k, 1, 0, 1]
        cfg[8:24] = acts
        cfg[24:40] = XC.slot_of_default()
        cfg[56:59] = [0, 0, 1]
        e = R.P3Inputs("personal", cfg, np.zeros((10, 90)), np.zeros((10, 90)), np.zeros((10, 90)), pools, [0, 1, 0, 0])
        d = R.Draft(0, 0, R.TEAM[k])
        for j in range(k):
            d.apply(int(acts[j]), R.TEAM[j], R.PICK[j])
        assert (R.p3_mask(d, e) == m_h).all(), trial
        n_chk += 1
        if R.PICK[k] and trial % 3 == 0:
            sm = SY.struct_mask(acts[None], k, m_h[None])[0]
            t = R.TEAM[k]
            own = [int(acts[j]) for j in range(k) if R.PICK[j] and R.TEAM[j] == t]
            left = 5 - len(own) - 1
            legal = np.flatnonzero(m_h)
            combos = [np.bincount(np.array(c, np.int64), minlength=6)
                      for c in itertools.combinations_with_replacement(range(6), left)]

            def feasible(h):
                rest = np.bincount(SY.HROLE[[x for x in legal if x != h]], minlength=6)
                cur = np.bincount(SY.HROLE[own + [int(h)]], minlength=6)
                return any(not (need > rest).any() and SY._valid_final(tuple(int(v) for v in cur + need))
                           for need in combos)
            ok = np.array([feasible(h) for h in legal])
            if not ok.any():
                ok[:] = True         # nothing feasible: the harness falls back to the legal mask
            assert (sm[legal] == ok).all() and not sm[~m_h].any(), trial
            n_struct += len(legal)
    REPORT["harness_masks"] = {"legal_mask_states": n_chk, "struct_mask_candidates": n_struct}


def test_schedules_and_seats():
    """every scheduled pair is also scheduled with seats swapped; no self pairs."""
    import p3t_tourney as TT
    out = {}
    for name in ("core", "spokes", "all", "tune", "ablation"):
        pairs = TT.schedule(name)
        ps = set(pairs)
        missing = [p for p in pairs if (p[1], p[0]) not in ps]
        selfp = [p for p in pairs if p[0] == p[1]]
        assert not missing and not selfp, (name, missing[:5], selfp[:5])
        out[name] = len(pairs)
    REPORT["schedules_both_seats"] = out


def test_no_judge_state_in_drafters():
    """the player state a drafter reads is never the judge's (judge, splitj)."""
    import p3t_systems as SY
    states = {}
    for name, spec in SY.SYSTEMS.items():
        st = spec.get("state", SY.vintage_state(spec["vint"]))
        states.setdefault(st, []).append(name)
        assert st not in ("judge", "splitj"), name
    REPORT["drafter_states"] = {k: len(v) for k, v in states.items()}


def test_kernel_seeds_distinct():
    """within one ordered pair, every (lobby, step, prior seed) search gets its
    own curand stream: base = kseed + 1000 k (play) + 7919 k + b0 + 104729 i
    (act_mcts), episode j adds j; the 2 x 16 steps of a pair never reuse a seed."""
    n, n_seeds = 300, 5
    seen = {}
    h = 0
    for k in range(16):
        kseed = h + 1000 * k
        idx = np.arange(n)
        seeds = idx % n_seeds
        for i in np.unique(seeds):
            sel = np.flatnonzero(seeds == i)
            for b0 in range(0, len(sel), 256):
                ii = sel[b0:b0 + 256]
                base = kseed + 7919 * k + b0 + 104729 * int(i)
                for j, lob in enumerate(ii):
                    s = base + j
                    assert s not in seen, (k, i, lob, seen.get(s))
                    seen[s] = (k, int(i), int(lob))
    REPORT["kernel_seeds_distinct"] = {"searches_per_pair_side": len(seen)}


def test_pool_bits_roundtrip():
    rng = np.random.default_rng(5)
    import p3_mcts_core as M
    pools = rng.random((4, 10, 90)) < 0.3
    b = XC.pool_bits(pools)
    for i in range(4):
        for s in range(10):
            assert (M.pool_bits(pools[i, s]) == b[i, s]).all()
    REPORT["pool_bits"] = "ok"


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failed = []
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}", flush=True)
        except Exception as ex:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            failed.append(t.__name__)
            print(f"FAIL {t.__name__}: {type(ex).__name__}: {str(ex)[:500]}", flush=True)
    REPORT["failed"] = failed
    out = os.path.join(TRAINING, "personalization", "results", "p3t_xverify_cpu.json")
    json.dump(REPORT, open(out, "w"), indent=1, default=str)
    print(json.dumps(REPORT, indent=1, default=str)[:3000])
    sys.exit(1 if failed else 0)
