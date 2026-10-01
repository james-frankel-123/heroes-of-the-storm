"""
Tests for the X2 search fix (search_v2.cuh).

CPU (no GPU needed)
  test_sampler_laws            v2_sample / v2_sample_existing laws (mirrors)
  test_widening_limit          progressive-widening limit sequence
  test_ref_backup_identities   chance/decision visit and value identities on a
                               reference tree
  test_ref_converges_to_expectimax
                               reference search picks the exact expectimax
                               action; its Q converges to the exact value
  test_ref_chance_average_unbiased
                               chance-node Q = sampled-outcome average,
                               unbiased without widening
GPU (kernel build from $X2_SO, default the installed training/cuda_mcts build)
  test_trace_tree              kernel tree dump vs an independent CPU replay:
                               every node's state rebuilt from the root by the
                               action path, kinds, legality, GD probabilities
                               at chance nodes and policy priors at decision
                               nodes recomputed in PyTorch on the rebuilt
                               state; visit/value identities; the tree reaches
                               later own picks and bans
  test_chance_statistics       child visit counts at chance nodes ~
                               Multinomial(N, GD) without widening
  test_determinism             same seed -> identical outputs and trees
  test_capacity_guard          tiny arena: guard trips, drafts stay valid
  test_valid_drafts            every mode: legal complete drafts
  test_legacy_bitexact         search_mode=0 == recorded pre-fix outputs

Usage (from training/):
  CUDA_VISIBLE_DEVICES=<gpu> python3 cuda_mcts/test_search_v2.py [--cpu-only]
  (or pytest cuda_mcts/test_search_v2.py). Writes x2_results/test_report.json.
"""
import os
import sys
import json
import math
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np

import ref_search as R

REPORT = {}
SO = os.environ.get("X2_SO") or None


# ── CPU ──────────────────────────────────────────────────────────────

def test_sampler_laws():
    rng = np.random.default_rng(1)
    p = np.array([0.0, 0.5, 0.0, 0.3, 0.2])
    child = [R.CHILD_NONE, 3, R.CHILD_NONE, R.CHILD_LAZY, 7]
    n = 200_000
    full = np.bincount([R.sample(p, 1.0 - rng.random()) for _ in range(n)], minlength=5) / n
    ex = np.bincount([R.sample_existing(p, child, 1.0 - rng.random()) for _ in range(n)], minlength=5) / n
    assert full[0] == 0 and full[2] == 0
    assert np.allclose(full, p, atol=0.005)
    target = np.array([0, 0.5, 0, 0, 0.2]) / 0.7
    assert np.allclose(ex, target, atol=0.005)
    # draw-full-then-resample-on-miss has the same law as sample_existing
    def full_then_resample():
        while True:
            a = R.sample(p, 1.0 - rng.random())
            if child[a] >= 0:
                return a
    alt = np.bincount([full_then_resample() for _ in range(n)], minlength=5) / n
    assert np.allclose(alt, target, atol=0.005)
    # rounding overrun falls back to the last positive entry, never a zero entry
    assert R.sample([0.25, 0.25, 0.25, 0.25 - 1e-7, 0.0], 1.0) == 3
    REPORT["sampler_laws"] = {"full_maxdev": float(np.abs(full - p).max()),
                              "existing_maxdev": float(np.abs(ex - target).max())}


def test_widening_limit():
    seq = [R.pw_limit(n, 1.0, 0.5) for n in (0, 1, 2, 4, 5, 9, 10, 100, 400)]
    assert seq == [1, 1, 2, 2, 3, 3, 4, 10, 20]
    assert R.pw_limit(10**6, 0.0, 0.5) >= 1 << 30


def _toy_tree(sims=3000, pw_k=1.0, seed=3, exact_leaf=False):
    g = R.ToyDraft(n=7, turns=(0, 1, 1, 0, 1, 0, 0), seed=seed, exact_leaf=exact_leaf)
    rs = R.RefSearch(g, c_puct=2.0, pw_k=pw_k, pw_alpha=0.5, seed=seed)
    nodes = rs.search(g.init(), sims)
    return g, rs, nodes


def _as_dump(nodes):
    slots, sp, sc = [], [], []
    for n in nodes:
        if n.p is None:
            slots.append(-1)
        else:
            slots.append(len(sp))
            sp.append(n.p)
            sc.append(n.child)
    return {"parent": [n.parent for n in nodes], "action": [n.action for n in nodes],
            "visits": [n.visits for n in nodes], "value_sum": [n.value_sum for n in nodes],
            "slot": slots, "n_kids": [n.n_kids for n in nodes],
            "step": [len(n.state[0]) + len(n.state[1]) for n in nodes],
            "kind": [n.kind for n in nodes], "slot_p": sp, "slot_child": sc}


def test_ref_backup_identities():
    for pw_k in (1.0, 0.0):
        g, rs, nodes = _toy_tree(sims=2000, pw_k=pw_k)
        dump = _as_dump(nodes)
        _, rep = R.check_tree(dump, 2000, pw_k, 0.5, g.kind, g.legal,
                              lambda s, st, a: g.apply(s, a), g.init(),
                              chance_probs_of=g.chance, prior_of=g.prior, prob_tol=1e-9)
        assert not rep["errors"], rep["errors"][:5]
        # chance node Q is exactly the visit-weighted mean of its children
        for i, n in enumerate(nodes):
            if n.kind == R.KIND_CHANCE and n.visits:
                ch = [nodes[c] for c in n.child if c >= 0]
                assert abs(n.q - sum(c.value_sum for c in ch) / n.visits) < 1e-9
    REPORT["ref_backup_identities"] = "ok"


def test_ref_converges_to_expectimax():
    """Decision nodes maximize, chance nodes average: on toy drafts the
    search recommends the exact expectimax action, and the recommended
    child's Q approaches the exact value as sims grow (from below: the mean
    includes exploratory visits deeper in the tree)."""
    out = []
    for seed in range(6):
        g = R.ToyDraft(n=6, turns=(0, 1, 1, 0, 1, 0), seed=seed, exact_leaf=True)
        exact = R.expectimax(g, g.init())
        gaps = []
        for sims in (2000, 50000):
            rs = R.RefSearch(g, c_puct=1.0, pw_k=1.0, pw_alpha=0.5, seed=seed)
            nodes = rs.search(g.init(), sims)
            best = max((nodes[c] for c in nodes[0].child if c >= 0), key=lambda n: n.visits)
            gaps.append(exact - best.q)
        assert abs(R.expectimax(g, g.apply(g.init(), best.action)) - exact) < 1e-12, seed
        assert abs(gaps[1]) < abs(gaps[0]) and abs(gaps[1]) < 0.015, (seed, gaps)
        out.append({"exact": float(exact), "gap_2000": float(gaps[0]), "gap_50000": float(gaps[1])})
    REPORT["ref_expectimax"] = out


def test_ref_chance_average_unbiased():
    """Below the root only chance moves: each root child's Q is the plain
    sampled-outcome average of exact terminal values, so without widening it
    is an unbiased estimate of the exact expectation (|z| < 4); with widening
    it converges to it."""
    zs, gaps_pw = [], []
    for seed in range(8):
        g = R.ToyDraft(n=7, turns=(0, 1, 1, 1), seed=seed, exact_leaf=True)
        for pw_k in (0.0, 1.0):
            rs = R.RefSearch(g, c_puct=2.0, pw_k=pw_k, pw_alpha=0.5, seed=100 + seed)
            nodes = rs.search(g.init(), 30000)
            for c in nodes[0].child:
                if c < 0 or nodes[c].visits < 500:
                    continue
                n = nodes[c]
                exact = R.expectimax(g, n.state)
                # per-visit values: terminal values of the sampled leaves
                leaves = [x for x in nodes if x.kind == R.KIND_TERMINAL]
                sub = []
                for x in leaves:
                    j = nodes.index(x)
                    while j > 0 and j != c:
                        j = nodes[j].parent
                    if j == c:
                        sub.append((g.terminal_value(x.state), x.visits))
                vals = np.repeat([v for v, _ in sub], [w for _, w in sub])
                assert len(vals) == n.visits
                assert abs(vals.mean() - n.q) < 1e-9
                if pw_k == 0.0:
                    zs.append((n.q - exact) / (vals.std(ddof=1) / math.sqrt(n.visits) + 1e-12))
                else:
                    gaps_pw.append(abs(n.q - exact))
    zs = np.array(zs)
    assert len(zs) >= 10 and np.abs(zs).max() < 4.0, zs
    assert max(gaps_pw) < 0.02, max(gaps_pw)
    REPORT["ref_chance_unbiased"] = {"n": int(len(zs)), "max_abs_z": float(np.abs(zs).max()),
                                     "mean_z": float(zs.mean()), "max_gap_widening": float(max(gaps_pw))}


# ── GPU ──────────────────────────────────────────────────────────────

_K = {}


def _kernel():
    if "k" not in _K:
        import torch
        if not torch.cuda.is_available():
            import pytest
            pytest.skip("no GPU")
        import x2_common as C
        _K["C"] = C
        _K["k"] = C.load_kernel(SO)
        _K["eng"] = C.make_engine(_K["k"], "new:F_oof_s0", gd_idx=0, max_concurrent=64)
    return _K["C"], _K["k"], _K["eng"]


def _nets():
    if "nets" not in _K:
        import torch
        from train_generic_draft import GenericDraftModel
        from overfit2026 import search
        from paper1_revision import bench_mcts
        gd = GenericDraftModel()
        gd.load_state_dict(torch.load(os.path.join(search.RR, "models", "generic_draft_0.pt"),
                                      weights_only=True, map_location="cpu"))
        gd.eval()
        pol = search.policy_net("path:" + bench_mcts.ckpt_of("new:F_oof_s0"))
        _K["nets"] = (gd, pol)
    return _K["nets"]


# draft state as a tuple: (t0, t1, bans, step, map, tier, our)

def decode_root(v):
    t0 = frozenset(np.flatnonzero(v[:90] > 0.5).tolist())
    t1 = frozenset(np.flatnonzero(v[90:180] > 0.5).tolist())
    bans = frozenset(np.flatnonzero(v[180:270] > 0.5).tolist())
    mp = int(np.argmax(v[270:284]))
    tier = int(np.argmax(v[284:287]))
    step = int(round(v[287] * 15))
    return (t0, t1, bans, step, mp, tier, int(round(v[289])))


def encode(s):
    t0, t1, bans, step, mp, tier, our = s
    v = np.zeros(290, np.float32)
    for h in t0:
        v[h] = 1
    for h in t1:
        v[90 + h] = 1
    for h in bans:
        v[180 + h] = 1
    v[270 + mp] = 1
    v[284 + tier] = 1
    if step < 16:
        v[287] = np.float32(step / 15.0)
        v[288] = 1.0 if R_IS_PICK[step] else 0.0
    else:
        v[287] = v[288] = 1.0
    v[289] = our
    return v


R_TEAM = [0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1]
R_IS_PICK = [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1]


def legal_of(s):
    m = np.ones(90, bool)
    for h in s[0] | s[1] | s[2]:
        m[h] = False
    return m


def apply_of(s, parent_step, a):
    t0, t1, bans, step, mp, tier, our = s
    assert step == parent_step
    if not R_IS_PICK[step]:
        bans = bans | {a}
    elif R_TEAM[step] == 0:
        t0 = t0 | {a}
    else:
        t1 = t1 | {a}
    return (t0, t1, bans, step + 1, mp, tier, our)


def kind_chance(s):
    if s[3] >= 16:
        return R.KIND_TERMINAL
    return R.KIND_DECISION if R_TEAM[s[3]] == s[6] else R.KIND_CHANCE


def _ref_probs(states, which):
    import torch
    gd, pol = _nets()
    X = np.stack([encode(s) for s in states])
    M = np.stack([legal_of(s) for s in states]).astype(np.float32)
    with torch.no_grad():
        if which == "gd":
            lg = gd(torch.tensor(X[:, :289]), torch.tensor(M))
        else:
            lg, _ = pol(torch.tensor(X), torch.tensor(M))
        return torch.softmax(lg, 1).numpy()


def _check_dump(d, sims, pw_k, pw_alpha):
    root = decode_root(d["root_state"])
    assert root[3] == d["root_step"]
    # pass 1: rebuild states; pass 2: compare probabilities in one batch
    states, rep = R.check_tree(d, sims, pw_k, pw_alpha, kind_chance, legal_of, apply_of, root)
    ch = [i for i in range(len(states)) if d["kind"][i] == R.KIND_CHANCE and d["slot"][i] >= 0]
    de = [i for i in range(len(states)) if d["kind"][i] == R.KIND_DECISION and d["slot"][i] >= 0]
    for idx, which in ((ch, "gd"), (de, "policy")):
        if not idx:
            continue
        ref = _ref_probs([states[i] for i in idx], which)
        got = np.stack([d["slot_p"][d["slot"][i]] for i in idx])
        diff = np.abs(ref - got).max(1)
        rep[f"max_{which}_diff"] = float(diff.max())
        bad = int((diff > 2e-4).sum())
        if bad:
            rep["errors"].append(f"{bad} {which} distributions differ by > 2e-4")
    # depth: own decisions below an opponent move (a later own block)
    beyond, steps = 0, set()
    for i in de:
        j, saw = i, False
        while j > 0:
            j = d["parent"][j]
            saw |= d["kind"][j] == R.KIND_CHANCE
        if saw:
            beyond += 1
            steps.add(int(d["step"][i]))
    rep["decisions_beyond_block"] = beyond
    rep["later_own_steps"] = sorted(steps)
    return root, rep


def test_trace_tree():
    C, k, eng = _kernel()
    cfg = C.configs(16, 20261002)
    out = []
    for turn in (0, 2, 4):
        trees = eng.debug_tree(cfg, 200, 2.0, 4321 + turn, turn, search_mode=1, pw_k=1.0, pw_alpha=0.5)
        for e, d in enumerate(trees):
            root, rep = _check_dump(d, 200, 1.0, 0.5)
            assert not rep["errors"], (turn, e, rep["errors"][:5])
            own_left = [s for s in range(root[3] + 1, 16) if R_TEAM[s] == root[6]]
            later = [s for s in own_left if any(R_TEAM[x] != root[6] for x in range(root[3] + 1, s))]
            if later:
                assert rep["decisions_beyond_block"] > 0, (turn, e)
                assert later[0] in rep["later_own_steps"], (turn, e, later[0], rep["later_own_steps"])
            bans_ahead = [s for s in later if not R_IS_PICK[s]]
            out.append({"turn": turn, "root_step": root[3], "our": root[6], "nodes": rep["nodes"],
                        "decisions_beyond_block": rep["decisions_beyond_block"],
                        "later_own_steps": rep["later_own_steps"],
                        "ban_steps_ahead": bans_ahead,
                        "ban_reached": bool(set(bans_ahead) & set(rep["later_own_steps"])),
                        "max_gd_diff": rep.get("max_gd_diff"),
                        "max_policy_diff": rep.get("max_policy_diff")})
    with_bans = [o for o in out if o["ban_steps_ahead"]]
    assert all(o["ban_reached"] for o in with_bans), [o for o in with_bans if not o["ban_reached"]]
    REPORT["trace_tree"] = {"trees": len(out), "all_invariants_ok": True,
                            "max_gd_diff": max(o["max_gd_diff"] or 0 for o in out),
                            "max_policy_diff": max(o["max_policy_diff"] or 0 for o in out),
                            "trees_with_bans_ahead": len(with_bans),
                            "trees_reaching_a_later_ban": sum(o["ban_reached"] for o in with_bans),
                            "per_tree": out}


def test_chance_statistics():
    import torch

    def chi2_sf(x, dof):
        return float(torch.special.gammaincc(torch.tensor(dof / 2.0, dtype=torch.float64),
                                             torch.tensor(x / 2.0, dtype=torch.float64)))
    C, k, eng = _kernel()
    cfg = C.configs(16, 20261003)
    pvals, n_nodes = [], 0
    for turn in (1, 3):
        trees = eng.debug_tree(cfg, 800, 2.0, 99 + turn, turn, search_mode=1, pw_k=0.0, pw_alpha=0.5)
        for d in trees:
            for i in range(len(d["kind"])):
                if d["kind"][i] != R.KIND_CHANCE or d["visits"][i] < 150 or d["slot"][i] < 0:
                    continue
                p = d["slot_p"][d["slot"][i]].astype(np.float64)
                child = d["slot_child"][d["slot"][i]]
                obs = np.array([d["visits"][c] if c >= 0 else 0 for c in child], np.float64)
                N = d["visits"][i]
                assert obs.sum() == N
                assert np.all(obs[p == 0] == 0)
                exp = N * p
                big = exp >= 5
                o = np.append(obs[big], obs[~big].sum())
                e = np.append(exp[big], exp[~big].sum())
                keep = e > 0
                stat = ((o[keep] - e[keep]) ** 2 / e[keep]).sum()
                dof = keep.sum() - 1
                if dof < 1:
                    continue
                pvals.append(chi2_sf(stat, dof))
                n_nodes += 1
    pvals = np.array(pvals)
    frac01 = float((pvals < 0.01).mean())
    # under the null the p-values are uniform; allow sampling slack
    assert n_nodes >= 30, n_nodes
    assert frac01 < 0.05, frac01
    assert 0.35 < pvals.mean() < 0.65, pvals.mean()
    REPORT["chance_statistics"] = {"nodes_tested": n_nodes, "mean_p": float(pvals.mean()),
                                   "frac_p_below_0.01": frac01}


def test_determinism():
    C, k, eng = _kernel()
    cfg = C.configs(32, 20261004)
    res = {}
    for mode in (1, 2):
        a = C.pack(eng.run_episodes(cfg, 200, 2.0, 55, 1.0, 0.3, 0.25, search_mode=mode))
        sa = eng.last_stats()
        b = C.pack(eng.run_episodes(cfg, 200, 2.0, 55, 1.0, 0.3, 0.25, search_mode=mode))
        sb = eng.last_stats()
        c = C.pack(eng.run_episodes(cfg, 200, 2.0, 56, 1.0, 0.3, 0.25, search_mode=mode))
        same = all(np.array_equal(a[x], b[x]) for x in a) and np.array_equal(sa, sb)
        diff = not np.array_equal(a["policies"], c["policies"])
        assert same and diff
        res[mode] = {"same_seed_identical": same, "other_seed_differs": diff}
    t1 = eng.debug_tree(cfg[:8], 300, 2.0, 7, 2)
    t2 = eng.debug_tree(cfg[:8], 300, 2.0, 7, 2)
    assert all(np.array_equal(x[f], y[f]) for x, y in zip(t1, t2) for f in x if isinstance(x[f], np.ndarray))
    REPORT["determinism"] = res


def _valid(packed, mode):
    for i in range(len(packed["wp"])):
        ts = packed["terminal"][i]
        t0, t1, bn = ts[:90] > 0.5, ts[90:180] > 0.5, ts[180:270] > 0.5
        assert t0.sum() == 5 and t1.sum() == 5 and bn.sum() == 6, (mode, i)
        assert not np.any((t0 & t1) | (t0 & bn) | (t1 & bn)), (mode, i)
        assert packed["n_turns"][i] == 8
        for t in range(8):
            pol, m = packed["policies"][i, t], packed["masks"][i, t]
            assert abs(pol.sum() - 1) < 1e-5 and np.all(pol[m < 0.5] == 0)
        assert 0.0 <= packed["wp"][i] <= 1.0


def test_capacity_guard():
    C, k, eng = _kernel()
    cfg = C.configs(16, 20261005)
    eng.set_tree_capacity(64, 64)
    try:
        p = C.pack(eng.run_episodes(cfg, 200, 2.0, 1, 0.0, 0.3, 0.0, search_mode=1))
        st = eng.last_stats()
    finally:
        eng.set_tree_capacity(0, 0)
    _valid(p, "guard")
    assert st[:, 4].min() > 0 and st[:, 2].max() <= 64 and st[:, 3].max() <= 64
    p = C.pack(eng.run_episodes(cfg, 1600, 2.0, 1, 0.0, 0.3, 0.0, search_mode=1))
    st2 = eng.last_stats()
    assert st2[:, 4].sum() == 0
    REPORT["capacity_guard"] = {"tiny_arena_cap_hits_mean": float(st[:, 4].mean()),
                                "default_arena_1600_sims_cap_hits": int(st2[:, 4].sum()),
                                "default_arena_1600_sims_max_nodes": int(st2[:, 2].max())}


def test_valid_drafts():
    C, k, eng = _kernel()
    cfg = C.configs(64, 20261006)
    for mode in (0, 1, 2):
        _valid(C.pack(eng.run_episodes(cfg, 100, 2.0, 3, 1.0, 0.3, 0.0, search_mode=mode)), mode)
    REPORT["valid_drafts"] = "ok (modes 0, 1, 2)"


def test_legacy_bitexact():
    cmd = [sys.executable, os.path.join(HERE, "x2_record_legacy.py"), "check"]
    if SO:
        cmd += ["--so", SO]
    r = subprocess.run(cmd, capture_output=True, text=True)
    REPORT["legacy_bitexact"] = r.stdout.strip().splitlines()[-3:]
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]


if __name__ == "__main__":
    cpu_only = "--cpu-only" in sys.argv
    tests = [test_sampler_laws, test_widening_limit, test_ref_backup_identities,
             test_ref_converges_to_expectimax, test_ref_chance_average_unbiased]
    if not cpu_only:
        tests += [test_trace_tree, test_chance_statistics, test_determinism,
                  test_capacity_guard, test_valid_drafts, test_legacy_bitexact]
    failed = []
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}", flush=True)
        except Exception as ex:  # noqa: BLE001
            failed.append(t.__name__)
            print(f"FAIL {t.__name__}: {type(ex).__name__}: {str(ex)[:600]}", flush=True)
    REPORT["failed"] = failed
    REPORT["so"] = SO or "installed"
    os.makedirs(os.path.join(HERE, "x2_results"), exist_ok=True)
    name = "test_report_cpu.json" if cpu_only else "test_report.json"
    json.dump(REPORT, open(os.path.join(HERE, "x2_results", name), "w"), indent=1, default=str)
    sys.exit(1 if failed else 0)
