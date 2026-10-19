"""
Regression test (GPU): several MCTSKernelEngine objects in one process must
give byte-identical outputs for the same inputs and seed, whatever ran before
them.

Guards the bug found on 2026-10-19 (personalization/results/P3_MCTS_VERIFICATION.md):
after the per-engine non-blocking stream (48ab696) the constructor still
uploaded weights with cudaMemcpy on the legacy stream, with no ordering
against launches on the engine's stream. On the WSL (WDDM) workers an engine
that was not the first in its process launched its first kernels before its
win-probability weights arrived: every episode differed and the leaf values
sat near 0.50.

Cases: (1) engine 1, deleted; other GPU work (a large torch tensor written and
freed); engine 2 -> same output as engine 1. (2) engine 3 built while engine 2
is alive -> same. (3) engine 2 again after engine 3 ran -> same. (4) three
engines running at once in three threads -> same as serial.

Usage (from training/, a GPU host): python cuda_mcts/test_engine_isolation.py [--so PATH]
  (or pytest; X2_SO selects the build). Exit 1 on failure.
"""
import os
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np

import x2_common as X

SO = os.environ.get("X2_SO") or None
N, SIMS, SEED = 64, 200, 1000003


def _flat(res):
    return [np.float32(r[0]).tobytes() + np.asarray(r[2], np.float32).tobytes()
            + b"".join(np.asarray(x[1], np.float32).tobytes() for x in r[1]) for r in res]


def make_engine(k, n):
    """paper-1 revision weights (x2_common) where present; else the P3 tournament's cur
    vintage (F policy seed 0, GD 0, enriched WP), which every worker has."""
    try:
        return X.make_engine(k, max_concurrent=n)
    except FileNotFoundError:
        sys.path.insert(0, os.path.join(os.path.dirname(HERE), "personalization"))
        os.environ.setdefault("P1_TIERS", "site")
        import p3t_systems as SY
        v = SY.vintage("cur")
        (pf, po), (gf, go), (wf, wo, lut) = v.policy_flat(0), v.gd_flat(0), v.wcfg()
        return k.MCTSKernelEngine(pf, gf, wf, po, go, wo, lut, max_concurrent=n, device_id=0)


def _run(eng, cfg):
    return _flat(eng.run_episodes(np.ascontiguousarray(cfg), SIMS, 2.0, SEED, 0.0, 0.3, 0.0, 1))


def _dirty_gpu():
    import torch
    t = torch.full((1 << 30,), 0x55, dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()
    del t
    torch.cuda.empty_cache()


def _diff(a, b):
    return sum(x != y for x, y in zip(a, b))


def test_engines_isolated():
    k = X.load_kernel(SO)
    rng = np.random.default_rng(20261019)
    cfg = np.stack([rng.integers(0, 14, N), rng.integers(0, 3, N), rng.integers(0, 2, N)], 1).astype(np.int32)
    e1 = make_engine(k, N)
    ref = _run(e1, cfg)
    del e1
    _dirty_gpu()
    e2 = make_engine(k, N)
    d = {"fresh_engine_after_delete": _diff(ref, _run(e2, cfg))}
    e3 = make_engine(k, N)
    d["engine_while_another_alive"] = _diff(ref, _run(e3, cfg))
    d["older_engine_again"] = _diff(ref, _run(e2, cfg))
    del e2, e3
    engs = [make_engine(k, N) for _ in range(3)]
    out = {}
    th = [threading.Thread(target=lambda i=i: out.__setitem__(i, _run(engs[i], cfg))) for i in range(3)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    d["concurrent_engines"] = sum(_diff(ref, out[i]) for i in range(3))
    print("episodes differing from the first engine:", d, flush=True)
    assert all(v == 0 for v in d.values()), d


def record(path):
    """single engine, fresh process: outputs for a byte comparison across builds."""
    k = X.load_kernel(SO)
    rng = np.random.default_rng(7)
    cfg = np.stack([rng.integers(0, 14, N), rng.integers(0, 3, N), rng.integers(0, 2, N)], 1).astype(np.int32)
    e = make_engine(k, N)
    out = {}
    for tag, sims, temp, eps, mode in (("argmax", 200, 0.0, 0.0, 1), ("train", 200, 1.0, 0.25, 1),
                                       ("rollfwd", 100, 1.0, 0.0, 2), ("legacy", 100, 1.0, 0.0, 0)):
        res = e.run_episodes(np.ascontiguousarray(cfg), sims, 2.0, SEED, temp, 0.3, eps, mode)
        out[tag] = np.frombuffer(b"".join(_flat(res)), np.uint8)
    np.savez(path, **out)
    print("recorded", path, {k_: len(v) for k_, v in out.items()})


if __name__ == "__main__":
    if "--so" in sys.argv:
        SO = sys.argv[sys.argv.index("--so") + 1]
    if "--record" in sys.argv:
        record(sys.argv[sys.argv.index("--record") + 1])
        sys.exit(0)
    try:
        test_engines_isolated()
        print("PASS test_engines_isolated")
    except AssertionError as ex:
        print("FAIL test_engines_isolated", ex)
        sys.exit(1)
