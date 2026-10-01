"""
Shared helpers for the X2 search fix (tests, reference recording, benchmark).

Engine inputs are the paper-1 revision setup: leak-free enriched WP with the
revision's own deploy statistics (paper1_revision.bench_mcts.leaf_config
"new:*"), the rerun2026 GD models (generic_draft_<i>.pt), and a policy
checkpoint as PUCT prior.
"""
import os
import sys
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
for p in (TRAINING_DIR, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch  # noqa: F401  (load libtorch before any kernel .so)

STAT_NAMES = ["searches", "sims", "max_nodes", "max_slots", "cap_hits", "max_eager_nodes",
              "sum_leaf_steps", "sum_own_ahead", "reached_mask", "gd_tree_fwd", "policy_fwd",
              "max_depth"] + [f"hist{k}" for k in range(8)]
MODES = {"legacy": 0, "chance": 1, "rollfwd": 2}
DRAFT_TEAM = [0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1]
DRAFT_IS_PICK = [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1]


def load_kernel(so_path=None, name="cuda_mcts_kernel"):
    """Load a built kernel module from an explicit .so path (default: the
    installed training/cuda_mcts build). One build per process: pybind11
    registers the engine type globally."""
    if so_path is None:
        so = [f for f in os.listdir(HERE) if f.startswith(name) and f.endswith(".so")]
        so_path = os.path.join(HERE, so[0])
    spec = importlib.util.spec_from_file_location(name, so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_CACHE = {}


def rev_leaf():
    """Kernel WP config of the paper-1 revision: leak-free enriched WP + own
    deploy statistics (flat weights, offsets, LUT blob)."""
    if "leaf" not in _CACHE:
        from paper1_revision import bench_mcts
        _CACHE["leaf"] = bench_mcts.leaf_config("new:x_s0")
    return _CACHE["leaf"]


def gd_flat(i):
    from overfit2026 import search
    return search.gd_flats()[i % 5]


def policy(spec):
    """spec: 'new:F_oof_s0' / 'old:J_800sim_s9' (paper-1 checkpoints), or
    'uniform' / 'bc' (overfit2026 priors)."""
    key = ("pol", spec)
    if key not in _CACHE:
        if spec.startswith("path:"):
            from overfit2026 import search
            _CACHE[key] = search.policy_flat(spec)
        elif ":" in spec:
            from paper1_revision import bench_mcts
            _CACHE[key] = bench_mcts.policy_flat(spec)
        else:
            from overfit2026 import search
            _CACHE[key] = search.policy_flat(spec)
    return _CACHE[key]


def make_engine(kernel, policy_spec="new:F_oof_s0", gd_idx=0, max_concurrent=128, device_id=0):
    pf, po = policy(policy_spec)
    gf, go = gd_flat(gd_idx)
    wf, wo, lut = rev_leaf()
    return kernel.MCTSKernelEngine(pf, gf, wf, po, go, wo, lut,
                                   max_concurrent=max_concurrent, device_id=device_id)


def configs(n, seed, tier_idx=1):
    """(map, tier, our_team) rows, same generator as overfit2026.search."""
    from overfit2026 import search
    return np.ascontiguousarray(search.draft_configs(n, seed, tier_idx))


def pack(results):
    """run_episodes output -> dict of arrays (fixed shapes for exact compare)."""
    n = len(results)
    wp = np.array([r[0] for r in results], np.float32)
    ts = np.stack([np.asarray(r[2], np.float32) for r in results])
    ot = np.array([r[3] for r in results], np.int32)
    nt = np.array([len(r[1]) for r in results], np.int32)
    S = np.zeros((n, 8, 290), np.float32)
    P = np.zeros((n, 8, 90), np.float32)
    M = np.zeros((n, 8, 90), np.float32)
    for i, r in enumerate(results):
        for t, (s, p, m) in enumerate(r[1]):
            S[i, t], P[i, t], M[i, t] = s, p, m
    return {"wp": wp, "terminal": ts, "our_team": ot, "n_turns": nt,
            "states": S, "policies": P, "masks": M}


def teams_from_terminal(ts, our):
    t0 = [j for j in range(90) if ts[j] > 0.5]
    t1 = [j for j in range(90) if ts[90 + j] > 0.5]
    return (t0, t1) if our == 0 else (t1, t0)
