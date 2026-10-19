"""
GPU side of the independent MCTS verification (paper 2). Runs the kernels on
fixed inputs and saves raw outputs; every comparison happens on the CPU in
xv_compare.py against the replica in xv_ref.py.

Kernels are loaded from XV_BUILD (separate builds of the committed sources, so
the tournament's builds are never touched):
  $XV_BUILD/cuda_personal  personal_kernel   $XV_BUILD/cuda_prior  prior_kernel
  $XV_BUILD/cuda_pvnet     pvnet_kernel      $XV_BUILD/cuda_explore explore_kernel
  $XV_BUILD/cuda_personal/ref pop_ref_kernel $XV_BUILD/cuda_mcts   cuda_mcts_kernel
Weights: the tournament's cur vintage (F policy seed 0, GD 0, enriched WP).

Stages (all by default): grid variants paper1 identity leaf toy opp stream determinism
Usage (from training/, worker env):
  XV_BUILD=~/hots/xv/build XV_OUT=~/hots/xv/out python <this> [stage ...]
"""
import importlib.util
import json
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING = os.environ.get("XV_TRAINING") or os.path.dirname(os.path.dirname(HERE))
for p in (HERE, TRAINING, os.path.join(TRAINING, "personalization"), os.path.join(TRAINING, "cuda_mcts"),
          os.path.join(TRAINING, "overfit2026")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("P1_TIERS", "site")
import numpy as np
import torch  # noqa: F401  (CUDA runtime before the extensions)

import xv_cases as XC

BUILD = os.path.expanduser(os.environ.get("XV_BUILD", "~/hots/xv/build"))
OUT = os.path.expanduser(os.environ.get("XV_OUT", "~/hots/xv/out"))
os.makedirs(OUT, exist_ok=True)
N_GRID = int(os.environ.get("XV_N", "128"))
GRID = list(range(1, 33)) + list(range(40, 401, 8))
BASE_SEED = 1000003
FSEED = {"personal": 101, "prior": 202, "pvnet": 303, "explore": 404}   # case seeds per flavor
MODS = {"personal": ("cuda_personal", "personal_kernel"), "prior": ("cuda_prior", "prior_kernel"),
        "pvnet": ("cuda_pvnet", "pvnet_kernel"), "explore": ("cuda_explore", "explore_kernel"),
        "popref": ("cuda_personal/ref", "pop_ref_kernel"),
        "paper1": (os.environ.get("XV_PAPER1_DIR", "cuda_mcts"), "cuda_mcts_kernel")}
_K = {}


def kmod(name):
    if name not in _K:
        d, m = MODS[name]
        d = os.path.join(BUILD, d)
        so = [f for f in os.listdir(d) if f.startswith(m) and f.endswith(".so")]
        assert len(so) == 1, (d, so)
        spec = importlib.util.spec_from_file_location(m, os.path.join(d, so[0]))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert getattr(mod, "COMP_FALLBACK_ORDER", None) == "mid_high_low", (name, getattr(mod, "COMP_FALLBACK_ORDER", None))
        _K[name] = mod
    return _K[name]


_W = {}


def weights():
    if not _W:
        import p3t_systems as SY
        v = SY.vintage("cur")
        _W["w"] = v.wcfg()
        _W["p"] = v.policy_flat(0)
        _W["g"] = v.gd_flat(0)
    return _W["p"], _W["g"], _W["w"]


def engine(name, n):
    (pf, po), (gf, go), (wf, wo, lut) = weights()
    k = kmod(name)
    if name == "paper1":
        return k.MCTSKernelEngine(pf, gf, wf, po, go, wo, lut, max_concurrent=n, device_id=0)
    cls = getattr(k, "PersonalEngine", None) or getattr(k, "PriorEngine", None) or getattr(k, "OfitEngine")
    return cls(pf, gf, wf, po, go, wo, lut, max_concurrent=n, device_id=0)


def run_p3(flavor, z, coefs, vnet, sims, c_puct=2.0, seed=BASE_SEED, temp=0.0, pw_k=1.0, pw_alpha=0.5,
           mode=1, eng=None):
    n = len(z["cfg"])
    e = eng or engine(flavor, n)
    pb = XC.pool_bits(z["pools"])
    a = [np.ascontiguousarray(z["cfg"]), np.ascontiguousarray(z["s"]), np.ascontiguousarray(z["off"]),
         np.ascontiguousarray(z["imit"]), np.ascontiguousarray(pb)]
    co = np.asarray(coefs, np.float32)
    if flavor in ("personal", "popref"):
        args = a + [co[:5]]
    elif flavor == "prior":
        args = a + [co, np.ascontiguousarray(z["bias"])]
    else:
        args = a + [co, np.ascontiguousarray(z["bias"]), np.ascontiguousarray(z["feat"]), vnet]
    r = e.run(*args, sims, c_puct, seed, temp, 0.3, 0.0, 1, search_mode=mode, pw_k=pw_k, pw_alpha=pw_alpha)
    return r


def stage_grid():
    """decide-only searches at every k in GRID (prefix property: a k-sim search
    is the first k sims of a longer one with the same seed)."""
    for fl in XC.FLAVORS:
        p = os.path.join(OUT, f"grid_{fl}.npz")
        if os.path.exists(p):
            continue
        t0 = time.time()
        z = XC.make_cases(fl, N_GRID, seed=FSEED[fl])
        vnet = XC.vnet_weights(7)
        e = engine(fl, N_GRID)
        pol, q, prior = [], [], None
        for k in GRID:
            r = run_p3(fl, z, XC.COEFS[fl], vnet, k, eng=e)
            pol.append(np.asarray(r[5])[:, 0])
            q.append(np.asarray(r[6])[:, 0])
            if prior is None and len(r) > 13:
                prior = np.asarray(r[13])[:, 0]
            dec = np.asarray(r[11])
        np.savez(p, pol=np.stack(pol, 1), q=np.stack(q, 1), grid=np.array(GRID), dec=dec,
                 prior=prior if prior is not None else np.zeros(0), vnet=vnet, coefs=np.array(XC.COEFS[fl]),
                 **{k: v for k, v in z.items()})
        print(f"grid {fl}: {time.time() - t0:.0f}s", flush=True)


def stage_variants():
    """c_puct and widening variants (personal kernel), sims 50 and 400."""
    p = os.path.join(OUT, "variants.npz")
    if os.path.exists(p):
        return
    z = XC.make_cases("personal", 64, seed=99)
    e = engine("personal", 64)
    out = {}
    for name, c, k_, a_ in (("c1", 1.0, 1.0, 0.5), ("c4", 4.0, 1.0, 0.5), ("pw0", 2.0, 0.0, 0.5),
                            ("pw2", 2.0, 2.0, 0.6)):
        for sims in (50, 400):
            r = run_p3("personal", z, XC.COEFS["personal"], None, sims, c_puct=c, pw_k=k_, pw_alpha=a_, eng=e)
            out[f"{name}_{sims}_pol"] = np.asarray(r[5])[:, 0]
            out[f"{name}_{sims}_q"] = np.asarray(r[6])[:, 0]
    np.savez(p, **out, **{k: v for k, v in z.items()})
    print("variants done", flush=True)


def paper1_cfg(n, seed):
    rng = np.random.default_rng(seed)
    return np.stack([rng.integers(0, 14, n), rng.integers(0, 3, n), rng.integers(0, 2, n)], 1).astype(np.int32)


def stage_paper1():
    """paper-1 kernel (cuda_mcts, chance search) full episodes, dir_eps 0."""
    p = os.path.join(OUT, "paper1.npz")
    if os.path.exists(p):
        return
    cfg = paper1_cfg(64, 5)
    e = engine("paper1", 64)
    out = {"cfg": cfg}
    for sims in (50, 200):
        for temp in (0.0, 1.0):
            res = e.run_episodes(np.ascontiguousarray(cfg), sims, 2.0, BASE_SEED, temp, 0.3, 0.0, 1)
            P = np.zeros((64, 8, 90), np.float32)
            T = np.zeros((64, 290), np.float32)
            W = np.zeros(64, np.float32)
            for i, r in enumerate(res):
                for t, (s_, pol, m_) in enumerate(r[1]):
                    P[i, t] = pol
                T[i] = r[2]
                W[i] = r[0]
            out[f"s{sims}_t{int(temp)}_pol"], out[f"s{sims}_t{int(temp)}_term"], out[f"s{sims}_t{int(temp)}_wp"] = P, T, W
    np.savez(p, **out)
    print("paper1 done", flush=True)


def stage_identity():
    """(1) P3 personal kernel, population settings (no pools, no forced bans,
    personal valuation off) == paper-1 kernel, full episodes, byte for byte;
    (2) personal valuation on with zero tables and b = (0, 1, *, *) == off;
    (3) cfg[7] = 0 == pop_ref_kernel (the P3 population reference build)."""
    p = os.path.join(OUT, "identity.npz")
    if os.path.exists(p):
        return
    n = 64
    c3 = paper1_cfg(n, 11)
    res = {}
    for sims in (50, 200):
        e1 = engine("paper1", n)
        r1 = e1.run_episodes(np.ascontiguousarray(c3), sims, 2.0, BASE_SEED, 0.0, 0.3, 0.0, 1)
        del e1
        P1 = np.zeros((n, 8, 90), np.float32)
        T1 = np.zeros((n, 290), np.float32)
        for i, r in enumerate(r1):
            for t, (s_, pol, m_) in enumerate(r[1]):
                P1[i, t] = pol
            T1[i] = r[2]
        cfg = np.full((n, 59), -1, np.int32)
        cfg[:, 0:3] = c3
        cfg[:, 3:8] = 0
        cfg[:, 4] = -1
        cfg[:, 40:56] = -1
        cfg[:, 56:59] = 0
        zz = dict(cfg=cfg, s=np.zeros((n, 10, 90), np.float32), off=np.zeros((n, 10, 90), np.float32),
                  imit=np.zeros((n, 10, 90), np.float32), pools=np.ones((n, 10, 90), bool))
        r2 = run_p3("personal", zz, [0, 1, 0, 0, 0], None, sims)
        P2 = np.asarray(r2[5])[:, :8]
        term2 = np.asarray(r2[3])
        T2 = np.zeros((n, 290), np.float32)
        for i in range(n):
            for k in range(16):
                h = term2[i, k]
                if XC.PICK[k]:
                    T2[i, (90 if XC.TEAM[k] else 0) + h] = 1
                else:
                    T2[i, 180 + h] = 1
        res[f"s{sims}_P1"], res[f"s{sims}_P2"], res[f"s{sims}_T1"], res[f"s{sims}_T2"] = P1, P2, T1, T2
        res[f"s{sims}_pol_equal"] = np.array(P1.tobytes() == np.ascontiguousarray(P2).tobytes())
        res[f"s{sims}_draft_equal"] = np.array(np.array_equal(T1[:, :270], T2[:, :270]))
        res[f"s{sims}_pol_maxdiff"] = np.array(np.abs(P1 - P2).max())
        # (2) aware with zero tables and b = (0, 1, x, y)
        zz2 = dict(zz)
        zz2["cfg"] = cfg.copy()
        zz2["cfg"][:, 7] = 1
        zz2["cfg"][:, 58] = 1
        r3 = run_p3("personal", zz2, [0, 1, 4.5, -0.05, 0], None, sims)
        res[f"s{sims}_zero_tables_equal"] = np.array(all(np.asarray(a).tobytes() == np.asarray(b).tobytes()
                                                         for a, b in zip(r2[:7], r3[:7])))
    # (3) the P3 population reference build (pop_ref_kernel, full episodes) == paper-1 kernel
    for sims in (50, 200):
        e1 = engine("paper1", n)
        ra = e1.run_episodes(np.ascontiguousarray(c3), sims, 2.0, BASE_SEED, 0.0, 0.3, 0.0, 1)
        del e1
        e2 = engine("popref", n)
        rb_ = e2.run_episodes(np.ascontiguousarray(c3), sims, 2.0, BASE_SEED, 0.0, 0.3, 0.0, 1, 1)
        del e2
        fl_ = lambda res: b"".join(np.asarray(r[2], np.float32).tobytes() + np.float32(r[0]).tobytes()
                                   + b"".join(np.asarray(x[1], np.float32).tobytes() for x in r[1]) for r in res)
        res[f"s{sims}_popref_equal"] = np.array(fl_(ra) == fl_(rb_))
    # (4) decide-only adversarial cases, personal valuation off
    z = XC.make_cases("personal", 64, seed=4242)
    z["cfg"][:, 7] = 0
    rp = run_p3("personal", z, XC.COEFS["personal"], None, 200)
    # (4) blind run invariant to the tables
    z2 = dict(z)
    z2["s"] = z["s"] * 7 + 1
    z2["off"] = 1 - z["off"]
    rb = run_p3("personal", z2, XC.COEFS["personal"], None, 200)
    res["blind_tables_invariant"] = np.array(all(np.asarray(a).tobytes() == np.asarray(b).tobytes()
                                                 for a, b in zip(rp[:7], rb[:7])))
    np.savez(p, **res)
    print("identity", {k: v.tolist() for k, v in res.items() if v.size == 1}, flush=True)


def leaf_args(flavor, cfg, s, off, imit, pools, coefs, bias, feat, vnet):
    pb = XC.pool_bits(pools)
    a = [np.ascontiguousarray(cfg), np.ascontiguousarray(s), np.ascontiguousarray(off),
         np.ascontiguousarray(imit), np.ascontiguousarray(pb), np.asarray(coefs, np.float32)]
    if flavor == "personal":
        a[-1] = a[-1][:5]
        return a
    if flavor == "prior":
        return a + [np.ascontiguousarray(bias)]
    return a + [np.ascontiguousarray(bias), np.ascontiguousarray(feat), vnet]


def stage_leaf():
    """leaf_eval on complete drafts (40% with one team stacked in one role, so
    rare or unknown compositions and the fallback order are exercised), with
    three symmetry variants: our side flipped; teams and tables swapped."""
    p = os.path.join(OUT, "leaf.npz")
    if os.path.exists(p):
        return
    rng = np.random.default_rng(31)
    n = 512
    acts = XC.complete_drafts(rng, n)
    base = XC.make_cases("pvnet", n, seed=77)
    s, off, imit, bias, feat, pools = (base[k] for k in ("s", "off", "imit", "bias", "feat", "pools"))
    # extreme rows on a quarter
    s[: n // 4] *= 10
    cfg = np.full((n, 59), -1, np.int32)
    cfg[:, 0], cfg[:, 1], cfg[:, 2] = rng.integers(0, 14, n), rng.integers(0, 3, n), rng.integers(0, 2, n)
    cfg[:, 3:8] = 0
    cfg[:, 7] = 1
    cfg[:, 8:24] = acts
    cfg[:, 24:40] = XC.slot_of_default()
    cfg[:, 56:59] = 0
    cfg[:, 58] = 1
    # swapped: team 1's picks at team 0's pick steps (in order) and vice versa, rows swapped
    acts_sw = acts.copy()
    p0 = [k for k in range(16) if XC.PICK[k] and XC.TEAM[k] == 0]
    p1 = [k for k in range(16) if XC.PICK[k] and XC.TEAM[k] == 1]
    acts_sw[:, p0], acts_sw[:, p1] = acts[:, p1], acts[:, p0]
    sw = lambda x: np.concatenate([x[:, 5:], x[:, :5]], 1)
    vnet = XC.vnet_weights(7)
    out = {"acts": acts, "cfg": cfg, "s": s, "off": off, "feat": feat, "vnet": vnet}
    for fl, pv in (("personal", 0), ("pvnet", 0), ("pvnet", 1)):
        co = XC.COEFS["personal"] if fl == "personal" else XC.COEFS["pvnet"]
        e = engine(fl, n)
        c = cfg.copy()
        c[:, 57] = pv
        tag = f"{fl}{pv}"
        out[f"{tag}_base"] = np.asarray(e.leaf_eval(*leaf_args(fl, c, s, off, imit, pools, co, bias, feat, vnet)))
        c2 = c.copy()
        c2[:, 2] = 1 - c[:, 2]
        out[f"{tag}_flip"] = np.asarray(e.leaf_eval(*leaf_args(fl, c2, s, off, imit, pools, co, bias, feat, vnet)))
        c3 = c.copy()
        c3[:, 8:24] = acts_sw
        out[f"{tag}_swap"] = np.asarray(e.leaf_eval(*leaf_args(fl, c3, sw(s), sw(off), sw(imit), sw(pools), co,
                                                               sw(bias), sw(feat), vnet)))
        out[f"{tag}_coefs"] = np.array(co)
        if fl == "personal":
            c4 = c.copy()
            c4[:, 58] = 0
            out[f"{tag}_slot"] = np.asarray(e.leaf_eval(*leaf_args(fl, c4, s, off, imit, pools, co, bias, feat,
                                                                   vnet)))
        del e
    np.savez(p, **out)
    print("leaf done", flush=True)


def stage_toy():
    """last decisions with exactly computable values: step 15 (our pick ends
    the draft: Q(a) = V(draft + a)) and step 14 (team 0's last pick, then one
    GD reply). sims 25/100/400/1600, 4 kernel seeds; plus leaf_eval of every
    completion at step 15 (the exact targets)."""
    p = os.path.join(OUT, "toy.npz")
    if os.path.exists(p):
        return
    rng = np.random.default_rng(55)
    out = {}
    for k in (15, 14):
        n = 48
        z = XC.make_cases("personal", n, seed=500 + k)
        for i in range(n):
            pools = rng.random((10, 90)) < 0.25
            acts = np.full(16, -1, np.int64)
            for j in range(k):
                m = XC.legal_mask(acts, j, pools)
                acts[j] = int(rng.choice(np.flatnonzero(m)))
            z["pools"][i] = pools
            z["cfg"][i, 8:24] = acts
            z["cfg"][i, 4] = k
            z["cfg"][i, 2] = XC.TEAM[k]
            z["cfg"][i, 5] = 1
            z["cfg"][i, 7] = 1
            z["cfg"][i, 58] = 1
            z["cfg"][i, 40:56] = -1
            z["s"][i] = rng.normal(0, 0.4, (10, 90))
        e = engine("personal", n)
        for sims in (25, 100, 400, 1600):
            for sd in range(4):
                r = run_p3("personal", z, XC.COEFS["personal"], None, sims, seed=BASE_SEED + 7919 * sd, eng=e)
                out[f"k{k}_s{sims}_sd{sd}_pol"] = np.asarray(r[5])[:, 0]
                out[f"k{k}_s{sims}_sd{sd}_q"] = np.asarray(r[6])[:, 0]
        for kk, v in z.items():
            out[f"k{k}_{kk}"] = v
        if k == 15:
            # exact targets: every legal completion
            rows = []
            for i in range(n):
                acts = z["cfg"][i, 8:24].copy()
                m = XC.legal_mask(acts, 15, z["pools"][i])
                for h in np.flatnonzero(m):
                    a2 = acts.copy()
                    a2[15] = h
                    rows.append((i, h, a2))
            c = np.repeat(z["cfg"][[r[0] for r in rows]], 1, 0)
            c[:, 8:24] = np.stack([r[2] for r in rows])
            ii = np.array([r[0] for r in rows])
            vals = []
            for b0 in range(0, len(rows), 1024):
                sl = slice(b0, b0 + 1024)
                e2 = engine("personal", 1024)
                vals.append(np.asarray(e2.leaf_eval(*leaf_args("personal", c[sl], z["s"][ii[sl]], z["off"][ii[sl]],
                                                              z["imit"][ii[sl]], z["pools"][ii[sl]],
                                                              XC.COEFS["personal"], None, None, None))))
                del e2
            out["k15_exact_i"] = ii
            out["k15_exact_h"] = np.array([r[1] for r in rows])
            out["k15_exact_v"] = np.concatenate(vals)
        del e
    np.savez(p, **out)
    print("toy done", flush=True)


def stage_opp():
    """main-loop opponent distribution at a fixed state, 8192 episodes each:
    explore cfg[6] in {0, 1, 2}, opponent = team 0 at step 4 (forced bans at
    steps 0-3 fix the state); paper-1 kernel: the opponent's first ban (team 0,
    empty draft). 1 sim per own turn (the own turns do not affect these
    draws: the opponent acts before our first search)."""
    p = os.path.join(OUT, "opp.npz")
    if os.path.exists(p):
        return
    n = 1024          # P3 episode memory is about 2 MB per episode
    nb = 8
    out = {}
    rng = np.random.default_rng(8)
    bans = rng.choice(90, 4, replace=False)
    imit_row = rng.normal(0, 1.5, 90).astype(np.float32)
    for mode in (0, 1, 2):
        cfg = np.full((n, 59), -1, np.int32)
        cfg[:, 0], cfg[:, 1], cfg[:, 2] = 4, 1, 1
        cfg[:, 3:8] = 0
        cfg[:, 4] = -1
        cfg[:, 6] = mode
        cfg[:, 24:40] = XC.slot_of_default()
        cfg[:, 40:44] = bans
        cfg[:, 56:59] = 0
        imit = np.zeros((n, 10, 90), np.float32)
        imit[:, 0] = imit_row          # slot of step 4 (team 0's first pick) is row 0
        z = dict(cfg=cfg, s=np.zeros((n, 10, 90), np.float32), off=np.zeros((n, 10, 90), np.float32), imit=imit,
                 bias=np.zeros((n, 10, 90), np.float32), feat=np.zeros((n, 10, 90), np.float32),
                 pools=np.ones((n, 10, 90), bool))
        draws = []
        e = engine("explore", n)
        for b in range(nb):
            r = run_p3("explore", z, [0, 1, 0, 0, 0.7, 1.0], XC.vnet_weights(7), 1, seed=BASE_SEED + 100000 * b + 1,
                       eng=e)
            draws.append(np.asarray(r[3])[:, 4])
        del e
        out[f"explore_m{mode}_step4"] = np.concatenate(draws)
    out["bans"] = bans
    out["imit_row"] = imit_row
    e = engine("paper1", 4096)
    c3 = np.zeros((4096, 3), np.int32)
    c3[:, 0], c3[:, 1], c3[:, 2] = 4, 1, 1
    first = []
    for b in range(2):   # 8192 episodes
        res = e.run_episodes(c3, 1, 2.0, BASE_SEED + 7 + 100000 * b, 0.0, 0.3, 0.0, 1)
        # the first ban (team 0, step 0) from turn 0's saved state (we act at step 1)
        st0 = np.stack([np.asarray(r[1][0][0]) for r in res])
        assert (st0[:, 180:270].sum(1) == 1).all()
        first.append(st0[:, 180:270].argmax(1))
    out["paper1_first_ban"] = np.concatenate(first)
    np.savez(p, **out)
    print("opp done", flush=True)


def stage_stream():
    """paper-1 engines: three engines in three threads at once (per-engine
    streams, GIL released) give byte-identical outputs to serial runs."""
    p = os.path.join(OUT, "stream.npz")
    if os.path.exists(p):
        return
    cfgs = [paper1_cfg(96, 100 + i) for i in range(3)]
    engs = [engine("paper1", 96) for _ in range(3)]

    def go(i, store):
        store[i] = engs[i].run_episodes(np.ascontiguousarray(cfgs[i]), 200, 2.0, BASE_SEED + i, 1.0, 0.3, 0.25, 1)
    ser = {}
    for i in range(3):
        go(i, ser)
    par = {}
    th = [threading.Thread(target=go, args=(i, par)) for i in range(3)]
    t0 = time.time()
    for t in th:
        t.start()
    for t in th:
        t.join()
    dt = time.time() - t0

    def flat(res):
        return b"".join(np.asarray(r[2], np.float32).tobytes() + np.float32(r[0]).tobytes()
                        + b"".join(np.asarray(x[1], np.float32).tobytes() for x in r[1]) for r in res)
    eq = [flat(ser[i]) == flat(par[i]) for i in range(3)]
    np.savez(p, equal=np.array(eq), secs=np.array(dt))
    print("stream", eq, flush=True)


def stage_determinism():
    p = os.path.join(OUT, "determinism.npz")
    if os.path.exists(p):
        return
    res = {}
    for fl in XC.FLAVORS:
        z = XC.make_cases(fl, 64, seed=FSEED[fl])
        vnet = XC.vnet_weights(7)
        a = run_p3(fl, z, XC.COEFS[fl], vnet, 400)
        b = run_p3(fl, z, XC.COEFS[fl], vnet, 400)
        c = run_p3(fl, z, XC.COEFS[fl], vnet, 400, seed=BASE_SEED + 1)
        res[f"{fl}_same"] = np.array(all(np.asarray(x).tobytes() == np.asarray(y).tobytes() for x, y in zip(a, b)))
        res[f"{fl}_other_seed_differs"] = np.array(np.asarray(a[5]).tobytes() != np.asarray(c[5]).tobytes())
    np.savez(p, **res)
    print("determinism", {k: bool(v) for k, v in res.items()}, flush=True)


STAGES = {"grid": stage_grid, "variants": stage_variants, "paper1": stage_paper1, "identity": stage_identity,
          "leaf": stage_leaf, "toy": stage_toy, "opp": stage_opp, "stream": stage_stream,
          "determinism": stage_determinism}



def stage_identity_raw():
    """raw outputs behind stage_identity for a CPU-side diagnosis."""
    p = os.path.join(OUT, "identity_raw.npz")
    n = 16
    c3 = paper1_cfg(n, 11)
    out = {"c3": c3}
    e1 = engine("paper1", n)
    r1 = e1.run_episodes(np.ascontiguousarray(c3), 50, 2.0, BASE_SEED, 0.0, 0.3, 0.0, 1)
    del e1
    P1 = np.zeros((n, 8, 90), np.float32)
    S1 = np.zeros((n, 8, 290), np.float32)
    T1 = np.zeros((n, 290), np.float32)
    for i, r in enumerate(r1):
        for t, (s_, pol, m_) in enumerate(r[1]):
            P1[i, t], S1[i, t] = pol, s_
        T1[i] = r[2]
    out.update(P1=P1, S1=S1, T1=T1)
    cfg = np.full((n, 59), -1, np.int32)
    cfg[:, 0:3] = c3
    cfg[:, 3:8] = 0
    cfg[:, 4] = -1
    cfg[:, 56:59] = 0
    zz = dict(cfg=cfg, s=np.zeros((n, 10, 90), np.float32), off=np.zeros((n, 10, 90), np.float32),
              imit=np.zeros((n, 10, 90), np.float32), pools=np.ones((n, 10, 90), bool))
    r2 = run_p3("personal", zz, [0, 1, 0, 0, 0], None, 50)
    for j, nm in enumerate(("win", "wpt0", "vt0", "term", "turns", "opol", "oq", "ostep", "oteam")):
        out["P3_" + nm] = np.asarray(r2[j])
    e2 = engine("popref", n)
    rb_ = e2.run_episodes(np.ascontiguousarray(c3), 50, 2.0, BASE_SEED, 0.0, 0.3, 0.0, 1, 1)
    PR = np.zeros((n, 8, 90), np.float32)
    TR = np.zeros((n, 290), np.float32)
    for i, r in enumerate(rb_):
        for t, (s_, pol, m_) in enumerate(r[1]):
            PR[i, t] = pol
        TR[i] = r[2]
    out.update(PR=PR, TR=TR)
    # explore full drafts as in stage_opp (mode 0, batch 0), full outputs
    g = np.load(os.path.join(OUT, "opp.npz"))
    n2 = 1024
    cfg = np.full((n2, 59), -1, np.int32)
    cfg[:, 0], cfg[:, 1], cfg[:, 2] = 4, 1, 1
    cfg[:, 3:8] = 0
    cfg[:, 4] = -1
    cfg[:, 24:40] = XC.slot_of_default()
    cfg[:, 40:44] = g["bans"]
    cfg[:, 56:59] = 0
    z = dict(cfg=cfg, s=np.zeros((n2, 10, 90), np.float32), off=np.zeros((n2, 10, 90), np.float32),
             imit=np.zeros((n2, 10, 90), np.float32), bias=np.zeros((n2, 10, 90), np.float32),
             feat=np.zeros((n2, 10, 90), np.float32), pools=np.ones((n2, 10, 90), bool))
    for sims in (1, 50):
        for fl in ("explore", "personal"):
            r = run_p3(fl, z, [0, 1, 0, 0, 0.7, 1.0], XC.vnet_weights(7), sims, seed=BASE_SEED + 1)
            for j, nm in enumerate(("win", "wpt0", "vt0", "term", "turns", "opol", "oq", "ostep", "oteam")):
                out[f"{fl}{sims}_{nm}"] = np.asarray(r[j])
            r = run_p3(fl, z, [0, 1, 0, 0, 0.7, 1.0], XC.vnet_weights(7), sims, seed=BASE_SEED + 1)
            out[f"{fl}{sims}_term_again"] = np.asarray(r[3])
    np.savez(p, **out)
    print("identity_raw saved", flush=True)


STAGES["identity_raw"] = stage_identity_raw


def stage_race():
    """rerun determinism at scale. (a) decide-only (the tournament mode: prefix
    replayed through the main loop's skip path), 1024 episodes, 3 runs per
    flavor; (b) full drafts with self-play and no skipped steps; (c) full drafts
    with forced bans (skipped steps). Counts episodes whose outputs differ
    between identical runs and drafts with a repeated hero."""
    p = os.path.join(OUT, "race.npz")
    out = {}
    n = 1024
    for fl in XC.FLAVORS:
        z = XC.make_cases(fl, n, seed=FSEED[fl] + 5)
        z["cfg"][:, 40:56] = -1                       # tournament: no forced bans
        vnet = XC.vnet_weights(7)
        e = engine(fl, n)
        runs = [run_p3(fl, z, XC.COEFS[fl], vnet, 400, eng=e) for _ in range(3)]
        pol = [np.asarray(r[5])[:, 0] for r in runs]
        diff = np.zeros(n, bool)
        for r in runs[1:]:
            diff |= (np.asarray(r[5])[:, 0] != pol[0]).any(1) | (np.asarray(r[6])[:, 0] != np.asarray(runs[0][6])[:, 0]).any(1)
        out[f"decide_{fl}_differ"] = diff
        out[f"decide_{fl}_k"] = z["cfg"][:, 4]
        del e
        print("race decide", fl, int(diff.sum()), flush=True)
    for tag, forced in (("selfplay", False), ("forced", True)):
        for fl in ("personal", "explore"):
            cfg = np.full((n, 59), -1, np.int32)
            rng = np.random.default_rng(9)
            cfg[:, 0], cfg[:, 1], cfg[:, 2] = rng.integers(0, 14, n), rng.integers(0, 3, n), rng.integers(0, 2, n)
            cfg[:, 3:8] = 0
            cfg[:, 3] = 0 if forced else 1
            cfg[:, 4] = -1
            cfg[:, 5] = 1
            cfg[:, 24:40] = XC.slot_of_default()
            cfg[:, 56:59] = 0
            cfg[:, 58] = 1
            if forced:
                for i in range(n):
                    cfg[i, 40:44] = rng.choice(90, 4, replace=False)
                    cfg[i, 49:51] = rng.choice(90, 2, replace=False)
            zz = XC.make_cases(fl, n, seed=31)
            zz["cfg"] = cfg
            e = engine(fl, n)
            runs = [np.asarray(run_p3(fl, zz, XC.COEFS[fl], XC.vnet_weights(7), 50, eng=e)[3]) for _ in range(3)]
            del e
            diff = np.zeros(n, bool)
            for r in runs[1:]:
                diff |= (r != runs[0]).any(1)
            dup = np.array([sum(len(set(r[i].tolist())) != 16 for r in runs) for i in range(n)])
            out[f"{tag}_{fl}_differ"] = diff
            out[f"{tag}_{fl}_dup"] = dup
            print("race", tag, fl, "differ", int(diff.sum()), "dup drafts", int((dup > 0).sum()), flush=True)
    np.savez(p, **out)


STAGES["race"] = stage_race


def stage_p1race():
    """paper-1 engine (per-engine non-blocking stream): first run right after
    construction vs a second run on the same engine vs a run after a pause, with
    other engines' allocations freed in between."""
    n = 64
    c3 = paper1_cfg(n, 11)
    flat = lambda res: [np.asarray(r[2], np.float32).tobytes() + b"".join(np.asarray(x[1], np.float32).tobytes()
                                                                        for x in r[1]) for r in res]
    z = XC.make_cases("personal", n, seed=4242)
    ref = None
    rows = []
    for trial in range(8):
        if trial % 2 == 0:
            run_p3("personal", z, XC.COEFS["personal"], None, 200)          # dirty the allocator
        e = engine("paper1", n)
        a = flat(e.run_episodes(np.ascontiguousarray(c3), 200, 2.0, BASE_SEED, 0.0, 0.3, 0.0, 1))
        b = flat(e.run_episodes(np.ascontiguousarray(c3), 200, 2.0, BASE_SEED, 0.0, 0.3, 0.0, 1))
        del e
        e = engine("paper1", n)
        time.sleep(1.0)
        c = flat(e.run_episodes(np.ascontiguousarray(c3), 200, 2.0, BASE_SEED, 0.0, 0.3, 0.0, 1))
        del e
        ref = ref or b
        row = [sum(x != y for x, y in zip(a, ref)), sum(x != y for x, y in zip(b, ref)), sum(x != y for x, y in zip(c, ref))]
        rows.append(row)
        print("p1race trial", trial, "episodes differing from reference: first run", row[0], "second run", row[1],
              "after 1 s", row[2], flush=True)
    np.savez(os.path.join(OUT, "p1race.npz"), rows=np.array(rows))


STAGES["p1race"] = stage_p1race


def stage_p1diag():
    """paper-1 engine: first engine in the process vs later engines, with the
    device allocator dirtied by a torch tensor of garbage in between."""
    n = 64
    c3 = paper1_cfg(n, 11)
    flat = lambda res: [np.asarray(r[2], np.float32).tobytes() + b"".join(np.asarray(x[1], np.float32).tobytes()
                                                                        for x in r[1]) for r in res]
    run = lambda: flat(engine("paper1", n).run_episodes(np.ascontiguousarray(c3), 200, 2.0, BASE_SEED, 0.0, 0.3, 0.0, 1))
    A = run()
    out = {"second_engine": sum(x != y for x, y in zip(A, run()))}
    for name, val in (("garbage_0x55", 0x55), ("garbage_zero", 0), ("garbage_0xff", 0xFF)):
        t = torch.full((3 * 2 ** 30,), val, dtype=torch.uint8, device="cuda")
        torch.cuda.synchronize()
        del t
        torch.cuda.empty_cache()
        out[name] = sum(x != y for x, y in zip(A, run()))
    print("p1diag", os.environ.get("XV_PAPER1_DIR"), json.dumps(out), flush=True)
    np.savez(os.path.join(OUT, f"p1diag_{os.path.basename(os.path.dirname(os.environ.get('XV_PAPER1_DIR', 'x/x')))}.npz"),
             **{k: np.array(v) for k, v in out.items()})


STAGES["p1diag"] = stage_p1diag


def stage_p1diag2():
    """where does the paper-1 first-engine/later-engine difference live?"""
    n = 64
    c3 = paper1_cfg(n, 11)
    flat = lambda res: [np.asarray(r[2], np.float32).tobytes() + b"".join(np.asarray(x[1], np.float32).tobytes()
                                                                        for x in r[1]) for r in res]
    run = lambda e: flat(e.run_episodes(np.ascontiguousarray(c3), 200, 2.0, BASE_SEED, 0.0, 0.3, 0.0, 1))
    diff = lambda a, b: sum(x != y for x, y in zip(a, b))
    out = {}
    e1 = engine("paper1", n)
    A = run(e1)
    z = XC.make_cases("personal", n, seed=4242)
    run_p3("personal", z, XC.COEFS["personal"], None, 200)
    out["e1_after_p3_kernel"] = diff(A, run(e1))
    e2 = engine("paper1", n)                 # e1 still alive
    out["e2_while_e1_alive"] = diff(A, run(e2))
    out["e1_after_e2"] = diff(A, run(e1))
    e3 = engine("paper1", n // 2)            # different size
    B = flat(e3.run_episodes(np.ascontiguousarray(c3[: n // 2]), 200, 2.0, BASE_SEED, 0.0, 0.3, 0.0, 1))
    out["e3_half_batch"] = diff(A[: n // 2], B)
    e4 = engine("paper1", n)
    e4.run_episodes(np.ascontiguousarray(c3), 1, 2.0, BASE_SEED, 0.0, 0.3, 0.0, 1)   # warm with a 1-sim run
    out["e4_after_1sim_run"] = diff(A, run(e4))
    e5 = engine("paper1", n)
    out["e5_sims_400_then_200"] = None
    e5.run_episodes(np.ascontiguousarray(c3), 400, 2.0, BASE_SEED, 0.0, 0.3, 0.0, 1)
    out["e5_sims_400_then_200"] = diff(A, run(e5))
    print("p1diag2", json.dumps(out), flush=True)


STAGES["p1diag2"] = stage_p1diag2


def stage_p1dbg():
    """instrumented paper-1 build: out-of-range arena reads (stats[19]) in clean and dirty runs."""
    n = 64
    c3 = paper1_cfg(n, 11)
    flat = lambda res: [np.asarray(r[2], np.float32).tobytes() + b"".join(np.asarray(x[1], np.float32).tobytes()
                                                                        for x in r[1]) for r in res]
    e = engine("paper1", n)
    A = flat(e.run_episodes(np.ascontiguousarray(c3), 200, 2.0, BASE_SEED, 0.0, 0.3, 0.0, 1))
    st = np.asarray(e.last_stats())
    print("p1dbg clean run: bad reads by check 12..19", st[:, 12:20].sum(0).tolist(), flush=True)
    e.run_episodes(np.ascontiguousarray(c3), 400, 2.0, BASE_SEED + 5, 0.0, 0.3, 0.0, 1)
    st = np.asarray(e.last_stats())
    print("p1dbg 400-sim run: bad reads by check", st[:, 12:20].sum(0).tolist(), flush=True)
    B = flat(e.run_episodes(np.ascontiguousarray(c3), 200, 2.0, BASE_SEED, 0.0, 0.3, 0.0, 1))
    st = np.asarray(e.last_stats())
    print("p1dbg dirty 200-sim run: differs", sum(x != y for x, y in zip(A, B)), "bad reads by check",
          st[:, 12:20].sum(0).tolist(), flush=True)


STAGES["p1dbg"] = stage_p1dbg


def stage_p1tree():
    """paper-1 engine trees, first engine vs a later engine (allocator dirtied), same inputs:
    first node where they differ and which field."""
    n = 8
    c3 = paper1_cfg(64, 11)[:n]
    out = {}
    e = engine("paper1", n)
    t1 = e.debug_tree(np.ascontiguousarray(c3), 200, 2.0, BASE_SEED, 0, 1, 1.0, 0.5, 0.0, 0.3, 0.0)
    del e
    z = XC.make_cases("personal", 64, seed=4242)
    run_p3("personal", z, XC.COEFS["personal"], None, 200)
    e = engine("paper1", 64)
    e.run_episodes(np.ascontiguousarray(paper1_cfg(64, 11)), 400, 2.0, BASE_SEED, 0.0, 0.3, 0.0, 1)
    del e
    e = engine("paper1", n)
    t2 = e.debug_tree(np.ascontiguousarray(c3), 200, 2.0, BASE_SEED, 0, 1, 1.0, 0.5, 0.0, 0.3, 0.0)
    for i, (a, b) in enumerate(zip(t1, t2)):
        rep = {}
        for f in ("parent", "action", "visits", "slot", "n_kids", "step", "kind", "value_sum"):
            x, y = np.asarray(a[f]), np.asarray(b[f])
            m = min(len(x), len(y))
            d = np.flatnonzero(x[:m] != y[:m])
            rep[f] = int(d[0]) if len(d) else (None if len(x) == len(y) else f"len {len(x)} vs {len(y)}")
        sp1, sp2 = np.asarray(a["slot_p"]), np.asarray(b["slot_p"])
        m = min(len(sp1), len(sp2))
        d = np.flatnonzero(np.abs(sp1[:m] - sp2[:m]).max(1) > 0)
        rep["slot_p_first"] = int(d[0]) if len(d) else None
        if len(d):
            k = int(d[0])
            rep["slot_p_maxdiff"] = float(np.abs(sp1[k] - sp2[k]).max())
            rep["slot_p_sum"] = [float(sp1[k].sum()), float(sp2[k].sum())]
            owner = [j for j in range(len(a["slot"])) if a["slot"][j] == k]
            rep["slot_owner_kind_step"] = [(int(a["kind"][j]), int(a["step"][j])) for j in owner]
        out[i] = rep
        print("p1tree ep", i, json.dumps(rep), flush=True)
    np.savez(os.path.join(OUT, "p1tree.npz"), t1=np.array(t1, dtype=object), t2=np.array(t2, dtype=object), allow_pickle=True)


STAGES["p1tree"] = stage_p1tree


def stage_p1host():
    """device weight buffers of the paper-1 engine read back and compared with the host arrays,
    for the first engine and for later engines, plus the root Q of a search."""
    n = 64
    c3 = paper1_cfg(n, 11)
    (pf, po), (gf, go), (wf, wo, lut) = weights()
    host = (np.asarray(pf, np.float32).tobytes(), np.asarray(wf, np.float32).tobytes(), np.asarray(lut).tobytes(),
            np.asarray(gf, np.float32).tobytes())
    z = XC.make_cases("personal", n, seed=4242)
    for k in range(4):
        if k:
            run_p3("personal", z, XC.COEFS["personal"], None, 200)
        e = engine("paper1", n)
        rb = e.read_back()
        same = [np.asarray(x).tobytes() == h for x, h in zip(rb, host)]
        res = e.run_episodes(np.ascontiguousarray(c3[:8]), 200, 2.0, BASE_SEED, 0.0, 0.3, 0.0, 1)
        rb2 = e.read_back()
        same2 = [np.asarray(x).tobytes() == h for x, h in zip(rb2, host)]
        print("p1host engine", k, "policy/wp/lut/gd equal to host before run", same, "after run", same2,
              "wp mean terminal", float(np.mean([r[0] for r in res])), flush=True)
        del e


STAGES["p1host"] = stage_p1host


if __name__ == "__main__":
    want = sys.argv[1:] or list(STAGES)
    t0 = time.time()
    for s in want:
        t1 = time.time()
        STAGES[s]()
        print(f"== {s} {time.time() - t1:.0f}s", flush=True)
    print(f"XV GPU DONE {time.time() - t0:.0f}s", flush=True)
