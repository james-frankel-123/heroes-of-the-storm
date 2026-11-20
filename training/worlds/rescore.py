"""
Rescore existing drafts under each world's truth (step 2 of the plan, item 3).

Paper 1 (stage p1): every tournament draft ood_frontier.records() finds (frontier agents, CQL / IQL /
MCQ, floor variants, ft, ft_sf, search_sf, search_ft_sf, ensemble LCB, ...), deduplicated as
ood_frontier.score. A strategy's value in a world = mean over the common opponent set (OPP8 minus
constrained_mcts, as frontier_excl_constrained_mcts.json; an entrant never plays itself) of its mean
true win probability, both seats. For each family and each dial cell of the frozen grid, the
per-world value of every strategy is stored (site/cache/rescore_p1_<fam>.npz, worlds x cells x
strategies); summaries in site/results/rescore_p1.json.

Paper 2 (stage p2): drafts in personalization/cache_def_def1027_hlsup/p3t/tourney (headline lobbies,
acts (300, 16)): the selected design and the support arms against each opponent, both seats.
"""
import os
import sys
import json
import time
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, TRAINING_DIR)
import data as WD  # noqa: E402
import model as M  # noqa: E402
import world as W  # noqa: E402

NPROC = int(os.environ.get("WORLDS_PROCS", "16"))
OPP8 = ["mcts", "constrained_mcts", "enriched", "constrained_greedy", "k_oof", "gourdeau", "gourdeau_disc", "gd"]
EXCL = ["constrained_mcts"]
CONTRASTS_P1 = [  # (X, Y): report X - Y per world
    ("constrained_mcts_sf", "constrained_mcts"), ("constrained_mcts_sf10", "constrained_mcts"),
    ("constrained_mcts_sf50", "constrained_mcts"), ("constrained_mcts_sf100", "constrained_mcts"),
    ("constrained_mcts_sf200", "constrained_mcts"), ("constrained_mcts_ft", "constrained_mcts"),
    ("constrained_mcts_ft_sf", "constrained_mcts"), ("constrained_mcts_ft_sf", "constrained_mcts_ft"),
    ("search_sf", "constrained_mcts"), ("search_ft_sf", "constrained_mcts"), ("search_sf", "mcts_search"),
    ("constrained_k_sf", "constrained_k"), ("constrained_greedy_sf", "constrained_greedy"),
    ("constrained_wr_sf", "constrained_wr"), ("mcts_lcb", "mcts_search"), ("mcts_lcb2", "mcts_search"),
    ("mcq_t0.5", "constrained_mcts"), ("cql_enr_a2.0", "constrained_mcts"), ("iql_td_t0.7_b10.0", "constrained_mcts"),
    ("mcts", "constrained_mcts")]


# ── paper 1 drafts ───────────────────────────────────────────────────────
def p1_drafts():
    p = os.path.join(WD.CACHE, "p1_drafts.npz")
    if os.path.exists(p):
        z = np.load(p, allow_pickle=False)
        return {k: z[k] for k in z.files}
    from paper1_revision import ood_frontier as of
    seen, recs = set(), []
    for a, b, r in of.records():
        if not (a in OPP8 or b in OPP8):
            continue
        k = (a, b, r["draft"], r["game_map"], tuple(r["team0"]["picks"]), tuple(r["team1"]["picks"]))
        if k not in seen:
            seen.add(k)
            recs.append((a, b, r))
    D = WD.from_rows([(r["team0"]["picks"], r["team1"]["picks"], r["game_map"], r["tier"]) for _, _, r in recs])
    D["S0"] = np.array([a for a, _, _ in recs])
    D["S1"] = np.array([b for _, b, _ in recs])
    np.savez(p, **D)
    print(f"paper-1 drafts: {len(recs):,}", flush=True)
    return D


def p1_index(D):
    """strategy -> list of (opponent, idx where strategy is team 0, idx where team 1), common opponent set."""
    names = sorted(set(D["S0"]) | set(D["S1"]))
    out = {}
    for s in names:
        opps = [o for o in OPP8 if o != s and o not in EXCL]
        L = []
        for o in opps:
            f = np.flatnonzero((D["S0"] == s) & (D["S1"] == o))
            g = np.flatnonzero((D["S0"] == o) & (D["S1"] == s))
            if len(f) + len(g):
                L.append((o, f, g))
        if len(L) == len(opps):
            out[s] = L
    return out


def strat_values(p, idx):
    """p: P(team 0 wins) per draft -> {strategy: mean over opponents of mean win prob}."""
    return {s: float(np.mean([np.concatenate([p[f], 1 - p[g]]).mean() for _, f, g in L])) for s, L in idx.items()}


def strat_support(D, idx):
    from paper1_revision import support_mitigation as sm
    C = sm.sup_corpus()
    m0, m1 = C.metrics(D["T0"])["min3"], C.metrics(D["T1"])["min3"]
    return {s: float(np.mean(np.concatenate([np.concatenate([m0[f], m1[g]]) for _, f, g in L]) < 25)) for s, L in idx.items()}


# Opt-in robustness runs (stage p1 only): WORLDS_P1_TAG suffixes every p1 output (rescore_p1<tag>.json,
# cache rescore_p1<tag>_<fam>.npz), required with WORLDS_GRADE=flat (world.py) so the default files are never
# touched; WORLDS_P1_SEEDS=1 adds per-seed entries "<strategy>@s<k>" for the seed-pooled policies (draft i of a
# pooled policy is played by checkpoint i % n_seeds: tournament.pooled_policy, floor_search.build_actor).
P1_TAG = os.environ.get("WORLDS_P1_TAG", "")
P1_SEEDS = {"constrained_mcts": 5, "constrained_mcts_sf": 5, "search_sf": 5,
            "constrained_mcts_ft": 3, "constrained_mcts_ft_sf": 3, "search_ft_sf": 3}


def p1_draft_numbers(D):
    """Tournament draft index of every paper-1 draft, in p1_drafts order (same dedupe as p1_drafts)."""
    p = os.path.join(WD.CACHE, "p1_drafts_draftno.npy")
    if os.path.exists(p):
        return np.load(p)
    from paper1_revision import ood_frontier as of
    seen, out, s0, s1 = set(), [], [], []
    for a, b, r in of.records():
        if not (a in OPP8 or b in OPP8):
            continue
        k = (a, b, r["draft"], r["game_map"], tuple(r["team0"]["picks"]), tuple(r["team1"]["picks"]))
        if k not in seen:
            seen.add(k)
            out.append(r["draft"])
            s0.append(a)
            s1.append(b)
    assert len(out) == len(D["S0"]) and (np.array(s0) == D["S0"]).all() and (np.array(s1) == D["S1"]).all()
    out = np.array(out, np.int64)
    np.save(p, out)
    return out


def p1_seed_index(D, idx):
    dn = p1_draft_numbers(D)
    out = {}
    for s, n in P1_SEEDS.items():
        for k in range(n):
            if s in idx:
                out[f"{s}@s{k}"] = [(o, f[dn[f] % n == k], g[dn[g] % n == k]) for o, f, g in idx[s]]
    return out


_G = {}


def p1_attach_lobbies(D):
    """Each paper-1 draft gets a real judge-window lobby (player_cov pool; draft i -> a fixed random lobby):
    team t's five players, their earlier games per hero as of that game, and the drafter state."""
    import player as PL
    cz = np.load(PL.COV)
    n = len(D["T0"])
    li = np.random.default_rng(23).integers(0, len(cz["pool_pid"]), n)
    return dict(D, H0=D["T0"], H1=D["T1"], pid=cz["pool_pid"][li], cnt=cz["pool_cnt"][li], lob=li,
                s=cz["pool_s"][li], off=cz["pool_off"][li])


def _p1_world(args):
    fam, wid, cells = args
    D, idx, names = _G["D"], _G["idx"], _G["names"]
    if fam in ("f", "fc"):
        import player as PL
        Ws = _G.setdefault("W", W.Worlds())
        PW = _G.setdefault("PW" + fam, PL.PlayerWorldsCov() if fam == "fc" else PL.PlayerWorlds2())
        Dl = _G["Dlob"] if fam == "fc" else {k: v for k, v in _G["Dlob"].items() if k not in ("s", "off")}
        zg = Ws.logit("a", D, wid)
        A = assign_terms(PW, Dl, wid, np.random.default_rng([wid, 4]))
        out = np.zeros((3, len(names)), np.float32)
        for i, nm in enumerate(("random", "human", "best")):
            v = strat_values(M.sig(zg + A[nm]), idx)
            out[i] = [v[s] for s in names]
        return out
    if fam == "g":
        p = np.asarray(_G["ens"][wid], float)
        v = strat_values(p, idx)
        return np.array([[v[s] for s in names]], np.float32)
    Ws = _G.setdefault("W", W.Worlds())
    c = Ws.components(fam, D, wid)
    out = np.zeros((len(cells), len(names)), np.float32)
    for i, cell in enumerate(cells):
        v = strat_values(M.sig(Ws.combine(c, fam, **cell)), idx)
        out[i] = [v[s] for s in names]
    return out


def cells_for(fam):
    Dd = W.dials()
    named = [dict(v) for v in Dd["named"].values()]
    G = Dd["grid"]
    grid = [dict(c, drift=1.0) for c in W.grid_cells(G)]
    cells = named + grid
    for c in cells:
        if fam != "e":
            c.pop("drift", None)
    if fam == "e":   # drift family: the dial is the drift scale (family a's dials at default)
        cells = [dict(k=1.0, delta=0.0, a=1.0, sigma=0.0, drift=s) for s in (0.0, 0.5, 1.0, 2.0, 3.0)]
    return cells, list(Dd["named"])


def stage_p1(fams, n):
    from multiprocessing import get_context
    D = p1_drafts()
    idx = p1_index(D)
    if W.GRADE != "log" and not P1_TAG:
        raise SystemExit("WORLDS_GRADE != log needs WORLDS_P1_TAG (never overwrite the default outputs)")
    if os.environ.get("WORLDS_P1_SEEDS") == "1":
        idx.update(p1_seed_index(D, idx))
    names = sorted(idx)
    _G.update(D=D, idx=idx, names=names)
    res_path = os.path.join(WD.RESULTS, f"rescore_p1{P1_TAG}.json")
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}
    res["strategies"] = names
    res["support_lt25"] = strat_support(D, idx)
    for fam in fams:
        out = os.path.join(WD.CACHE, f"rescore_p1{P1_TAG}_{fam}.npz")
        if os.path.exists(out):
            continue
        t0 = time.time()
        if fam == "g":
            import ensemble as EN
            _G["ens"] = EN.values(EN.rows_of(D), "p1drafts")
            cells, nn = [{}], ["member"]
            jobs = [(fam, w, cells) for w in range(EN.N_MEMBERS)]
        elif fam in ("f", "fc"):
            _G["Dlob"] = p1_attach_lobbies(D)
            cells = [{"assign": x} for x in ("random", "human", "best")]
            jobs = [(fam, w, cells) for w in range(n)]
        else:
            cells, nn = cells_for(fam)
            jobs = [(fam, w, cells) for w in range(n)]
        if fam != "g":
            W.Worlds().base_draws([0])   # write the draw block once, before workers read it
        with get_context("fork").Pool(NPROC) as pool:
            V = np.stack(pool.map(_p1_world, jobs, chunksize=4))     # worlds x cells x strategies
        np.savez(out, V=V, cells=json.dumps(cells), names=np.array(names))
        print(f"p1 family {fam}: {V.shape} in {time.time() - t0:.0f}s", flush=True)
    json.dump(res, open(res_path, "w"), indent=1)


# ── paper 2 drafts ───────────────────────────────────────────────────────
P2DIR = os.path.join(TRAINING_DIR, "personalization", "cache_def_def1027_hlsup", "p3t")
DRAFT_TEAM = [0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1]
IS_PICK = [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1]
SEL = "x-combo1-dr-chance"
P2_OPP = {"headline": ["pMCTS-cur", "aMCTS-pp", "GD", "DraftRec", "Real"], "tune": ["pMCTS-cur", "aMCTS-pp"]}
P2_ARMS = ["s-floor10", "s-floor15", "s-floor20", "s-floor25", "s-pcal", "s-pcal-floor25", "s-leaf1", "s-leaf3", "s-leaf10",
           "s-leaf30", "s-gd025", "s-gd05", "s-gd1", "s-gd2", "s-ffloor", "s-ffloor-floor25"]
PERMS = np.array(list(__import__("itertools").permutations(range(5))), np.int64)


def p2_drafts(lset):
    """Every (arm or SEL) vs opponent file of the lobby set, both seats -> arrays + pair labels."""
    import player as PL
    z = np.load(PL.P2_LOBBIES)
    tdir = os.path.join(P2DIR, "tourney" if lset == "headline" else "tourney_tune")
    PK = [[k for k in range(16) if IS_PICK[k] and DRAFT_TEAM[k] == t] for t in (0, 1)]
    rows, S0, S1, lob = [], [], [], []
    for a in [SEL] + P2_ARMS:
        for o in P2_OPP[lset]:
            for x, y in ((a, o), (o, a)):
                f = os.path.join(tdir, f"{x}__{y}.npz")
                if not os.path.exists(f):
                    continue
                d = np.load(f)
                assert (d["gi"] == z[f"{lset}_gi"]).all()
                rows.append(d["acts"])
                S0 += [x] * len(d["acts"])
                S1 += [y] * len(d["acts"])
                lob.append(np.arange(len(d["acts"])))
    A = np.concatenate(rows)
    lob = np.concatenate(lob)
    maps, tiers = WD.meta()
    D = {"T0": np.sort(A[:, PK[0]], 1), "T1": np.sort(A[:, PK[1]], 1),
         "H0": A[:, PK[0]], "H1": A[:, PK[1]],
         "m": np.array([maps.index(str(x)) if str(x) in maps else len(maps) for x in z[f"{lset}_map"][lob]]),
         "t": np.array([tiers.index(str(x)) if str(x) in tiers else len(tiers) for x in z[f"{lset}_tier"][lob]]),
         "S0": np.array(S0), "S1": np.array(S1), "lob": lob,
         "pid": z[f"{lset}_pid"][lob], "cnt": z[f"{lset}_cnt"][lob]}
    return D


def p2_values(p, D, lset):
    """-> {(arm, opp): mean true P(arm wins), both seats}."""
    out = {}
    for a in [SEL] + P2_ARMS:
        for o in P2_OPP[lset]:
            f = (D["S0"] == a) & (D["S1"] == o)
            g = (D["S0"] == o) & (D["S1"] == a)
            if f.any() and g.any():
                out[(a, o)] = float(np.concatenate([p[f], 1 - p[g]]).mean())
    return out


def assign_terms(PW, D, wid, rng):
    """Player term (team 0 minus team 1, logit) of world wid under three assignments of each draft team's
    five players to its five heroes: random, human-like (P(perm) proportional to prod (n_ph + 0.5)), best
    (the world's own max)."""
    n = len(D["lob"])
    out = {"random": np.zeros(n), "human": np.zeros(n), "best": np.zeros(n)}
    for t, Hk in ((0, "H0"), (1, "H1")):
        pid = D["pid"][:, t]                                      # (n, 5) players
        her = D[Hk]                                               # (n, 5) heroes in pick order
        P = np.repeat(pid[:, :, None], 5, 2)                      # player i x hero j
        Hh = np.repeat(her[:, None, :], 5, 1)
        N = np.take_along_axis(D["cnt"][:, t], Hh, 2)             # n_ph as of the lobby day
        if "s" in D:                                              # f-cov: the drafter state's s and off-role
            Sv = np.take_along_axis(D["s"][:, t], Hh, 2)
            Ov = np.take_along_axis(D["off"][:, t], Hh, 2).astype(float)
            U = PW.u(P, Hh, N, wid, s=Sv, off=Ov)
        else:
            U = PW.u(P, Hh, N, wid)                               # (n, 5, 5)
        tot = U[:, np.arange(5)[None, :], PERMS].sum(2)          # (n, 120): perm k gives player i hero PERMS[k, i]
        lw = np.log(N + 0.5)[:, np.arange(5)[None, :], PERMS].sum(2)
        pr = np.exp(lw - lw.max(1, keepdims=True))
        pr /= pr.sum(1, keepdims=True)
        kh = (pr.cumsum(1) > rng.random(n)[:, None]).argmax(1)
        kr = rng.integers(0, 120, n)
        sg = 1.0 if t == 0 else -1.0
        out["random"] += sg * tot[np.arange(n), kr]
        out["human"] += sg * tot[np.arange(n), kh]
        out["best"] += sg * tot.max(1)
    return out


def _p2_world(args):
    fam, wid, cells, lset = args
    D = _G["D2"]
    Ws = _G.setdefault("W", W.Worlds())
    keys = _G["keys2"]
    if fam in ("f", "fc"):
        import player as PL
        PW = _G.setdefault("PW" + fam, PL.PlayerWorldsCov() if fam == "fc" else PL.PlayerWorlds2())
        if fam == "fc" and "s" not in D:
            D = dict(D, s=_G["cov_s"][lset], off=_G["cov_off"][lset])
        zg = Ws.logit("a", D, wid)
        A = assign_terms(PW, D, wid, np.random.default_rng([wid, 3]))
        res = []
        for nm in ("random", "human", "best"):
            v = p2_values(M.sig(zg + A[nm]), D, lset)
            res.append([v.get(k, np.nan) for k in keys])
        return np.array(res, np.float32)
    c = Ws.components(fam, D, wid)
    out = np.zeros((len(cells), len(keys)), np.float32)
    for i, cell in enumerate(cells):
        v = p2_values(M.sig(Ws.combine(c, fam, **cell)), D, lset)
        out[i] = [v.get(k, np.nan) for k in keys]
    return out


def stage_p2(fams, n, lsets):
    from multiprocessing import get_context
    for lset in lsets:
        D = p2_drafts(lset)
        keys = sorted({(a, o) for a in [SEL] + P2_ARMS for o in P2_OPP[lset]
                       if ((D["S0"] == a) & (D["S1"] == o)).any()})
        _G.update(D2=D, keys2=keys)
        W.Worlds().base_draws([0])
        for fam in fams:
            out = os.path.join(WD.CACHE, f"rescore_p2_{lset}_{fam}.npz")
            if os.path.exists(out):
                continue
            t0 = time.time()
            cells = [{"assign": x} for x in ("random", "human", "best")] if fam in ("f", "fc") else cells_for(fam)[0]
            if fam == "fc":
                import player as PL
                cz = np.load(PL.COV)
                _G["cov_s"] = {lset: cz[f"{lset}_s"][D["lob"]]}
                _G["cov_off"] = {lset: cz[f"{lset}_off"][D["lob"]]}
            with get_context("fork").Pool(NPROC) as pool:
                V = np.stack(pool.map(_p2_world, [(fam, w, cells, lset) for w in range(n)], chunksize=4))
            np.savez(out, V=V, cells=json.dumps(cells), keys=json.dumps(keys))
            print(f"p2 {lset} family {fam}: {V.shape} in {time.time() - t0:.0f}s", flush=True)


def summarize_p2():
    res = {}
    for lset in ("headline", "tune"):
        for fam in ("a", "b", "c", "d", "e", "f", "fc"):
            p = os.path.join(WD.CACHE, f"rescore_p2_{lset}_{fam}.npz")
            if not os.path.exists(p):
                continue
            z = np.load(p)
            V, cells, keys = z["V"], json.loads(str(z["cells"])), [tuple(k) for k in json.loads(str(z["keys"]))]
            named = list(W.dials()["named"]) if fam not in ("e", "f", "fc") else None
            F = {"worlds": int(V.shape[0]), "cells": cells, "arms": {}}
            for ci, cell in enumerate(cells):
                tag = named[ci] if named and ci < len(named) else json.dumps(cell, sort_keys=True)
                T = {}
                for a in [SEL] + P2_ARMS:
                    opps = [o for o in P2_OPP[lset] if (a, o) in keys and (SEL, o) in keys and o != a]
                    if not opps:
                        continue
                    va = np.mean([V[:, ci, keys.index((a, o))] for o in opps], 0)
                    vs = np.mean([V[:, ci, keys.index((SEL, o))] for o in opps], 0)
                    T[a] = {"value_pts": float(100 * va.mean()), "opponents": opps}
                    if a != SEL:
                        T[a]["minus_sel"] = contrast(va - vs)
                F["arms"][tag] = T
            res.setdefault(lset, {})[fam] = F
    json.dump(res, open(os.path.join(WD.RESULTS, "rescore_p2.json"), "w"), indent=1)


# ── summaries ────────────────────────────────────────────────────────────
def contrast(x):
    """per-world paired differences (points) -> mean, between-world 95% interval, P(>0), worst-10% mean."""
    x = 100 * np.asarray(x, float)
    q = np.sort(x)
    k = max(1, int(np.ceil(0.1 * len(x))))
    return {"mean": float(x.mean()), "lo95": float(np.percentile(x, 2.5)), "hi95": float(np.percentile(x, 97.5)),
            "p_gt0": float((x > 0).mean()), "worst10_mean": float(q[:k].mean()), "worlds": int(len(x))}


BE_P1 = [("constrained_mcts_sf", "constrained_mcts"), ("constrained_mcts_ft_sf", "constrained_mcts"),
         ("constrained_mcts_ft_sf", "constrained_mcts_ft"), ("search_sf", "constrained_mcts"), ("search_ft_sf", "constrained_mcts")]


def breakeven(V, cells, ix, iy, allowed=None):
    """Per plane of the grid (delta x sigma, delta x k, delta x a): mean paired difference (points) at each
    delta, and the delta where it crosses zero (linear interpolation; None if no crossing in the grid)."""
    out = []
    for key, vals in (("sigma", None), ("k", None), ("a", None)):
        pl = {}
        for ci, c in enumerate(cells):
            if "delta" not in c or "sigma" not in c:
                continue
            other = {kk: c[kk] for kk in ("k", "a", "sigma") if kk != key}
            if any(other[kk] != (0.0 if kk == "sigma" else 1.0) for kk in other):
                continue
            pl.setdefault(c[key], []).append((c["delta"], float(100 * (V[:, ci, ix] - V[:, ci, iy]).mean()),
                                              float(100 * np.mean(V[:, ci, ix] > V[:, ci, iy]))))
        for v, pts in sorted(pl.items()):
            pts.sort()
            cross = None
            for (d0, m0, _), (d1, m1, _) in zip(pts[:-1], pts[1:]):
                if (m0 <= 0 < m1) or (m1 <= 0 < m0) or m0 == 0:
                    cross = d0 + (d1 - d0) * (0 - m0) / (m1 - m0) if m1 != m0 else d0
            row = {"plane": key, "value": v, "by_delta": [{"delta": d, "mean_pts": m, "p_gt0_pct": p} for d, m, p in pts],
                   "breakeven_delta": cross}
            if allowed is not None:
                row["allowed_deltas"] = sorted(d for (kk, vv, d) in allowed if kk == key and vv == v)
            out.append(row)
    return out


def allowed_cells(fam):
    p = os.path.join(WD.RESULTS, f"check_allowed_{fam}.json")
    if not os.path.exists(p):
        return None
    out = []
    for c in json.load(open(p))["cells"]:
        if not c["allowed"]:
            continue
        for key in ("sigma", "k", "a"):
            other = {kk: c[kk] for kk in ("k", "a", "sigma") if kk != key}
            if all(other[kk] == (0.0 if kk == "sigma" else 1.0) for kk in other):
                out.append((key, c[key], c["delta"]))
    return out


def summarize_p1():
    res_path = os.path.join(WD.RESULTS, f"rescore_p1{P1_TAG}.json")
    res = json.load(open(res_path))
    names = res["strategies"]
    res["families"] = {}
    for fam in ("a", "b", "c", "d", "e", "g", "f", "fc"):
        p = os.path.join(WD.CACHE, f"rescore_p1{P1_TAG}_{fam}.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p)
        V, cells = z["V"], json.loads(str(z["cells"]))
        F = {"cells": cells, "worlds": int(V.shape[0]), "value": {}, "contrasts": {}}
        named = list(W.dials()["named"]) if fam not in ("e", "g", "f", "fc") else None
        for ci, cell in enumerate(cells):
            tag = named[ci] if named and ci < len(named) else json.dumps(cell, sort_keys=True)
            F["value"][tag] = {s: [float(100 * V[:, ci, j].mean()), float(100 * V[:, ci, j].std())] for j, s in enumerate(names)}
            F["contrasts"][tag] = {f"{x} - {y}": contrast(V[:, ci, names.index(x)] - V[:, ci, names.index(y)])
                                   for x, y in CONTRASTS_P1 if x in names and y in names}
        if fam in ("a", "b", "c", "d"):
            al = allowed_cells(fam) if W.GRADE == "log" else None   # check_allowed_* were run under the log grade
            gcells = [{} if ci < len(named) else c for ci, c in enumerate(cells)]   # grid planes only
            F["breakeven"] = {f"{x} - {y}": breakeven(V, gcells, names.index(x), names.index(y), al)
                              for x, y in BE_P1 if x in names and y in names}
        res["families"][fam] = F
    json.dump(res, open(res_path, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["p1", "p2", "summary"])
    ap.add_argument("--lsets", default="headline,tune")
    ap.add_argument("--fams", default="a,b,c,d,e,g")
    ap.add_argument("--n", type=int, default=1000)
    a = ap.parse_args()
    if a.stage == "p1":
        stage_p1(a.fams.split(","), a.n)
        summarize_p1()
    elif a.stage == "p2":
        stage_p2([f for f in a.fams.split(",") if f != "g"], a.n, a.lsets.split(","))
        summarize_p2()
    else:
        summarize_p1()
        summarize_p2()
