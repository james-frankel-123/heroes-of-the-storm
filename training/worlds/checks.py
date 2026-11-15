"""
Plausibility checks of the hidden worlds, per family (from training/, p1site env, on the 3090).

  step1     Real drafts, world outcomes. On the NODRIFT games, y ~ Bernoulli(world p); the drafter's
            own leaf (enriched leak-free WP, deploy statistics; overrating.py's cached values) is
            rescored against them with step 1's exact estimators (ood_frontier.certify bins,
            overrating.pooled / bin_table): certify error by own-min3 bin, the GLM slope of the
            interaction claim below and above the floor, the bias and the top-decile overvaluation.
            The real value of each statistic is placed in the distribution over worlds (percentile).
            -> site/results/check_step1_<fam>.json
  allowed   For each dial cell of the pre-registered grid: the EXPECTED step-1 statistics (y replaced
            by the world p, so no sampling noise) averaged over worlds; a cell is allowed by step 1
            when every statistic is within 2 SE of its real value. -> check_allowed_<fam>.json
  logloss   The world model's posterior-mean prediction vs the enriched network on NODRIFT, overall
            and by own-min3 bin; plus the worlds' spread: SD over worlds of the logit by support bin.
            -> check_logloss.json
  pairs     World outcomes on the training games; counter_checks' real-data estimator (main effects,
            residual pair means, empirical-Bayes true SD) -> the SD of true counters and synergies,
            to set against counter_checks/real.json (1.09 and 1.72 pts). -> check_pairs.json
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

NPROC = int(os.environ.get("WORLDS_PROCS", "8"))
CERT_EDGES = [0, 1, 5, 10, 25, 50, 100, 200, 400, 10 ** 12]
_CTX = {}


def ctx():
    """NODRIFT arrays, the agent's cached leaf values, step-1's baseline, min3 (paper 1's corpus)."""
    if not _CTX:
        from paper1_revision import overrating as OV
        from paper1_revision import support_mitigation as sm
        D = WD.load("NODRIFT")
        z = np.load(os.path.join(OV.CACHE, "vals_nodrift.npz"))
        maps, tiers = WD.meta()
        mn = [maps[i] if i < len(maps) else "?" for i in D["m"]]
        tn = [tiers[i] if i < len(tiers) else "?" for i in D["t"]]
        E = OV.MainEffects(OV.base_path("current"))
        C = sm.sup_corpus()
        _CTX.update(D=D, p=np.asarray(z["agent"], float), b=OV.sig(E.logit(D["T0"], D["T1"], mn, tn)),
                    m0=C.metrics(D["T0"])["min3"], m1=C.metrics(D["T1"])["min3"], OV=OV)
        assert len(_CTX["p"]) == len(D["y"])
    return _CTX


def step1_stats(y0, full=True):
    """Step 1's statistics for team-0 outcomes y0 (0/1, or expected values) on NODRIFT."""
    c = ctx()
    OV = c["OV"]
    y, p, b, own, opp, gid = OV.team_rows(y0, c["p"], c["b"], c["m0"], c["m1"])
    P = OV.pooled(y, p, b, own, opp, gid)
    out = {"below_bias": P["below_floor"]["bias_y_minus_p"][0], "below_beta": P["below_floor"]["beta"][0],
           "in_beta": P["in_support"]["beta"][0], "beta_diff": P["beta_diff_below_minus_in"][0],
           "topdec_below": P["below_floor"]["top_decile"]["overval_p_minus_y"][0],
           "topdec_in": P["in_support"]["top_decile"]["overval_p_minus_y"][0],
           "topdec_diff": P["topdec_overval_diff_below_minus_in"][0]}
    # certify's raw gap per own-min3 bin (opponent in support)
    for lo, hi in zip(CERT_EDGES[:-1], CERT_EDGES[1:]):
        sel = (own >= lo) & (own < hi) & (opp >= 25)
        out[f"cert_{lo}"] = float(100 * (y[sel] - p[sel]).mean())
    if full:
        bt = OV.bin_table(y, p, b, own, opp, gid)
        for r in bt:
            if "beta" in r:
                out[f"beta_{r['min3_lo']}"] = r["beta"][0]
    return out


def real_step1():
    c = ctx()
    return step1_stats(c["D"]["y"])


def ens_nodrift():
    import ensemble as EN
    c = ctx()
    if "ens" not in c:
        c["ens"] = EN.values(EN.rows_of(c["D"]), "nodrift")
    return c["ens"]


def _fam_world(args):
    fam, wid, dial, sample = args
    c = ctx()
    if fam == "g":
        p = np.asarray(ens_nodrift()[wid], float)
    else:
        p = M.sig(W.Worlds().logit(fam, c["D"], wid, **dial))
    if sample:
        y0 = (np.random.default_rng([int(wid), 5]).random(len(p)) < p).astype(float)
    else:
        y0 = p
    return step1_stats(y0, full=sample)


def pool_map(f, jobs):
    from multiprocessing import get_context
    ctx()   # load before forking
    W.Worlds().base_draws([0])   # write the draw block once, before workers read it
    with get_context("fork").Pool(NPROC) as pool:
        return pool.map(f, jobs, chunksize=1)


def stage_step1(fam, n, dial_name):
    Dd = W.dials()
    dial = dict(Dd["named"][dial_name])
    if fam != "e":
        dial.pop("drift", None)
    t0 = time.time()
    real = real_step1()
    if fam == "g":
        import ensemble as EN
        ens_nodrift()
        n = min(n, EN.N_MEMBERS)
    res = pool_map(_fam_world, [(fam, w, dial, True) for w in range(n)])
    out = {"family": fam, "dial": dial_name, "dial_values": dial, "worlds": n, "stats": {}}
    for k, rv in real.items():
        v = np.array([r.get(k, np.nan) for r in res], float)
        v = v[np.isfinite(v)]
        if not len(v):
            continue
        out["stats"][k] = {"real": rv, "world_mean": float(v.mean()), "world_sd": float(v.std()),
                           "world_p05": float(np.percentile(v, 5)), "world_p95": float(np.percentile(v, 95)),
                           "real_percentile": float(100 * (v < rv).mean())}
    out["secs"] = time.time() - t0
    json.dump(out, open(os.path.join(WD.RESULTS, f"check_step1_{fam}_{dial_name}.json"), "w"), indent=1)
    for k, s in out["stats"].items():
        print(f"{fam} {dial_name} {k:14s} real {s['real']:+.3f}  worlds {s['world_mean']:+.3f} ± {s['world_sd']:.3f} "
              f"[{s['world_p05']:+.3f}, {s['world_p95']:+.3f}]  pct {s['real_percentile']:.0f}", flush=True)


# real step-1 constraints (overrating slope.json: agent, opponent in support), value and SE
def constraints():
    from paper1_revision import overrating as OV
    S = json.load(open(os.path.join(OV.OUT, "slope.json")))["bins"]["agent"]
    P = S["pooled"]
    b = {r["min3_lo"]: r for r in S["opp_in_support"]}
    return {"below_bias": P["below_floor"]["bias_y_minus_p"], "beta_diff": P["beta_diff_below_minus_in"],
            "topdec_diff": P["topdec_overval_diff_below_minus_in"], "cert_1": b[1]["bias_y_minus_p"],
            "cert_5": b[5]["bias_y_minus_p"], "cert_10": b[10]["bias_y_minus_p"]}


def stage_allowed(fam, n):
    Dd = W.dials()
    G = Dd["grid"]
    cells = W.grid_cells(G)
    real = constraints()
    jobs = [(fam, w, cell, False) for cell in cells for w in range(n)]
    t0 = time.time()
    res = pool_map(_fam_world, jobs)
    out = {"family": fam, "worlds_per_cell": n, "constraints": real, "cells": []}
    for i, cell in enumerate(cells):
        R = res[i * n:(i + 1) * n]
        st = {k: float(np.mean([r[k] for r in R])) for k in real}
        ok = {k: bool(abs(st[k] - v[0]) <= 2 * v[1]) for k, v in real.items()}
        out["cells"].append(dict(cell, expected=st, ok=ok, allowed=all(ok.values())))
    out["secs"] = time.time() - t0
    json.dump(out, open(os.path.join(WD.RESULTS, f"check_allowed_{fam}.json"), "w"), indent=1)
    for c in out["cells"]:
        print(fam, {k: c[k] for k in ("k", "delta", "a")}, "allowed" if c["allowed"] else
              "x " + ",".join(k for k, v in c["ok"].items() if not v), flush=True)


def stage_logloss(n):
    c = ctx()
    D, y, pa = c["D"], c["D"]["y"], c["p"]
    own = np.minimum(c["m0"], c["m1"])
    Ws = W.Worlds()
    bins = [0, 5, 25, 100, 400, 10 ** 12]
    out = {"games": int(len(y)), "agent_nll": M.nll(pa, y), "families": {}}
    pb = M.sig(Ws.B.logit(Ws.theta, D))
    out["base_only_nll"] = M.nll(pb, y)
    for fam in ("a", "b", "c", "d"):
        pw = M.sig(Ws.posterior_mean_logit(fam, D))
        r = {"world_mean_nll": M.nll(pw, y), "by_game_min_min3": []}
        Z = np.stack([Ws.logit(fam, D, w) for w in range(n)])
        for lo, hi in zip(bins[:-1], bins[1:]):
            sel = (own >= lo) & (own < hi)
            r["by_game_min_min3"].append({"lo": lo, "hi": hi if hi < 10 ** 12 else None, "n": int(sel.sum()),
                                          "agent_nll": M.nll(pa[sel], y[sel]), "world_nll": M.nll(pw[sel], y[sel]),
                                          "worlds_logit_sd": float(Z[:, sel].std(0).mean()),
                                          "worlds_p_sd_pts": float(100 * M.sig(Z[:, sel]).std(0).mean())})
        out["families"][fam] = r
        print(fam, json.dumps(r, indent=None), flush=True)
    Pg = np.asarray(ens_nodrift(), float)
    pg = M.sig(np.mean(np.log(Pg / (1 - Pg)), 0))
    r = {"world_mean_nll": M.nll(pg, y), "members_nll_mean": float(np.mean([M.nll(x, y) for x in Pg])), "by_game_min_min3": []}
    Z = np.log(Pg / (1 - Pg))
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (own >= lo) & (own < hi)
        r["by_game_min_min3"].append({"lo": lo, "hi": hi if hi < 10 ** 12 else None, "n": int(sel.sum()),
                                      "agent_nll": M.nll(pa[sel], y[sel]), "world_nll": M.nll(pg[sel], y[sel]),
                                      "worlds_logit_sd": float(Z[:, sel].std(0).mean()),
                                      "worlds_p_sd_pts": float(100 * Pg[:, sel].std(0).mean())})
    out["families"]["g"] = r
    print("g", json.dumps(r), flush=True)
    json.dump(out, open(os.path.join(WD.RESULTS, "check_logloss.json"), "w"), indent=1)


def _pairs_one(args):
    fam, wid = args
    from paper1_revision import counter_checks as CC
    D = WD.load("train")
    if fam == "real":
        y = D["y"]
    else:
        p = M.sig(W.Worlds().logit(fam, D, wid))
        y = (np.random.default_rng([int(wid), 9]).random(len(p)) < p).astype(float)
    maps, tiers = WD.meta()
    ti = np.where(D["t"] < len(tiers), D["t"], -1)
    mi = np.where(D["m"] < len(maps), D["m"], -1)
    par, _ = CC.fit(D["T0"], D["T1"], ti, mi, y, len(tiers), len(maps))
    p0 = CC.predict(par, D["T0"], D["T1"], ti, mi)
    pr = CC.pair_resid(D["T0"], D["T1"], y - p0, p0 * (1 - p0))
    out = {"family": fam, "world": wid}
    for name in ("counter", "synergy"):
        est, se, n = pr[name]
        est = est.copy()
        est[np.eye(M.H, dtype=bool)] = np.nan
        s, _ = CC.eb(est, se, n)
        out[name] = s
    print(json.dumps(out), flush=True)
    return out


def stage_pairs(n):
    from multiprocessing import get_context
    jobs = [("real", 0)] + [(f, w) for f in ("a", "b", "c", "d") for w in range(n)]
    with get_context("spawn").Pool(min(NPROC, 6)) as pool:
        res = pool.map(_pairs_one, jobs, chunksize=1)
    out = {"real_reference": "paper1_revision/site/results/counter_checks/real.json", "runs": res, "summary": {}}
    for f in ("real", "a", "b", "c", "d"):
        R = [r for r in res if r["family"] == f]
        out["summary"][f] = {nm: {"sd_true_pts_mean": float(np.mean([r[nm]["sd_true_pts"] for r in R])),
                                  "sd_true_pts_sd": float(np.std([r[nm]["sd_true_pts"] for r in R]))}
                             for nm in ("counter", "synergy")}
    json.dump(out, open(os.path.join(WD.RESULTS, "check_pairs.json"), "w"), indent=1)
    print(json.dumps(out["summary"], indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["step1", "allowed", "logloss", "pairs"])
    ap.add_argument("--fam", default="a")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--dial", default="default")
    a = ap.parse_args()
    if a.stage == "step1":
        stage_step1(a.fam, a.n, a.dial)
    elif a.stage == "allowed":
        stage_allowed(a.fam, a.n)
    elif a.stage == "logloss":
        stage_logloss(a.n)
    else:
        stage_pairs(a.n)
