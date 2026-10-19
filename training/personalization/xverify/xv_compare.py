"""
CPU side of the independent MCTS verification: compares the GPU outputs saved
by xv_gpu.py with the CPU replica (xv_ref.py) and checks properties.

Usage (from training/, CPU only):
  nice -n 19 taskset -c 48-51 python3 personalization/xverify/xv_compare.py --gpu-out DIR [--procs 4] [stage ...]
Writes DIR/compare.json (one entry per stage) and prints a summary.
"""
import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np

import xv_cases as XC
import xv_ref as R

BASE_SEED = 1000003
TIE = 1e-5          # float near-tie: GPU and CPU numerics differ at about this level
Q_TOL = 2e-4        # root Q agreement while trees agree (leaf parity is ~1e-5; float32 sums of 400 values)
_NETS = {}


def nets():
    if "n" not in _NETS:
        _NETS["n"] = R.Nets("cur", 0, gd_i=0)
    return _NETS["n"]


def inputs(fl, z, i, coefs, vnet):
    return R.P3Inputs(fl, z["cfg"][i], z["s"][i], z["off"][i], z["imit"][i], z["pools"][i], coefs,
                      z["bias"][i] if "bias" in z else None, z["feat"][i] if "feat" in z else None, vnet)


# ------------------------------------------------------------------ grid: exact replay

def _grid_one(args):
    fl, z, i, coefs, vnet, grid, pol, q, prior, c_puct, pw_k, pw_alpha, sims_max = args
    e = inputs(fl, z, i, coefs, vnet)
    out = R.decide_p3(nets(), e, sims_max, c_puct, BASE_SEED + i, pw_k=pw_k, pw_alpha=pw_alpha, snap=grid)
    rep = {"i": i, "cat": str(z["cat"][i]), "k": int(z["cfg"][i, 4]), "cfg7": int(z["cfg"][i, 7]),
           "cfg56": int(z["cfg"][i, 56]), "cfg57": int(z["cfg"][i, 57]), "cfg6": int(z["cfg"][i, 6]),
           "assign": int(z["cfg"][i, 58])}
    if out is None:
        rep["skipped"] = "forced ban at the decide step"
        return rep
    st = out["state"]
    rmask = R.p3_mask(st, e)
    first_bad, worst_q = None, 0.0
    for j, k in enumerate(grid):
        vis_c, q_c = out["snaps"][k]
        vis_g = np.rint(pol[j] * k).astype(np.int64)
        if vis_g.sum() != k:
            rep["gpu_visit_sum_error"] = int(vis_g.sum())
        if (vis_g[~rmask] > 0).any():
            rep["illegal_visit"] = [int(a) for a in np.flatnonzero((vis_g > 0) & ~rmask)]
        same = np.array_equal(vis_g, vis_c)
        dq = np.abs(q[j] - q_c)[vis_c > 0].max() if (vis_c > 0).any() else 0.0
        if same and dq <= Q_TOL:
            worst_q = max(worst_q, float(dq))
            continue
        first_bad = (j, k, bool(same), float(dq))
        break
    rep["max_dq_while_equal"] = worst_q
    if prior is not None and len(prior):
        rep["prior_maxdiff"] = float(np.abs(prior - out["root_prior"]).max())
        rep["prior_zero_legal"] = int(((out["root_prior"] == 0) & rmask).sum())
        rep["prior_zero_legal_gpu"] = int(((prior == 0) & rmask).sum())
    rep["n_legal"] = int(rmask.sum())
    if first_bad is None:
        rep["match"] = True
        return rep
    j, k, same, dq = first_bad
    lo = grid[j - 1] if j > 0 else 0
    mg = out["margins"][lo:k]
    mp = min(m[0] for m in mg)
    ms = min(m[1] for m in mg)
    rep.update(match=False, first_bad_k=k, last_good_k=lo, visits_equal_at_bad=same, dq_at_bad=dq,
               min_puct_gap=float(mp), min_sample_gap=float(ms),
               explained=bool(min(mp, ms) < TIE))
    return rep


def compare_grid(gdir, procs, flavors=XC.FLAVORS):
    import multiprocessing as mp
    out = {}
    for fl in flavors:
        p = os.path.join(gdir, f"grid_{fl}.npz")
        if not os.path.exists(p):
            continue
        g = np.load(p, allow_pickle=True)
        z = {k: g[k] for k in ("cfg", "s", "off", "imit", "bias", "feat", "pools", "cat")}
        grid = [int(x) for x in g["grid"]]
        prior = g["prior"]
        vnet = g["vnet"]
        coefs = g["coefs"]
        n = len(z["cfg"])
        tasks = [(fl, z, i, coefs, vnet, grid, g["pol"][i], g["q"][i], prior[i] if len(prior) else None,
                  2.0, 1.0, 0.5, grid[-1]) for i in range(n)]
        t0 = time.time()
        with mp.get_context("fork").Pool(procs) as pool:
            reps = pool.map(_grid_one, tasks, chunksize=1)
        out[fl] = summarize_grid(reps)
        out[fl]["secs"] = round(time.time() - t0)
        print(fl, json.dumps({k: v for k, v in out[fl].items() if k != "cases"}), flush=True)
    return out


def summarize_grid(reps):
    done = [r for r in reps if "skipped" not in r]
    full = [r for r in done if r["match"]]
    bad = [r for r in done if not r["match"]]
    unexpl = [r for r in bad if not r["explained"]]
    s = {"cases": reps, "n": len(done), "skipped": len(reps) - len(done),
         "exact_through_400": len(full), "diverged_float_tie": len(bad) - len(unexpl),
         "diverged_unexplained": len(unexpl),
         "first_divergence_k": sorted(r["first_bad_k"] for r in bad),
         "max_dq_while_equal": max([r["max_dq_while_equal"] for r in done] + [0.0]),
         "illegal_visits": sum(1 for r in done if "illegal_visit" in r),
         "visit_sum_errors": sum(1 for r in done if "gpu_visit_sum_error" in r)}
    pz = [r for r in done if "prior_maxdiff" in r]
    if pz:
        s["root_prior_maxdiff"] = max(r["prior_maxdiff"] for r in pz)
        s["prior_zero_legal_cases"] = sum(1 for r in pz if r["prior_zero_legal_gpu"] > 0)
        s["prior_zero_legal_mean"] = float(np.mean([r["prior_zero_legal_gpu"] for r in pz]))
    cats = {}
    for r in done:
        c = cats.setdefault(r["cat"], [0, 0])
        c[0] += 1
        c[1] += r["match"] or r.get("explained", False)
    s["by_category"] = cats
    s["unexplained_cases"] = [{k: v for k, v in r.items()} for r in unexpl]
    return s


def compare_variants(gdir, procs):
    import multiprocessing as mp
    p = os.path.join(gdir, "variants.npz")
    if not os.path.exists(p):
        return {}
    g = np.load(p, allow_pickle=True)
    z = {k: g[k] for k in ("cfg", "s", "off", "imit", "bias", "feat", "pools", "cat")}
    out = {}
    for name, c, k_, a_ in (("c1", 1.0, 1.0, 0.5), ("c4", 4.0, 1.0, 0.5), ("pw0", 2.0, 0.0, 0.5),
                            ("pw2", 2.0, 2.0, 0.6)):
        grid = [50, 400]
        tasks = [("personal", z, i, np.array(XC.COEFS["personal"]), None, grid,
                  np.stack([g[f"{name}_50_pol"][i], g[f"{name}_400_pol"][i]]),
                  np.stack([g[f"{name}_50_q"][i], g[f"{name}_400_q"][i]]), None, c, k_, a_, 400)
                 for i in range(len(z["cfg"]))]
        with mp.get_context("fork").Pool(procs) as pool:
            reps = pool.map(_grid_one, tasks, chunksize=1)
        s = summarize_grid(reps)
        out[name] = {k: v for k, v in s.items() if k != "cases"}
        print("variant", name, json.dumps({k: v for k, v in out[name].items() if k not in ("by_category",)}), flush=True)
    return out


# ------------------------------------------------------------------ paper-1 full episodes

def _p1_one(args):
    i, cfg, sims, temp, pol_g, term_g, wp_g = args
    o = R.episode_paper1(nets(), int(cfg[0]), int(cfg[1]), int(cfg[2]), sims, 2.0, BASE_SEED + i, root_temp=temp)
    P = np.zeros((8, 90), np.float32)
    for t, p in enumerate(o["policies"]):
        P[t] = p
    acts = o["acts"]
    T = np.zeros(270, np.float32)
    for k, h in enumerate(acts):
        T[(90 if R.TEAM[k] else 0) + h if R.PICK[k] else 180 + h] = 1
    turn_eq = [bool(np.array_equal(np.rint(P[t] * sims), np.rint(pol_g[t] * sims))) for t in range(8)]
    first = next((t for t, x in enumerate(turn_eq) if not x), None)
    return {"i": i, "draft_equal": bool(np.array_equal(T, term_g[:270])), "turns_equal": turn_eq,
            "first_bad_turn": first, "wp_diff": float(abs(o["wp"] - wp_g)), "min_margin": float(min(o["margins"]))}


def compare_paper1(gdir, procs):
    import multiprocessing as mp
    p = os.path.join(gdir, "paper1.npz")
    if not os.path.exists(p):
        return {}
    g = np.load(p)
    out = {}
    for sims in (50, 200):
        for temp in (0, 1):
            tag = f"s{sims}_t{temp}"
            tasks = [(i, g["cfg"][i], sims, float(temp), g[f"{tag}_pol"][i], g[f"{tag}_term"][i], g[f"{tag}_wp"][i])
                     for i in range(len(g["cfg"]))]
            with mp.get_context("fork").Pool(procs) as pool:
                reps = pool.map(_p1_one, tasks, chunksize=1)
            eq = [r for r in reps if r["draft_equal"] and all(r["turns_equal"])]
            out[tag] = {"n": len(reps), "episodes_identical": len(eq),
                        "max_wp_diff_identical": max([r["wp_diff"] for r in eq] + [0.0]),
                        "diverged": [{k: r[k] for k in ("i", "first_bad_turn", "min_margin")}
                                     for r in reps if r not in eq]}
            print("paper1", tag, json.dumps({k: v for k, v in out[tag].items() if k != "diverged"}),
                  "diverged min margins", [round(d["min_margin"], 8) for d in out[tag]["diverged"]], flush=True)
    return out


# ------------------------------------------------------------------ leaf parity and symmetry

def _leaf_one(args):
    acts, cfg, s, off, feat, vnet, fl, pv, coefs = args
    e = R.P3Inputs("personal" if fl == "personal" else "pvnet", cfg, s, off, np.zeros((10, 90)), np.ones((10, 90), bool),
                   coefs, None, feat, vnet)
    e.cfg[57] = pv
    d = R.Draft(int(cfg[0]), int(cfg[1]), int(cfg[2]))
    for k in range(16):
        d.apply(int(acts[k]), R.TEAM[k], R.PICK[k])
    w0 = nets().wp_t0(d.team_heroes(0), d.team_heroes(1), d.map, d.tier)
    p_our = w0 if d.our == 0 else 1 - w0
    return p_our, R.personal_value(p_our, d, d.our, e)


def compare_leaf(gdir, procs):
    import multiprocessing as mp
    p = os.path.join(gdir, "leaf.npz")
    if not os.path.exists(p):
        return {}
    g = np.load(p)
    out = {}
    lg = lambda x: np.log(np.clip(x, 1e-7, 1 - 1e-7) / np.clip(1 - x, 1e-7, 1 - 1e-7))
    for fl, pv in (("personal", 0), ("pvnet", 0), ("pvnet", 1)):
        tag = f"{fl}{pv}"
        co = g[f"{tag}_coefs"]
        n = len(g["acts"])
        tasks = [(g["acts"][i], g["cfg"][i], g["s"][i], g["off"][i], g["feat"][i], g["vnet"], fl, pv, co)
                 for i in range(n)]
        with mp.get_context("fork").Pool(procs) as pool:
            ref = np.array(pool.map(_leaf_one, tasks, chunksize=8))
        b, f, sw = g[f"{tag}_base"], g[f"{tag}_flip"], g[f"{tag}_swap"]
        ours = g["cfg"][:, 2]
        b0s = np.where(ours == 0, co[0], -co[0])
        weird = np.zeros(n, bool)
        r = {"n": n,
             "wp_maxdiff_vs_cpu": float(np.abs(b[:, 0] - ref[:, 0]).max()),
             "value_maxdiff_vs_cpu": float(np.abs(b[:, 1] - ref[:, 1]).max()),
             "wp_flip_sum_maxdev": float(np.abs(b[:, 0] + f[:, 0] - 1).max()),
             "value_flip_sum_maxdev": float(np.abs(b[:, 1] + f[:, 1] - 1).max()),
             "wp_swap_sum_maxdev": float(np.abs(b[:, 0] + sw[:, 0] - 1).max()),
             "value_swap_logit_maxdev": float(np.abs(lg(b[:, 1]) + lg(sw[:, 1]) - 2 * b0s)[
                 (b[:, 1] > 1e-5) & (b[:, 1] < 1 - 1e-5)].max()),
             "personal_shift_mean_abs": float(np.abs(b[:, 1] - b[:, 0]).mean())}
        if f"{tag}_slot" in g.files:
            r["slot_mode_value_differs_frac"] = float((np.abs(g[f"{tag}_slot"][:, 1] - b[:, 1]) > 1e-6).mean())
        del weird
        out[tag] = r
        print("leaf", tag, json.dumps(r), flush=True)
    return out


# ------------------------------------------------------------------ toy optimum and monotonicity

def _exact14(args):
    i, cfg, s, off, pools = args
    e = R.P3Inputs("personal", cfg, s, off, np.zeros((10, 90)), pools, XC.COEFS["personal"])
    d = R.Draft(int(cfg[0]), int(cfg[1]), int(cfg[2]))
    for k in range(14):
        d.apply(int(cfg[8 + k]), R.TEAM[k], R.PICK[k])
    d.our = 0
    m = R.p3_mask(d, e)
    N = nets()
    vals = np.full(90, np.nan)
    for a in np.flatnonzero(m):
        d2 = d.copy()
        d2.apply(int(a), 0, 1)
        m2 = R.p3_mask(d2, e)
        pg = N.gd_probs(d2, m2)
        v = 0.0
        for b in np.flatnonzero(pg > 0):
            d3 = d2.copy()
            d3.apply(int(b), 1, 1)
            w0 = N.wp_t0(d3.team_heroes(0), d3.team_heroes(1), d3.map, d3.tier)
            v += float(pg[b]) * R.personal_value(w0, d3, 0, e)
        vals[a] = v
    return i, vals


def compare_toy(gdir, procs):
    import multiprocessing as mp
    p = os.path.join(gdir, "toy.npz")
    if not os.path.exists(p):
        return {}
    g = np.load(p)
    out = {}
    # step 15: exact values from leaf_eval
    n = len(g["k15_cfg"])
    V = np.full((n, 90), np.nan)
    V[g["k15_exact_i"], g["k15_exact_h"]] = g["k15_exact_v"][:, 1]
    # step 14: exact expectation over the GD reply (CPU replica nets + values)
    tasks = [(i, g["k14_cfg"][i], g["k14_s"][i], g["k14_off"][i], g["k14_pools"][i]) for i in range(len(g["k14_cfg"]))]
    with mp.get_context("fork").Pool(procs) as pool:
        ex14 = dict(pool.map(_exact14, tasks, chunksize=1))
    V14 = np.stack([ex14[i] for i in range(len(tasks))])
    for k, VV in ((15, V), (14, V14)):
        reg, opt, qerr = {}, {}, []
        for sims in (25, 100, 400, 1600):
            rr, oo = [], []
            for sd in range(4):
                pol, q = g[f"k{k}_s{sims}_sd{sd}_pol"], g[f"k{k}_s{sims}_sd{sd}_q"]
                for i in range(len(pol)):
                    legal = np.isfinite(VV[i])
                    ch = int(np.argmax(np.where(legal, pol[i] + 1e-6 * q[i], -1e30)))
                    best = np.nanmax(VV[i])
                    rr.append(best - VV[i, ch])
                    oo.append(VV[i, ch] >= best - 1e-6)
                    if k == 15:
                        vis = pol[i] > 0
                        qerr.append(float(np.abs(q[i][vis] - VV[i][vis]).max()))
            rr = np.array(rr)
            reg[sims] = {"mean_regret": float(rr.mean()), "se": float(rr.std(ddof=1) / math.sqrt(len(rr))),
                         "max_regret": float(rr.max())}
            opt[sims] = float(np.mean(oo))
        out[f"step{k}"] = {"regret": reg, "optimal_frac": opt}
        if k == 15:
            out["step15"]["max_q_vs_exact_value"] = float(max(qerr))
        print("toy", k, json.dumps(out[f"step{k}"]), flush=True)
    m = [out["step14"]["regret"][s]["mean_regret"] for s in (25, 100, 400, 1600)]
    out["step14"]["regret_nonincreasing"] = bool(all(m[j + 1] <= m[j] + 2 * out["step14"]["regret"][(25, 100, 400, 1600)[j + 1]]["se"]
                                                    for j in range(3)))
    return out


# ------------------------------------------------------------------ opponent model

def chi2_test(counts, p):
    from scipy.stats import chi2
    n = counts.sum()
    e = n * p
    big = e >= 5
    o2 = np.append(counts[big], counts[~big].sum())
    e2 = np.append(e[big], e[~big].sum())
    keep = e2 > 0
    stat = float(((o2[keep] - e2[keep]) ** 2 / e2[keep]).sum())
    dof = int(keep.sum() - 1)
    zero_mass_hits = int(counts[p == 0].sum())
    return {"n": int(n), "chi2": stat, "dof": dof, "p": float(chi2.sf(stat, dof)), "draws_on_zero_prob": zero_mass_hits,
            "tv": float(0.5 * np.abs(counts / n - p).sum())}


def compare_opp(gdir):
    p = os.path.join(gdir, "opp.npz")
    if not os.path.exists(p):
        return {}
    g = np.load(p)
    N = nets()
    d = R.Draft(4, 1, 1)
    for k in range(4):
        d.apply(int(g["bans"][k]), R.TEAM[k], 0)
    m = d.free()
    pg = N.gd_probs(d, m).astype(np.float64)
    ex = R.P3Inputs("explore", np.r_[[4, 1, 1, 0, -1, 0, 2, 0], np.full(16, -1), XC.slot_of_default(), np.full(16, -1),
                                     [0, 0, 0]], np.zeros((10, 90)), np.zeros((10, 90)),
                    np.stack([g["imit_row"]] + [np.zeros(90)] * 9), np.ones((10, 90), bool), [0, 1, 0, 0, 0.7, 1.0])
    pp = R.opp_personalize(pg.astype(np.float32), m, 4, 1, ex).astype(np.float64)
    out = {}
    for mode, target in ((0, pg), (1, pp), (2, pp)):
        c = np.bincount(g[f"explore_m{mode}_step4"], minlength=90).astype(np.float64)
        out[f"explore_cfg6_{mode}"] = chi2_test(c, target / target.sum())
    d0 = R.Draft(4, 1, 1)
    p0 = N.gd_probs(d0, d0.free()).astype(np.float64)
    c = np.bincount(g["paper1_first_ban"], minlength=90).astype(np.float64)
    out["paper1_first_ban"] = chi2_test(c, p0 / p0.sum())
    out["personalized_vs_gd_tv"] = float(0.5 * np.abs(pp - pg).sum())
    print("opp", json.dumps(out), flush=True)
    return out


def read_flags(gdir, name):
    p = os.path.join(gdir, f"{name}.npz")
    if not os.path.exists(p):
        return {}
    g = np.load(p, allow_pickle=True)
    r = {k: g[k].tolist() for k in g.files}
    print(name, json.dumps(r), flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-out", required=True)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("stages", nargs="*")
    a = ap.parse_args()
    st = a.stages or ["grid", "variants", "paper1", "leaf", "toy", "opp", "flags"]
    jp = os.path.join(a.gpu_out, "compare.json")
    res = json.load(open(jp)) if os.path.exists(jp) else {}
    for s in st:
        t0 = time.time()
        if s == "grid":
            res["grid"] = compare_grid(a.gpu_out, a.procs)
        elif s == "variants":
            res["variants"] = compare_variants(a.gpu_out, a.procs)
        elif s == "paper1":
            res["paper1"] = compare_paper1(a.gpu_out, a.procs)
        elif s == "leaf":
            res["leaf"] = compare_leaf(a.gpu_out, a.procs)
        elif s == "toy":
            res["toy"] = compare_toy(a.gpu_out, a.procs)
        elif s == "opp":
            res["opp"] = compare_opp(a.gpu_out)
        elif s == "flags":
            res["identity"] = read_flags(a.gpu_out, "identity")
            res["stream"] = read_flags(a.gpu_out, "stream")
            res["determinism"] = read_flags(a.gpu_out, "determinism")
        print(f"== {s} {time.time() - t0:.0f}s", flush=True)
        json.dump(res, open(jp, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
