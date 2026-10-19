"""
P3 tournament: checks for the learned personal value net kernel (cuda_pvnet,
module pvnet_kernel; leaf option (c), cfg[57] = 1).

Inputs: paper 1's population side via the cur vintage (WP, statistics,
F policy seed 0, GD 0), random lobbies with realistic personal tensors
s ~ N(0, 0.2^2), off ~ Bernoulli(0.1), feat = log1p(Poisson(5)), random
pools of 25 heroes per slot, the cur combiner (b0, b1, b2, b3).

(a) cfg[57] = 0: pvnet run() vs cuda_prior run() on the same inputs (feat and
    vnet random, so they must be ignored): every output array byte-identical
    (win probs, terminal actions, root visits, Q, priors, ...), decide-only and
    full drafts, 64 episodes, 100 and 400 sims, cfg[56] = 0 and 1, chance search.
(b) cfg[57] = 1: leaf_eval on 256 random complete drafts vs a float64 mirror
    (assignment from p3_assign.best_assignment; F_t, h, sigmoid in float64).
    Gate: max |V_kernel - V_mirror| < 1e-5 with the kernel's own population
    WP as the mirror's input (the new code). Also reported: the float64 WP
    mirror (paper-1 featurization, WP net in float64) and the end-to-end value
    difference, and the WP column identical to cuda_prior's leaf_eval.
(c) population path: cfg[7] = 0 (personal valuation off) identical to the
    unmodified population kernel cuda_personal/ref (pop_ref_kernel), as in
    p3_mcts_verify: win probs, terminal drafts, root visit distributions; also
    with cfg[57] = 1, assign mode and random tensors (blind ignores them),
    and with personal terms zeroed (b = 0, 1, 0, 0).
Also: cfg[57] = 1 without assign mode is refused.

Run (from training/, GPU host, a few minutes):
  python personalization/p3t_pvnet_verify.py
Output: $P3_RESULTS/p3t/verify_pvnet.json; prints VERIFY OK / FAIL (exit 1).
"""
import copy
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("P1_TIERS", "site")
import numpy as np

import p3_hs_core as C
import p3t_systems as SY

H = 90
N_EP = 64
N_LEAF = 256


def load_kernels():
    import p3_mcts_core as M
    pv = M.load_module("pvnet_kernel", os.path.join(HERE, "cuda_pvnet"))
    pr = M.load_module("prior_kernel", os.path.join(HERE, "cuda_prior"))
    return M, pv, pr, M.ref_kernel()


def engine(kmod, v, n):
    wf, wo, lut = v.wcfg()
    pf, po = v.policy_flat(0)
    gf, go = v.gd_flat(0)
    cls = getattr(kmod, "PersonalEngine", None) or getattr(kmod, "PriorEngine", None) or kmod.OfitEngine
    return cls(pf, gf, wf, po, go, wo, lut, max_concurrent=n, device_id=0)


def random_vnet(rng):
    W1 = rng.normal(0, 0.3, (16, 12))
    c1 = rng.normal(0, 0.1, 16)
    w2 = rng.normal(0, 0.3, 16)
    c2 = np.array([0.1])
    return np.concatenate([W1.ravel(), c1, w2, c2]).astype(np.float32)


def random_tensors(rng, n):
    s = rng.normal(0, 0.2, (n, 10, H)).astype(np.float32)
    off = (rng.rand(n, 10, H) < 0.1).astype(np.float32)
    feat = np.log1p(rng.poisson(5.0, (n, 10, H))).astype(np.float32)
    imit = rng.normal(0, 0.5, (n, 10, H)).astype(np.float32)
    bias = rng.normal(0, 0.5, (n, 10, H)).astype(np.float32)
    pool = np.zeros((n, 10, 3), np.uint32)
    for i in range(n):
        for sl in range(10):
            for h in rng.choice(H, 25, replace=False):
                pool[i, sl, h // 32] |= np.uint32(1 << (h % 32))
    return s, off, feat, imit, bias, pool


def random_drafts(rng, n):
    return np.array([rng.choice(H, 16, replace=False) for _ in range(n)], np.int32)


def slot_of_default():
    """step -> slot: pick j of team t -> slot 5t + j; bans -1."""
    out, cnt = [], [0, 0]
    for k in range(16):
        if SY.IS_PICK[k]:
            t = SY.DRAFT_TEAM[k]
            out.append(5 * t + cnt[t])
            cnt[t] += 1
        else:
            out.append(-1)
    return np.array(out, np.int32)


# ---------------------------------------------------------------- mirror

def h64(F, vnet):
    v = vnet.astype(np.float64)
    W1 = v[:192].reshape(16, 12)
    c1, w2, c2 = v[192:208], v[208:224], v[224]
    return float(w2 @ np.maximum(W1 @ F + c1, 0.0) + c2)


def pvnet_value64(p_our, our, s, off, feat, acts, coefs, vnet):
    """float64 mirror of the cfg[57] = 1 leaf for a complete draft (16 kernel-order actions)."""
    from p3_assign import best_assignment, team_picks
    b0, b1, b2, b3 = (float(x) for x in coefs[:4])
    hs = []
    nonident = False
    for t, picks in enumerate(team_picks(acts)):
        sl = slice(5 * t, 5 * t + 5)
        S, O, perm = best_assignment(s[sl].astype(np.float64), off[sl].astype(np.float64), picks, b2, b3)
        nonident |= perm != (0, 1, 2, 3, 4)
        sv = [float(s[5 * t + i, picks[j]]) if j < len(picks) else 0.0 for i, j in enumerate(perm)]
        fv = [float(feat[5 * t + i, picks[j]]) if j < len(picks) else 0.0 for i, j in enumerate(perm)]
        F = np.array([S, O] + sorted(sv, reverse=True) + sorted(fv, reverse=True), np.float64)
        hs.append(h64(F, vnet))
    dh = hs[our] - hs[1 - our]
    b0s = b0 if our == 0 else -b0
    pc = min(max(float(p_our), 1e-6), 1 - 1e-6)
    z = b0s + b1 * np.log(pc / (1 - pc)) + dh
    return 1 / (1 + np.exp(-z)), nonident


def wp64(v, rows):
    """float64 version of Vintage.wp_drafts: symmetrized P(team 0 wins)."""
    import torch
    from paper1_revision import core
    m, cols = v.wp_model()
    m64 = copy.deepcopy(m).double().eval()
    Xf, Xs = core._chunk((rows, core.stats_dict(v.stats())))
    with torch.no_grad():
        a = m64(torch.tensor(np.asarray(Xf, np.float64)[:, cols])).view(-1).numpy()
        b = m64(torch.tensor(np.asarray(Xs, np.float64)[:, cols])).view(-1).numpy()
    return 0.5 * (a + 1 - b)


# ---------------------------------------------------------------- checks

def check_identity_vs_prior(M, PV, PR, v, coefs, rng, out):
    n = N_EP
    s, off, feat, imit, bias, pool = random_tensors(rng, n)
    vnet = random_vnet(rng)
    drafts = random_drafts(rng, n)
    so = slot_of_default()
    res = {}
    allsame = True
    for mode in ("decide", "full"):
        cfg = M.empty_cfg(n)
        cfg[:, 0] = rng.randint(14, size=n)
        cfg[:, 1] = rng.randint(3, size=n)
        cfg[:, 5] = 1
        cfg[:, 6] = rng.randint(2, size=n)          # GD or imitation opponent
        cfg[:, 7] = 1
        cfg[:, 24:40] = so
        cfg[:, 40:56] = -1
        cfg[:, M.CFG_ASSIGN] = 1
        if mode == "decide":
            ks = rng.choice([2, 5, 7, 9, 13], n)
            cfg[:, 4] = ks
            cfg[:, 2] = [SY.DRAFT_TEAM[k] for k in ks]
            cfg[:, 8:24] = np.where(np.arange(16)[None] < ks[:, None], drafts, -1)
        else:
            cfg[:, 2] = rng.randint(2, size=n)
            cfg[:, 3] = rng.randint(2, size=n)      # self-play on half
        for c56 in (0, 1):
            cfg[:, 56] = c56
            cfg[:, 57] = 0
            for sims in (100, 400):
                seed = 1000 + 10 * sims + c56
                ep = engine(PR, v, n)
                rp = ep.run(cfg, s, off, imit, pool, coefs, bias, sims, 2.0, seed, 0.0, 0.3, 0.0, 1,
                            search_mode=1)
                del ep
                ev = engine(PV, v, n)
                rv = ev.run(cfg, s, off, imit, pool, coefs, bias, feat, vnet, sims, 2.0, seed, 0.0, 0.3, 0.0, 1,
                            search_mode=1)
                del ev
                names = ("win_prob", "wp_t0", "v_t0", "terminal_actions", "num_turns", "root_visits", "root_q",
                         "out_step", "out_team", "max_nodes", "cap_hits", "decided", "our_team", "root_prior")
                same = {nm: np.asarray(a).tobytes() == np.asarray(b).tobytes() for nm, a, b in zip(names, rp, rv)}
                ok = all(same.values())
                allsame &= ok
                key = f"{mode} | cfg56={c56} | sims {sims}"
                res[key] = {"episodes": n, "all_outputs_byte_identical": ok,
                            "differing_outputs": [k for k, x in same.items() if not x],
                            "root_visits_nonzero_turns": int((np.asarray(rv[5]).sum(-1) > 0).sum())}
                print(key, json.dumps(res[key]), flush=True)
    out["a_identity_vs_prior"] = res
    return allsame


def check_leaf(M, PV, PR, v, coefs, rng, out):
    from shared import HEROES, MAPS, SKILL_TIERS
    n = N_LEAF
    s, off, feat, imit, bias, pool = random_tensors(rng, n)
    vnet = random_vnet(rng)
    drafts = random_drafts(rng, n)
    cfg = M.empty_cfg(n)
    cfg[:, 0] = rng.randint(14, size=n)
    cfg[:, 1] = rng.randint(3, size=n)
    cfg[:, 2] = rng.randint(2, size=n)
    cfg[:, 7] = 1
    cfg[:, 8:24] = drafts
    cfg[:, 24:40] = slot_of_default()
    cfg[:, M.CFG_ASSIGN] = 1
    cfg[:, 57] = 1
    e = engine(PV, v, n)
    r = e.leaf_eval(cfg, s, off, imit, pool, coefs, bias, feat, vnet)
    cfg0 = cfg.copy()
    cfg0[:, 57] = 0
    r0 = e.leaf_eval(cfg0, s, off, imit, pool, coefs, bias, feat, vnet)
    refused = False
    bad = cfg.copy()
    bad[0, M.CFG_ASSIGN] = 0
    try:
        e.leaf_eval(bad, s, off, imit, pool, coefs, bias, feat, vnet)
    except RuntimeError as ex:
        refused = "assign" in str(ex)
    del e
    ep = engine(PR, v, n)
    rp = ep.leaf_eval(cfg0, s, off, imit, pool, coefs, bias)
    del ep
    rows = []
    for i in range(n):
        a = drafts[i]
        t0 = [HEROES[a[k]] for k in range(16) if SY.IS_PICK[k] and SY.DRAFT_TEAM[k] == 0]
        t1 = [HEROES[a[k]] for k in range(16) if SY.IS_PICK[k] and SY.DRAFT_TEAM[k] == 1]
        rows.append((t0, t1, MAPS[cfg[i, 0]], SKILL_TIERS[cfg[i, 1]]))
    w0 = wp64(v, rows)
    d_val, d_wp, d_e2e, nonid, shift = [], [], [], 0, []
    for i in range(n):
        our = int(cfg[i, 2])
        p64 = w0[i] if our == 0 else 1 - w0[i]
        V, ni = pvnet_value64(float(r[i, 0]), our, s[i], off[i], feat[i], drafts[i], coefs, vnet)
        Ve, _ = pvnet_value64(p64, our, s[i], off[i], feat[i], drafts[i], coefs, vnet)
        nonid += ni
        d_val.append(abs(float(r[i, 1]) - V))
        d_wp.append(abs(float(r[i, 0]) - p64))
        d_e2e.append(abs(float(r[i, 1]) - Ve))
        shift.append(abs(float(r[i, 1]) - float(r[i, 0])))
    out["b_leaf"] = {
        "drafts": n,
        "max_abs_diff_value": float(np.max(d_val)), "mean_abs_diff_value": float(np.mean(d_val)),
        "max_abs_diff_wp_vs_float64_mirror": float(np.max(d_wp)),
        "max_abs_diff_value_end_to_end_float64": float(np.max(d_e2e)),
        "mean_abs_diff_value_end_to_end_float64": float(np.mean(d_e2e)),
        "teams_with_non_identity_assignment_drafts": int(nonid),
        "mean_abs_pvnet_shift_from_wp": float(np.mean(shift)),
        "wp_column_identical_to_prior_kernel": bool(r[:, 0].tobytes() == rp[:, 0].tobytes()),
        "cfg57_0_values_identical_to_prior_kernel": bool(r0.tobytes() == rp.tobytes()),
        "cfg57_without_assign_refused": bool(refused),
    }
    print("leaf", json.dumps(out["b_leaf"]), flush=True)
    b = out["b_leaf"]
    return (b["max_abs_diff_value"] < 1e-5 and b["max_abs_diff_value_end_to_end_float64"] < 1e-4
            and b["wp_column_identical_to_prior_kernel"] and b["cfg57_0_values_identical_to_prior_kernel"]
            and b["cfg57_without_assign_refused"] and b["teams_with_non_identity_assignment_drafts"] > 0)


def check_population(M, PV, R, v, coefs, rng, out):
    n = N_EP
    base = np.stack([rng.randint(14, size=n), rng.randint(3, size=n), rng.randint(2, size=n)], 1).astype(np.int32)
    s_r, off_r, feat_r, imit_r, bias_r, pool_r = random_tensors(rng, n)
    vnet = random_vnet(rng)
    zero = np.zeros((n, 10, H), np.float32)
    res = {}
    ok = True
    for sims in (100, 400):
        er = engine(R, v, n)
        rr = er.run_episodes(np.ascontiguousarray(base), sims, 2.0, 777, 0.0, 0.3, 0.0, 1, search_mode=1)
        del er
        variants = (
            ("valuation off", 0, 0, 0, np.array([0, 1, 0, 0, 0.8, 0], np.float32), zero, zero, zero),
            ("valuation off, cfg57=1 assign, random tensors", 0, 1, 1, coefs, s_r, off_r, feat_r),
            ("personal terms zeroed", 1, 0, 0, np.array([0, 1, 0, 0, 0.8, 0], np.float32), zero, zero, zero),
        )
        for label, c7, c57, c58, cf, s, off, feat in variants:
            cfg = M.empty_cfg(n)
            cfg[:, :3] = base
            cfg[:, 7] = c7
            cfg[:, 24:40] = -1
            cfg[:, 57] = c57
            cfg[:, M.CFG_ASSIGN] = c58
            ek = engine(PV, v, n)
            kk = ek.run(cfg, s, off, zero, np.zeros((n, 10, 3), np.uint32), cf, zero, feat, vnet,
                        sims, 2.0, 777, 0.0, 0.3, 0.0, 1, search_mode=1)
            del ek
            same_wp = all(float(rr[i][0]) == float(kk[0][i]) for i in range(n))
            same_draft, same_pol = True, True
            for i in range(n):
                tsv = np.array(rr[i][2])
                acts = kk[3][i]
                t0 = sorted(int(a) for k, a in enumerate(acts) if SY.IS_PICK[k] and SY.DRAFT_TEAM[k] == 0)
                t1 = sorted(int(a) for k, a in enumerate(acts) if SY.IS_PICK[k] and SY.DRAFT_TEAM[k] == 1)
                bn = sorted(int(a) for k, a in enumerate(acts) if not SY.IS_PICK[k])
                if (t0 != sorted(np.flatnonzero(tsv[:H] > 0.5).tolist())
                        or t1 != sorted(np.flatnonzero(tsv[H:2 * H] > 0.5).tolist())
                        or bn != sorted(np.flatnonzero(tsv[2 * H:3 * H] > 0.5).tolist())):
                    same_draft = False
                for t, ex in enumerate(rr[i][1]):
                    if not np.array_equal(np.array(ex[1]), kk[5][i, t]):
                        same_pol = False
            key = f"{label} | sims {sims}"
            res[key] = {"episodes": n, "identical_win_prob": same_wp, "identical_drafts": same_draft,
                        "identical_root_visit_distributions": same_pol,
                        "max_nodes_used": int(kk[9].max()), "capacity_hits": int(kk[10].sum())}
            ok &= same_wp and same_draft and same_pol
            print(key, json.dumps(res[key]), flush=True)
    out["c_population_identity"] = res
    return ok


def main():
    import torch
    torch.set_num_threads(2)
    t0 = time.time()
    M, PV, PR, R = load_kernels()
    v = SY.vintage("cur")
    coefs = np.array(list(v.spec["combiner"]) + [0.8, 0.7], np.float32)   # b0..b3, imit_w0, prior alpha
    out = {"coefs": coefs.tolist(), "search_mode": "chance",
           "comp_fallback_order": {"pvnet": PV.COMP_FALLBACK_ORDER, "prior": PR.COMP_FALLBACK_ORDER,
                                   "ref": R.COMP_FALLBACK_ORDER},
           "tensors": "s ~ N(0, 0.2^2), off ~ Bernoulli(0.1), feat = log1p(Poisson(5)), pools 25 heroes/slot, "
                      "imit and prior bias ~ N(0, 0.5^2), vnet W1, w2 ~ N(0, 0.3^2), c1 ~ N(0, 0.1^2), c2 0.1"}
    gates = {}
    gates["a_identity_vs_prior"] = check_identity_vs_prior(M, PV, PR, v, coefs, np.random.RandomState(1), out)
    gates["b_leaf_parity"] = check_leaf(M, PV, PR, v, coefs, np.random.RandomState(2), out)
    gates["c_population_identity"] = check_population(M, PV, R, v, coefs, np.random.RandomState(3), out)
    out["gates"] = {k: bool(x) for k, x in gates.items()}
    out["pass"] = all(gates.values())
    out["seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.join(C.RESULTS, "p3t"), exist_ok=True)
    with open(os.path.join(C.RESULTS, "p3t", "verify_pvnet.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("gates", json.dumps(out["gates"]), flush=True)
    print("VERIFY", "OK" if out["pass"] else "FAIL", flush=True)
    sys.exit(0 if out["pass"] else 1)


if __name__ == "__main__":
    main()
