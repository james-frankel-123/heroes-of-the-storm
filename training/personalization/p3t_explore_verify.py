"""
P3 tournament: checks for the explore kernel (cuda_explore, module
explore_kernel): pvnet kernel plus cfg[56] = 2 (prior term at our ban steps
too) and cfg[6] = 2 (player-aware opponent inside the search).

Inputs as p3t_pvnet_verify.py: the cur vintage (WP, statistics, F policy
seed 0, GD 0), random lobbies, s ~ N(0, 0.2^2), off ~ Bernoulli(0.1),
feat = log1p(Poisson(5)), imit and prior bias ~ N(0, 0.5^2), pools of 25
heroes per slot, assign mode, the cur combiner.

(a) cfg[56] in {0, 1}, cfg[6] in {0, 1}: explore run() vs pvnet run(), every
    output byte-identical, decide-only and full drafts (self-play on half),
    64 episodes, 100 and 400 sims, chance search; cfg[57] random in {0, 1};
    ban-step rows set to random rows on half the episodes (both kernels must
    ignore them in these modes). Also legacy and rollfwd search at 100 sims.
(b) cfg[56] = 2: (i) with ban rows -1 byte-identical to cfg[56] = 1;
    (ii) decide-only at our ban steps with the ban row pointing to a row with
    a strong bias (+4) on one hero: the root prior (out_prior) equals
    softmax(alpha log p + bias[row]) over valid heroes, p = the root prior of
    the same run with cfg[56] = 0, to 1e-5.
(c) cfg[6] = 2: (i) imit all zeros, imit_w0 = 1: byte-identical to cfg[6] = 0
    (decide-only and full drafts, chance, legacy and rollfwd search); also
    reported, not gated: imit rows constant (0.7) with w0 = 1, which runs the
    full log/exp path (identical up to float rounding in sampling);
    (ii) imit +50 on one hero h for every opponent row (picks and bans): in
    full drafts (no self-play) the opponent takes h at its first step where h
    is legal, in 100% of episodes; and decide-only root Q values change versus
    cfg[6] = 0 (in-search effect only, the main loop never runs an opponent
    step in decide-only mode).
(d) population path: cfg[7] = 0 identical to cuda_personal/ref (pop_ref_kernel)
    as in p3_mcts_verify, including with cfg[6] = 2 and cfg[56] = 2 set and
    zero tensors / -1 rows.

Run (from training/, GPU host, a few minutes):
  python personalization/p3t_explore_verify.py
Output: $P3_RESULTS/p3t/verify_explore.json; prints VERIFY OK / FAIL (exit 1).
"""
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
from p3t_pvnet_verify import engine, random_drafts, random_tensors, random_vnet, slot_of_default

H = 90
N_EP = 64
NAMES = ("win_prob", "wp_t0", "v_t0", "terminal_actions", "num_turns", "root_visits", "root_q",
         "out_step", "out_team", "max_nodes", "cap_hits", "decided", "our_team", "root_prior")
BAN_STEPS = [k for k in range(16) if not SY.IS_PICK[k]]


def load_kernels():
    import p3_mcts_core as M
    ex = M.load_module("explore_kernel", os.path.join(HERE, "cuda_explore"))
    pv = M.load_module("pvnet_kernel", os.path.join(HERE, "cuda_pvnet"))
    return M, ex, pv, M.ref_kernel()


def run(kmod, v, cfg, s, off, imit, pool, coefs, bias, feat, vnet, sims, seed, mode):
    e = engine(kmod, v, len(cfg))
    r = e.run(np.ascontiguousarray(cfg), s, off, imit, pool, coefs, bias, feat, vnet, sims, 2.0, seed,
              0.0, 0.3, 0.0, 1, search_mode=mode)
    del e
    return r


def same_outputs(ra, rb):
    same = {nm: np.asarray(a).tobytes() == np.asarray(b).tobytes() for nm, a, b in zip(NAMES, ra, rb)}
    return all(same.values()), [k for k, x in same.items() if not x]


def base_cfg(M, rng, n, mode, drafts, ks=None):
    cfg = M.empty_cfg(n)
    cfg[:, 0] = rng.randint(14, size=n)
    cfg[:, 1] = rng.randint(3, size=n)
    cfg[:, 5] = 1
    cfg[:, 7] = 1
    cfg[:, 24:40] = slot_of_default()
    cfg[:, 40:56] = -1
    cfg[:, M.CFG_ASSIGN] = 1
    if mode == "decide":
        if ks is None:
            ks = rng.choice([2, 5, 7, 9, 13], n)
        cfg[:, 4] = ks
        cfg[:, 2] = [SY.DRAFT_TEAM[k] for k in ks]
        cfg[:, 8:24] = np.where(np.arange(16)[None] < np.asarray(ks)[:, None], drafts, -1)
    else:
        cfg[:, 2] = rng.randint(2, size=n)
        cfg[:, 3] = rng.randint(2, size=n)
    return cfg


def check_identity_vs_pvnet(M, EX, PV, v, coefs, rng, out):
    n = N_EP
    s, off, feat, imit, bias, pool = random_tensors(rng, n)
    vnet = random_vnet(rng)
    drafts = random_drafts(rng, n)
    res, ok = {}, True
    for mode in ("decide", "full"):
        cfg = base_cfg(M, rng, n, mode, drafts)
        cfg[:, 57] = rng.randint(2, size=n)
        half = rng.rand(n) < 0.5
        for k in BAN_STEPS:
            cfg[half, 24 + k] = rng.randint(10, size=int(half.sum()))
        grid = [(c56, c6, sims, 1) for c56 in (0, 1) for c6 in (0, 1) for sims in (100, 400)]
        grid += [(1, 1, 100, 0), (1, 1, 100, 2)]
        for c56, c6, sims, sm in grid:
            cfg[:, 56] = c56
            cfg[:, 6] = c6
            seed = 1000 + 10 * sims + 2 * c56 + c6 + 100 * sm
            ra = run(PV, v, cfg, s, off, imit, pool, coefs, bias, feat, vnet, sims, seed, sm)
            rb = run(EX, v, cfg, s, off, imit, pool, coefs, bias, feat, vnet, sims, seed, sm)
            same, diff = same_outputs(ra, rb)
            ok &= same
            key = f"{mode} | cfg56={c56} cfg6={c6} | sims {sims} | search {sm}"
            res[key] = {"episodes": n, "all_outputs_byte_identical": same, "differing_outputs": diff}
            print(key, json.dumps(res[key]), flush=True)
    out["a_identity_vs_pvnet"] = res
    return ok


def check_ban_bias(M, EX, v, coefs, rng, out):
    n = N_EP
    s, off, feat, imit, bias, pool = random_tensors(rng, n)
    vnet = np.zeros(0, np.float32)
    feat[:] = 0
    drafts = random_drafts(rng, n)
    res = {}
    # (i) ban rows -1: cfg56 = 2 == cfg56 = 1
    ok_i = True
    for mode in ("decide", "full"):
        cfg = base_cfg(M, rng, n, mode, drafts)
        cfg[:, 6] = rng.randint(2, size=n)
        for sims in (100, 400):
            cfg[:, 56] = 1
            ra = run(EX, v, cfg, s, off, imit, pool, coefs, bias, feat, vnet, sims, 50 + sims, 1)
            cfg[:, 56] = 2
            rb = run(EX, v, cfg, s, off, imit, pool, coefs, bias, feat, vnet, sims, 50 + sims, 1)
            same, diff = same_outputs(ra, rb)
            ok_i &= same
            key = f"i ban rows -1 | {mode} | sims {sims}"
            res[key] = {"all_outputs_byte_identical_to_cfg56_1": same, "differing_outputs": diff}
            print(key, json.dumps(res[key]), flush=True)
    # (ii) our ban step, strong bias row
    ks = rng.choice(BAN_STEPS, n)
    cfg = base_cfg(M, rng, n, "decide", drafts, ks)
    cfg[:, 6] = 0
    row = rng.randint(10, size=n)
    hero = np.empty(n, np.int64)
    sb = np.zeros_like(bias)
    for i in range(n):
        taken = set(int(a) for a in drafts[i, :ks[i]])
        hero[i] = rng.choice([h for h in range(H) if h not in taken])
        sb[i] = bias[i]
        sb[i, row[i], hero[i]] += 4.0
        cfg[i, 24 + ks[i]] = row[i]
    cfg[:, 56] = 0
    r0 = run(EX, v, cfg, s, off, imit, pool, coefs, sb, feat, vnet, 100, 77, 1)
    cfg[:, 56] = 1
    r1 = run(EX, v, cfg, s, off, imit, pool, coefs, sb, feat, vnet, 100, 77, 1)
    cfg[:, 56] = 2
    r2 = run(EX, v, cfg, s, off, imit, pool, coefs, sb, feat, vnet, 100, 77, 1)
    alpha = float(coefs[5])
    p0 = np.asarray(r0[13])[:, 0, :].astype(np.float64)
    p2 = np.asarray(r2[13])[:, 0, :].astype(np.float64)
    diffs, mass0, mass2 = [], [], []
    for i in range(n):
        ok = p0[i] > 0
        z = np.full(H, -np.inf)
        z[ok] = alpha * np.log(p0[i, ok]) + sb[i, row[i]].astype(np.float64)[ok]
        q = np.exp(z - z[ok].max())
        q /= q.sum()
        diffs.append(float(np.abs(q - p2[i]).max()))
        mass0.append(p0[i, hero[i]])
        mass2.append(p2[i, hero[i]])
    cfg1_unbiased = np.asarray(r1[13]).tobytes() == np.asarray(r0[13]).tobytes()
    res["ii our ban step, bias +4 on one hero"] = {
        "episodes": n, "ban_steps": sorted(set(int(k) for k in ks)),
        "max_abs_diff_root_prior_vs_python_softmax": max(diffs),
        "mean_root_prior_on_hero_cfg56_0": float(np.mean(mass0)),
        "mean_root_prior_on_hero_cfg56_2": float(np.mean(mass2)),
        "cfg56_1_root_prior_identical_to_cfg56_0_at_bans": bool(cfg1_unbiased),
        "root_visits_changed_episodes": int((np.abs(np.asarray(r2[5])[:, 0] - np.asarray(r0[5])[:, 0]).sum(1) > 0).sum()),
    }
    print("ban bias", json.dumps(res["ii our ban step, bias +4 on one hero"]), flush=True)
    out["b_ban_bias"] = res
    return ok_i and max(diffs) < 1e-5 and cfg1_unbiased and np.mean(mass2) > np.mean(mass0)


def opp_rows(cfg):
    """Every opponent step (picks and bans) gets a row: picks keep their slot,
    bans get the opponent team's first slot."""
    for k in BAN_STEPS:
        cfg[:, 24 + k] = 5 * SY.DRAFT_TEAM[k]


def check_opp(M, EX, v, coefs, rng, out):
    n = N_EP
    s, off, feat, imit_r, bias, pool = random_tensors(rng, n)
    vnet = np.zeros(0, np.float32)
    feat[:] = 0
    drafts = random_drafts(rng, n)
    zero = np.zeros((n, 10, H), np.float32)
    c1 = coefs.copy()
    c1[4] = 1.0
    res = {}
    ok_i = True
    for mode in ("decide", "full"):
        cfg = base_cfg(M, rng, n, mode, drafts)
        if mode == "full":
            cfg[:, 3] = 0
        opp_rows(cfg)
        cfg[:, 56] = rng.randint(3, size=n)
        for sims, sm in ((100, 1), (400, 1), (100, 0), (100, 2)):
            cfg[:, 6] = 0
            ra = run(EX, v, cfg, s, off, zero, pool, c1, bias, feat, vnet, sims, 300 + sims + sm, sm)
            cfg[:, 6] = 2
            rb = run(EX, v, cfg, s, off, zero, pool, c1, bias, feat, vnet, sims, 300 + sims + sm, sm)
            same, diff = same_outputs(ra, rb)
            ok_i &= same
            const = np.full((n, 10, H), 0.7, np.float32)
            rc = run(EX, v, cfg, s, off, const, pool, c1, bias, feat, vnet, sims, 300 + sims + sm, sm)
            key = f"i zero imit, w0 = 1 | {mode} | sims {sims} | search {sm}"
            res[key] = {"all_outputs_byte_identical_to_cfg6_0": same, "differing_outputs": diff,
                        "constant_rows_full_path_identical_terminal_drafts":
                            float((np.asarray(rc[3]) == np.asarray(ra[3])).all(1).mean()),
                        "constant_rows_full_path_identical_root_visits":
                            float((np.asarray(rc[5]) == np.asarray(ra[5])).all((1, 2)).mean())}
            print(key, json.dumps(res[key]), flush=True)
    # (ii) huge bias on one hero for every opponent row
    hero = rng.randint(H, size=n)
    big = imit_r.copy()
    for i in range(n):
        big[i, :, hero[i]] = 50.0
    cfg = base_cfg(M, rng, n, "full", drafts)
    cfg[:, 3] = 0
    opp_rows(cfg)
    rates = {}
    ok_ii = True
    for c6 in (0, 1, 2):
        cfg[:, 6] = c6
        hits = legal_n = 0
        for sims in (100, 400):
            r = run(EX, v, cfg, s, off, big, pool, coefs, bias, feat, vnet, sims, 900 + sims, 1)
            acts = np.asarray(r[3])
            for i in range(n):
                our = int(cfg[i, 2])
                for k in range(16):
                    if SY.DRAFT_TEAM[k] == our:
                        if hero[i] in acts[i, :k + 1]:
                            break
                        continue
                    if hero[i] in acts[i, :k]:
                        break
                    if SY.IS_PICK[k]:
                        tm = SY.DRAFT_TEAM[k]
                        u = np.bitwise_or.reduce(pool[i, 5 * tm:5 * tm + 5], 0)
                        inpool = lambda h: (int(u[h // 32]) >> (h % 32)) & 1
                        free = [h for h in range(H) if h not in acts[i, :k]]
                        if any(inpool(h) for h in free) and not inpool(hero[i]):
                            continue
                    legal_n += 1
                    hits += int(acts[i, k] == hero[i])
                    break
        rates[f"cfg6={c6}"] = {"episodes_with_a_legal_opp_step": legal_n, "opp_took_hero": hits,
                               "rate": hits / max(legal_n, 1)}
    ok_ii &= rates["cfg6=2"]["rate"] == 1.0 and rates["cfg6=2"]["episodes_with_a_legal_opp_step"] > 0
    res["ii full drafts, imit +50 on one hero"] = rates
    print("opp full", json.dumps(rates), flush=True)
    # decide-only: root Q changes (in-search effect)
    cfg = base_cfg(M, rng, n, "decide", drafts)
    opp_rows(cfg)
    qs = {}
    for c6 in (0, 2):
        cfg[:, 6] = c6
        r = run(EX, v, cfg, s, off, big, pool, coefs, bias, feat, vnet, 400, 4242, 1)
        qs[c6] = (np.asarray(r[6])[:, 0], np.asarray(r[5])[:, 0])
    dq = np.abs(qs[2][0] - qs[0][0])
    changed = int((dq.max(1) > 0).sum())
    # episodes where the opponent still acts after the decide step
    later = int(sum(any(SY.DRAFT_TEAM[k] != SY.DRAFT_TEAM[int(cfg[i, 4])] for k in range(int(cfg[i, 4]) + 1, 16))
                    for i in range(n)))
    res["ii decide-only root Q, imit +50 vs cfg6=0"] = {
        "episodes": n, "episodes_with_later_opp_steps": later, "episodes_root_q_changed": changed,
        "mean_max_abs_root_q_change": float(dq.max(1).mean()),
        "episodes_root_visits_changed": int((np.abs(qs[2][1] - qs[0][1]).sum(1) > 0).sum())}
    print("opp decide", json.dumps(res["ii decide-only root Q, imit +50 vs cfg6=0"]), flush=True)
    # imit rows are random on every hero, so any later opponent step changes the search
    ok_ii &= changed >= later and later > 0
    out["c_opp_personal"] = res
    return ok_i and ok_ii


def check_population(M, EX, R, v, coefs, rng, out):
    n = N_EP
    base = np.stack([rng.randint(14, size=n), rng.randint(3, size=n), rng.randint(2, size=n)], 1).astype(np.int32)
    s_r, off_r, feat_r, imit_r, bias_r, pool_r = random_tensors(rng, n)
    vnet = random_vnet(rng)
    zero = np.zeros((n, 10, H), np.float32)
    plain = np.array([0, 1, 0, 0, 0.8, 0], np.float32)
    res, ok = {}, True
    for sims in (100, 400):
        er = engine(R, v, n)
        rr = er.run_episodes(np.ascontiguousarray(base), sims, 2.0, 777, 0.0, 0.3, 0.0, 1, search_mode=1)
        del er
        variants = (
            ("valuation off", 0, 0, 0, 0, 0, plain, zero, zero, zero),
            ("valuation off, cfg57=1 assign, random tensors", 0, 1, 1, 0, 0, coefs, s_r, off_r, feat_r),
            ("valuation off, cfg6=2 cfg56=2, rows -1", 0, 0, 0, 2, 2, coefs, s_r, off_r, feat_r),
            ("personal terms zeroed", 1, 0, 0, 0, 0, plain, zero, zero, zero),
        )
        for label, c7, c57, c58, c6, c56, cf, s, off, feat in variants:
            cfg = M.empty_cfg(n)
            cfg[:, :3] = base
            cfg[:, 6] = c6
            cfg[:, 7] = c7
            cfg[:, 24:40] = -1
            cfg[:, 56] = c56
            cfg[:, 57] = c57
            cfg[:, M.CFG_ASSIGN] = c58
            imit = imit_r if c6 else zero
            bias = bias_r if c56 else zero
            kk = run(EX, v, cfg, s, off, imit, np.zeros((n, 10, 3), np.uint32), cf, bias, feat, vnet, sims, 777, 1)
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
                        "identical_root_visit_distributions": same_pol}
            ok &= same_wp and same_draft and same_pol
            print(key, json.dumps(res[key]), flush=True)
    out["d_population_identity"] = res
    return ok


def main():
    import torch
    torch.set_num_threads(2)
    t0 = time.time()
    M, EX, PV, R = load_kernels()
    v = SY.vintage("cur")
    coefs = np.array(list(v.spec["combiner"]) + [0.8, 0.7], np.float32)   # b0..b3, imit_w0, prior alpha
    out = {"coefs": coefs.tolist(),
           "attrs": {"BAN_BIAS_MODE": EX.BAN_BIAS_MODE, "OPP_PERSONAL_MODE": EX.OPP_PERSONAL_MODE,
                     "SEARCH": [EX.SEARCH_LEGACY, EX.SEARCH_CHANCE, EX.SEARCH_ROLLFWD],
                     "CFG_LEN": EX.CFG_LEN, "PVNET_LEN": EX.PVNET_LEN},
           "comp_fallback_order": {"explore": EX.COMP_FALLBACK_ORDER, "pvnet": PV.COMP_FALLBACK_ORDER,
                                   "ref": R.COMP_FALLBACK_ORDER},
           "device": torch.cuda.get_device_name(0)}
    gates = {"attrs": EX.BAN_BIAS_MODE == 2 and EX.OPP_PERSONAL_MODE == 2}
    gates["a_identity_vs_pvnet"] = check_identity_vs_pvnet(M, EX, PV, v, coefs, np.random.RandomState(1), out)
    gates["b_ban_bias"] = check_ban_bias(M, EX, v, coefs, np.random.RandomState(2), out)
    gates["c_opp_personal"] = check_opp(M, EX, v, coefs, np.random.RandomState(3), out)
    gates["d_population_identity"] = check_population(M, EX, R, v, coefs, np.random.RandomState(4), out)
    out["gates"] = {k: bool(x) for k, x in gates.items()}
    out["pass"] = all(gates.values())
    out["seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.join(C.RESULTS, "p3t"), exist_ok=True)
    with open(os.path.join(C.RESULTS, "p3t", "verify_explore.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("gates", json.dumps(out["gates"]), flush=True)
    print("VERIFY", "OK" if out["pass"] else "FAIL", flush=True)
    sys.exit(0 if out["pass"] else 1)


if __name__ == "__main__":
    main()
