"""
P3 tournament: kernel checks for the paper-1 leaf in cuda_personal.

1. Leaf parity: real tournament-window drafts (kernel labels), the cur
   vintage's personal tensors, assign mode; kernel leaf_eval vs the float64
   Python mirror (paper-1 featurization + the vintage WP, symmetrized; then
   the personal terms with the best assignment). Gate: max |diff| < 1e-4.
2. Blind invariance: with personal valuation off, decide-only searches are
   identical whatever the personal tensors hold.
3. Decide-only smoke and cost: one search per draft at a ban step and at a
   pick step, aware and blind, legality and GPU seconds per decision.
Run (from training/, GPU host): python personalization/p3t_verify.py --vintage cur
Output: $P3_RESULTS/p3t/verify_<vintage>.json; exit 1 if a gate fails.
"""
import argparse
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
import p3t_data as TD
import p3t_systems as SY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vintage", default="cur")
    ap.add_argument("--n", type=int, default=256)
    a = ap.parse_args()
    import torch
    import p3_mcts_core as M
    from p3_assign import assignment_terms
    from shared import HEROES, MAPS, SKILL_TIERS
    torch.set_num_threads(2)
    out = {"vintage": a.vintage}
    L = TD.load_lobbies()
    zz = np.load(os.path.join(TD.OUT, "state_cur.npz"))
    z = {k: zz[k] for k in ("s", "off", "n")}
    rng = np.random.RandomState(5)
    win = np.flatnonzero(L["day"] <= TD.LOBBY_END)
    gl = np.sort(rng.choice(win, a.n, replace=False))
    v = SY.vintage(a.vintage)
    K = SY.kernel_module()
    coefs = np.array([0.003, 1.0, 4.5, -0.05, 0.0], np.float32)
    n = len(gl)
    cfg = M.empty_cfg(n)
    s = np.zeros((n, 10, 90), np.float32)
    off = np.zeros((n, 10, 90), np.float32)
    imit = np.zeros((n, 10, 90), np.float32)
    pool = np.zeros((n, 10, 3), np.uint32)
    rows, acts_all = [], []
    for i, gi in enumerate(gl):
        st = L["steps"][gi]
        first = int(L["first_team"][gi])
        acts, slots, cnt, j = [], [], [0, 0], 0
        for k in range(16):
            h, ty, tm, r = st[k]
            kt = tm ^ first
            assert kt == SY.DRAFT_TEAM[k]
            acts.append(int(h))
            if ty == 1:
                sl = 5 * kt + cnt[kt]
                cnt[kt] += 1
                slots.append(sl)
                s[i, sl] = z["s"][gi * 10 + j].astype(np.float32)
                off[i, sl] = z["off"][gi * 10 + j]
                pool[i, sl] = M.pool_bits(z["n"][gi * 10 + j] > 0)
                j += 1
            else:
                slots.append(-1)
        cfg[i, 0] = MAPS.index(str(L["map"][gi]))
        cfg[i, 1] = SKILL_TIERS.index(str(L["tier"][gi]))
        cfg[i, 2] = rng.randint(2)
        cfg[i, 7] = 1
        cfg[i, 8:24] = acts
        cfg[i, 24:40] = slots
        cfg[i, M.CFG_ASSIGN] = 1
        t0 = [HEROES[acts[k]] for k in range(16) if SY.IS_PICK[k] and SY.DRAFT_TEAM[k] == 0]
        t1 = [HEROES[acts[k]] for k in range(16) if SY.IS_PICK[k] and SY.DRAFT_TEAM[k] == 1]
        rows.append((t0, t1, str(L["map"][gi]), str(L["tier"][gi])))
        acts_all.append(acts)
    wf, wo, lut = v.wcfg()
    pf, po = v.policy_flat(0)
    gf, go = v.gd_flat(0)
    eng = K.PersonalEngine(pf, gf, wf, po, go, wo, lut, max_concurrent=n, device_id=0)
    res = eng.leaf_eval(cfg, s, off, imit, pool, coefs)
    del eng
    # mirror (float64 for the personal part; the WP net in float32 as the kernel)
    wp0 = v.wp_drafts(rows)
    dp, dv = [], []
    for i in range(n):
        ot = int(cfg[i, 2])
        p_our = wp0[i] if ot == 0 else 1 - wp0[i]
        S, O, _ = assignment_terms(s[i].astype(np.float64), off[i].astype(np.float64), acts_all[i],
                                   float(coefs[2]), float(coefs[3]))
        b0s = coefs[0] if ot == 0 else -coefs[0]
        pc = min(max(p_our, 1e-6), 1 - 1e-6)
        V = 1 / (1 + np.exp(-(b0s + coefs[1] * np.log(pc / (1 - pc)) + coefs[2] * (S[ot] - S[1 - ot])
                              + coefs[3] * (O[ot] - O[1 - ot]))))
        dp.append(abs(res[i, 0] - p_our))
        dv.append(abs(res[i, 1] - V))
    out["leaf"] = {"drafts": n, "max_abs_diff_wp": float(np.max(dp)), "mean_abs_diff_wp": float(np.mean(dp)),
                   "max_abs_diff_value": float(np.max(dv)), "mean_abs_diff_value": float(np.mean(dv)),
                   "mean_abs_personal_shift": float(np.mean(np.abs(res[:, 1] - res[:, 0])))}
    print("leaf", json.dumps(out["leaf"]), flush=True)
    ok = out["leaf"]["max_abs_diff_wp"] < 1e-4 and out["leaf"]["max_abs_diff_value"] < 1e-4

    # 2-3: decide-only at step 2 (ban) and 7 (pick), aware vs blind
    team_pool = np.zeros((n, 2, 90), bool)
    for i in range(n):
        for sl in range(10):
            m = np.zeros(90, bool)
            for h in range(90):
                m[h] = bool(pool[i, sl, h // 32] >> np.uint32(h % 32) & np.uint32(1))
            team_pool[i, sl // 5] |= m
    acts_np = np.array(acts_all, np.int64)
    out["decide"] = {}
    for k in (2, 7, 13):
        base = {"acts": acts_np.copy(), "k": k, "map": cfg[:, 0], "tier": cfg[:, 1], "team_pool": team_pool,
                "pool_bits": pool, "slot_of": cfg[:, 24:40], "s": s, "off": off,
                "seed_i": np.arange(n), "kseed": 99}
        v.spec["combiner"] = list(map(float, coefs[:4]))
        r = {}
        for aware in (1, 0):
            t1 = time.time()
            r[aware] = SY.act_mcts(v, dict(base), aware)
            r[f"sec{aware}"] = time.time() - t1
        zero = dict(base)
        zero["s"] = np.zeros_like(s)
        zero["off"] = np.zeros_like(off)
        rz = SY.act_mcts(v, zero, 0)
        same = float((rz == r[0]).mean())
        out["decide"][str(k)] = {"agree_aware_blind": float((r[1] == r[0]).mean()),
                                 "blind_invariant_to_tensors": same,
                                 "sec_per_decision_aware": r["sec1"] / n, "sec_per_decision_blind": r["sec0"] / n,
                                 "sims": SY.SIMS}
        ok = ok and same == 1.0
        print(k, json.dumps(out["decide"][str(k)]), flush=True)
    os.makedirs(os.path.join(C.RESULTS, "p3t"), exist_ok=True)
    out["pass"] = bool(ok)
    json.dump(out, open(os.path.join(C.RESULTS, "p3t", f"verify_{a.vintage}.json"), "w"), indent=1)
    print("VERIFY", "OK" if ok else "FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
