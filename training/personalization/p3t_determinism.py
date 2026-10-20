"""
P3 kernels after the race fix: identical reruns must be byte-identical, and
no draft may repeat a hero.

For each P3 kernel module (personal, prior, pvnet, explore, explore with cfg[6] = 3): decide-only
searches at steps 2, 7 and 13 (prefix replay, the tournament's mode) on the
probe lobbies, run twice with the same seeds; root visits, Q and win
probabilities compared exactly. Then full self-play drafts with forced bans
(two real bans forced per draft): every draft must have 16 distinct heroes.
Run (GPU host): python personalization/p3t_determinism.py
Output: $P3_RESULTS/p3t/determinism.json; exit 1 on any difference.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("P1_TIERS", "site")
import numpy as np

import p3_hs_core as C
import p3t_data as TD


def main():
    import torch
    import p3_mcts_core as M
    import p3t_systems as SY
    torch.set_num_threads(2)
    K = dict(np.load(os.path.join(TD.OUT, "probe_lobbies.npz")))
    n = min(512, len(K["gi"]))
    v = SY.vintage("cur")
    wf, wo, lut = v.wcfg()
    pf, po = v.policy_flat(0)
    gf, go = v.gd_flat(0)
    mods = {"personal": ("personal_kernel", "cuda_personal", 0), "prior": ("prior_kernel", "cuda_prior", 1),
            "pvnet": ("pvnet_kernel", "cuda_pvnet", 3), "explore": ("explore_kernel", "cuda_explore", 3),
            "explore_chance": ("explore_kernel", "cuda_explore", 3)}
    out, ok = {}, True
    rng = np.random.RandomState(0)
    bias = rng.normal(0, 0.5, (n, 10, 90)).astype(np.float32)
    feat = np.log1p(rng.poisson(5, (n, 10, 90))).astype(np.float32)
    vnet = (rng.normal(0, 0.1, 225)).astype(np.float32)
    for name, (mod, d, nextra) in mods.items():
        km = M.load_module(mod, os.path.join(HERE, d))
        Eng = getattr(km, "PersonalEngine", None) or getattr(km, "PriorEngine")
        res = {}
        for k in (2, 7, 13):
            cfg = np.full((n, M.CFG_LEN), -1, np.int32)
            cfg[:, 0], cfg[:, 1], cfg[:, 2] = K["map"][:n], K["tier"][:n], SY.DRAFT_TEAM[k]
            cfg[:, 3], cfg[:, 4], cfg[:, 5], cfg[:, 6], cfg[:, 7] = 0, k, 1, 0, 1
            cfg[:, 8:24] = np.where(np.arange(16)[None] < k, K["acts"][:n], -1)
            cfg[:, 24:40] = K["slot_of"][:n]
            cfg[:, 40:56] = -1
            cfg[:, 56:M.CFG_LEN] = 0
            cfg[:, M.CFG_ASSIGN] = 1
            if nextra >= 1:
                cfg[:, 56] = 1
            if name.startswith("explore"):
                cfg[:, 6] = 3 if name == "explore_chance" else 2
            coefs = np.array([0, 1.2, 4.5, -0.02, 0.8, 0.3], np.float32)
            args = [cfg, K["s"][:n].astype(np.float32), K["off"][:n].astype(np.float32),
                    np.zeros((n, 10, 90), np.float32) if not name.startswith("explore") else bias * 0.5,
                    K["pool_bits"][:n], coefs if nextra else coefs[:5]]
            if nextra >= 1:
                args.append(bias)
            if nextra >= 3:
                args += [feat, vnet]
            outs = []
            for rep in range(2):
                eng = Eng(pf, gf, wf, po, go, wo, lut, max_concurrent=n, device_id=0)
                r = eng.run(*[np.ascontiguousarray(a) for a in args], 400, 2.0, 12345 + k, 0.0, 0.3, 0.0, 1,
                            search_mode=1)
                del eng
                outs.append((r[5][:, 0, :].copy(), np.nan_to_num(r[6][:, 0, :]).copy(), r[0].copy()))
            same = all(np.array_equal(a, b) for a, b in zip(outs[0], outs[1]))
            ndiff = int((outs[0][0] != outs[1][0]).any(1).sum())
            res[f"decide_k{k}_identical"] = same
            res[f"decide_k{k}_episodes_differing"] = ndiff
            ok &= same
        # full self-play drafts with two forced bans
        cfg = np.full((n, M.CFG_LEN), -1, np.int32)
        cfg[:, 0], cfg[:, 1], cfg[:, 2] = K["map"][:n], K["tier"][:n], 0
        cfg[:, 3], cfg[:, 4], cfg[:, 5], cfg[:, 6], cfg[:, 7] = 1, -1, 1, 0, 1
        cfg[:, 24:40] = K["slot_of"][:n]
        cfg[:, 40:56] = -1
        cfg[:, 40] = K["acts"][:n, 0]
        cfg[:, 43] = K["acts"][:n, 3]
        cfg[:, 56:M.CFG_LEN] = 0
        cfg[:, M.CFG_ASSIGN] = 1
        coefs = np.array([0, 1.2, 4.5, -0.02, 0.8, 0.3], np.float32)
        args = [cfg, K["s"][:n].astype(np.float32), K["off"][:n].astype(np.float32), np.zeros((n, 10, 90), np.float32),
                K["pool_bits"][:n], coefs if nextra else coefs[:5]]
        if nextra >= 1:
            args.append(np.zeros((n, 10, 90), np.float32))
        if nextra >= 3:
            args += [feat, vnet]
        acts = []
        for rep in range(2):
            eng = Eng(pf, gf, wf, po, go, wo, lut, max_concurrent=n, device_id=0)
            r = eng.run(*[np.ascontiguousarray(a) for a in args], 100, 2.0, 999, 1.0, 0.3, 0.0, 1, search_mode=1)
            del eng
            acts.append(np.array(r[3]))
        dup = int(sum(len(set(a.tolist())) != 16 for a in acts[0]))
        res["full_forced_bans_drafts_with_repeat"] = dup
        res["full_identical"] = bool(np.array_equal(acts[0], acts[1]))
        ok &= dup == 0 and res["full_identical"]
        out[name] = res
        print(name, json.dumps(res), flush=True)
    # cfg[6] = 3 (chance nodes only, 2026-10-20): identity rows give the cfg[6] = 2 search exactly,
    # and real rows change the search (the gate is live)
    km = M.load_module("explore_kernel", os.path.join(HERE, "cuda_explore"))
    k = 7
    res = {}
    for label, rows_, w0 in (("identity_rows", np.zeros((n, 10, 90), np.float32), 1.0), ("bias_rows", bias * 0.5, 0.8)):
        r6 = {}
        for c6 in (2, 3):
            cfg = np.full((n, M.CFG_LEN), -1, np.int32)
            cfg[:, 0], cfg[:, 1], cfg[:, 2] = K["map"][:n], K["tier"][:n], SY.DRAFT_TEAM[k]
            cfg[:, 3], cfg[:, 4], cfg[:, 5], cfg[:, 6], cfg[:, 7] = 0, k, 1, c6, 1
            cfg[:, 8:24] = np.where(np.arange(16)[None] < k, K["acts"][:n], -1)
            cfg[:, 24:40] = K["slot_of"][:n]
            cfg[:, 40:56] = -1
            cfg[:, 56:M.CFG_LEN] = 0
            cfg[:, M.CFG_ASSIGN] = 1
            cfg[:, 56] = 1
            coefs = np.array([0, 1.2, 4.5, -0.02, w0, 0.3], np.float32)
            args = [cfg, K["s"][:n].astype(np.float32), K["off"][:n].astype(np.float32), rows_, K["pool_bits"][:n],
                    coefs, bias, feat, vnet]
            eng = (getattr(km, "PersonalEngine", None) or km.PriorEngine)(pf, gf, wf, po, go, wo, lut, max_concurrent=n,
                                                                       device_id=0)
            r = eng.run(*[np.ascontiguousarray(a) for a in args], 400, 2.0, 4242, 0.0, 0.3, 0.0, 1, search_mode=1)
            del eng
            r6[c6] = r[5][:, 0, :].copy()
        res[f"{label}_cfg6_3_equals_2"] = bool(np.array_equal(r6[2], r6[3]))
    res["pass"] = res["identity_rows_cfg6_3_equals_2"] and not res["bias_rows_cfg6_3_equals_2"]
    ok &= res["pass"]
    out["explore_cfg6_3_gate"] = res
    print("explore cfg6=3 gate", json.dumps(res), flush=True)
    out["pass"] = bool(ok)
    os.makedirs(os.path.join(C.RESULTS, "p3t"), exist_ok=True)
    json.dump(out, open(os.path.join(C.RESULTS, "p3t", "determinism.json"), "w"), indent=1)
    print("DETERMINISM", "OK" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
