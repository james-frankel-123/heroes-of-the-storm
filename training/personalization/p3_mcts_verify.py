"""
P3 personalized MCTS kernel verification.

1. Leaf values: complete drafts (real held-out drafts, re-labeled to the
   kernel order) with realistic personal tensors (rows of the personal
   tables) and random slot assignments; kernel leaf_eval vs the float64
   Python mirror (p3_mcts_core.reference_value). Reported: max and mean
   absolute difference of the population WP and of the personal value.
   Also the gap between the kernel's mean-logit ensemble and the
   mean-probability ensemble used by the one-step drafter.
2. Population identity: the unmodified reference kernel (cuda_personal/ref,
   = overfit2026/cuda_ofit) vs the personalized kernel with personal terms
   zeroed (b = 0, 1, 0, 0; no pools, no forced bans), same configs, seeds,
   sims: terminal drafts, win probabilities and root visit distributions
   must be identical. Also with personal valuation switched off.
3. Tree guard: max nodes used and capacity hits at the sim counts used.

Run (from training/):
  CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 nice -n 19 taskset -c 48-63 python3 personalization/p3_mcts_verify.py --search-mode chance
Output: results/p3_mcts_verify_<mode>.json; exits 1 if a gate fails (leaf parity
1e-5, population identity, tree reaching a later own pick block).
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from p3_heroes import NUM_HEROES
import p3_mcts_core as M
import p3_hs_core as C
import p3_dr_core as D


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--search-mode", required=True, choices=["chance", "rollfwd"])
    M.set_search_mode(ap.parse_args().search_mode)
    torch.set_num_threads(4)
    rng = np.random.RandomState(0)
    W = M.Weights()
    K = M.personal_kernel()
    R = M.ref_kernel()
    from shared import HEROES, MAPS, SKILL_TIERS
    d = C.load_slots()
    gd = D.GDPolicy(d["hero_names"])
    to_sh = gd.to_shared
    L = D.load_lobbies()
    T = D.load_personal()
    b = D.combiner()
    coefs = np.array([0.0, b[1], b[2], b[3], 0.8], np.float32)
    out = {"coefs": coefs.tolist()}
    wmap = np.load(C.WP)
    bidx_all = wmap["build_idx"][np.argsort(wmap["replay_ids"])]
    # ---------- 1. leaf values
    games = rng.choice(len(L["replay_id"]), 1000, replace=False)
    items = {}
    for gi in games:
        key = W.stats_key(bidx_all[L["g"][gi]])
        items.setdefault(key, []).append(gi)
    diffs_p, diffs_v, gap_ens = [], [], []
    wpE = D.WPEval(d["hero_names"])
    for key, gl in items.items():
        n = len(gl)
        cfg = M.empty_cfg(n)
        s, off, imit, pool = M.zeros_personal(n)
        refs = []
        for i, gi in enumerate(gl):
            st = L["steps"][gi]
            first = st[[k for k in range(16) if st[k][1] == 1][0]][2]
            acts, slots = [], []
            cnt = [0, 0]
            for k in range(16):
                h, ty, tm, row = st[k]
                kt = tm if first == 0 else 1 - tm
                assert kt == M.DRAFT_TEAM[k] and ty == M.IS_PICK[k], "draft order mismatch"
                acts.append(int(to_sh[h]))
                if ty == 1:
                    sl = cnt[kt] + 5 * kt
                    cnt[kt] += 1
                    slots.append(sl)
                    rrow = T["pos"][int(row)] if rng.rand() < 0.5 else rng.randint(len(T["rows"]))
                    s[i, sl, to_sh] = T["s"][rrow]
                    off[i, sl, to_sh] = T["off"][rrow]
                else:
                    slots.append(-1)
            cfg[i, 0] = MAPS.index(str(L["map"][gi]))
            cfg[i, 1] = SKILL_TIERS.index(str(L["tier"][gi]))
            cfg[i, 2] = rng.randint(2)
            cfg[i, 7] = 1
            cfg[i, 8:24] = acts
            cfg[i, 24:40] = slots
            t0 = [HEROES[acts[k]] for k in range(16) if M.IS_PICK[k] and M.DRAFT_TEAM[k] == 0]
            t1 = [HEROES[acts[k]] for k in range(16) if M.IS_PICK[k] and M.DRAFT_TEAM[k] == 1]
            Sx, Ox = [0.0, 0.0], [0.0, 0.0]
            for k in range(16):
                if M.IS_PICK[k]:
                    sl = slots[k]
                    tm = 0 if sl < 5 else 1
                    Sx[tm] += float(s[i, sl, acts[k]])
                    Ox[tm] += float(off[i, sl, acts[k]])
            refs.append(M.reference_value(W, key, str(L["map"][gi]), str(L["tier"][gi]), t0, t1, int(cfg[i, 2]),
                                          Sx, Ox, coefs.astype(np.float64)))
            # mean-probability ensemble (one-step drafter) for the same draft, team-0 side
            my = {HEROES[a]: j for j, a in enumerate(to_sh)}
            wp_mp = wpE.wp([(tuple(np.argsort(to_sh)[[HEROES.index(x) for x in t0]]),
                             tuple(np.argsort(to_sh)[[HEROES.index(x) for x in t1]]))],
                           bidx_all[L["g"][gi]], str(L["map"][gi]), str(L["tier"][gi]))[0]
            ref_t0 = refs[-1][0] if cfg[i, 2] == 0 else 1 - refs[-1][0]
            gap_ens.append(abs(wp_mp - ref_t0))
        eng = M.make_engine(K, W, key, 0, n, 0)
        res = eng.leaf_eval(cfg, s, off, imit, pool, coefs)
        del eng
        refs = np.array(refs)
        diffs_p += list(np.abs(res[:, 0] - refs[:, 0]))
        diffs_v += list(np.abs(res[:, 1] - refs[:, 1]))
    out["leaf"] = {"drafts": len(diffs_p), "max_abs_diff_wp": float(np.max(diffs_p)),
                   "mean_abs_diff_wp": float(np.mean(diffs_p)), "max_abs_diff_personal_value": float(np.max(diffs_v)),
                   "mean_abs_diff_personal_value": float(np.mean(diffs_v)),
                   "mean_logit_vs_mean_prob_ensemble_max_gap": float(np.max(gap_ens)),
                   "mean_logit_vs_mean_prob_ensemble_mean_gap": float(np.mean(gap_ens))}
    print("leaf", json.dumps(out["leaf"]), flush=True)

    # ---------- 2. population identity vs the reference kernel
    key = W.stats_key(bidx_all[L["g"][games[0]]])
    n = 64
    base = np.array([[rng.randint(14), rng.randint(3), rng.randint(2)] for _ in range(n)], np.int32)
    ident = {}
    for sims in (100, 400):
        er = M.make_engine(R, W, key, 1, n, 0)
        rr = er.run_episodes(np.ascontiguousarray(base), sims, 2.0, 777, 0.0, 0.3, 0.0, 1,
                    search_mode=M.search_mode())
        del er
        for label, personal in (("personal terms zeroed", 1), ("personal valuation off", 0)):
            cfg = M.empty_cfg(n)
            cfg[:, :3] = base
            cfg[:, 7] = personal
            cfg[:, 24:40] = -1
            s, off, imit, pool = M.zeros_personal(n)
            ek = M.make_engine(K, W, key, 1, n, 0)
            kk = ek.run(cfg, s, off, imit, pool, np.array([0, 1, 0, 0, 0.8], np.float32),
                        sims, 2.0, 777, 0.0, 0.3, 0.0, 1,
                    search_mode=M.search_mode())
            del ek
            same_wp = all(float(rr[i][0]) == float(kk[0][i]) for i in range(n))
            same_draft = True
            same_pol = True
            for i in range(n):
                tsv = np.array(rr[i][2])
                acts = kk[3][i]
                t0 = sorted(int(a) for k, a in enumerate(acts) if M.IS_PICK[k] and M.DRAFT_TEAM[k] == 0)
                t1 = sorted(int(a) for k, a in enumerate(acts) if M.IS_PICK[k] and M.DRAFT_TEAM[k] == 1)
                bn = sorted(int(a) for k, a in enumerate(acts) if not M.IS_PICK[k])
                if (t0 != sorted(np.flatnonzero(tsv[:NUM_HEROES] > 0.5).tolist()) or
                        t1 != sorted(np.flatnonzero(tsv[NUM_HEROES:2 * NUM_HEROES] > 0.5).tolist()) or
                        bn != sorted(np.flatnonzero(tsv[2 * NUM_HEROES:3 * NUM_HEROES] > 0.5).tolist())):
                    same_draft = False
                for t, ex in enumerate(rr[i][1]):
                    if not np.array_equal(np.array(ex[1]), kk[5][i, t]):
                        same_pol = False
            ident[f"{label} | sims {sims}"] = {"episodes": n, "identical_win_prob": same_wp,
                                                "identical_drafts": same_draft,
                                                "identical_root_visit_distributions": same_pol,
                                                "max_nodes_used": int(kk[9].max()), "capacity_hits": int(kk[10].sum())}
            print(label, sims, ident[f"{label} | sims {sims}"], flush=True)
    out["population_identity"] = ident
    # ---------- 3. guard at high sims
    cfg = M.empty_cfg(16)
    cfg[:, :3] = base[:16]
    s, off, imit, pool = M.zeros_personal(16)
    ek = M.make_engine(K, W, key, 1, 16, 0)
    g = {}
    for sims in (1000, 3000):
        kk = ek.run(cfg, s, off, imit, pool, np.array([0, 1, 0, 0, 0.8], np.float32), sims, 2.0, 5, 0.0, 0.3, 0.0, 1,
                    search_mode=M.search_mode())
        g[f"sims {sims}"] = {"max_nodes_used": int(kk[9].max()), "capacity_hits_total": int(kk[10].sum()),
                             "episodes_with_hits": int((kk[10] > 0).sum())}
    out["guard"] = g
    print("guard", g, flush=True)
    # ---------- 4. tree depth: the search must reach our later pick blocks
    depth = {}
    for sims in (100, 400):
        cfg = M.empty_cfg(16)
        cfg[:, :3] = base[:16]
        cfg[:, 2] = 0
        kk = ek.run(cfg, s, off, imit, pool, np.array([0, 1, 0, 0, 0.8], np.float32), sims, 2.0, 11,
                    0.0, 0.3, 0.0, 1, search_mode=M.search_mode())
        st = np.asarray(ek.last_stats())
        own_steps = [k for k in range(16) if M.DRAFT_TEAM[k] == 0]
        later = [k for k in own_steps if k > 4]
        mask = int(np.bitwise_or.reduce(st[:, 8]))
        depth[f"sims {sims}"] = {"max_depth_steps": int(st[:, 11].max()),
                                 "mean_own_decisions_ahead_per_sim": float(st[:, 7].sum() / max(st[:, 1].sum(), 1)),
                                 "later_own_steps_reached": [k for k in later if mask >> k & 1]}
    out["depth"] = depth
    out["search_mode"] = M.search_mode_name()
    out["comp_fallback_order"] = {"personal": K.COMP_FALLBACK_ORDER, "ref": R.COMP_FALLBACK_ORDER}
    print("depth", depth, flush=True)
    # ---------- gates: refuse downstream runs on a failed build
    fails = []
    if out["leaf"]["max_abs_diff_wp"] > 1e-5 or out["leaf"]["max_abs_diff_personal_value"] > 1e-5:
        fails.append("leaf parity above 1e-5")
    if not all(v["identical_win_prob"] and v["identical_drafts"] for v in ident.values()):
        fails.append("population identity vs the reference kernel")
    if not depth["sims 400"]["later_own_steps_reached"]:
        fails.append("tree never reaches a later own pick block")
    out["gates_failed"] = fails
    out["ok"] = not fails
    with open(os.path.join(C.RESULTS, f"p3_mcts_verify_{M.search_mode_name()}.json"), "w") as f:
        json.dump(out, f, indent=1)
    if fails:
        raise SystemExit("VERIFY FAILED: " + "; ".join(fails))
    print("VERIFY OK", flush=True)


if __name__ == "__main__":
    main()
