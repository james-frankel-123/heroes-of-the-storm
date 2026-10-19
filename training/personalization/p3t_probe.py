"""
P3 tournament probe: how much does a player-aware LEAF change decisions of
a search whose prior was trained player-blind?

At every real pick state of probe lobbies (tournament window, not tournament
lobbies), decide-only searches with the personal leaf on and off, same seeds.
Reported per configuration (prior x sims): share of decisions where the
aware pick differs from the blind pick, and, when they differ, the aware
search's own Q gain (aware choice minus blind choice under the aware Q;
model-internal), plus prior entropy at the root.

Stages: extract (3090: kernel tensors of the probe lobbies -> p3t/probe_lobbies.npz)
        run [--configs F:400,F:1600,BC:400,BC:1600] (any GPU host with the file)
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

PICKS = [4, 5, 6, 7, 8, 11, 12, 13, 14, 15]
PRIORS = {"F": None, "BC": "personalization/cache/gd_trainonly/bc_prior.pt"}


def extract(n):
    import p3t_backtest as BT
    L = TD.load_lobbies()
    z = np.load(os.path.join(TD.OUT, "state_cur.npz"))["n"]
    rich = (z.astype(np.float64).sum(1) >= 50).reshape(-1, 10).all(1)
    tl = os.path.join(TD.OUT, "tourney", "lobbies.npz")
    excl = set(np.load(tl)["gi"].tolist()) if os.path.exists(tl) else set()
    cand = np.array([g for g in np.flatnonzero(rich & (L["day"] <= TD.LOBBY_END)) if g not in excl])
    gi = np.sort(np.random.RandomState(77).choice(cand, n, replace=False))
    K = BT.kernel_lobbies(gi)
    np.savez(os.path.join(TD.OUT, "probe_lobbies.npz"), gi=gi, **K)
    print(f"probe lobbies: {len(gi)} (excluded {len(excl)} tournament lobbies)")


def run(configs, device=0):
    import torch
    import p3_mcts_core as M
    import p3t_systems as SY
    torch.set_num_threads(2)
    K = dict(np.load(os.path.join(TD.OUT, "probe_lobbies.npz")))
    n = len(K["gi"])
    base = SY.vintage("cur")
    comb = base.spec["combiner"]
    out = {"lobbies": n, "combiner": comb}
    kmod = SY.kernel_module()
    for cf in configs:
        pr, sims = cf.split(":")
        sims = int(sims)
        SY.SIMS = sims
        v = SY.Vintage("cur_probe", dict(base.spec))
        if PRIORS[pr]:
            v.spec["policies"] = [PRIORS[pr]]
        res = {"diff": [], "qgain": [], "ent": [], "sec": 0.0}
        for k in PICKS:
            ctx = {"acts": K["acts"], "k": k, "map": K["map"], "tier": K["tier"], "team_pool": K["team_pool"],
                   "pool_bits": K["pool_bits"], "slot_of": K["slot_of"], "s": K["s"], "off": K["off"],
                   "seed_i": np.arange(n), "kseed": 4242 + k}
            t0 = time.time()
            qa = _search(SY, M, kmod, v, ctx, 1, device)
            qb = _search(SY, M, kmod, v, ctx, 0, device)
            res["sec"] += time.time() - t0
            a_pick = qa["visits"].argmax(1)
            b_pick = qb["visits"].argmax(1)
            d = a_pick != b_pick
            res["diff"] += d.tolist()
            ii = np.flatnonzero(d)
            res["qgain"] += (qa["q"][ii, a_pick[ii]] - qa["q"][ii, b_pick[ii]]).tolist()
            p = qb["visits"] / np.maximum(qb["visits"].sum(1, keepdims=True), 1e-9)
            res["ent"] += (-(p * np.log(np.maximum(p, 1e-12))).sum(1)).tolist()
        q = np.array(res["qgain"])
        out[cf] = {"decisions": len(res["diff"]), "share_differ": float(np.mean(res["diff"])),
                   "own_q_gain_when_differ_pp": float(100 * np.nanmean(q)) if len(q) else None,
                   "root_visit_entropy_blind": float(np.mean(res["ent"])), "gpu_sec": res["sec"]}
        print(cf, json.dumps(out[cf]), flush=True)
    os.makedirs(os.path.join(C.RESULTS, "p3t"), exist_ok=True)
    p = os.path.join(C.RESULTS, "p3t", "probe.json")
    old = json.load(open(p)) if os.path.exists(p) else {}
    old.update(out)
    json.dump(old, open(p, "w"), indent=1)


def _search(SY, M, kmod, v, ctx, aware, device):
    """One decide-only search per draft; returns root visits and Q (n, 90)."""
    k = ctx["k"]
    B = len(ctx["acts"])
    cfg = np.full((B, M.CFG_LEN), -1, np.int32)
    cfg[:, 0], cfg[:, 1], cfg[:, 2] = ctx["map"], ctx["tier"], SY.DRAFT_TEAM[k]
    cfg[:, 3], cfg[:, 4], cfg[:, 5], cfg[:, 6], cfg[:, 7] = 0, k, 1, 0, aware
    cfg[:, 8:24] = np.where(np.arange(16)[None] < k, ctx["acts"], -1)
    cfg[:, 24:40] = ctx["slot_of"]
    cfg[:, 40:56] = -1
    cfg[:, 56:M.CFG_LEN] = 0
    cfg[:, M.CFG_ASSIGN] = 1
    coefs = np.array(list(v.spec["combiner"]) + [0.0], np.float32)
    wf, wo, lut = v.wcfg()
    vis = np.zeros((B, 90), np.float32)
    q = np.full((B, 90), np.nan, np.float32)
    for i in range(v.n_seeds):
        sel = np.flatnonzero(np.arange(B) % v.n_seeds == i)
        pf, po = v.policy_flat(i)
        gf, go = v.gd_flat(i)
        for b0 in range(0, len(sel), 256):
            ii = sel[b0:b0 + 256]
            eng = (getattr(kmod, "PersonalEngine", None) or kmod.PriorEngine)(pf, gf, wf, po, go, wo, lut, max_concurrent=len(ii), device_id=device)
            r = eng.run(np.ascontiguousarray(cfg[ii]), np.ascontiguousarray(ctx["s"][ii].astype(np.float32)),
                        np.ascontiguousarray(ctx["off"][ii].astype(np.float32)),
                        np.zeros((len(ii), 10, 90), np.float32), np.ascontiguousarray(ctx["pool_bits"][ii]),
                        coefs, SY.SIMS, 2.0, int(ctx["kseed"] + b0 + 1000 * i), 0.0, 0.3, 0.0, 1, search_mode=1)
            del eng
            vis[ii] = r[5][:, 0, :]
            q[ii] = r[6][:, 0, :]
    return {"visits": vis, "q": q}


def timing(systems, reps=2, device=0):
    """GPU wall seconds per decide-only search for each search system, alone on
    the GPU: all 10 pick states of the probe lobbies, 400 sims, repeated."""
    import torch
    import p3t_systems as SY
    import p3t_tourney as TT
    torch.set_num_threads(2)
    K = dict(np.load(os.path.join(TD.OUT, "probe_lobbies.npz")))
    n = len(K["gi"])
    out = {"lobbies": n, "sims": SY.SIMS}
    for name in systems:
        spec = SY.SYSTEMS[name]
        v = SY.vintage(spec["vint"])
        secs, nd = 0.0, 0
        for r in range(reps):
            for k in PICKS:
                ctx = {"acts": K["acts"], "k": k, "map": K["map"], "tier": K["tier"], "team_pool": K["team_pool"],
                       "pool_bits": K["pool_bits"], "slot_of": K["slot_of"], "s": K["s"], "off": K["off"],
                       "seed_i": np.arange(n), "kseed": 77 + k + 1000 * r,
                       "n": np.zeros((n, 10, 90), np.float32) if "n" not in K else K["n"]}
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                t0 = time.time()
                if spec["kind"] == "mcts":
                    SY.act_mcts(v, ctx, spec["aware"], device=device)
                elif spec["kind"] == "mcts_pp":
                    SY.act_mcts(v, ctx, spec["aware"], device=device, pp=True, prior=spec["prior"])
                elif spec["kind"] == "mcts_vn":
                    SY.act_mcts(v, ctx, spec["aware"], device=device, vn=True)
                secs += time.time() - t0
                nd += n
        out[name] = {"ms_per_decision": round(1000 * secs / nd, 3), "ms_per_draft_8_decisions": round(8000 * secs / nd, 2)}
        print(name, out[name], flush=True)
    p = os.path.join(C.RESULTS, "p3t", "timing.json")
    json.dump(out, open(p, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["extract", "run", "timing"])
    ap.add_argument("--systems", default="pMCTS-cur,aMCTS-cur,pMCTS-bc,aMCTS-bc,aMCTS-pp,aMCTS-sp,aMCTS-vn")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--configs", default="F:400,F:1600,BC:400,BC:1600")
    a = ap.parse_args()
    if a.stage == "extract":
        extract(a.n)
    elif a.stage == "timing":
        timing(a.systems.split(","))
    else:
        run(a.configs.split(","))


if __name__ == "__main__":
    main()
