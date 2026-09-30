"""
Stage E: optimization-pressure sweeps against side-A proxies, scored on
side-B gold (within-snapshot split, no meta drift) and on post-snapshot gold.

A run = (proxy set, prior, sims, root temperature, lcb lambda). Pressure is
varied at inference time: sims = 0 means no search (the prior policy itself
drafts, sampled or argmax); sims > 0 runs the CUDA MCTS kernel with the proxy
at the leaves (GD rollouts to the terminal draft, then the proxy). Every run
uses the same 400 draft configurations (map, mid tier, side) and the same GD
opponent pool, so conditions are paired.

Usage:
  python3 overfit2026/stage_e.py run --grid main [--gpu 2]
  python3 overfit2026/stage_e.py summarize
"""
import os
import sys
import json
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)

import numpy as np
import torch

OUT_DIR = os.path.join(HERE, "results", "split_search")
N_DRAFTS = 400
SEED = 20260929


def run_name(proxies, prior, sims, temp, lam, stats=None):
    p = "+".join(proxies)
    s = f"{p}__{prior}__sims{sims}__T{temp}"
    if lam:
        s += f"__lcb{lam}"
    if stats:
        s += f"__st{stats}"
    return s


def grid(name):
    G = []
    SIMS = [0, 16, 64, 256, 1024, 4096]
    if name == "main":
        for px in ("pA8_leak", "pA8_oof"):
            for prior in ("bc", "uniform"):
                for t in (1.0, 0.0):
                    for s in SIMS:
                        if prior == "uniform" and s == 0 and t == 0.0:
                            continue
                        G.append(((px,), prior, s, t, 0.0))
    elif name == "deep":
        for px in ("pA8_leak", "pA8_oof", "pA1_leak", "pA1_oof"):
            G.append(((px,), "bc", 16384, 0.0, 0.0))
    elif name == "size":
        for e in (1, 2, 4):
            for mode in ("leak", "oof"):
                for s in (0, 64, 256, 1024, 4096):
                    G.append(((f"pA{e}_{mode}",), "bc", s, 0.0, 0.0))
    elif name == "capacity":
        # trimmed 2026-09-29 (GPU contention): width variants ran at 64/1024/4096;
        # the rest at 1024 only
        for px in ("pA8_leak_w64", "pA8_leak_w1024", "pA8_leak_deep"):
            for s in (64, 1024, 4096):
                G.append(((px,), "bc", s, 0.0, 0.0))
        for px in ("pA8_leak_noreg", "pA8_leak_ep1", "pA8_leak_ep10", "pA8_leak_noreg_ep40",
                   "pA8_leak_s1", "pA8_oof_s1", "pA8_naive"):
            for s in (1024,):
                G.append(((px,), "bc", s, 0.0, 0.0))
    elif name == "lcb":
        for ens in (("pA8_leak", "pA8_leak_s1", "pA8_leak_s2", "pA8_leak_s3"),
                    ("pA8_oof", "pA8_oof_s1", "pA8_oof_s2", "pA8_oof_s3")):
            for lam in (0.0, 1.0, 2.0, 4.0, 8.0):
                for s in (1024,):
                    G.append((ens, "bc", s, 0.0, lam))
    return G


def policy_only(prior, n, seed, argmax):
    """No search: the prior drafts for us, a GD model for the opponent."""
    from train_draft_policy import DraftState, DRAFT_ORDER
    from train_generic_draft import GenericDraftModel
    from shared import HEROES, MAPS, SKILL_TIERS
    from overfit2026 import search
    net = search.policy_net(prior)
    gds = []
    for i in range(5):
        g = GenericDraftModel()
        g.load_state_dict(torch.load(os.path.join(TRAINING_DIR, "rerun2026", "models",
                                                  f"generic_draft_{i}.pt"),
                                     weights_only=True, map_location="cpu"))
        g.eval()
        gds.append(g)
    cfgs = search.draft_configs(n, seed)
    rng = np.random.RandomState(seed + 1)
    out = []
    for i, (mi, ti, ou) in enumerate(cfgs):
        gd = gds[(i // 100) % len(gds)]
        st = DraftState(MAPS[mi], SKILL_TIERS[ti], our_team=int(ou))
        while not st.is_terminal():
            team, kind = DRAFT_ORDER[st.step]
            mask = torch.tensor(st.valid_mask_np()).unsqueeze(0)
            with torch.no_grad():
                if team == ou:
                    x = torch.tensor(st.to_numpy()).unsqueeze(0)
                    logits, _ = net(x, mask)
                    t = 0.0 if argmax else 1.0
                else:
                    x = torch.tensor(st.to_numpy()[:-1]).unsqueeze(0)
                    logits = gd(x, mask)
                    t = 1.0
            p = torch.softmax(logits[0], -1).numpy().astype(np.float64)
            p = p / p.sum()
            a = int(np.argmax(p)) if t == 0.0 else int(rng.choice(len(p), p=p))
            st.apply_action(a, team, kind)
        t0 = [HEROES[j] for j in range(90) if st.team0_picks[j] > 0.5]
        t1 = [HEROES[j] for j in range(90) if st.team1_picks[j] > 0.5]
        out.append({"our": t0 if ou == 0 else t1, "opp": t1 if ou == 0 else t0,
                    "map": MAPS[mi], "tier": SKILL_TIERS[ti], "side": int(ou)})
    return out


def do_run(spec, kernel, device_id, force=False):
    from overfit2026 import search, score, split
    proxies, prior, sims, temp, lam = spec
    name = run_name(proxies, prior, sims, temp, lam)
    path = os.path.join(OUT_DIR, name + ".json")
    if os.path.exists(path) and not force:
        return
    t0 = time.time()
    if sims == 0:
        drafts = policy_only(prior, N_DRAFTS, SEED, argmax=(temp == 0.0))
    else:
        models = [score.model(p)[0] for p in proxies]
        stats = score._stats(score.deploy_stats_name(proxies[0]))
        wcfg = search.wp_config(models, stats, lcb_lambda=lam)
        pf = search.policy_flat(prior)
        drafts = search.run(kernel, pf, wcfg, n=N_DRAFTS, sims=sims, seed=SEED,
                            root_temp=temp, guard=1, device_id=0)
    sc = score.score(drafts, proxies=proxies)
    if len(proxies) > 1:
        sc["proxy_ens_mean"] = np.mean([sc[f"proxy:{p}"] for p in proxies], 0)
    res = {"spec": {"proxies": list(proxies), "prior": prior, "sims": sims, "temp": temp,
                    "lcb": lam},
           "n": len(drafts), "summary": score.summarize(sc),
           "diversity": score.diversity(drafts),
           "cap_hits": float(np.mean([d.get("cap_hits", 0) for d in drafts])),
           "drafts": drafts,
           "per_draft": {k: np.round(np.asarray(v, float), 5).tolist() for k, v in sc.items()},
           "secs": time.time() - t0}
    with open(path, "w") as f:
        json.dump(res, f)
    s = res["summary"]
    px = s[f"proxy:{proxies[0]}"][0]
    print(f"{name:70s} proxy={px:.4f} gB={s['gB'][0]:.4f} RB={s['RB'][0]:.4f} "
          f"RN={s['RN'][0]:.4f} QM={s['QM2026'][0]:.4f} deg={s['degen'][0]:.3f} "
          f"H={res['diversity']['entropy_bits']:.2f} ({res['secs']:.0f}s)", flush=True)


def cmd_run(a):
    from overfit2026 import search
    os.makedirs(OUT_DIR, exist_ok=True)
    kernel = search.load_kernel("ofit")
    for g in a.grid.split(","):
        for spec in grid(g):
            do_run(spec, kernel, 0, a.force)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--grid", default="main")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    {"run": cmd_run}[a.cmd](a)
