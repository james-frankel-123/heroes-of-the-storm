"""
Splitting the root-held control's gain over player-aware greedy (paper 2, follow-up to p3t_matched.py).

g1-pcal-floor25 (the final drafter held to the root) beats player-aware greedy (a1step-cur) by 3.3 points on the
paired record. It differs from greedy in what it proposes (learned player-aware prior with the ten-player
assignment and denial-ban terms, against greedy's top 15 by the imitation drafter plus top 15 by personal term,
and the imitation drafter's bans) and in budget (400 PUCT-allocated evaluations against 4 rollouts x 30). Two arms,
both the root-held control with one part swapped for greedy's:
  g1s120-pcal-floor25   greedy's budget: 120 simulations per decision instead of 400 (proposals unchanged)
  g1gc-pcal-floor25     greedy's proposals: the root sees only greedy's candidates under a flat prior, and the
                        bans are the imitation drafter's (p3t_systems.act_mcts_x greedy_cand / gd_bans); 400 sims
Everything else (kernel, floor, mask, leaf value, rollout policy, choice rule) is the root-held control's.

Stages (from training/, the env of remote/p3t_split_chain.sh): list | smoke | run --shard i/n | score | summary
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

FINAL = "s-pcal-floor25"
CTRL = "g1-pcal-floor25"
GREEDY = "a1step-cur"
SPLIT = {"g1s120-pcal-floor25": dict(sims=120),
         "g1gc-pcal-floor25": dict(greedy_cand=True, gd_bans=True)}
OPPS = [GREEDY, CTRL, FINAL, "Real", "pMCTS-cur"]


def split_systems():
    import p3t_support as SP
    arms = SP.arms()
    out = {}
    if FINAL in arms:
        for n, extra in SPLIT.items():
            c = dict(arms[FINAL], depth1=True, **extra)
            out[n] = dict(kind="mcts_x", vint=c.get("base", "cur"), aware=1, x=c)
    return out


def pairs():
    out = []
    for a in SPLIT:
        for o in OPPS:
            out += [(a, o), (o, a)]
    return out


def smoke(n=4, sims=16, device=0):
    import p3t_systems as SY
    import p3t_tourney as TT
    import torch
    torch.set_num_threads(2)
    LB = TT.Lobbies()
    idx = np.arange(n)
    for a in SPLIT:
        SY.SYSTEMS[a]["x"] = dict(SY.SYSTEMS[a]["x"], sims=sims)
        acts = np.full((n, 16), -1, np.int64)
        t0 = time.time()
        for k in range(16):
            s = a if SY.DRAFT_TEAM[k] == 0 else "GD"
            ctx = TT.ctx_for(LB, idx, acts, k, s, kseed=1000 * k)
            acts[:, k] = TT.act(s, ctx, LB, idx, device)
        assert (acts >= 0).all() and all(len(set(x)) == 16 for x in acts)
        print("smoke ok", a, acts.tolist(), f"{time.time() - t0:.1f}s", flush=True)


def run(shard, device=0):
    import hashlib
    import p3t_support as SP
    import p3t_tourney as TT
    SP.check_vintages()
    i, n = (int(x) for x in shard.split("/"))
    todo = [p for p in pairs() if int(hashlib.md5(f"{p[0]}__{p[1]}".encode()).hexdigest()[:8], 16) % n == i]
    TT.SHARD = shard
    import torch
    torch.set_num_threads(2)
    LB = TT.Lobbies()
    print(f"{len(todo)} pairs, {LB.n} lobbies", flush=True)
    for a, b in todo:
        TT.play(LB, a, b, device=device)
    for o in TT._ONESTEP.values():
        o.close()
    print("SPLIT RUN DONE", flush=True)


def score():
    import p3t_matched as PM
    PM.pairs = pairs
    PM.score()


def summary(out_dir):
    import p3t_support as SP
    names = list(SPLIT)
    SP.summary(names, out_dir, opponents=OPPS + names, stem="split_greedy", ref=names[0])
    js = json.load(open(os.path.join(out_dir, "split_greedy.json")))["arms"]
    out = {"what": "paired record (pp; 95% bootstrap over lobbies), J and CONS (win %, both seats), headline lobbies; "
                   "root-held control with greedy's budget (g1s120) or greedy's proposals and bans (g1gc)",
           "arms": {a: {o: (None if v is None else {k: v[k] for k in ("record_pp", "J", "CONS")})
                        for o, v in js[a]["vs"].items()} for a in names}}
    json.dump(out, open(os.path.join(out_dir, "split_greedy_contrasts.json"), "w"), indent=1)
    print(json.dumps(out, indent=1), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["list", "smoke", "run", "score", "summary"])
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--lobbies", type=int, default=4)
    ap.add_argument("--sims", type=int, default=16)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.stage == "list":
        for p in pairs():
            print("__".join(p))
    elif a.stage == "smoke":
        smoke(a.lobbies, a.sims, a.device)
    elif a.stage == "run":
        run(a.shard, a.device)
    elif a.stage == "score":
        score()
    else:
        import p3_hs_core as C
        summary(a.out or os.path.join(C.RESULTS, "p3t_support"))


if __name__ == "__main__":
    main()
