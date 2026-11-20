"""
Compute-matched one-step control for paper 2's final drafter (s-pcal-floor25).

Question: the final drafter (MCTS, player-aware value and proposals) beats player-aware greedy (a1step-cur,
p3_dr_drafter's one-step search: top 15 GD + top 15 personal candidates, 4 GD rollouts each) by about
2 points on the judge-free paired record. Is that lookahead, or more and better candidate evaluations?

The control (g1-pcal-floor25): the final drafter itself with its search held to the root. Same kernel code
(cuda_support1 = cuda_support plus run(slot_cap)), same root priors (player-aware prior, assignment and
denial-ban biases), same support floor and structural mask at the root, same c_puct, same leaf value
(population WP + scaled personal terms), same 400 simulations per decision and the same root choice rule
(most visits among allowed heroes). With slot_cap = 1 the v2 search has only the root's slot: each
simulation picks a root action by PUCT, finds no slot for the next node (opponent chance node or our next
decision) and rolls the draft out to the end with the search's rollout policy (GD for both teams under the
pool masks, the floor on our side, plain GD for the opponent) before the leaf value. So every root
candidate is scored by rollouts, with as many value evaluations per decision as the search; what is
removed is the tree below the root (our later decisions searched, the DraftRec opponent at chance nodes).

Stages (from training/, the paper-2 env of p3t_promote_pcalf_chain.sh):
  list                       the pairs (MATCHED vs every MAIN system and the final drafter, both seats)
  verify                     kernel equivalence: s-pcal-floor25 vs GD redrafted through support1_kernel
                             with slot_cap 0 into <cache>/p3t/tourney_verify_m1, compared with the file
  smoke [--lobbies n] [--sims s]
                             a tiny run of the control (first n lobbies, s sims), no file written
  run --shard i/n            play the pairs (p3t_tourney.play; new pair files only)
  score                      score only this module's pair files (p3t_judge.score_drafts; new score files)
  summary                    p3t_support.summary for the control -> results/p3t_support/matched_greedy.{json,md}
                             plus matched_greedy_contrasts.json (the three contrasts the paper asks for)
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
MATCHED = {"g1-pcal-floor25": FINAL}      # control name -> the support arm it holds to the root
CTRL = "g1-pcal-floor25"
GREEDY = "a1step-cur"


def matched_systems():
    import p3t_support as SP
    arms = SP.arms()
    out = {}
    for n, base in MATCHED.items():
        if base in arms:
            c = dict(arms[base], depth1=True)
            out[n] = dict(kind="mcts_x", vint=c.get("base", "cur"), aware=1, x=c)
    return out


def opponents():
    import p3t_tourney as TT
    return [o for o in TT.MAIN if o != CTRL] + [FINAL]


def pairs():
    out = []
    for o in opponents():
        out += [(CTRL, o), (o, CTRL)]
    return out


def verify(device=0):
    """Kernel equivalence: the final drafter through support1_kernel with slot_cap 0 must redraft its own file."""
    import p3t_support as SP
    import p3t_systems as SY
    import p3t_tourney as TT
    SP.check_vintages()
    pair = f"{FINAL}__GD"
    SY.SYSTEMS[FINAL] = dict(SY.SYSTEMS[FINAL], x=dict(SY.SYSTEMS[FINAL]["x"], depth1=True, slot_cap=0))
    vdir = TT.TDIR + "_verify_m1"
    os.makedirs(vdir, exist_ok=True)
    ref_dir = TT.TDIR
    TT.pair_file = lambda x, y: os.path.join(vdir, f"{x}__{y}.npz")
    import torch
    torch.set_num_threads(2)
    LB = TT.Lobbies()
    TT.play(LB, FINAL, "GD", device=device)
    new = np.load(os.path.join(vdir, f"{pair}.npz"))
    ref = np.load(os.path.join(ref_dir, f"{pair}.npz"))
    assert (ref["gi"] == new["gi"]).all()
    same = (new["acts"] == ref["acts"]).all(1)
    res = {"pair": pair, "kernel": "support1_kernel slot_cap 0", "lobbies": int(len(same)),
           "identical_drafts": int(same.sum()), "first_diff_lobbies": np.flatnonzero(~same)[:10].tolist(),
           "ref_prov": json.loads(str(ref["prov"])), "new_prov": json.loads(str(new["prov"]))}
    print(json.dumps(res, indent=1), flush=True)
    assert same.sum() >= 0.97 * len(same), "support1_kernel (slot_cap 0) does not reproduce the final drafter"
    print("VERIFY OK", flush=True)
    return res


def smoke(n=4, sims=16, device=0):
    """Tiny run of the control on the first n lobbies (no file written): legal, complete, distinct drafts."""
    import p3t_systems as SY
    import p3t_tourney as TT
    SY.SYSTEMS[CTRL]["x"] = dict(SY.SYSTEMS[CTRL]["x"], sims=sims)
    import torch
    torch.set_num_threads(2)
    LB = TT.Lobbies()
    idx = np.arange(n)
    acts = np.full((n, 16), -1, np.int64)
    t0 = time.time()
    for k in range(16):
        s = CTRL if SY.DRAFT_TEAM[k] == 0 else "GD"
        ctx = TT.ctx_for(LB, idx, acts, k, s, kseed=1000 * k)
        acts[:, k] = TT.act(s, ctx, LB, idx, device)
    assert (acts >= 0).all() and all(len(set(a)) == 16 for a in acts)
    print("smoke ok", acts.tolist(), f"{time.time() - t0:.1f}s", flush=True)


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
    print("MATCHED RUN DONE", flush=True)


def score():
    """p3t_judge.score_all restricted to this module's pair files; never rewrites an existing score file."""
    import p3t_judge as JG
    import p3t_tourney as TT
    LB = TT.Lobbies()
    judge = json.load(open(os.path.join(JG.RES, JG.JNAME)))
    truth = JG.Truth()
    key = JG.score_key(LB, os.path.join(TT.TDIR, "lobbies.npz"))
    for a, b in pairs():
        name = f"{a}__{b}"
        src = os.path.join(TT.TDIR, f"{name}.npz")
        p = os.path.join(JG.SDIR, f"{name}.npz")
        if not os.path.exists(src):
            print("missing", name, flush=True)
            continue
        if os.path.exists(p):
            continue
        z = np.load(src)
        t0 = time.time()
        sc = JG.score_drafts(LB, z["acts"], judge, truth)
        for side, s_ in (("a", a), ("b", b)):
            wp, V = JG.own_values(LB, z["acts"], s_)
            sc[f"own_wp_{side}"], sc[f"own_V_{side}"] = wp, V
        sc["judge_c2"] = np.array(judge["c2_skill"])
        sc["score_key"] = np.array(key)
        np.savez(p + ".tmp.npz", **sc)
        os.replace(p + ".tmp.npz", p)
        print(f"scored {name} ({time.time() - t0:.0f}s)", flush=True)
    print("MATCHED SCORE DONE", flush=True)


def summary(out_dir):
    import p3t_support as SP
    opp = opponents()
    res = SP.summary([CTRL], out_dir, opponents=opp, stem="matched_greedy", ref=CTRL)
    row = json.load(open(os.path.join(out_dir, "matched_greedy.json")))["arms"][CTRL]["vs"]
    neg = lambda v: [-v[0], -v[2], -v[1]]
    out = {"what": "judge-free paired record (pp; 95% bootstrap over lobbies) and J (win %, both seats), headline "
                   "lobbies; matched greedy = g1-pcal-floor25 (the final drafter's search held to the root, same "
                   "priors, leaf value and 400 simulations per decision)",
           "final_vs_matched": {"record_pp": neg(row[FINAL]["record_pp"]), "J": [100 - row[FINAL]["J"][0], 100 - row[FINAL]["J"][2],
                                                                                  100 - row[FINAL]["J"][1]],
                                "coverage_both": row[FINAL]["coverage"]["cov_both"]} if row.get(FINAL) else None,
           "matched_vs_a1step": {k: row[GREEDY][k] for k in ("record_pp", "J", "CONS")} if row.get(GREEDY) else None,
           "matched_vs_real": {k: row["Real"][k] for k in ("record_pp", "J", "CONS")} if row.get("Real") else None,
           "per_opponent": {o: (None if v is None else {k: v[k] for k in ("record_pp", "J", "CONS")}) for o, v in row.items()}}
    json.dump(out, open(os.path.join(out_dir, "matched_greedy_contrasts.json"), "w"), indent=1)
    print(json.dumps(out, indent=1), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["list", "verify", "smoke", "run", "score", "summary"])
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--lobbies", type=int, default=4)
    ap.add_argument("--sims", type=int, default=16)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.stage == "list":
        for p in pairs():
            print("__".join(p))
    elif a.stage == "verify":
        verify(a.device)
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
