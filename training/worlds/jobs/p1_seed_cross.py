"""Map/tier-composition check for seed splits: a strategy's value on the draft subset draft % m == k, for m = 3 and 5.
A pooled policy's real seed split is its own modulus; the other modulus (and the same modulus for a different policy)
gives a split with the same map/tier composition but mixed checkpoints. Family a, default dials, worlds 0..NW-1."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.expanduser("~/hots/repo/training/worlds"))
sys.path.insert(0, os.path.expanduser("~/hots/repo/training"))
import rescore as R, world as W, model as M
NW = int(os.environ.get("NW", "40"))
D = R.p1_drafts(); idx = R.p1_index(D); dn = R.p1_draft_numbers(D)
Ws = W.Worlds()
P = np.mean([M.sig(Ws.logit("a", D, w)) for w in range(NW)], 0)
out = {}
for s in ["constrained_mcts", "constrained_mcts_sf", "constrained_mcts_ft", "constrained_mcts_ft_sf", "search_ft_sf", "search_sf", "gd", "k_oof", "enriched"]:
    if s not in idx:
        continue
    L = idx[s]
    val = lambda sub: float(np.mean([np.concatenate([P[f], 1 - P[g]]).mean() for _, f, g in sub]))
    out[s] = {}
    for m in (3, 5):
        v = [100 * val([(o, f[dn[f] % m == k], g[dn[g] % m == k]) for o, f, g in L]) for k in range(m)]
        out[s][f"mod{m}"] = {"values": v, "sd": float(np.std(v, ddof=1))}
    print(s, json.dumps(out[s]), flush=True)
json.dump(out, open(os.path.expanduser("~/hots/repo/training/worlds/site/results/rescore_p1_seed_cross.json"), "w"), indent=1)
