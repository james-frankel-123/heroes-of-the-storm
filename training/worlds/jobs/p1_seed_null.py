"""Seed spread vs a draft-resampling null (family a, default dials, log grade): per strategy, value of each seed's
draft subset (draft % n == k) against random partitions of the same drafts into n groups of matching sizes.
Values are the mean over worlds 0..NW-1 of the strategy value (rescore.strat_values)."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.expanduser("~/hots/repo/training/worlds"))
sys.path.insert(0, os.path.expanduser("~/hots/repo/training"))
import rescore as R, world as W, model as M
NW = int(os.environ.get("NW", "40"))
D = R.p1_drafts(); idx = R.p1_index(D); dn = R.p1_draft_numbers(D)
Ws = W.Worlds()
P = np.mean([M.sig(Ws.logit("a", D, w)) for w in range(NW)], 0)   # world-mean true P(team0 wins) per draft
rng = np.random.default_rng(5)
out = {}
for s, n in [("constrained_mcts", 5), ("constrained_mcts_sf", 5), ("constrained_mcts_ft", 3), ("constrained_mcts_ft_sf", 3), ("search_ft_sf", 3), ("search_sf", 5)]:
    L = idx[s]
    val = lambda sub: float(np.mean([np.concatenate([P[f], 1 - P[g]]).mean() for _, f, g in sub]))
    obs = [100 * val([(o, f[dn[f] % n == k], g[dn[g] % n == k]) for o, f, g in L]) for k in range(n)]
    null_sd = []
    for _ in range(2000):
        sub_k = []
        perm = [(o, rng.permutation(dn[f] % n), rng.permutation(dn[g] % n), f, g) for o, f, g in L]
        v = [100 * val([(o, f[pf == k], g[pg == k]) for o, pf, pg, f, g in perm]) for k in range(n)]
        null_sd.append(np.std(v, ddof=1))
    out[s] = {"n_seeds": n, "pooled": 100 * val(L), "seed_values": obs, "seed_sd": float(np.std(obs, ddof=1)),
              "null_seed_sd_mean": float(np.mean(null_sd)), "null_seed_sd_p95": float(np.percentile(null_sd, 95)),
              "p_null_sd_ge_obs": float(np.mean(np.array(null_sd) >= np.std(obs, ddof=1))),
              "drafts": int(sum(len(f) + len(g) for _, f, g in L))}
    print(s, json.dumps(out[s]), flush=True)
json.dump({"worlds": NW, "family": "a", "cell": "default", "strategies": out},
          open(os.path.expanduser("~/hots/repo/training/worlds/site/results/rescore_p1_seed_null.json"), "w"), indent=1)
