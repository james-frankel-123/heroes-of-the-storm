# Runs on a worker (~/hots/repo layout): P3_CACHE=<cache_full> python ~/hots/p2f_stale_cell.py
# Year-stale table for P2_NARRATIVE_FACTS.md section 7, from the tournament saved states (read only).
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.expanduser("~/hots/repo/training/personalization"))
sys.path.insert(0, os.path.expanduser("~/hots/repo/training"))
import p3t_data as TD
L = TD.load_lobbies()
g = np.flatnonzero(L["day"] <= TD.LOBBY_END)
P = np.load(os.path.join(TD.OUT, "predeval_wp.npz"))
assert (P["g"] == g).all()
y = P["y"]
res_dir = os.path.join(os.path.dirname(TD.OUT).replace("cache_full", "results"), "p3t")
comb = json.load(open(os.path.expanduser("~/hots/repo/training/personalization/results/rcf_full/p3t/combiner.json")))
st = {v: np.load(os.path.join(TD.OUT, f"state_{v}.npz")) for v in ("cur", "s4m", "s1y")}
team = L["steps"][g][:, :, 2][L["steps"][g][:, :, 1] == 1].reshape(len(g), 10)
hero = L["steps"][g][:, :, 0][L["steps"][g][:, :, 1] == 1].reshape(len(g), 10)
sign = np.where(team == 0, 1.0, -1.0)
def ll(p): p = np.clip(p, 1e-9, 1-1e-9); return float(-(y*np.log(p)+(1-y)*np.log(1-p)).mean())
out = {"games": int(len(g))}
for wv in ("cur", "s4m", "s1y"):
    wp = P[f"wp_{wv}"]; lo = np.log(np.clip(wp,1e-6,1-1e-6)/np.clip(1-wp,1e-6,1-1e-6))
    b = comb[wv]["coef"]
    for sv in ("cur", "s4m", "s1y"):
        base = (g[:, None]*10 + np.arange(10)[None]) if sv == "cur" else (np.arange(len(g))[:, None]*10 + np.arange(10)[None])
        sp = st[sv]["s"].astype(np.float32)[base.ravel(), hero.ravel()].reshape(len(g), 10)
        op = st[sv]["off"][base.ravel(), hero.ravel()].reshape(len(g), 10)
        p = 1/(1+np.exp(-(b[0]+b[1]*lo+b[2]*(sign*sp).sum(1)+b[3]*(sign*op).sum(1))))
        out[f"wp {wv} + state {sv}"] = {"acc": float(((p>0.5)==(y==1)).mean()), "ll": ll(p)}
    out[f"wp {wv} alone"] = {"acc": float(((wp>0.5)==(y==1)).mean()), "ll": ll(wp)}
print(json.dumps(out, indent=1))
