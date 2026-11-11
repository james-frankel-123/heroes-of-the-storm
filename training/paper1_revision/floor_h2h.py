"""Head to head of floored and base search policies vs key opponents, per judge and correction form, with config-clustered SE."""
import os, sys, json, glob
import numpy as np
T = os.path.expanduser("~/hots/repo/training"); sys.path.insert(0, T)
from paper1_revision import core
from paper1_revision import support_mitigation as sm
from paper1_revision import judge_ood_calibration as joc
from shared import HERO_TO_IDX
ROWS = [("constrained_mcts_sf", "constrained_wr_sf"), ("constrained_k_sf", "constrained_wr_sf"),
        ("constrained_mcts", "constrained_wr"), ("constrained_mcts_sf", "constrained_wr"),
        ("constrained_mcts_sf", "constrained_greedy"), ("constrained_mcts", "constrained_greedy"),
        ("constrained_mcts_sf", "gd"), ("constrained_mcts", "gd"), ("constrained_k_sf", "constrained_greedy"),
        ("constrained_mcts_sf", "constrained_mcts"), ("constrained_k_sf", "constrained_k")]
want = {x for r in ROWS for x in r}
recs = []
for d in ("tournament", "wr_baseline/tournament", "masked_k/tournament", "support_mitigation/tournament"):
    for f in sorted(glob.glob(os.path.join(core.RESULTS, d, "*.json"))):
        R = json.load(open(f)); a, b = R["team0_strategy"], R["team1_strategy"]
        if a in want and b in want:
            recs += [(a, b, r) for r in R["records"]]
games = [(None, r["tier"], r["game_map"], r["team0"]["picks"], r["team1"]["picks"]) for _, _, r in recs]
sc = joc.judge_scores(games)
C = sm.sup_corpus()
A = lambda side: np.array([sorted(HERO_TO_IDX[h] for h in r[side]["picks"]) for _, _, r in recs], np.int64)
m0, m1 = C.metrics(A("team0")), C.metrics(A("team1"))
jc = json.load(open(os.path.join(sm.OUT, "judge_corr.json")))
S0 = np.array([a for a, _, _ in recs]); S1 = np.array([b for _, b, _ in recs])
cfg = np.array([f"{r['game_map']}|{r['tier']}" for _, _, r in recs])
out = {}
for j in ("consensus", "gN", "RN", "QM2026"):
    wl, wb = np.array(jc["judges"][j]["w_linear"]), np.array(jc["judges"][j]["w_binned"])
    forms = {"none": np.asarray(sc[j]), "linear": sm.sig(sm.logit(sc[j]) + (sm.lin_feats(m0) - sm.lin_feats(m1)) @ wl),
             "binned": sm.sig(sm.logit(sc[j]) + (sm.bin_feats(m0, jc["edges"]) - sm.bin_feats(m1, jc["edges"])) @ wb)}
    for f, v in forms.items():
        for a, b in ROWS:
            x = np.concatenate([v[(S0 == a) & (S1 == b)], 1 - v[(S0 == b) & (S1 == a)]])
            c = np.concatenate([cfg[(S0 == a) & (S1 == b)], cfg[(S0 == b) & (S1 == a)]])
            if not len(x):
                continue
            mu = x.mean(); u = np.unique(c, return_inverse=True)[1]
            se = np.sqrt((np.bincount(u, x - mu) ** 2).sum()) / len(x)
            out.setdefault(f"{a} vs {b}", {})[f"{j}/{f}"] = [round(float(mu), 4), round(float(se), 4), int(len(x))]
json.dump(out, open(os.path.join(sm.OUT, "floor_h2h.json"), "w"), indent=1)
for k, v in out.items():
    print(f"{k:44s}", "  ".join(f"{jf} {m:.3f}±{s:.3f}" for jf, (m, s, n) in v.items() if jf.split('/')[0] in ('consensus', 'gN', 'RN') and jf.split('/')[1] in ('none', 'binned')))
