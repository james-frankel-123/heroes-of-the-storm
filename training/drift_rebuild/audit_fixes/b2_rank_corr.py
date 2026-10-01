"""Consolidated audit B2: rank correlation of the QM-2021 and QM-2026 judges'
strategy standings with paper 1's consensus order, with the two strategy keys
that qm2026/train_qm_wp.py's CONSENSUS_ORDER misspells (gourdeau_est -> gourdeau,
mcq -> mcq_t0.5) matched, so all 11 strategies enter.
Output: drift_rebuild/results/b2_rank_corr.json"""
import json
import os
import numpy as np

T = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONS = {"constrained_mcts": 0.670, "mcts": 0.658, "constrained_greedy": 0.595,
        "enriched": 0.595, "enriched_aug": 0.580, "gourdeau": 0.519,
        "cql_enr_a2.0": 0.420, "cql_naive_a1.0": 0.419, "gourdeau_disc": 0.417,
        "gd": 0.404, "mcq_t0.5": 0.225}


def avg_rank(x):
    x = np.asarray(x, float)
    r = np.empty(len(x))
    r[np.argsort(x)] = np.arange(len(x))
    for v in np.unique(x):
        r[x == v] = r[x == v].mean()
    return r


out = {}
for f, k in (("qm2026/results/qm_wp_v0.json", "strategy_standings_qm"),
             ("qm2026/results/qm2026_judge.json", "strategy_standings")):
    st = json.load(open(os.path.join(T, f)))[k]
    s = [x for x in CONS if x in st]
    a, b = np.array([st[x] for x in s]), np.array([CONS[x] for x in s])
    ordinal = float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])
    avg = float(np.corrcoef(avg_rank(a), avg_rank(b))[0, 1])
    out[f] = {"n": len(s), "spearman_ordinal": round(ordinal, 4), "spearman_avg_ties": round(avg, 4)}
    print(f, out[f])
json.dump(out, open(os.path.join(T, "drift_rebuild/results/b2_rank_corr.json"), "w"), indent=1)
