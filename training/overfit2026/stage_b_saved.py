"""
Stage B: rescore every saved paper-1 draft record with gold references that
share no data with the proxy.

Proxy  = the paper's enriched WP (wp_enriched_256 + frozen 2026-05-19 stats),
         symmetrized; the number the paper reports as "Avg WP" (stored in
         each dump as wp_sym).
Gold   = RealizedIndex over post-snapshot games (gold.GOLD_SETS), and the
         Quick-Match 2026 judge (different game mode, naive features).

Inputs (read-only):
  rerun2026/results/diversity/*.json          200 drafts per run vs GD pool
  rerun2026/results/roundrobin/*.json         July tournament records
  rerun2026/results/constrained/roundrobin/*.json
Output: overfit2026/results/stage_b.json
"""
import os
import sys
import glob
import json

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)

import numpy as np
import torch

from overfit2026 import gold

RR = os.path.join(TRAINING_DIR, "rerun2026", "results")
OUT = os.path.join(HERE, "results", "stage_b.json")
GOLDS = ["NODRIFT", "BF", "T97", "T17", "FUT"]


def load_qm():
    from qm2026.train_qm_wp import MLP
    out = {}
    for tag in ("2026", "2022"):
        p = os.path.join(TRAINING_DIR, "qm2026", "results", f"qm_wp_{tag}.pt")
        if os.path.exists(p):
            m = MLP()
            m.load_state_dict(torch.load(p, map_location="cpu", weights_only=True))
            m.eval()
            out[f"QM{tag}"] = m
    return out


def qm_scores(models, rows):
    """rows: list of (own, opp, map, tier) -> {name: array P(own wins)}."""
    from qm2026.train_qm_wp import featurize
    Xf = np.stack([featurize(o, p, m, t) for o, p, m, t in rows])
    Xr = np.stack([featurize(p, o, m, t) for o, p, m, t in rows])
    res = {}
    with torch.no_grad():
        for n, m in models.items():
            a = m(torch.tensor(Xf)).numpy()
            b = m(torch.tensor(Xr)).numpy()
            res[n] = 0.5 * (a + 1 - b)
    return res


def mean_se(v):
    v = np.asarray(v, dtype=float)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0


def score_rows(rows, idx, qm):
    """rows: (own, opp, map, tier). Returns dict of per-draft arrays."""
    out = {}
    for g in GOLDS:
        out[g] = np.array([idx[g].score(tuple(o), tuple(p), t) for o, p, m, t in rows])
    comps = [idx["NODRIFT"].components(tuple(o), tuple(p), t) for o, p, m, t in rows]
    for k in gold.NAMES:
        out["nd_" + k] = np.array([c[k] for c in comps])
    out.update(qm_scores(qm, rows))
    return out


def main():
    idx = {g: gold.get_index(g) for g in GOLDS}
    qm = load_qm()
    result = {"gold_sets": {g: idx[g].describe() for g in GOLDS}, "dumps": {},
              "tournament": {}}

    # ── diversity dumps (MCTS configs / inference variants vs GD pool) ──
    for path in sorted(glob.glob(os.path.join(RR, "diversity", "*__*.json"))):
        d = json.load(open(path))
        if "drafts" not in d:
            continue
        rows = [(x["our"], x["opp"], x["map"], "mid") for x in d["drafts"]]
        sc = score_rows(rows, idx, qm)
        sc["proxy"] = np.array([x["wp_sym"] for x in d["drafts"]])
        name = os.path.basename(path)[:-5]
        result["dumps"][name] = {
            "experiment": d["experiment"], "seed": d["seed"],
            "variant": d["variant"], "n": len(rows),
            "distinct": d.get("distinct"), "entropy": d.get("entropy_bits"),
            "degen": d.get("degen"), "healer": d.get("healer"),
            "means": {k: mean_se(v) for k, v in sc.items()},
            "per_draft": {k: np.round(v, 5).tolist() for k, v in sc.items()},
        }
        print(f"{name:40s} proxy {sc['proxy'].mean():.4f} NODRIFT "
              f"{sc['NODRIFT'].mean():.4f} T17 {sc['T17'].mean():.4f} "
              f"QM {sc.get('QM2026', np.array([np.nan])).mean():.4f}", flush=True)

    # ── tournament records ──
    files = (glob.glob(os.path.join(RR, "roundrobin", "*.json"))
             + glob.glob(os.path.join(RR, "constrained", "roundrobin", "*.json")))
    per_strat = {}
    for path in sorted(files):
        d = json.load(open(path))
        s0, s1 = d["team0_strategy"], d["team1_strategy"]
        rows, prox = [], []
        for r in d["records"]:
            rows.append((r["team0"]["picks"], r["team1"]["picks"], r["game_map"], r["tier"]))
            prox.append(np.mean([r["wp"][j]["sym"] for j in
                                 ("naive", "herostrength", "enriched", "augmented")]))
        sc = score_rows(rows, idx, qm)
        sc["consensus"] = np.array(prox)
        pair = f"{s0}__{s1}"
        if pair in result["tournament"]:
            continue
        result["tournament"][pair] = {k: mean_se(v) for k, v in sc.items()}
        for s, sign in ((s0, 1), (s1, -1)):
            e = per_strat.setdefault(s, {})
            for k, v in sc.items():
                vals = v if sign == 1 else (1 - v if k in GOLDS + list(qm) + ["consensus"] else -v)
                e.setdefault(k, []).append(float(np.mean(vals)))
    result["standings"] = {s: {k: float(np.mean(v)) for k, v in e.items()}
                           for s, e in per_strat.items()}
    result["standings_wins"] = {s: {k: int(sum(x > 0.5 for x in v)) for k, v in e.items()
                                    if k in GOLDS + list(qm) + ["consensus"]}
                                for s, e in per_strat.items()}
    result["standings_n_pairs"] = {s: len(e["consensus"]) for s, e in per_strat.items()}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(result, f)
    print("\nSTANDINGS (mean P over ordered pairs)")
    ks = ["consensus"] + GOLDS + list(qm)
    print(f"{'strategy':22s}" + "".join(f"{k:>10s}" for k in ks))
    for s, e in sorted(result["standings"].items(), key=lambda kv: -kv[1]["consensus"]):
        print(f"{s:22s}" + "".join(f"{e[k]:10.4f}" for k in ks))


if __name__ == "__main__":
    main()
