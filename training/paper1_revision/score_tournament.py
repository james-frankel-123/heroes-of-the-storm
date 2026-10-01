"""
Score the revised tournament with judges that are no contestant's value
function, and write standings.

Judges, each P(team0 wins), team-order symmetrized:
  gN        mean of 3 out-of-fold enriched judges trained only on 291,837
            post-snapshot no-drift games (overfit2026; own statistics of
            those games)
  gN_naive  hero-identity judge on the same games
  RN        realized-outcome index on the same games (cross-fitted; weak but
            model-free; it barely penalizes degenerate teams)
  QM2026    Quick Match judge, 116,589 games, draft-free mode
  QM2021    Quick Match judge, 91,365 games, 2021 era
  naive_rev revision hero-identity judge, snapshot train split (no statistics;
            not any contestant's objective, but same corpus)
  R17       realized index on drifted 2.55.17 games (reported, not in consensus)
Diagnostic only (contestants' own value functions): self_enriched (the
revision's leak-free enriched WP, objective of MCTS / greedy / constrained).

Consensus (fixed before scoring) = mean of gN, gN_naive, RN, QM2026.

Standing of a strategy = mean over its 22 ordered pairs of its mean P(win).
SE: within each pair, drafts are clustered by (map, tier) configuration (42
configs; argmax agents repeat drafts across the 200), pair variances are
summed over the fixed opponent set.
Output: results/tournament_scores.json, results/tournament_standings.json
"""
import os
import sys
import json
import itertools

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)
from paper1_revision import core
from paper1_revision.tournament import STRATEGIES, UNCHANGED, pair_path, reused_path

import numpy as np
import torch

JUDGES = ["gN", "gN_naive", "RN", "QM2026", "QM2021", "naive_rev", "R17", "self_enriched"]
CONSENSUS = ["gN", "gN_naive", "RN", "QM2026"]


def load_pair(a, b):
    p = reused_path(a, b) if (a in UNCHANGED and b in UNCHANGED) else pair_path(a, b)
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def qm2021():
    from qm2026.train_qm_wp import MLP
    m = MLP()
    m.load_state_dict(torch.load(os.path.join(TRAINING_DIR, "qm2026", "results", "qm_wp_v0.pt"),
                                 map_location="cpu", weights_only=True))
    m.eval()
    return m


def score_rows(rows):
    from overfit2026 import score as oscore
    from overfit2026.stage_b_saved import qm_scores
    from paper1_revision import train_wp
    GN = oscore.GN
    ms = oscore.model_scores(GN + [oscore.GN_NAIVE], rows, dev="cpu")
    out = {"gN": np.mean([ms[n] for n in GN], 0), "gN_naive": ms["gN8_naive"]}
    rn, r17 = oscore.realized("NODRIFT"), oscore.realized("T17")
    out["RN"] = np.array([rn.score(tuple(o), tuple(p), t) for o, p, m, t in rows])
    out["R17"] = np.array([r17.score(tuple(o), tuple(p), t) for o, p, m, t in rows])
    q = qm_scores({"QM2026": oscore.qm()["QM2026"], "QM2021": qm2021()}, rows)
    out.update(q)
    Xf, Xs = core.featurize(rows, core.load_stats("deploy"), nproc=int(os.environ.get("P1R_NPROC", "6")))
    for name, key in (("naive", "naive_rev"), ("enriched", "self_enriched")):
        m, cols = train_wp.load(name)
        with torch.no_grad():
            a = m(torch.tensor(Xf[:, cols])).view(-1).numpy()
            b = m(torch.tensor(Xs[:, cols])).view(-1).numpy()
        out[key] = 0.5 * (a + 1 - b)
    return out


def pair_stats(v, cfg):
    """mean and config-clustered variance of the pair mean."""
    v = np.asarray(v, float)
    mu = v.mean()
    groups = {}
    for x, c in zip(v, cfg):
        groups.setdefault(c, []).append(x - mu)
    n = len(v)
    var = sum(np.sum(g) ** 2 for g in groups.values()) / n ** 2 * len(groups) / max(len(groups) - 1, 1)
    return float(mu), float(var)


def _ranks(a):
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a))
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(x, y):
    """Spearman rank correlation, ties averaged (numpy only)."""
    rx, ry = _ranks(x), _ranks(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))


def main():
    pairs = {}
    rows_all, idx = [], {}
    for a, b in itertools.permutations(STRATEGIES, 2):
        d = load_pair(a, b)
        if d is None:
            continue
        recs = d["records"]
        rows = [(r["team0"]["picks"], r["team1"]["picks"], r["game_map"], r["tier"]) for r in recs]
        idx[(a, b)] = (len(rows_all), len(rows_all) + len(rows))
        rows_all += rows
        pairs[(a, b)] = recs
    print(f"{len(pairs)} ordered pairs, {len(rows_all):,} drafts", flush=True)
    sc_path = os.path.join(core.RESULTS, "tournament_scores.npz")
    if os.path.exists(sc_path) and len(np.load(sc_path)["gN"]) == len(rows_all):
        sc = dict(np.load(sc_path))
    else:
        sc = score_rows(rows_all)
        np.savez(sc_path, **sc)

    per = {s: {j: [] for j in JUDGES + ["consensus"]} for s in STRATEGIES}
    pair_out = {}
    for (a, b), (i0, i1) in idx.items():
        recs = pairs[(a, b)]
        cfg = [(r["game_map"], r["tier"]) for r in recs]
        res = {}
        for j in JUDGES + ["consensus"]:
            v = (np.mean([sc[k][i0:i1] for k in CONSENSUS], 0) if j == "consensus"
                 else sc[j][i0:i1])
            mu, var = pair_stats(v, cfg)
            res[j] = (mu, var)
            per[a][j].append((mu, var))
            per[b][j].append((1 - mu, var))
        side = {}
        for s, key in ((a, "team0"), (b, "team1")):
            side[s] = {m: float(np.mean([float(r[key][f]) for r in recs]))
                       for m, f in (("healer", "has_healer"), ("degen", "is_degen"),
                                    ("synergy", "synergy"), ("counter", "counter"),
                                    ("resilience_late", "resilience_late"),
                                    ("resilience_avg", "resilience_avg"))}
        pair_out[f"{a}__{b}"] = {"team0_p": {j: res[j][0] for j in res},
                                 "se": {j: float(np.sqrt(res[j][1])) for j in res},
                                 "n": i1 - i0, "sides": side}
    stand = {}
    for s in STRATEGIES:
        e = per[s]
        if not e["consensus"]:
            continue
        stand[s] = {"n_pairs": len(e["consensus"])}
        for j in JUDGES + ["consensus"]:
            mus = np.array([x[0] for x in e[j]])
            vs = np.array([x[1] for x in e[j]])
            stand[s][j] = {"mean": float(mus.mean()), "se": float(np.sqrt(vs.sum()) / len(vs)),
                           "wins": int((mus > 0.5).sum())}
    # per-strategy draft metrics pooled over all its pairs and both sides
    metrics = {}
    for (a, b), recs in pairs.items():
        for s, key in ((a, "team0"), (b, "team1")):
            m = metrics.setdefault(s, {"healer": [], "degen": [], "synergy": [], "counter": [],
                                       "resilience_late": [], "picks": []})
            for r in recs:
                m["healer"].append(float(r[key]["has_healer"]))
                m["degen"].append(float(r[key]["is_degen"]))
                m["synergy"].append(r[key]["synergy"])
                m["counter"].append(r[key]["counter"])
                m["resilience_late"].append(r[key]["resilience_late"])
                m["picks"] += r[key]["picks"]
    from collections import Counter
    msum = {}
    for s, m in metrics.items():
        c = Counter(m.pop("picks"))
        tot = sum(c.values())
        p = np.array(list(c.values())) / tot
        msum[s] = {k: float(np.mean(v)) for k, v in m.items()}
        msum[s].update({"distinct": len(c), "entropy": float(-(p * np.log2(p)).sum()),
                        "top10": float(sum(v for _, v in c.most_common(10)) / tot)})
    ss = [s for s in STRATEGIES if s in stand]
    cons = [stand[s]["consensus"]["mean"] for s in ss]
    rank_corr = {}
    top6 = sorted(ss, key=lambda s: -stand[s]["consensus"]["mean"])[:6]
    for j in JUDGES:
        rank_corr[j] = {"all": spearman([stand[s][j]["mean"] for s in ss], cons),
                        "top6": spearman([stand[s][j]["mean"] for s in top6],
                                         [stand[s]["consensus"]["mean"] for s in top6])}
    out = {"consensus_judges": CONSENSUS, "standings": stand, "draft_metrics": msum,
           "spearman_vs_consensus": rank_corr, "top6": top6, "pairs": pair_out}
    json.dump(out, open(os.path.join(core.RESULTS, "tournament_standings.json"), "w"), indent=1)
    print(f"\n{'strategy':20s}" + "".join(f"{j:>12s}" for j in ["consensus"] + JUDGES))
    for s in sorted(ss, key=lambda s: -stand[s]["consensus"]["mean"]):
        print(f"{s:20s}" + "".join(f"{stand[s][j]['mean']:8.3f}({stand[s][j]['wins']:2d})"
                                   for j in ["consensus"] + JUDGES))
    print("spearman", {j: (round(v['all'], 3), round(v['top6'], 3)) for j, v in rank_corr.items()})


if __name__ == "__main__":
    main()
