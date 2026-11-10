"""
Support correction of the judges, and the tournament rescored with it.

judge_ood_calibration.py found that every judge overrates real teams with little
support in the training data (rare triples, hero pairs fielded together less than
their popularity predicts), by about 2.5 points on teams shaped like the search
agents'. Here we fit that bias as a per-team term and rescore the tournament.

Fit (per judge), game level, team0 orientation, on the leave-one-out training-split
games (judge_ood_calibration_teams.npz, set == train_loo):
    logit P(team0 wins) = logit p_judge + w . (x(team0) - x(team1))
x(team) = support features: log1p(rarest triple), log1p(rarest pair), mean pair
affinity, and indicators for one swap and two-plus swaps from a seen team. No
intercept: the term is antisymmetric, so it moves no game whose two teams are
equally supported. Checked on the held-out validation and paper-test games, which
the support counts never include: log loss and the binned calibration gap before
and after.

Then every tournament draft (12-strategy round robin, the win-rate drafter, masked
K, the decision-time search agents incl. the ensemble-LCB ones, TD and MCQ variants)
is rescored with each judge before and after the correction: standings, and the
head-to-head pairs the paper quotes.

Usage (from training/, p1site env): python3 paper1_revision/support_correction.py
Output: site/results/support_correction.json
"""
import os
import sys
import json
import glob

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
T = os.path.dirname(HERE)
sys.path.insert(0, T)
from paper1_revision import core
from paper1_revision import team_support as ts
from paper1_revision import judge_ood_calibration as joc
from shared import HERO_TO_IDX

RES = ts.RES
JUDGES = joc.JUDGES
CONS = ["gN", "gN_naive", "RN", "QM2026"]
OUT = os.path.join(RES, "support_correction.json")
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
sig = lambda z: 1 / (1 + np.exp(-z))


def feats(m):
    d = np.asarray(m["dist"])
    return np.stack([np.log1p(np.maximum(m["min3"], 0)), np.log1p(np.maximum(m["min2"], 0)), m["pmi"],
                     (d == 1).astype(float), (d >= 2).astype(float)], 1)


FEATS = ["log1p_min3", "log1p_min2", "pmi", "one_swap", "two_plus_swaps"]


def fit_offset(off, X, y, l2=1e-3, iters=30):
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = sig(off + X @ w)
        g = X.T @ (p - y) / len(y) + l2 * w
        Hm = (X * (p * (1 - p))[:, None]).T @ X / len(y) + l2 * np.eye(X.shape[1])
        step = np.linalg.solve(Hm, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def ll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def games(z, s):
    """Game-level arrays for one set: team0 rows and the matching team1 rows."""
    sel = z["set"] == s
    rid, team = z["rid"][sel], z["team"][sel]
    o0 = np.where(team == 0)[0]
    o1 = np.where(team == 1)[0]
    o0 = o0[np.argsort(rid[o0], kind="stable")]
    o1 = o1[np.argsort(rid[o1], kind="stable")]
    assert (rid[o0] == rid[o1]).all()
    g = lambda k, o: z[k][sel][o]
    m0 = {k: g("m_" + k, o0) for k in ("min3", "min2", "pmi", "dist")}
    m1 = {k: g("m_" + k, o1) for k in ("min3", "min2", "pmi", "dist")}
    return {"y": g("y", o0), "p": {j: g("p_" + j, o0) for j in JUDGES}, "x0": feats(m0), "x1": feats(m1), "m0": m0}


def calib_bins(p, y, m, edges):
    b = np.clip(np.searchsorted(edges, m, side="right") - 1, 0, len(edges) - 2)
    return [float((y[b == i] - p[b == i]).mean()) if (b == i).sum() > 50 else None for i in range(len(edges) - 1)]


def tournament_records():
    """(strategy0, strategy1, record) for every scored tournament draft we have."""
    out = []
    for d in ("tournament", "wr_baseline/tournament", "masked_k/tournament", "td/*/tournament",
              "mcq_real/tournament", "search_agents/tournament"):
        for f in sorted(glob.glob(os.path.join(RES, d, "*.json"))):
            R = json.load(open(f))
            for r in R["records"]:
                out.append((R["team0_strategy"], R["team1_strategy"], r))
    return out


MAIN12 = ["mcts", "constrained_mcts", "enriched", "constrained_greedy", "k_oof", "gourdeau", "gourdeau_disc",
          "cql_naive_a1.0", "cql_enr_a2.0", "gd", "mcq_t0.5", "constrained_wr"]


def standings(S0, S1, P, names):
    """Mean win probability over ordered pairs (both seats), the paper's protocol: a round-robin
    entrant against the other eleven entrants; any other strategy against the eleven learned ones."""
    res = {}
    pairs = {}
    for a, b, p in zip(S0, S1, P):
        pairs.setdefault((a, b), []).append(p)
    for s in names:
        vals, won = [], 0
        opp = set(MAIN12) - {s} if s in MAIN12 else set(MAIN12) - {"constrained_wr"}
        for (a, b), v in pairs.items():
            if (a == s and b not in opp) or (b == s and a not in opp):
                continue
            if a == s and b != s:
                m = float(np.mean(v))
            elif b == s and a != s:
                m = 1 - float(np.mean(v))
            else:
                continue
            vals.append(m)
            won += m > 0.5
        if vals:
            res[s] = {"mean": float(np.mean(vals)), "pairs": len(vals), "won": int(won)}
    return res, pairs


def h2h(pairs, a, b):
    v = []
    if (a, b) in pairs:
        v += list(pairs[(a, b)])
    if (b, a) in pairs:
        v += [1 - x for x in pairs[(b, a)]]
    return float(np.mean(v)) if v else None


def main():
    z = dict(np.load(os.path.join(RES, "judge_ood_calibration_teams.npz"), allow_pickle=True))
    tr, ho = games(z, "train_loo"), games(z, "heldout")
    res = {"features": FEATS, "fit": {}, "heldout": {}}
    W = {}
    for j in JUDGES:
        w = fit_offset(logit(tr["p"][j]), tr["x0"] - tr["x1"], tr["y"])
        W[j] = w
        p0 = ho["p"][j]
        p1 = sig(logit(p0) + (ho["x0"] - ho["x1"]) @ w)
        # binned gap per team0 support, before and after (held-out)
        e3 = np.percentile(ho["m0"]["min3"], [0, 1, 5, 25, 100])
        ep = np.percentile(ho["m0"]["pmi"], [0, 1, 5, 25, 100])
        res["fit"][j] = dict(zip(FEATS, w.tolist()))
        res["heldout"][j] = {"ll_before": ll(p0, ho["y"]), "ll_after": ll(p1, ho["y"]),
                             "gap_by_min3_before": calib_bins(p0, ho["y"], ho["m0"]["min3"], e3),
                             "gap_by_min3_after": calib_bins(p1, ho["y"], ho["m0"]["min3"], e3),
                             "gap_by_pmi_before": calib_bins(p0, ho["y"], ho["m0"]["pmi"], ep),
                             "gap_by_pmi_after": calib_bins(p1, ho["y"], ho["m0"]["pmi"], ep)}
        print(j, "w", np.round(w, 3), "heldout ll", round(res["heldout"][j]["ll_before"], 5), "->",
              round(res["heldout"][j]["ll_after"], 5), flush=True)

    # tournament drafts
    recs = tournament_records()
    games_t = [(None, r["tier"], r["game_map"], r["team0"]["picks"], r["team1"]["picks"]) for _, _, r in recs]
    sc = joc.judge_scores(games_t)
    G = core.split_games()
    C = ts.Corpus(np.concatenate([joc.idx_teams(G["train"], 3), joc.idx_teams(G["train"], 4)]))
    A0 = np.array([sorted(HERO_TO_IDX[h] for h in r["team0"]["picks"]) for _, _, r in recs], np.int64)
    A1 = np.array([sorted(HERO_TO_IDX[h] for h in r["team1"]["picks"]) for _, _, r in recs], np.int64)
    X0, X1 = feats(C.metrics(A0)), feats(C.metrics(A1))
    S0 = [a for a, _, _ in recs]
    S1 = [b for _, b, _ in recs]
    corr = {j: sig(logit(sc[j]) + (X0 - X1) @ W[j]) for j in JUDGES}
    corr["consensus_of_corrected"] = np.mean([corr[j] for j in CONS], 0)
    base = {j: np.asarray(sc[j]) for j in JUDGES}
    names = sorted(set(S0) | set(S1))
    res["standings"], res["h2h"] = {}, {}
    key_pairs = [("constrained_mcts", "constrained_wr"), ("mcts", "constrained_wr"), ("k_oof", "constrained_wr"),
                 ("constrained_k", "constrained_wr"), ("constrained_mcts", "constrained_greedy"),
                 ("mcts", "enriched"), ("constrained_mcts", "gd"), ("constrained_mcts", "k_oof"),
                 ("constrained_mcts", "mcts"), ("constrained_k", "constrained_mcts"),
                 ("mcts_lcb", "constrained_mcts"), ("mcts_lcb2", "constrained_mcts"), ("mcts_search", "mcts"),
                 ("constrained_wr", "constrained_greedy"), ("constrained_wr", "enriched")]
    for tag, P in (("before", base), ("after", corr)):
        res["standings"][tag], res["h2h"][tag] = {}, {}
        for j in list(P):
            st, pairs = standings(S0, S1, P[j], names)
            res["standings"][tag][j] = st
            res["h2h"][tag][j] = {f"{a}__vs__{b}": h2h(pairs, a, b) for a, b in key_pairs}
    # support and expected bias of each strategy's teams (team as that strategy)
    sup = {}
    for s in names:
        xs = np.concatenate([X0[[a == s for a in S0]], X1[[b == s for b in S1]]])
        sup[s] = {"n_teams": int(len(xs)), "mean_feats": dict(zip(FEATS, xs.mean(0).round(3).tolist()))}
    res["team_support"] = sup
    json.dump(res, open(OUT, "w"), indent=1)
    print("wrote", OUT)
    main12 = MAIN12 + ["constrained_k", "mcts_search", "mcts_lcb", "mcts_lcb2", "mcq_real", "cql_td_a1.0", "iql_td_t0.7_b3.0"]
    for j in ("consensus", "gN", "RN"):
        b, a = res["standings"]["before"][j], res["standings"]["after"][j]
        print("==", j)
        for s in sorted(main12, key=lambda s: -a.get(s, {"mean": 0})["mean"]):
            if s in b:
                print(f"  {s:20s} {b[s]['mean']:.3f} ({b[s]['won']:2d}) -> {a[s]['mean']:.3f} ({a[s]['won']:2d})")
        for k, v in res["h2h"]["after"][j].items():
            if v is not None:
                print(f"  h2h {k:45s} {res['h2h']['before'][j][k]:.3f} -> {v:.3f}")


if __name__ == "__main__":
    main()
