"""
P3 tournament diagnostics: why is the judged personal gain smaller than the
drafter's own estimate?

For every side of every tournament draft and for the real drafts:
  S_D  the drafter-state personal sum (state cur, the drafter's combiner b2/b3 assignment)
  S_J  the judge-state personal sum (judge window evidence, other-half fit, judge assignment)
Reported:
  - per system: mean S_D and S_J of its own side (pp, scaled by b2 and c2 to logit units too);
  - regression slope of S_J on S_D across sides, separately for real teams
    (no selection on the drafter's estimates) and for each drafter's teams
    (selected by the drafter): a slope well below the real-team slope means
    the drafter selects on estimation noise (winner's curse);
  - pool sizes (heroes a team may pick) and the share of real picks outside the pool.
Output: $P3_RESULTS/p3t/diag.json
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("P1_TIERS", "site")
import numpy as np

import p3_hs_core as C
import p3t_data as TD
import p3t_judge as JG

PICK_T = [(k, t) for k, t in zip(range(16), [0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1])
          if k in (4, 5, 6, 7, 8, 11, 12, 13, 14, 15)]


def side_sums(LB, acts, b2, b3, c2, c3):
    n = len(acts)
    SD = np.zeros((n, 2))
    SJ = np.zeros((n, 2))
    sd, od = LB.state["cur"]["s"], LB.state["cur"]["off"].astype(float)
    sj, oj = LB.state["judge"]["s"], LB.state["judge"]["off"].astype(float)
    for i in range(n):
        for t in (0, 1):
            hs = [acts[i, k] for k, tt in PICK_T if tt == t]
            sl = slice(5 * t, 5 * t + 5)
            SD[i, t] = JG.best_assign(sd[i, sl], od[i, sl], hs, b2, b3)[0]
            SJ[i, t] = JG.best_assign(sj[i, sl], oj[i, sl], hs, c2, c3)[0]
    return SD, SJ


def slope(x, y):
    x = x - x.mean()
    return float((x * (y - y.mean())).sum() / max((x * x).sum(), 1e-12))


def main():
    import p3t_systems as SY
    import p3t_tourney as TT
    LB = TT.Lobbies()
    judge = json.load(open(os.path.join(C.RESULTS, "p3t", "judge.json")))
    c2, c3 = judge["c2_skill"], judge["c3_offrole"]
    b = SY.vintage("cur").spec["combiner"]
    out = {"b2_drafter": b[2], "c2_judge": c2}
    SDr, SJr = side_sums(LB, LB.real_acts, b[2], b[3], c2, c3)
    out["real"] = {"S_D_pp": round(100 * SDr.mean(), 3), "S_J_pp": round(100 * SJr.mean(), 3),
                   "slope_SJ_on_SD": round(slope(SDr.ravel(), SJr.ravel()), 3),
                   "corr": round(float(np.corrcoef(SDr.ravel(), SJr.ravel())[0, 1]), 3)}
    per = {}
    for f in sorted(os.listdir(TT.TDIR)):
        if "__" not in f or not f.endswith(".npz"):
            continue
        z = np.load(os.path.join(TT.TDIR, f))
        a_, b_ = str(z["a"]), str(z["b"])
        SD, SJ = side_sums(LB, z["acts"], b[2], b[3], c2, c3)
        for side, s_ in ((0, a_), (1, b_)):
            per.setdefault(s_, ([], []))
            per[s_][0].append(SD[:, side])
            per[s_][1].append(SJ[:, side])
    out["systems"] = {}
    for s_, (xd, xj) in per.items():
        xd, xj = np.concatenate(xd), np.concatenate(xj)
        out["systems"][s_] = {"S_D_pp": round(100 * xd.mean(), 3), "S_J_pp": round(100 * xj.mean(), 3),
                              "S_D_minus_real_pp": round(100 * (xd.mean() - SDr.mean()), 3),
                              "S_J_minus_real_pp": round(100 * (xj.mean() - SJr.mean()), 3),
                              "slope_SJ_on_SD": round(slope(xd, xj), 3)}
        print(s_, out["systems"][s_], flush=True)
    tp = LB.team_pool.sum(-1)
    out["team_pool_size"] = {"median": float(np.median(tp)), "p10": float(np.percentile(tp, 10)),
                             "p90": float(np.percentile(tp, 90))}
    inpool = []
    for i in range(LB.n):
        for k, t in PICK_T:
            inpool.append(bool(LB.team_pool[i, t, LB.real_acts[i, k]]))
    out["real_picks_in_team_pool"] = float(np.mean(inpool))
    print(json.dumps({k: v for k, v in out.items() if k != "systems"}, indent=1))
    json.dump(out, open(os.path.join(C.RESULTS, "p3t", "diag.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
