"""
Audit fixes P3-01, P3-15, P3-25: the drafters' "realized agreement" check,
restated as team-level calibration of the skill model.

Why: agreement ranks the real picks by a value that contains the skill term
s, and is scored against y - WP_pop, the residual s was fit to predict. So a
high agreement quintile re-tests the skill model (AUDIT P3-01). What would be
evidence for the drafter is a REMAINDER: realized minus the skill model's own
predicted gap for the same real draft.

For every drafter run under the fixed protocol (cache/fix/), at the real
pick states of its held-out lobbies:
  agreement (one definition for all drafters, P3-25)
      pers  = team mean of the real picks' percentile in the personalized
              ranking; pop = the same in the population ranking;
      personal component = pers - pop
  realized   = y_team - WP_pop(team) of the real game
  predicted  = V(real draft) - WP_pop(team), V with the fixed combiner and the
               fixed personal tables (the model's own forecast of the gap)
  remainder  = realized - predicted
Quintile means of the three, Q5 - Q1 with a GAME bootstrap (both teams of a
game resampled together, P3-15), and the OLS of realized on agreement and
predicted gap (agreement coefficient, game bootstrap).

Run (from training/): OMP_NUM_THREADS=4 nice -n 19 taskset -c 48-63 python3 personalization/p3_fix_realized.py
     With --old: the same on the pre-audit runs (cache/), with the pre-audit
     tables and combiner as the predicted gap.
Output: results/fix/p3_fix_realized.json (--old: p3_fix_realized_old.json)
"""
import os
import sys
import gzip
import json
import pickle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import p3_hs_core as C
import p3_mcts_core as M
import p3_fix_drafters as FD

FIXC = FD.FIX_CACHE
OLD = "--old" in sys.argv  # pre-audit runs, tables and combiner (cache/), for the old-to-new record
if OLD:
    FIXC = C.CACHE


def lp(name):
    p = os.path.join(FIXC, name)
    return pickle.load(gzip.open(p)) if os.path.exists(p) else None


class Ctx:
    def __init__(self):
        import p3_dr_core as D
        if not OLD:
            D.PTAB = FD.FIX_PTAB
        self.d = C.load_slots()
        self.T = D.load_personal()
        self.L = D.load_lobbies()
        gd = D.GDPolicy(self.d["hero_names"])
        self.to_sh = gd.to_shared
        self.from_sh = np.argsort(self.to_sh)
        self.b = D.combiner() if OLD else FD.fixed_combiner()
        w = np.load(C.WP)
        o = np.argsort(w["replay_ids"])
        self.y, self.wp = w["y"][o], w["wp0"][o]

    def team_pred_gap(self, gi):
        """Predicted gap V - WP_pop for real team 0 and team 1 of lobby gi."""
        st = self.L["steps"][gi]
        S, O = [0.0, 0.0], [0.0, 0.0]
        for h, ty, tm, row in st:
            if ty == 1:
                p = self.T["pos"][int(row)]
                S[tm] += float(self.T["s"][p][h])
                O[tm] += float(self.T["off"][p][h])
        g = self.L["g"][gi]
        wp0 = min(max(float(self.wp[g]), 1e-6), 1 - 1e-6)
        z = self.b[1] * np.log(wp0 / (1 - wp0)) + self.b[2] * (S[0] - S[1]) + self.b[3] * (O[0] - O[1])
        V0 = 1 / (1 + np.exp(-z))
        return (V0 - wp0, -(V0 - wp0)), (float(self.y[g]) - wp0, -(float(self.y[g]) - wp0))


def pct(score, cand, actual):
    key = np.array([score[c] for c in cand])
    ka = score[actual]
    return ((key < ka).sum() + 0.5 * ((key == ka).sum() - 1)) / max(len(cand) - 1, 1)


def teams_onestep(X, z):
    recs = []
    for l in z["lobbies"]:
        gi = l["gi"]
        (pg0, pg1), (r0, r1) = X.team_pred_gap(gi)
        for t, pg, rr in ((0, pg0, r0), (1, pg1, r1)):
            pp, pq = [], []
            for dd in l["real"]["decisions"]:
                if dd["team"] != t or len(dd["cand"]) < 2:
                    continue
                cand = list(range(len(dd["cand"])))
                i = int(np.flatnonzero(dd["cand"] == dd["actual"])[0])
                pp.append(pct(dd["V"], cand, i))
                pq.append(pct(dd["wp"], cand, i))
            if len(pp) >= 3:
                recs.append((gi, t, np.mean(pp), np.mean(pq), rr, pg))
    return recs


def teams_mcts(X, lobs, decs_pers, decs_pop):
    """decs: dict (li, k) -> (pol, q) in shared order."""
    recs = []
    for li, lob in enumerate(lobs):
        gi = lob["gi"]
        (pg0, pg1), (r0, r1) = X.team_pred_gap(gi)
        per = {0: ([], []), 1: ([], [])}
        for k in range(16):
            if not M.IS_PICK[k] or (li, k) not in decs_pers or (li, k) not in decs_pop:
                continue
            actual = int(lob["acts_real"][k])
            sl = int(lob["slot_of"][k])
            row = int(lob["rows"][sl])
            pool = X.T["pool"][X.T["pos"][row]]
            taken = set(int(a) for a in lob["acts_real"][:k])
            cand = [int(X.to_sh[h]) for h in np.flatnonzero(pool) if int(X.to_sh[h]) not in taken]
            if actual not in cand:
                cand.append(actual)
            if len(cand) < 2:
                continue
            real_team = M.DRAFT_TEAM[k] ^ lob["first"]
            for j, dd in ((0, decs_pers), (1, decs_pop)):
                pol, q = dd[(li, k)]
                sc = {c: pol[c] + 1e-3 * q[c] for c in cand}
                per[real_team][j].append(pct(sc, cand, actual))
        for t, pg, rr in ((0, pg0, r0), (1, pg1, r1)):
            if len(per[t][0]) >= 3:
                recs.append((gi, t, np.mean(per[t][0]), np.mean(per[t][1]), rr, pg))
    return recs


def summarize(recs, rng, nboot=500):
    A = np.array(recs, float)
    gi = A[:, 0].astype(np.int64)
    ag = {"personalized agreement": A[:, 2], "population agreement": A[:, 3],
          "personal component": A[:, 2] - A[:, 3]}
    real, pred = A[:, 4], A[:, 5]
    rem = real - pred
    ug, inv = np.unique(gi, return_inverse=True)
    out = {"teams": int(len(A)), "games": int(len(ug))}
    W = [np.bincount(rng.randint(0, len(ug), len(ug)), minlength=len(ug))[inv].astype(float) for _ in range(nboot)]
    for nm, a in ag.items():
        qs = np.percentile(a, [20, 40, 60, 80])
        b = np.searchsorted(qs, a)
        r = {}
        for lab, val in (("realized", real), ("predicted by the skill model", pred), ("remainder", rem)):
            qm = [float(100 * val[b == q].mean()) for q in range(5)]
            diff = lambda w: 100 * (np.average(val[b == 4], weights=w[b == 4]) - np.average(val[b == 0], weights=w[b == 0]))
            bs = [diff(w) for w in W]
            r[lab] = {"quintile_means_pp": qm, "Q5_minus_Q1_pp": [qm[4] - qm[0], float(np.percentile(bs, 2.5)),
                                                                  float(np.percentile(bs, 97.5))]}
        X_ = np.column_stack([np.ones(len(a)), a - a.mean(), pred - pred.mean()])
        beta = np.linalg.lstsq(X_, real, rcond=None)[0]
        bs = []
        for w in W[:300]:
            sw = np.sqrt(w)
            bs.append(np.linalg.lstsq(X_ * sw[:, None], real * sw, rcond=None)[0])
        bs = np.array(bs)
        r["ols realized ~ agreement + predicted gap"] = {
            "agreement_pp_per_unit": [float(100 * beta[1]), float(100 * np.percentile(bs[:, 1], 2.5)),
                                      float(100 * np.percentile(bs[:, 1], 97.5))],
            "predicted_gap_coefficient": [float(beta[2]), float(np.percentile(bs[:, 2], 2.5)),
                                          float(np.percentile(bs[:, 2], 97.5))]}
        out[nm] = r
    return out


def main():
    rng = np.random.RandomState(0)
    X = Ctx()
    out = {}
    z = lp("dr_runs.pkl.gz")
    if z is not None:
        out["one-step"] = summarize(teams_onestep(X, z), rng)
        print("one-step", json.dumps(out["one-step"]["personalized agreement"]), flush=True)
    mr = lp("mcts_real_s400.pkl.gz")
    if mr is not None:
        dp = {(li, k): (pol, q) for li, k, pers, pol, q in mr["decisions"] if pers == 1}
        dq = {(li, k): (pol, q) for li, k, pers, pol, q in mr["decisions"] if pers == 0}
        out["MCTS 400, BC prior"] = summarize(teams_mcts(X, mr["lobbies"], dp, dq), rng)
        print("mcts", json.dumps(out["MCTS 400, BC prior"]["personalized agreement"]), flush=True)
        ds = lp("dsmcts_real_s400.pkl.gz")
        if ds is not None:
            n = len(ds["lobbies"])
            assert [l["gi"] for l in ds["lobbies"]] == [l["gi"] for l in mr["lobbies"][:n]]
            dsp = {(li, k): (pol, q) for li, k, pers, pol, q in ds["decisions"]}
            out["MCTS 400, distilled prior"] = summarize(teams_mcts(X, ds["lobbies"], dsp, dq), rng)
    for sims in (25, 100, 1500):
        m2 = lp(f"mcts_real_s{sims}.pkl.gz")
        if m2 is not None:
            a2 = {(li, k): (pol, q) for li, k, pers, pol, q in m2["decisions"] if pers == 1}
            b2 = {(li, k): (pol, q) for li, k, pers, pol, q in m2["decisions"] if pers == 0}
            out[f"MCTS {sims}, BC prior"] = summarize(teams_mcts(X, m2["lobbies"], a2, b2), rng)
    d100 = lp("dsmcts_real_s100.pkl.gz")
    m100 = lp("mcts_real_s100.pkl.gz")
    if d100 is not None and m100 is not None:
        q100 = {(li, k): (pol, q) for li, k, pers, pol, q in m100["decisions"] if pers == 0}
        dsp = {(li, k): (pol, q) for li, k, pers, pol, q in d100["decisions"]}
        out["MCTS 100, distilled prior"] = summarize(teams_mcts(X, d100["lobbies"], dsp, q100), rng)
    gr = lp("dsgreedy_real.pkl.gz")
    if gr is not None:
        gp = {(li, k): (pol, q) for li, k, pers, pol, q in gr["decisions"] if pers == 1}
        gq = {(li, k): (pol, q) for li, k, pers, pol, q in gr["decisions"] if pers == 0}
        out["greedy distilled (vs greedy BC)"] = summarize(teams_mcts(X, gr["lobbies"], gp, gq), rng)
    pg = lp("pgdmcts_real_s400.pkl.gz")
    if pg is not None:
        a = {(li, k): (pol, q) for li, k, pol, q in pg["R2 pers-PGD"]}
        b = {(li, k): (pol, q) for li, k, pol, q in pg["P2 pop-PGD"]}
        out["MCTS 400, personal-GD opponents"] = summarize(teams_mcts(X, pg["lobbies"], a, b), rng)
    for k, v in out.items():
        print(k, {a: v[a]["remainder"]["Q5_minus_Q1_pp"] for a in ("personalized agreement", "personal component")},
              flush=True)
    os.makedirs(FD.FIX_RESULTS, exist_ok=True)
    with open(os.path.join(FD.FIX_RESULTS, "p3_fix_realized_old.json" if OLD else "p3_fix_realized.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)


if __name__ == "__main__":
    main()
