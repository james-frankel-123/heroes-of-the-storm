"""
P3 personalized GD, stage 5: analysis of the MCTS runs with the personalized
GD (cache/pgdmcts_real_s400.pkl.gz, cache/pgdmcts_full_s400.pkl.gz).

At the real pick and ban states of 2,000 held-out lobbies:
  - how often the recommendation changes when opponents are modeled as
    themselves (R2, R3 vs R1)
  - personal gain per decision: Q in the personalized tree of the
    personalized choice minus the population choice under the same opponent
    model (R1 vs P1; R2 vs P2), for picks and for bans (the redone ban value),
    overall and when a one-trick opponent is still to pick with his main free
  - share of recommended bans that hit a remaining opponent's main
Full drafts (1,000 lobbies): controlled-team V and WP_pop vs the population
drafter with the same opponent model; self-play meta breadth and off-role.
The realized-agreement check is not used (AUDIT_P3 A1).
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
from p3_mcts_analyze import Ctx, ci_mean, eff


def main():
    rng = np.random.RandomState(0)
    X = Ctx()
    R = pickle.load(gzip.open(os.path.join(C.CACHE, "pgdmcts_real_s400.pkl.gz")))
    F = pickle.load(gzip.open(os.path.join(C.CACHE, "pgdmcts_full_s400.pkl.gz")))
    lobs = R["lobbies"]
    names = ["R1 pers-GD", "R2 pers-PGD", "R3 pers-PGD+P", "P1 pop-GD", "P2 pop-PGD"]
    dec = {nm: {(li, k): (pol, q) for li, k, pol, q in R[nm]} for nm in names}
    keys = sorted(dec[names[0]])
    out = {"decisions": len(keys)}
    # opponent mains (from personal tables: counts before the day)
    def one_trick_main(li, k):
        lob = lobs[li]
        t = M.DRAFT_TEAM[k]
        mains = []
        for sl in range(10):
            ks = [j for j in range(16) if M.IS_PICK[j] and lob["slot_of"][j] == sl][0]
            if (sl < 5) == (t == 0) or ks <= k:
                continue
            n = X.T["n"][X.T["pos"][int(lob["rows"][sl])]]
            tot = n.sum()
            if tot >= 30 and n.max() / tot >= 0.5:
                mains.append(int(X.to_sh[int(np.argmax(n))]))
        taken = set(int(a) for a in lob["acts_real"][:k])
        return [m for m in mains if m not in taken]

    def opp_mains_any(li, k):
        lob = lobs[li]
        t = M.DRAFT_TEAM[k]
        mains = set()
        for sl in range(10):
            ks = [j for j in range(16) if M.IS_PICK[j] and lob["slot_of"][j] == sl][0]
            if (sl < 5) == (t == 0) or ks <= k:
                continue
            n = X.T["n"][X.T["pos"][int(lob["rows"][sl])]]
            if n.sum() >= 10:
                mains.add(int(X.to_sh[int(np.argmax(n))]))
        return mains

    rec = {}
    for kind, steps in (("picks", [k for k in range(16) if M.IS_PICK[k]]), ("bans", [k for k in range(16) if not M.IS_PICK[k]])):
        ks = [x for x in keys if x[1] in steps]
        top = {nm: np.array([int(np.argmax(dec[nm][x][0])) for x in ks]) for nm in names}
        real = np.array([int(lobs[x[0]]["acts_real"][x[1]]) for x in ks])
        r = {"states": len(ks),
             "R2 vs R1 recommendation differs": float((top["R2 pers-PGD"] != top["R1 pers-GD"]).mean()),
             "R3 vs R1 recommendation differs": float((top["R3 pers-PGD+P"] != top["R1 pers-GD"]).mean()),
             "P2 vs P1 recommendation differs": float((top["P2 pop-PGD"] != top["P1 pop-GD"]).mean()),
             "recommendation equals the real action": {nm: float((top[nm] == real).mean()) for nm in names}}
        for pers, pop in (("R1 pers-GD", "P1 pop-GD"), ("R2 pers-PGD", "P2 pop-PGD"), ("R3 pers-PGD+P", "P2 pop-PGD")):
            gains, gw, vis, ot = [], [], [], []
            for x in ks:
                pp, qp = dec[pers][x]
                pq, qq = dec[pop][x]
                a, b = int(np.argmax(pp)), int(np.argmax(pq))
                if a == b:
                    gains.append(0.0)
                    vis.append(True)
                elif pp[b] > 0:
                    gains.append(qp[a] - qp[b])
                    vis.append(True)
                else:
                    gains.append(np.nan)
                    vis.append(False)
                if kind == "bans":
                    ot.append(len(one_trick_main(*x)) > 0)
            g = np.array(gains)
            e = {"share_differ": float(np.mean([int(np.argmax(dec[pers][x][0])) != int(np.argmax(dec[pop][x][0]))
                                               for x in ks])),
                 "share_pop_choice_visited": float(np.mean(vis)),
                 "gain_pp_per_decision": ci_mean(100 * g, rng)}
            if kind == "bans":
                ot = np.array(ot)
                e["gain_pp_one_trick_opponent_with_main_free"] = ci_mean(100 * g[ot], rng)
                e["gain_pp_otherwise"] = ci_mean(100 * g[~ot], rng)
                e["share_decisions_with_one_trick_opponent"] = float(ot.mean())
                e["recommended_ban_is_a_remaining_opponents_main"] = float(np.mean(
                    [int(np.argmax(dec[pers][x][0])) in opp_mains_any(*x) for x in ks]))
                e["population_ban_is_a_remaining_opponents_main"] = float(np.mean(
                    [int(np.argmax(dec[pop][x][0])) in opp_mains_any(*x) for x in ks]))
            r[f"{pers} vs {pop}"] = e
        r["real_ban_is_a_remaining_opponents_main"] = float(np.mean([real[i] in opp_mains_any(*x)
                                                                     for i, x in enumerate(ks)])) if kind == "bans" else None
        rec[kind] = r
        print(kind, json.dumps(r), flush=True)
    out["real_states"] = rec
    # ---------- full drafts
    full = {}
    for pers, pop in (("R1 pers-GD", "P1 pop-GD"), ("R2 pers-PGD", "P2 pop-PGD"), ("R3 pers-PGD+P", "P2 pop-PGD")):
        dV, dW, vp = [], [], []
        for lob, a, b in zip(F["lobbies"], F[f"{pers}|ctrl"], F[f"{pop}|ctrl"]):
            Va, Vb = X.value(lob, a["acts"], a["wp_t0"]), X.value(lob, b["acts"], b["wp_t0"])
            Wa, Wb = a["wp_t0"], b["wp_t0"]
            if lob["kctrl"] == 1:
                Va, Vb, Wa, Wb = 1 - Va, 1 - Vb, 1 - Wa, 1 - Wb
            dV.append(Va - Vb)
            dW.append(Wa - Wb)
            vp.append(Va)
        full[f"{pers} vs {pop}"] = {"dV_pp": ci_mean(100 * np.array(dV), rng), "dWPpop_pp": ci_mean(100 * np.array(dW), rng),
                                    "mean_V_pers": float(np.mean(vp))}
    creal = np.zeros(90)
    for lob in F["lobbies"]:
        for tm, row, h in X.slot_heroes(lob, lob["acts_real"]):
            creal[h] += 1
    meta = {"real": {"effective_heroes": eff(creal)}}
    for nm in names:
        c = np.zeros(90)
        o, n = 0, 0
        for lob, r in zip(F["lobbies"], F[f"{nm}|self"]):
            for tm, row, h in X.slot_heroes(lob, r["acts"]):
                c[h] += 1
                o += X.T["off"][X.T["pos"][row]][h]
                n += 1
        sh = c / c.sum()
        meta[nm] = {"effective_heroes": eff(c), "top10_share": float(np.sort(sh)[-10:].sum()),
                    "corr_with_real": float(np.corrcoef(sh, creal / creal.sum())[0, 1]), "off_role": float(o / n)}
    out["full_drafts"] = full
    out["self_play_meta"] = meta
    print("full", json.dumps(full), "\nmeta", json.dumps(meta), flush=True)
    with open(os.path.join(C.RESULTS, "p3_pgd_mcts.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)


if __name__ == "__main__":
    main()
