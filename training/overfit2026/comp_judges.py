"""
Composition audit, steps 1 and 3: (a) the held-out calibration audit of every
independent judge on real degenerate teams, by type; (b) fitting the v2
structural corrections; (c) validating them on held-out real games.

(a) Audit. Each judge is scored only on games it never trained on
    (comp_audit_data.py): SNAP for all judges; T17 for the no-drift judges and
    QM; N for R17 and QM; in-family cross-fits RN_cf (N) and R17_cf (T17).
    Per team type: realized WR, mean prediction, gap, and the gap after a
    global slope/intercept recalibration of the judge on all teams of that set
    (separates blindness to structure from overall over/under-confidence).

(b) Fit. beta_J (3 structure coefficients, judge prediction as fixed offset)
    on the no-drift post-snapshot games N (no agent saw them), always with a
    prediction the judge family makes for a game it did not train on:
      gN, gN_naive  two-fold refits of the same recipe on halves of N
                    (comp_gn_folds.py), each scoring the other half
      RN            the two-way cross-fit RN_cf
      QM2026/QM2021, R17  held out on N as they are
    A first fit on the drifted 2.55.17 games under-corrected snapshot and N
    games (the penalty is smaller in that meta); it is kept in
    results/comp_judges_t17fit.json.
    variants
      realized  outcome ~ offset + structure
      causal    outcome ~ offset + structure + skill controls (M2 of
                comp_causal.py: team skill difference from the "clean" P3
                state, skill by within-team pick position, first-pick side),
                fitted on the subset with full player rows; only beta is kept,
                so v2 judges a draft as played by equally skilled teams.

(c) Validation on SNAP (1.95M games, no judge and no correction saw them):
    log loss and accuracy, v1 vs v2; gap by type; and, for the causal variant,
    the gap after conditioning on the same skill controls (games from
    2025-06-01 with player rows). Drift check on the 2.55.17 games (T17).

Usage (from training/): nice -n 19 taskset -c 48-53 python3 overfit2026/comp_judges.py
Outputs: overfit2026/results/comp_audit.json, overfit2026/results/comp_judges.json
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np

from overfit2026.comp_common import (CONSENSUS, fit_offset, gap_table, logit, logloss, sig,
                                     teams)
from overfit2026.comp_causal import controls
from overfit2026.structure import STRUCT_NAMES

from overfit2026 import data as _odata
CACHE = _odata.art(HERE, "cache")
RES = _odata.art(HERE, "results")
HELD = {"SNAP": ["gN", "gN_naive", "RN", "R17", "QM2026", "QM2021", "consensus"],
        "T17": ["gN", "gN_naive", "RN", "QM2026", "QM2021", "consensus", "R17_cf"],
        "N": ["gN_oof", "gN_naive_oof", "RN_cf", "R17", "QM2026", "QM2021", "consensus_oof"]}
FIT_SET = {j: "N" for j in ("gN", "gN_naive", "RN", "QM2026", "QM2021", "R17")}
# offset on N: held-out prediction of the judge family on each N game
FIT_OFFSET = {"gN": "gN_oof", "gN_naive": "gN_naive_oof", "RN": "RN_cf",
              "QM2026": "QM2026", "QM2021": "QM2021", "R17": "R17"}


def load(name):
    z = dict(np.load(os.path.join(CACHE, f"comp_audit_{name}.npz")))
    qpath = os.path.join(CACHE, f"comp_audit_{name}_qm2021.npy")
    if not os.path.exists(qpath):
        z["QM2021"] = qm2021_scores(name, z)
        np.save(qpath, z["QM2021"])
    else:
        z["QM2021"] = np.load(qpath)
    z["consensus"] = np.mean([z[k] for k in CONSENSUS], 0)
    if name == "N":
        o = dict(np.load(os.path.join(CACHE, "comp_gn_oof_N.npz")))
        assert np.array_equal(o["rid"], z["rid"]) and np.array_equal(o["y"], z["y"])
        z["gN_oof"], z["gN_naive_oof"] = o["gN"], o["gN_naive"]
        z["consensus_oof"] = np.mean([z[k] for k in ("gN_oof", "gN_naive_oof", "RN_cf", "QM2026")], 0)
    return z


def qm2021_scores(name, z):
    import torch
    from overfit2026 import data, gold
    from overfit2026.stage_b_saved import qm_scores
    from paper1_revision.score_tournament import qm2021
    torch.set_num_threads(4)
    games = (data.load_snapshot() if name == "SNAP"
             else gold.GOLD_SETS["NODRIFT" if name == "N" else "T17"](data))
    assert np.array_equal(np.array([g[0] for g in games]), z["rid"])
    m = qm2021()
    rows = [(g[3], g[4], g[2], g[1]) for g in games]
    return np.concatenate([qm_scores({"q": m}, rows[i:i + 200000])["q"]
                           for i in range(0, len(rows), 200000)])


def causal_join(z):
    """Rows of z with full player data -> (mask over z, controls matrix)."""
    c = dict(np.load(os.path.join(CACHE, "comp_causal_games.npz")))
    o = np.argsort(c["rid"])
    rid = c["rid"][o]
    j = np.searchsorted(rid, z["rid"])
    ok = (j < len(rid)) & (rid[np.minimum(j, len(rid) - 1)] == z["rid"])
    jj = o[j[ok]]
    sub = {k: v[jj] for k, v in c.items()}
    good = sub["rank_ok"] & (sub["first_pick_team"] >= 0)
    # orientation: the slot table's team0 outcome equals the slim game's
    agree = float(np.mean(sub["y"] == z["y"][ok]))
    assert agree > 0.999, agree
    mask = np.zeros(len(z["y"]), bool)
    mask[np.flatnonzero(ok)[good]] = True
    C = controls({k: v[good] for k, v in sub.items()}, 2, "clean")
    return mask, C


def corrected(p, z, beta):
    return sig(logit(p) + (z["s0"] - z["s1"]) @ beta)


def main():
    sets = {n: load(n) for n in ("SNAP", "T17", "N")}
    # (a) audit
    audit = {n: {j: gap_table(z[j], z["y"], z["s0"], z["s1"]) for j in HELD[n]}
             for n, z in sets.items()}
    json.dump(audit, open(os.path.join(RES, "comp_audit.json"), "w"), indent=1)
    print("AUDIT gap pp (realized - predicted) / after recalibration")
    for n in audit:
        for j, t in audit[n].items():
            print(f"  {n:5s} {j:10s} ll={t['logloss']:.4f} slope={t['slope']:.2f} "
                  + " ".join(f"{k}:{v['gap_pp']:+.1f}±{v['se_pp']:.1f}/{v['gap_recal_pp']:+.1f}"
                             for k, v in t["by_kind"].items() if k != "ok")
                  + f" ok:{t['by_kind']['ok']['gap_pp']:+.2f}")
    # (b) fit
    beta = {"realized": {}, "causal": {}}
    fit_info = {}
    for j, fs in FIT_SET.items():
        z = sets[fs]
        dS = z["s0"] - z["s1"]
        off = logit(z[FIT_OFFSET[j]])
        w, se = fit_offset(dS, z["y"], off)
        beta["realized"][j] = w[1:4].tolist()
        m, C = causal_join(z)
        wc, sec = fit_offset(np.column_stack([dS[m], C]), z["y"][m], off[m])
        wr, ser = fit_offset(dS[m], z["y"][m], off[m])   # same subset, no controls
        beta["causal"][j] = wc[1:4].tolist()
        fit_info[j] = {"fit_set": fs, "n": int(len(z["y"])), "n_causal": int(m.sum()),
                       "realized": {"beta": w[1:4].tolist(), "se": se[1:4].tolist()},
                       "realized_same_subset": {"beta": wr[1:4].tolist(), "se": ser[1:4].tolist()},
                       "causal": {"beta": wc[1:4].tolist(), "se": sec[1:4].tolist(),
                                  "controls": wc[4:].tolist()}}
        print(f"FIT {j:9s} on {fs} n={len(z['y']):,} realized {np.round(w[1:4], 3)} "
              f"(se {np.round(se[1:4], 3)}) | subset n={m.sum():,} no-ctrl {np.round(wr[1:4], 3)} "
              f"causal {np.round(wc[1:4], 3)}", flush=True)
    # diagnostic fits on SNAP (never used): stability of beta across eras
    diag = {}
    for j in ("gN", "gN_naive", "RN", "QM2026"):
        z = sets["SNAP"]
        w, se = fit_offset(z["s0"] - z["s1"], z["y"], logit(z[j]))
        diag[j] = {"beta": w[1:4].tolist(), "se": se[1:4].tolist()}
    z = sets["N"]
    w, se = fit_offset(z["s0"] - z["s1"], z["y"], logit(z["RN_cf"]))
    diag["RN_cf_on_N"] = {"beta": w[1:4].tolist(), "se": se[1:4].tolist()}
    print("diagnostic beta (SNAP; RN_cf on N):", {k: np.round(v["beta"], 3).tolist() for k, v in diag.items()})
    # (c) validation
    val = {}
    zS = sets["SNAP"]
    mS, CS = causal_join(zS)
    for variant in ("realized", "causal"):
        val[variant] = {}
        b = {k: np.array(v) for k, v in beta[variant].items()}
        cor = {j: corrected(zS[j], zS, b[j]) for j in b}
        cor["consensus"] = np.mean([cor[j] for j in CONSENSUS], 0)
        for j in list(b) + ["consensus"]:
            t = gap_table(cor[j], zS["y"], zS["s0"], zS["s1"])
            t["logloss_v1"] = logloss(zS[j], zS["y"])
            t["delta_logloss_x1e4"] = 1e4 * (t["logloss"] - t["logloss_v1"])
            # skill-conditioned calibration on the subset with player rows:
            # refit only the control coefficients (offset = v2 prediction)
            off = logit(cor[j][mS])
            wc, _ = fit_offset(CS, zS["y"][mS], off)
            pc = sig(off + wc[0] + CS @ wc[1:])
            t["skill_conditioned"] = gap_table(pc, zS["y"][mS], zS["s0"][mS], zS["s1"][mS],
                                               recal=False)["by_kind"]
            t1 = gap_table(zS[j][mS], zS["y"][mS], zS["s0"][mS], zS["s1"][mS], recal=False)
            off1 = logit(zS[j][mS])
            w1, _ = fit_offset(CS, zS["y"][mS], off1)
            p1 = sig(off1 + w1[0] + CS @ w1[1:])
            t["skill_conditioned_v1"] = gap_table(p1, zS["y"][mS], zS["s0"][mS], zS["s1"][mS],
                                                  recal=False)["by_kind"]
            val[variant][j] = t
            bk = t["by_kind"]
            print(f"VAL {variant:8s} {j:9s} ll {t['logloss_v1']:.5f}->{t['logloss']:.5f} "
                  f"({t['delta_logloss_x1e4']:+.2f}e-4) slope {t['slope']:.2f} "
                  + " ".join(f"{k}:{v['gap_pp']:+.1f}" for k, v in bk.items())
                  + " | skill-cond any: v1 "
                  f"{t['skill_conditioned_v1']['any_degenerate']['gap_pp']:+.1f} v2 "
                  f"{t['skill_conditioned']['any_degenerate']['gap_pp']:+.1f}", flush=True)
    zT = sets["T17"]
    for variant in ("realized", "causal"):
        b = {k: np.array(v) for k, v in beta[variant].items()}
        cor = {j: corrected(zT[j], zT, b[j]) for j in CONSENSUS + ["QM2021"]}
        cor["consensus"] = np.mean([cor[j] for j in CONSENSUS], 0)
        for j in cor:
            t = gap_table(cor[j], zT["y"], zT["s0"], zT["s1"])
            t["logloss_v1"] = logloss(zT[j], zT["y"])
            val[variant][f"{j}_on_T17"] = t
            print(f"VAL {variant} {j} on T17 (drifted): ll {t['logloss_v1']:.5f}->{t['logloss']:.5f} "
                  + " ".join(f"{k}:{v['gap_pp']:+.1f}" for k, v in t["by_kind"].items()))
    params = json.load(open(os.path.join(RES, "comp_judges.json"))) if os.path.exists(
        os.path.join(RES, "comp_judges.json")) else {}
    out = {"chosen": params.get("chosen", "causal"), "struct_names": STRUCT_NAMES,
           "beta": beta, "fit": fit_info, "diagnostic_beta": diag, "validation": val}
    json.dump(out, open(os.path.join(RES, "comp_judges.json"), "w"), indent=1)
    print("wrote comp_judges.json (chosen:", out["chosen"], ")")


if __name__ == "__main__":
    main()
