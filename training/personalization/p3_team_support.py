"""
Team-composition support of the P3 drafters, on paper 1's yardstick: every arm's final teams
(both seats, against the other arms of p3_ood_audit.ARMS, headline tournament files) scored
against paper 1's training-split corpus (support_mitigation.sup_corpus): share with a rarest
triple below 25 / 10 / 5, median rarest triple, mean pair PMI, share seen exactly. Then the
selected design's composition-judge scores (consensus, gN, RN) against five arms, raw and with
paper 1's binned support correction (support_mitigation judge_corr.json).

Usage (from training/): python3 personalization/p3_team_support.py --hl <copy of
cache_def_def1027_headline/p3t> -> personalization/results/p3_team_support.json
"""
import os
import sys
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
from paper1_revision import support_mitigation as sm

SP = [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1]
ST = [0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1]
PK = {t: [k for k in range(16) if SP[k] and ST[k] == t] for t in (0, 1)}
SEL = "x-combo1-dr-chance"
ARMS = [SEL, "pMCTS-cur", "aMCTS-pp", "aMCTS-cur", "a1step-cur", "p1step-cur", "DraftRec", "DraftRec-pv", "GD", "Real"]
CORR_OPPS = ["pMCTS-cur", "aMCTS-pp", "GD", "DraftRec", "Real"]
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
sg = lambda z: 1 / (1 + np.exp(-z))


def teams(hl, a, b, t):
    acts = np.load(os.path.join(hl, "tourney", f"{a}__{b}.npz"))["acts"]
    A = np.sort(acts[:, PK[t]], 1).astype(np.int64)
    assert (A >= 0).all() and (A < 90).all() and all(len(set(r)) == 5 for r in A)
    return A


def main(hl, out):
    C = sm.sup_corpus()
    res = {"support": {}, "corrected_vs": {}}
    for a in ARMS:
        T = [teams(hl, a, b, 0) for b in ARMS if b != a and os.path.exists(os.path.join(hl, "tourney", f"{a}__{b}.npz"))]
        T += [teams(hl, b, a, 1) for b in ARMS if b != a and os.path.exists(os.path.join(hl, "tourney", f"{b}__{a}.npz"))]
        m = C.metrics(np.concatenate(T))
        mn = np.asarray(m["min3"])
        res["support"][a] = {"teams": int(len(mn)), "lt25": float((mn < 25).mean()), "lt10": float((mn < 10).mean()),
                             "lt5": float((mn < 5).mean()), "min3_median": float(np.median(mn)),
                             "pmi_mean": float(np.mean(m["pmi"])), "seen_exact": float((np.asarray(m["n5"]) > 0).mean())}
        print(a, res["support"][a], flush=True)
    jc = json.load(open(os.path.join(sm.OUT, "judge_corr.json")))
    for o in CORR_OPPS:
        row = {}
        for j in ("consensus", "gN", "RN"):
            raw, cor = [], []
            for a, b, first in ((SEL, o, True), (o, SEL, False)):
                p = np.asarray(np.load(os.path.join(hl, "scores", f"{a}__{b}.npz"))[j], float)
                m0, m1 = C.metrics(teams(hl, a, b, 0)), C.metrics(teams(hl, a, b, 1))
                pc = sg(lg(p) + (sm.bin_feats(m0, jc["edges"]) - sm.bin_feats(m1, jc["edges"])) @ np.array(jc["judges"][j]["w_binned"]))
                raw.append(p if first else 1 - p)
                cor.append(pc if first else 1 - pc)
            raw, cor = np.concatenate(raw), np.concatenate(cor)
            row[j] = {"raw": float(raw.mean()), "corrected": float(cor.mean()), "se": float(cor.std(ddof=1) / np.sqrt(len(cor)))}
        res["corrected_vs"][o] = row
        print(o, row, flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hl", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "p3_team_support.json"))
    a = ap.parse_args()
    main(a.hl, a.out)
