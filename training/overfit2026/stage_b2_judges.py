"""
Stage B2: rescore the saved paper-1 drafts (diversity dumps + tournament)
with learned judges trained ONLY on post-snapshot no-drift games
(backfill + 2.55.16.97039, 291,837 games; statistics from those games, OOF
for training rows): gN = mean of 3 enriched OOF judges, gN_naive = hero-identity
judge. Neither shares a single game with the paper proxy.
Output: results/stage_b2.json
"""
import os
import sys
import glob
import json

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)

import numpy as np

from overfit2026 import score

RR = os.path.join(TRAINING_DIR, "rerun2026", "results")
GN = ["gN8_oof_s0", "gN8_oof_s1", "gN8_oof_s2"]


def judge(rows):
    ms = score.model_scores(GN + ["gN8_naive"], rows)
    return {"gN": np.mean([ms[n] for n in GN], 0), "gN_naive": ms["gN8_naive"]}


def main():
    out = {"dumps": {}, "tournament": {}}
    for p in sorted(glob.glob(os.path.join(RR, "diversity", "*__*.json"))):
        d = json.load(open(p))
        if "drafts" not in d:
            continue
        rows = [(tuple(x["our"]), tuple(x["opp"]), x["map"], "mid") for x in d["drafts"]]
        sc = judge(rows)
        out["dumps"][os.path.basename(p)[:-5]] = {k: v.tolist() for k, v in sc.items()}
    files = (glob.glob(os.path.join(RR, "roundrobin", "*.json"))
             + glob.glob(os.path.join(RR, "constrained", "roundrobin", "*.json")))
    for p in sorted(files):
        d = json.load(open(p))
        pair = f"{d['team0_strategy']}__{d['team1_strategy']}"
        if pair in out["tournament"]:
            continue
        rows = [(tuple(r["team0"]["picks"]), tuple(r["team1"]["picks"]), r["game_map"], r["tier"])
                for r in d["records"]]
        sc = judge(rows)
        out["tournament"][pair] = {k: float(v.mean()) for k, v in sc.items()}
    json.dump(out, open(os.path.join(HERE, "results", "stage_b2.json"), "w"))


if __name__ == "__main__":
    main()
