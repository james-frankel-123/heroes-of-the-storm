"""
How far are each strategy's final five-hero teams from the teams in the training data?
Teams only: opponent, map and tier are ignored. Every metric is also computed for
held-out human teams (the p1val validation replays, none in the training split), so
"normal" has a reference.

Per team, against all training-split teams (two per game):
  n5          games containing this exact team
  dist        fewest hero swaps to reach an observed team (0 = seen, 1 = some 4 of its
              heroes were seen together, ... ; 3 means even every triple fails)
  min4/min3/min2   count of the rarest 4-, 3-, 2-hero subset (the support behind the
              pieces a model interpolates from)
  pmi         mean pointwise mutual information of its 10 pairs, log(c_ij N / c_i c_j):
              > 0 heroes co-occur more than their popularity predicts, < 0 less
  rc          games with its role composition (HP roles, tier-agnostic)

Usage (from training/, p1site env): python3 paper1_revision/team_support.py
Output: paper1_revision/site/results/team_support.json
"""
import os
import sys
import json
import glob
from itertools import combinations

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
T = os.path.dirname(HERE)
sys.path.insert(0, T)
from rerun2026 import common
from shared import HEROES, HERO_TO_IDX, HERO_ROLE_FINE

H = len(HEROES)
HP_ROLE = {"tank": "Tank", "bruiser": "Bruiser", "healer": "Healer", "ranged_aa": "Ranged Assassin",
           "ranged_mage": "Ranged Assassin", "melee_assassin": "Melee Assassin",
           "support_utility": "Support", "varian": "Bruiser", "pusher": "Ranged Assassin",
           "unknown": "Ranged Assassin"}
ROLES = sorted(set(HP_ROLE.values()))
ROLE_IDX = np.array([ROLES.index(HP_ROLE.get(HERO_ROLE_FINE.get(h, "unknown"), "Ranged Assassin")) for h in HEROES])
RES = os.path.join(HERE, "site", "results")


def teams_of(rows):
    out = []
    for r in rows:
        for k in ("team0_heroes", "team1_heroes"):
            t = [HERO_TO_IDX.get(h) for h in (r.get(k) or [])]
            if len(t) == 5 and None not in t and len(set(t)) == 5:
                out.append(sorted(t))
    return np.array(out, dtype=np.int64)


def key(cols, A):
    k = np.zeros(len(A), np.int64)
    for c in cols:
        k = k * H + A[:, c]
    return k


class Corpus:
    def __init__(self, A):
        self.n = len(A)
        self.k5 = np.unique(key(range(5), A), return_counts=True)
        self.c4 = np.bincount(np.concatenate([key(c, A) for c in combinations(range(5), 4)]), minlength=H ** 4)
        self.c3 = np.bincount(np.concatenate([key(c, A) for c in combinations(range(5), 3)]), minlength=H ** 3)
        self.c2 = np.bincount(np.concatenate([key(c, A) for c in combinations(range(5), 2)]), minlength=H ** 2)
        self.c1 = np.bincount(A.ravel(), minlength=H)
        rk = key(range(5), np.sort(ROLE_IDX[A], 1).astype(np.int64))
        self.rc = np.unique(rk, return_counts=True)

    @staticmethod
    def _lookup(u, x):
        keys, cnt = u
        i = np.clip(np.searchsorted(keys, x), 0, len(keys) - 1)
        return np.where(keys[i] == x, cnt[i], 0)

    def metrics(self, A):
        n5 = self._lookup(self.k5, key(range(5), A))
        s4 = np.stack([self.c4[key(c, A)] for c in combinations(range(5), 4)], 1)
        s3 = np.stack([self.c3[key(c, A)] for c in combinations(range(5), 3)], 1)
        s2 = np.stack([self.c2[key(c, A)] for c in combinations(range(5), 2)], 1)
        dist = np.where(n5 > 0, 0, np.where(s4.max(1) > 0, 1, np.where(s3.max(1) > 0, 2, 3)))
        p = self.c1 / self.n  # share of teams containing each hero
        expect = np.stack([self.n * p[A[:, a]] * p[A[:, b]] for a, b in combinations(range(5), 2)], 1)
        pmi = np.log((s2 + 0.5) / expect).mean(1)  # observed vs independent co-occurrence
        rc = self._lookup(self.rc, key(range(5), np.sort(ROLE_IDX[A], 1).astype(np.int64)))
        return dict(n5=n5, dist=dist, min4=s4.min(1), min3=s3.min(1), min2=s2.min(1), pmi=pmi, rc=rc)


def strategy_teams():
    """Final teams per strategy from the tournament records (both seatings)."""
    by = {}
    for d in ("tournament", "wr_baseline/tournament", "masked_k/tournament", "td/*/tournament", "mcq_real/tournament"):
        for f in glob.glob(os.path.join(RES, d, "*.json")):
            R = json.load(open(f))
            for side in ("team0", "team1"):
                name = R[f"{side}_strategy"]
                for rec in R["records"]:
                    t = [HERO_TO_IDX[h] for h in rec[side]["picks"]]
                    by.setdefault(name, set()).add((os.path.basename(f), rec["draft"], side, tuple(sorted(t))))
    return {k: np.array([x[3] for x in v], np.int64) for k, v in by.items()}


def summarize(m, ref):
    q = lambda x, p: float(np.percentile(x, p))
    out = {"n": int(len(m["n5"])),
           "seen_exact": float((m["n5"] > 0).mean()),
           "dist_share": [float((m["dist"] == d).mean()) for d in range(4)],
           "min4_median": q(m["min4"], 50), "min3_median": q(m["min3"], 50), "min2_median": q(m["min2"], 50),
           "pmi_mean": float(m["pmi"].mean()),
           "rc_ge50": float((m["rc"] >= 50).mean()), "rc_ge1000": float((m["rc"] >= 1000).mean())}
    if ref is not None:  # share of teams rarer than the rarest 5% of held-out human teams
        for k in ("min4", "min3", "min2", "pmi"):
            out[f"{k}_below_human_p5"] = float((m[k] < np.percentile(ref[k], 5)).mean())
    return out


def main():
    common.setup()
    train, val = common.load_split()
    C = Corpus(teams_of(train))
    print(f"corpus teams {C.n:,}", flush=True)
    rng = np.random.default_rng(0)
    V = teams_of(val)
    V = V[rng.choice(len(V), min(20000, len(V)), replace=False)]
    ref = C.metrics(V)
    res = {"corpus_teams": C.n, "human_heldout": summarize(ref, None)}
    for name, A in sorted(strategy_teams().items()):
        res[name] = summarize(C.metrics(A), ref)
    res["human_heldout_p5"] = {k: float(np.percentile(ref[k], 5)) for k in ("min4", "min3", "min2", "pmi")}
    out = os.path.join(RES, "team_support.json")
    json.dump(res, open(out, "w"), indent=1)
    cols = ["n", "seen_exact", "dist_share", "min4_median", "min3_median", "min2_median", "pmi_mean", "rc_ge50",
            "min4_below_human_p5", "min2_below_human_p5", "pmi_below_human_p5"]
    for k, v in res.items():
        if isinstance(v, dict) and "n" in v:
            print(k.ljust(22), {c: (round(v[c], 3) if isinstance(v.get(c), float) else
                                    [round(x, 3) for x in v[c]] if isinstance(v.get(c), list) else v.get(c)) for c in cols})


if __name__ == "__main__":
    main()
