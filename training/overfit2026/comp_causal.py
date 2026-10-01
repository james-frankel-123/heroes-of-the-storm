"""
Composition audit, step 2: is the degenerate-team penalty causal, or partly
who fields degenerate teams?

Data: overfit2026/cache/comp_causal_games.npz (comp_causal_data.py): every
snapshot game from 2025-06-01 and every post-snapshot game with all ten
player rows, with causal (lag 1 day) per-player skill (P3 "+CF rank 2" GP
plus experience offsets) and pick order.

Base judge (offset): the realized-outcome index RN where it is held out
(snapshot games: RN; 2.55.17 games: RN; post-snapshot no-drift games: the
two-way cross-fitted RN_cf), joined by replay id from comp_audit_<set>.npz.
Outcome model, team0 orientation:
  y ~ sigmoid(logit RN + c + beta . dS + controls)
dS = structure(team0) - structure(team1) (no_healer, no_frontline, stack).
Control sets, added cumulatively:
  M0 none
  M1 team skill sum difference ("clean" state: degenerate-team games never
     enter any player's skill estimate)
  M2 + skill by within-team pick position (1st..5th pick of each team), and
     first-pick side
  M3 + partied-player count and never-played-hero slot count differences
Also with the "all" skill state, and with raw outcomes (no judge offset).
Penalty in pp = mean over degenerate teams of p(fitted) - p(fitted with that
team's structure terms set to zero), i.e. the average effect on the teams
that carry it.

Usage (from training/): python3 overfit2026/comp_causal.py
Output: overfit2026/results/comp_causal.json
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np

from overfit2026.comp_common import fit_offset, logit, sig
from overfit2026.structure import STRUCT_NAMES

from overfit2026 import data as _odata
CACHE = _odata.art(HERE, "cache")
OUT = _odata.art(HERE, "results", "comp_causal.json")


def judge_by_rid():
    """rid -> held-out RN prediction (P team0 wins)."""
    rid, p, src = [], [], []
    for name, key in (("SNAP", "RN"), ("T17", "RN"), ("N", "RN_cf")):
        z = np.load(os.path.join(CACHE, f"comp_audit_{name}.npz"))
        rid.append(z["rid"])
        p.append(z[key])
        src.append(np.full(len(z["rid"]), name))
    rid, p, src = np.concatenate(rid), np.concatenate(p), np.concatenate(src)
    o = np.argsort(rid)
    return rid[o], p[o], src[o]


def controls(z, level, var="clean"):
    sk = z[f"skill_{var}"]
    cols = []
    if level >= 1:
        cols.append(sk[:, 0] - sk[:, 1])
    if level >= 2:
        pos = np.nan_to_num(z["skill_pos"])
        for k in range(5):
            cols.append(pos[:, 0, k] - pos[:, 1, k])
        cols.append((z["first_pick_team"] == 0).astype(float) - (z["first_pick_team"] == 1))
    if level >= 3:
        cols.append(z["partied"][:, 0] - z["partied"][:, 1])
        cols.append(z["never_played_slots"][:, 0] - z["never_played_slots"][:, 1])
    return np.column_stack(cols) if cols else np.zeros((len(z["y"]), 0))


def penalty(w, X, off, dS, s0, s1):
    """Average effect on degenerate teams (pp) of their own structure terms,
    by type, plus any-degenerate."""
    b = w[1:4]
    z = off + w[0] + X @ w[1:]
    p = sig(z)
    res = {}
    # team0's own terms enter +b.s0, team1's enter -b.s1
    p0_wo = sig(z - s0 @ b)                 # team0 without its own structure
    p1_wo = 1 - sig(z + s1 @ b)             # team1 P(win) without its own structure
    eff0 = p - p0_wo
    eff1 = (1 - p) - p1_wo
    eff = np.r_[eff0, eff1]
    S = np.r_[s0, s1]
    for k, n in enumerate(STRUCT_NAMES):
        m = S[:, k] > 0
        res[n] = float(100 * eff[m].mean())
    m = S.max(1) > 0
    res["any_degenerate"] = float(100 * eff[m].mean())
    return res


def run(z, off, mask, level, var="clean"):
    dS = z["s0"] - z["s1"]
    C = controls(z, level, var)
    X = np.column_stack([dS, C])[mask]
    w, se = fit_offset(X, z["y"][mask], off[mask])
    pen = penalty(w, X, off[mask], dS[mask], z["s0"][mask], z["s1"][mask])
    # SE of the any-degenerate penalty via the delta method on beta only
    # (approximated by scaling beta SEs with the mean p(1-p) on those teams)
    out = {"n_games": int(mask.sum()), "beta": dict(zip(STRUCT_NAMES, w[1:4].tolist())),
           "beta_se": dict(zip(STRUCT_NAMES, se[1:4].tolist())), "penalty_pp": pen,
           "control_coef": w[4:].tolist()}
    return out


def boot_any(z, off, mask, level, var="clean", n=100, seed=0):
    """Game-bootstrap SE of the any-degenerate penalty and of the M0-minus-Mk
    difference (paired)."""
    rng = np.random.RandomState(seed)
    idx = np.flatnonzero(mask)
    vals, diffs = [], []
    for _ in range(n):
        b = np.zeros(len(mask), bool)
        s = rng.choice(idx, len(idx))
        w = np.bincount(s, minlength=len(mask)).astype(float)
        # weighted fit via replication is costly; use row duplication
        sub = np.repeat(np.arange(len(mask)), w.astype(int))
        zz = {k: (v[sub] if hasattr(v, "shape") and v.shape[:1] == mask.shape else v)
              for k, v in z.items()}
        m = np.ones(len(sub), bool)
        a = run(zz, off[sub], m, level, var)["penalty_pp"]["any_degenerate"]
        a0 = run(zz, off[sub], m, 0, var)["penalty_pp"]["any_degenerate"]
        vals.append(a)
        diffs.append(a0 - a)
    return float(np.std(vals, ddof=1)), float(np.std(diffs, ddof=1))


def main():
    z = dict(np.load(os.path.join(CACHE, "comp_causal_games.npz")))
    rid, p, src = judge_by_rid()
    j = np.searchsorted(rid, z["rid"])
    ok = (j < len(rid)) & (rid[np.minimum(j, len(rid) - 1)] == z["rid"])
    pr = np.full(len(z["rid"]), np.nan)
    pr[ok] = p[j[ok]]
    srcg = np.full(len(z["rid"]), "", dtype=object)
    srcg[ok] = src[j[ok]]
    # orientation check: the slot table's team0 must be the slim game's team0
    base = ok & z["rank_ok"] & (z["first_pick_team"] >= 0)
    off_rn = logit(np.where(ok, pr, 0.5))
    off_raw = np.zeros(len(pr))
    out = {"n_games": int(len(pr)), "joined_judge": int(ok.sum()),
           "by_source": {s: int((srcg == s).sum()) for s in ("SNAP", "T17", "N")}}
    samples = {"snapshot_2025-06+": base & ~z["post"], "post": base & z["post"],
               "all": base}
    res = {}
    for sname, m in samples.items():
        r = {}
        for off_name, off in (("RN", off_rn), ("raw", off_raw)):
            for level in (0, 1, 2, 3):
                r[f"{off_name}|M{level}|clean"] = run(z, off, m, level, "clean")
            r[f"{off_name}|M2|all"] = run(z, off, m, 2, "all")
        res[sname] = r
        a = r["RN|M0|clean"]["penalty_pp"]["any_degenerate"]
        b = r["RN|M2|clean"]["penalty_pp"]["any_degenerate"]
        print(f"{sname:18s} n={m.sum():,} RN offset: M0 {a:+.2f}pp  M1 "
              f"{r['RN|M1|clean']['penalty_pp']['any_degenerate']:+.2f}  M2 {b:+.2f}  M3 "
              f"{r['RN|M3|clean']['penalty_pp']['any_degenerate']:+.2f}  M2(all-state) "
              f"{r['RN|M2|all']['penalty_pp']['any_degenerate']:+.2f} | raw M0 "
              f"{r['raw|M0|clean']['penalty_pp']['any_degenerate']:+.2f} raw M2 "
              f"{r['raw|M2|clean']['penalty_pp']['any_degenerate']:+.2f}", flush=True)
    out["fits"] = res
    se, se_d = boot_any(z, off_rn, samples["all"], 2, "clean", n=50)
    out["bootstrap_all_RN_M2"] = {"se_penalty_pp": se, "se_selection_share_pp": se_d}
    print("bootstrap SE (all, RN, M2):", out["bootstrap_all_RN_M2"], flush=True)
    # who fields degenerate teams: mean skill of degenerate vs normal teams
    S = np.r_[z["s0"], z["s1"]]
    sk = np.r_[z["skill_clean"][:, 0], z["skill_clean"][:, 1]]
    m = np.r_[base, base]
    dg = S.max(1) > 0
    out["team_skill_pp"] = {"degenerate": float(100 * sk[m & dg].mean()),
                            "normal": float(100 * sk[m & ~dg].mean())}
    print("team skill (sum of 5 slots, pp):", out["team_skill_pp"], flush=True)
    json.dump(out, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
