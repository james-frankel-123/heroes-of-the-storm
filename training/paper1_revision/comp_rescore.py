"""
Rescore every paper-1 revision evaluation under the composition-aware v2
judges (overfit2026/judges_v2.py) from the SAVED draft records. No draft is
regenerated.

  mcts       results/mcts_bench/*.json (per-draft v1 scores are stored; v2 =
             v1 + structural correction): Table VI, sim scaling (incl. J_oof),
             feature gap (B_fullwp vs K_truebase; B_oof vs K_truebase),
             operating point (F_oof vs J_oof vs E_1M, argmax root)
  tournament results/tournament_scores.npz + pair records (score_tournament
             ordering): standings, wins, Spearman, head to head
  h2h_sub    results/h2h_submitted.json stores only means and degenerate
             counts; v2 is approximated by the mean structural shift of the
             counted degenerate teams (reported as approximate)
  crosseval  results/crosseval_records.json (Table II "Ind."): v1 judges
             rescored on the saved drafts, then corrected
  greedy     results/greedy/rich_*.json (Table V greedy/MCTS rows, gN)

Usage (from training/): nice -n 19 taskset -c 48-53 python3 paper1_revision/comp_rescore.py
Output: results/comp_rescore.json
"""
import os
import sys
import json
import glob
import itertools

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)

import numpy as np

from paper1_revision import core
from overfit2026 import judges_v2
from overfit2026.structure import struct_matrix

R = core.RESULTS
JUDGES = ["gN", "gN_naive", "RN", "R17", "QM2026", "QM2021"]
GN_CACHE = os.path.join(core.CACHE, "comp_rescore_gn.npz")


def regn(key, rows):
    """gN (own-composition rebuild, comp_gn_rebuild.py) for stored drafts,
    cached by key. Stored per-draft gN values predate the rebuild."""
    z = dict(np.load(GN_CACHE)) if os.path.exists(GN_CACHE) else {}
    if key not in z or len(z[key]) != len(rows):
        z[key] = judges_v2.score_base(rows, judges=["gN"])["gN"]
        np.savez(GN_CACHE, **z)
    return z[key]
CONS = judges_v2.CONSENSUS
OUT = os.path.join(R, "comp_rescore.json")


def v1_v2(base, rows):
    """base: per-draft v1 arrays. Returns {'v1': {...+consensus}, 'v2': {...}}."""
    v1 = {j: np.asarray(base[j], float) for j in JUDGES if j in base}
    v1["consensus"] = np.mean([v1[j] for j in CONS], 0)
    v2 = judges_v2.correct(v1, rows, judges=[j for j in JUDGES if j in v1])
    return {"v1": v1, "v2": v2}


# ── MCTS benchmark ───────────────────────────────────────────────────

PAIRS = [("new:F_oof", "new:B_oof"), ("new:J_oof", "new:F_oof"), ("new:J_oof", "new:B_oof"),
         ("old:F_400sim", "old:B_fullwp"), ("old:J_800sim", "old:F_400sim"),
         ("old:I_600sim", "old:F_400sim"), ("old:J_800sim", "old:B_fullwp"),
         ("old:B_fullwp", "old:K_truebase"), ("new:B_oof", "old:K_truebase"),
         ("new:F_oof", "old:K_truebase"), ("new:J_oof", "old:K_truebase"),
         ("new:B_oof", "old:B_fullwp"), ("new:F_oof", "old:F_400sim"),
         ("new:J_oof", "old:J_800sim"), ("old:E_1M", "old:B_fullwp"),
         ("new:F_oof", "old:E_1M"), ("new:J_oof", "old:E_1M"), ("old:C_large", "old:B_fullwp"),
         ("old:N2_absolute", "old:B_fullwp"), ("old:M2_relational", "old:B_fullwp")]
REFS = [j for j in JUDGES if j != "QM2021"] + ["consensus"]   # benchmark has no QM2021
TREFS = JUDGES + ["consensus"]


def mcts():
    groups = {}
    files = sorted(glob.glob(os.path.join(R, "mcts_bench", "*.json")))
    data = [json.load(open(p)) for p in files]
    rows_all = [(x["our"], x["opp"], x["map"], "mid") for d in data for x in d["drafts"]]
    gn_all = regn("mcts_bench", rows_all)
    off = 0
    for p, d in zip(files, data):
        kind, name = d["run"].split(":", 1)
        cfg = f"{kind}:{name.rsplit('_s', 1)[0]}"
        rows = [(x["our"], x["opp"], x["map"], "mid") for x in d["drafts"]]
        base = {k: v for k, v in d["per_draft"].items() if k != "QM2021"}   # QM2021 key held QM2022
        base["gN"] = gn_all[off:off + len(rows)]
        off += len(rows)
        sc = v1_v2(base, rows)
        groups.setdefault((cfg, d["temp"]), []).append(
            {"run": d["run"], "v1": {k: float(v.mean()) for k, v in sc["v1"].items()},
             "v2": {k: float(v.mean()) for k, v in sc["v2"].items()},
             "degen": float(np.mean(d["per_draft"]["degen"])),
             "proxy": float(np.mean(d["per_draft"]["proxy_rev" if kind == "new" else "proxy_sub"]))})
    per = {}
    configs = {}
    for (cfg, T), runs in groups.items():
        k = f"{cfg}|T{T:g}"
        runs = sorted(runs, key=lambda r: r["run"])
        per[k] = {v: {r: np.array([x[v][r] for x in runs]) for r in REFS} for v in ("v1", "v2")}
        configs[k] = {"n_seeds": len(runs),
                      "degen": float(np.mean([x["degen"] for x in runs])),
                      "proxy": float(np.mean([x["proxy"] for x in runs])),
                      **{v: {r: float(per[k][v][r].mean()) for r in REFS} for v in ("v1", "v2")}}
    contrasts = {}
    for T in ("1", "0"):
        for a, b in PAIRS:
            ka, kb = f"{a}|T{T}", f"{b}|T{T}"
            if ka not in per or kb not in per:
                continue
            c = {}
            for v in ("v1", "v2"):
                c[v] = {}
                for r in REFS:
                    x, y = per[ka][v][r], per[kb][v][r]
                    se = float(np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y)))
                    c[v][r] = [float(x.mean() - y.mean()), se]
            contrasts[f"{a} - {b}|T{T}"] = c
    for cfg in {k.split("|")[0] for k in per}:
        k1, k0 = f"{cfg}|T1", f"{cfg}|T0"
        if k1 in per and k0 in per and len(per[k1]["v1"]["gN"]) == len(per[k0]["v1"]["gN"]):
            c = {}
            for v in ("v1", "v2"):
                c[v] = {}
                for r in REFS:
                    d = per[k0][v][r] - per[k1][v][r]
                    c[v][r] = [float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))]
            contrasts[f"{cfg}: T0 - T1"] = c
    return {"configs": configs, "contrasts": contrasts}


# ── Tournament ───────────────────────────────────────────────────────

def tournament():
    from paper1_revision.tournament import STRATEGIES
    from paper1_revision.score_tournament import load_pair, pair_stats, spearman
    sc = dict(np.load(os.path.join(R, "tournament_scores.npz")))
    # tournament_scores.npz holds the pairs of the stored 12-strategy run in
    # this order; standings are computed over the current STRATEGIES only
    stored = ["mcts", "constrained_mcts", "enriched", "enriched_aug", "constrained_greedy",
              "k_truebase", "gourdeau", "gourdeau_disc", "cql_naive_a1.0", "cql_enr_a2.0",
              "gd", "mcq_t0.5"]
    rows_all, idx, pairs = [], {}, {}
    for a, b in itertools.permutations(stored, 2):
        d = load_pair(a, b)
        if d is None:
            continue
        recs = d["records"]
        rows = [(r["team0"]["picks"], r["team1"]["picks"], r["game_map"], r["tier"]) for r in recs]
        idx[(a, b)] = (len(rows_all), len(rows_all) + len(rows))
        rows_all += rows
        pairs[(a, b)] = recs
    assert len(sc["gN"]) == len(rows_all)
    sc["gN"] = regn("tournament", rows_all)
    S = v1_v2(sc, rows_all)
    out = {}
    for v in ("v1", "v2"):
        per = {s: {j: [] for j in TREFS} for s in STRATEGIES}
        h2h = {}
        for (a, b), (i0, i1) in idx.items():
            if a not in STRATEGIES or b not in STRATEGIES:
                continue
            cfg = [(r["game_map"], r["tier"]) for r in pairs[(a, b)]]
            for j in TREFS:
                mu, var = pair_stats(S[v][j][i0:i1], cfg)
                per[a][j].append((mu, var))
                per[b][j].append((1 - mu, var))
                h2h.setdefault(j, {})[(a, b)] = (mu, var)
        stand = {}
        for s in STRATEGIES:
            stand[s] = {}
            for j in TREFS:
                mus = np.array([x[0] for x in per[s][j]])
                vs = np.array([x[1] for x in per[s][j]])
                stand[s][j] = {"mean": float(mus.mean()), "se": float(np.sqrt(vs.sum()) / len(vs)),
                               "wins": int((mus > 0.5).sum())}
        cons = [stand[s]["consensus"]["mean"] for s in STRATEGIES]
        top6 = sorted(STRATEGIES, key=lambda s: -stand[s]["consensus"]["mean"])[:6]
        rc = {j: {"all": spearman([stand[s][j]["mean"] for s in STRATEGIES], cons),
                  "top6": spearman([stand[s][j]["mean"] for s in top6],
                                   [stand[s]["consensus"]["mean"] for s in top6])} for j in TREFS}
        hh = {}
        for a, b in (("constrained_mcts", "mcts"), ("constrained_greedy", "enriched"),
                     ("mcts", "k_truebase"), ("constrained_mcts", "k_truebase"),
                     ):
            hh[f"{a} vs {b}"] = {}
            for j in TREFS:
                m1, v1_ = h2h[j][(a, b)]
                m2, v2_ = h2h[j][(b, a)]
                hh[f"{a} vs {b}"][j] = [0.5 * (m1 + 1 - m2), 0.5 * float(np.sqrt(v1_ + v2_))]
        out[v] = {"standings": stand, "spearman": rc, "top6": top6, "h2h": hh,
                  "order": sorted(STRATEGIES, key=lambda s: -stand[s]["consensus"]["mean"])}
    return out


# ── Submitted matched h2h (approximate) ──────────────────────────────

def h2h_submitted(mcts_res):
    """Only means and degenerate counts were stored. Approximate the v2 shift
    with the mean logit shift a degenerate team of the submitted agent's kind
    gets (no-healer, the dominant type for J_800sim) at the stored mean."""
    d = json.load(open(os.path.join(R, "h2h_submitted.json")))
    beta = judges_v2.params()
    out = {}
    for key, deg_key in (("constrained__plain", "team1"), ("plain__constrained", "team0")):
        n_deg = d[key]["degen"][deg_key]
        res = {}
        for j in CONS:
            p = d[key]["team0_p"][j]
            z = np.log(p / (1 - p))
            # p*(team0) = sigmoid(z + beta.(s0 - s1)): team0 degenerate adds
            # +beta, team1 degenerate adds -beta, on n_deg of 200 drafts
            sgn = 1.0 if deg_key == "team0" else -1.0
            shift = sgn * float(beta[j][0])
            p_deg = 1 / (1 + np.exp(-(z + shift)))
            res[j] = p + (n_deg / 200.0) * (p_deg - p)
        res["consensus"] = float(np.mean([res[j] for j in CONS]))
        out[key] = res
    c = {j: 0.5 * (out["constrained__plain"][j] + 1 - out["plain__constrained"][j])
         for j in CONS + ["consensus"]}
    return {"approx_v2_constrained_mean_p": c,
            "v1_constrained_mean_p": d["constrained_mean_p"],
            "note": "approximate: drafts not stored; no-healer beta applied to the stored degenerate counts"}


# ── Greedy cross-evaluation and rich evaluation ──────────────────────

def crosseval():
    recs = json.load(open(os.path.join(R, "crosseval_records.json")))
    rows = [(r["our_picks"], r["opp_picks"], r["game_map"], r["skill_tier"]) for r in recs]
    base = judges_v2.score_base(rows, judges=CONS)
    S = v1_v2(base, rows)
    out = {}
    for m in sorted({r["wp_model"] for r in recs}):
        sel = np.array([r["wp_model"] == m for r in recs])
        out[m] = {v: {j: float(S[v][j][sel].mean()) for j in S[v]} for v in ("v1", "v2")}
        out[m]["degen"] = float(struct_matrix([r[0] for r, s in zip(rows, sel) if s]).max(1).mean())
    return out


def greedy():
    out = {}
    for name in ("enriched", "enriched_aug", "constrained_greedy", "mcts", "constrained_mcts"):
        rows = []
        for p in sorted(glob.glob(os.path.join(R, "greedy", f"rich_{name}_s*.json"))):
            rows += [(x["our"], x["opp"], x["map"], x["tier"]) for x in json.load(open(p))]
        if not rows:
            continue
        base = judges_v2.score_base(rows, judges=CONS)
        S = v1_v2(base, rows)
        out[name] = {"n": len(rows), **{v: {j: float(S[v][j].mean()) for j in S[v]} for v in ("v1", "v2")}}
    return out


def main():
    res = {}
    res["mcts"] = mcts()
    print("mcts done", flush=True)
    res["tournament"] = tournament()
    print("tournament done", flush=True)
    res["h2h_submitted"] = h2h_submitted(res["mcts"])
    res["crosseval"] = crosseval()
    print("crosseval done", flush=True)
    res["greedy"] = greedy()
    res["judges"] = {"variant": json.load(open(judges_v2.PARAMS))["chosen"],
                     "beta": {k: v.tolist() for k, v in judges_v2.params().items()}}
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
