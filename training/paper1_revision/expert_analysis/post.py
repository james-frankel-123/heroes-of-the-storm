"""
Expert study v6.2 (prereg v3): the non-GLMM parts and the report.

  python3 post.py <dir>          (dir holds pairs.csv, anchors.csv, raters.json, models.json)

  * S4: Bradley-Terry strengths over the 10 strategies from all machine-pair
    forced choices (a judgment credits the chosen side's strategy), compared
    with the tournament ordering by Spearman rho, one-sided exact permutation
    p over all 10! orderings.
    Tournament ordering (prereg: "round-robin tournament win-rate ordering"):
    each strategy's mean structure-corrected consensus P(win) over every
    tournament draft it played (round-robin and constrained-search files of
    ns/oct2026), from its own side. The uncorrected-consensus ordering is
    reported too.
  * S3 status: confirmatory only if >= 3 Group-A raters completed 240 items.
  * Per-rater and per-tier Wilson 95% CIs (descriptive), the gate table and
    the calibration flags.
Writes results.json and results.md into dir. Prints no outcome numbers.
"""
import os
import sys
import csv
import json
import math
import itertools
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.dirname(TRAINING)
sys.path.insert(0, TRAINING)


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return {"k": k, "n": n}
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return {"k": k, "n": n, "rate": p, "wilson_lo": c - h, "wilson_hi": c + h}


def bradley_terry(wins, strategies, iters=10000, tol=1e-12):
    """MM algorithm (Hunter 2004); log-strengths centred at 0."""
    idx = {s: i for i, s in enumerate(strategies)}
    W = np.zeros((len(strategies), len(strategies)))
    for (a, b), n in wins.items():
        W[idx[a], idx[b]] += n
    N = W + W.T
    p = np.ones(len(strategies))
    for _ in range(iters):
        denom = (N / (p[:, None] + p[None, :])).sum(1)
        new = W.sum(1) / denom
        new /= np.exp(np.log(new).mean())
        if np.max(np.abs(new - p)) < tol:
            p = new
            break
        p = new
    return dict(zip(strategies, np.log(p)))


def ranks(v):
    order = np.argsort(np.argsort(v))
    return order.astype(float)


def spearman_exact_one_sided(x, y):
    rx, ry = ranks(np.asarray(x)), ranks(np.asarray(y))
    n = len(rx)
    obs = np.corrcoef(rx, ry)[0, 1]
    d0 = ((rx - ry) ** 2).sum()
    count = total = 0
    base = np.arange(n, dtype=np.int16)
    batch = []
    for perm in itertools.permutations(range(n)):
        batch.append(perm)
        if len(batch) == 200000:
            P = np.array(batch, dtype=np.int16)
            d = ((rx[None, :] - P) ** 2).sum(1)
            count += int((d <= d0 + 1e-9).sum())
            total += len(P)
            batch = []
    if batch:
        P = np.array(batch, dtype=np.int16)
        d = ((rx[None, :] - P) ** 2).sum(1)
        count += int((d <= d0 + 1e-9).sum())
        total += len(P)
    # ry is a permutation of 0..n-1 (no ties); rho >= obs  <=>  sum d^2 <= d0
    return float(obs), count / total, total


def tournament_strength():
    from overfit2026.structure import struct_matrix
    beta = np.array(json.load(open(os.path.join(TRAINING, "paper1_revision", "results", "expert_v6",
                                                "struct_correction.json")))["beta"]["consensus"])
    ns = os.path.join(TRAINING, "rerun2026", "ns", "oct2026", "results")
    acc = defaultdict(lambda: [0.0, 0.0, 0])
    for d in (os.path.join(ns, "roundrobin"), os.path.join(ns, "constrained", "roundrobin")):
        for f in sorted(os.listdir(d)):
            if not f.endswith(".json"):
                continue
            j = json.load(open(os.path.join(d, f)))
            s0, s1, recs = j["team0_strategy"], j["team1_strategy"], j["records"]
            u = np.array([np.mean([r["wp"][e]["sym"] for e in ("naive", "herostrength", "enriched")]) for r in recs])
            D = struct_matrix([r["team0"]["picks"] for r in recs]) - struct_matrix([r["team1"]["picks"] for r in recs])
            uc = np.clip(u, 1e-6, 1 - 1e-6)
            c = 1 / (1 + np.exp(-(np.log(uc / (1 - uc)) + D @ beta)))
            for s, pc, pu in ((s0, c, u), (s1, 1 - c, 1 - u)):
                acc[s][0] += pc.sum()
                acc[s][1] += pu.sum()
                acc[s][2] += len(recs)
    return {s: {"corrected": v[0] / v[2], "uncorrected": v[1] / v[2], "drafts": v[2]} for s, v in acc.items()}


def main():
    d = sys.argv[1]
    cfg = json.load(open(os.path.join(HERE, "config.json")))
    P = list(csv.DictReader(open(os.path.join(d, "pairs.csv"))))
    A = list(csv.DictReader(open(os.path.join(d, "anchors.csv"))))
    R = json.load(open(os.path.join(d, "raters.json")))
    M = json.load(open(os.path.join(d, "models.json")))
    out = {"models": M, "raters": R}

    # S4
    wins = defaultdict(int)
    for r in P:
        wins[(r["chosen"], r["other"])] += 1
    strategies = sorted({r["s0"] for r in P} | {r["s1"] for r in P})
    pairs_present = {tuple(sorted(k)) for k in wins}
    bt = bradley_terry(wins, strategies)
    T = tournament_strength()
    missing = [s for s in strategies if s not in T]
    if missing:
        raise SystemExit(f"tournament strength missing for {missing}")
    x = [bt[s] for s in strategies]
    rho, p, nperm = spearman_exact_one_sided(x, [T[s]["corrected"] for s in strategies])
    rho_u, p_u, _ = spearman_exact_one_sided(x, [T[s]["uncorrected"] for s in strategies])
    out["S4"] = {"n_judgments": len(P), "strategies": strategies,
                 "pairings_with_judgments": len(pairs_present),
                 "bt_log_strength": bt, "tournament": T,
                 "spearman_rho": rho, "p_one_sided_exact": p, "permutations": nperm,
                 "significant": p < 0.05 and rho > 0,
                 "uncorrected_ordering": {"spearman_rho": rho_u, "p_one_sided_exact": p_u}}

    # S3 status
    completeA = [c for c, r in R["raters"].items() if r["group"] == "A" and r["complete"]]
    includedA = [c for c in R["included"] if R["raters"][c]["group"] == "A"]
    out["S3_status"] = {"group_A_completed_240": completeA, "group_A_included": includedA,
                        "confirmatory": len(completeA) >= 3}

    # descriptive: per rater / tier accuracy, per rater agreement
    per = {}
    for c in R["included"]:
        a = [r for r in A if r["rater"] == c]
        pr = [r for r in P if r["rater"] == c and abs(float(r["p_cons"]) - 0.5) > 0.02]
        per[c] = {"anchors_all": wilson(sum(int(r["correct"]) for r in a), len(a)),
                  "anchors_by_tier": {t: wilson(sum(int(r["correct"]) for r in a if r["tier"] == t),
                                                sum(r["tier"] == t for r in a)) for t in ("low", "mid", "high")},
                  "h1_agreement": wilson(sum(int(r["agree_cons"]) for r in pr), len(pr))}
    out["per_rater"] = per
    out["per_tier_pooled"] = {t: wilson(sum(int(r["correct"]) for r in A if r["tier"] == t),
                                        sum(r["tier"] == t for r in A)) for t in ("low", "mid", "high")}
    json.dump(out, open(os.path.join(d, "results.json"), "w"), indent=1, default=float)
    write_md(d, out, cfg)
    print(f"post done -> {d}/results.json, {d}/results.md")


def f(x, k=3):
    return "n/a" if x is None else (f"{x:.{k}f}" if isinstance(x, (int, float)) else str(x))


def ci(r):
    return f"{f(r.get('rate'))} [{f(r.get('wilson_lo'))}, {f(r.get('wilson_hi'))}] ({r['k']}/{r['n']})" if r.get("n") else "n/a"


def agree_line(name, t):
    if not t or "error" in t:
        return f"| {name} | – | – | – | {t.get('error', 'not run') if t else 'not run'} |"
    if "intercept" in t:
        i, pr = t["intercept"], t["prob_scale"]
        est = f"{f(i['est'])} [{f(i['ci_lo'])}, {f(i['ci_hi'])}]; p {f(i['p_two_sided'], 4)}; prob {f(pr['est'])} [{f(pr['ci_lo'])}, {f(pr['ci_hi'])}]"
    else:
        mj = t.get("majority", {})
        est = f"majority {mj.get('k')}/{mj.get('n')}, p {f(mj.get('p_two_sided'), 4)}"
    return f"| {name} | {t['n_items']} | {t['n_judgments']} | {est} | {t.get('model')}; success={t.get('success')} |"


def write_md(d, o, cfg):
    M, R = o["models"], o["raters"]
    L = ["# Expert study v6.2: confirmatory analysis (prereg v3, osf.io/fspuc)", "",
         f"Partials rule: {R['partials_rule']}. Included raters: {', '.join(R['included'])}.", "",
         "## Gates (§3)", "",
         "| rater | group | slot | rated | screener | catch | median ms | calibration acc | flag ≤50% | included (strict / lenient) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for c, r in sorted(R["raters"].items()):
        L.append(f"| {c} | {r['group']} | {r['slot']} | {r['n_rated']} | {r['screener'][0]}/{r['screener'][1]} | "
                 f"{r['catch'][0]}/{r['catch'][1]} | {f(r['median_ms'], 0)} | {f(r['calibration_acc'])} | "
                 f"{r['calibration_flag_le_50pct']} | {r['included_strict']} / {r['included_lenient']} |")
    L += ["", "## H1 and S5 (§4, §6): agreement with the model-preferred side", "",
          "| test | items | judgments | intercept log-odds [95% Wald CI]; p; probability scale | model; success |",
          "|---|---|---|---|---|",
          agree_line("H1 (confirmatory, 0.02)", M["H1"])]
    for k, v in M["H1_sens_threshold"].items():
        L.append(agree_line(f"H1 near-tie {k}", v))
    for k, v in M["H1_sens_evaluator"].items():
        L.append(agree_line(f"H1 evaluator {k}", v))
    L.append(agree_line("H1 uncorrected consensus (descriptive)", M["H1_uncorrected_descriptive"]))
    L.append(agree_line("H1 patch-2.57 (a)", M["H1_patch257"]["a"]))
    L.append(agree_line("H1 patch-2.57 (b, descriptive)", M["H1_patch257"]["b"]))
    L.append(agree_line("S5 (confirmatory)", M["S5"]))
    L.append(agree_line("S5 patch-2.57 (a)", M["S5_patch257"]["a"]))
    L.append(agree_line("S5 patch-2.57 (b, descriptive)", M["S5_patch257"]["b"]))
    L += ["", "Pooled agreement rates (descriptive): H1 " + ci(M["H1"]["pooled"]) + "; S5 " + ci(M["S5"]["pooled"]), "",
          "## H2 (§5): moderation by OOD covariate", "",
          "| test | items | judgments | slope [95% Wald CI]; p | class (δ = −0.25) | class (δ = −0.20) | model |",
          "|---|---|---|---|---|---|---|"]

    def h2line(name, t):
        if not t or "error" in t:
            return f"| {name} | – | – | – | {t.get('error') if t else 'not run'} | | |"
        s = t.get("slope")
        est = f"{f(s['est'])} [{f(s['ci_lo'])}, {f(s['ci_hi'])}]; p {f(s['p_two_sided'], 4)}" if s else f"spearman {t.get('spearman')}"
        return f"| {name} | {t['n_items']} | {t['n_judgments']} | {est} | {t.get('class_confirmatory')} | {t.get('class_margin_020')} | {t.get('model')} |"
    L.append(h2line("H2 ood_var_max (confirmatory)", M["H2"]))
    for k, v in M["H2_sens_covariate"].items():
        L.append(h2line(f"H2 {k}", v))
    L.append(h2line("H2 patch-2.57 (a)", M["H2_patch257"]["a"]))
    L.append(h2line("H2 patch-2.57 (b, descriptive)", M["H2_patch257"]["b"]))
    L += ["", "## S1 (§6): anchor accuracy by tier, Holm family of 3", "",
          "| set | tier | items | judgments | intercept [95% CI]; p | Holm p | above chance | pooled accuracy |",
          "|---|---|---|---|---|---|---|---|"]
    for name, fam in (("full", M["S1"]), ("patch-2.57 (a)", M["S1_patch257"]["a"]), ("patch-2.57 (b)", M["S1_patch257"]["b"])):
        if "error" in fam:
            L.append(f"| {name} | – | | | {fam['error']} | | | |")
            continue
        for t in ("low", "mid", "high"):
            r = fam[t]
            i = r.get("intercept")
            est = f"{f(i['est'])} [{f(i['ci_lo'])}, {f(i['ci_hi'])}]; p {f(i['p_two_sided'], 4)}" if i else "no fit"
            L.append(f"| {name} | {t} | {r['n_items']} | {r['n_judgments']} | {est} | {f(r.get('p_holm'), 4)} | {r.get('above_chance')} | {ci(r['pooled'])} |")
    s2 = M["S2"]
    L += ["", "## S2 (§6): slider calibration against the consensus", "",
          f"{s2.get('model')}: slope {f(s2.get('slope'))} [{f(s2.get('ci_lo'))}, {f(s2.get('ci_hi'))}], "
          f"p {f(s2.get('p_two_sided'), 4)}, n = {s2['n_judgments']}; positive calibration = {s2.get('positive_calibration')}.", ""]
    s3, st = M["S3"], o["S3_status"]
    eff = s3.get("effect")
    L += ["## S3 (§6): low-tier anchors, Group A vs B", "",
          f"Status: {'confirmatory' if st['confirmatory'] else 'descriptive case study'} "
          f"(Group A raters with 240/240: {', '.join(st['group_A_completed_240'])}; included: {', '.join(st['group_A_included'])}).",
          f"{s3.get('model')}: group effect {f(eff['est']) if eff else 'n/a'} [{f(eff['ci_lo']) if eff else ''}, {f(eff['ci_hi']) if eff else ''}], "
          f"one-sided p {f(s3.get('p_one_sided'), 4)}. Pooled A {ci(s3['pooled_A'])}; B {ci(s3['pooled_B'])}.", ""]
    s4 = o["S4"]
    L += ["## S4 (§6): Bradley-Terry vs tournament ordering", "",
          f"{s4['n_judgments']} judgments, {len(s4['strategies'])} strategies, {s4['pairings_with_judgments']} of 45 pairings judged. "
          f"Spearman rho {f(s4['spearman_rho'])}, one-sided exact p {f(s4['p_one_sided_exact'], 4)} over {s4['permutations']} orderings "
          f"(uncorrected tournament ordering: rho {f(s4['uncorrected_ordering']['spearman_rho'])}, p {f(s4['uncorrected_ordering']['p_one_sided_exact'], 4)}).", "",
          "| strategy | BT log-strength | tournament mean P(win), corrected | uncorrected |", "|---|---|---|---|"]
    for s in sorted(s4["strategies"], key=lambda z: -s4["bt_log_strength"][z]):
        L.append(f"| {s} | {f(s4['bt_log_strength'][s])} | {f(s4['tournament'][s]['corrected'])} | {f(s4['tournament'][s]['uncorrected'])} |")
    L += ["", "## Per rater (descriptive)", "", "| rater | anchors | low | mid | high | H1 agreement (0.02) |", "|---|---|---|---|---|---|"]
    for c, r in o["per_rater"].items():
        L.append(f"| {c} | {ci(r['anchors_all'])} | {ci(r['anchors_by_tier']['low'])} | {ci(r['anchors_by_tier']['mid'])} | "
                 f"{ci(r['anchors_by_tier']['high'])} | {ci(r['h1_agreement'])} |")
    open(os.path.join(d, "results.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
