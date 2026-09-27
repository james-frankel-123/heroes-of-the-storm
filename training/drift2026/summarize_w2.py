"""
Wave-2 summaries -> results/W2{A,B,C}_SUMMARY.md (+ machine-readable JSONs).

  W2A  within-patch-causal local enrichment verdict (d2c_causal_* cells vs
       the cumprev deployable champion and the leaky d2c_local oracle), plus
       a within-build depth analysis (does accuracy grow later in a test
       build as local stats accumulate?).
  W2B  accuracy-vs-policy table for the wave-1 regime cells + the W4 trigger
       (fires if degen-rate spread > 3pp OR sanity-28 spread >= 3).
  W2C  auto-retrain policy simulation over the build timeline from the
       results/w2c/ accuracy matrix: cumulative games-weighted accuracy
       regret vs the always-retrain oracle.

Usage:
    python drift2026/summarize_w2.py            # all three
    python drift2026/summarize_w2.py --only c --skip-depth
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drift2026 import common

import numpy as np

R = common.RESULTS_DIR
C0_POS = 8   # keep in sync with phase_w2.py


def jload(*p):
    with open(os.path.join(R, *p)) as f:
        return json.load(f)


def builds_255():
    return [b for b in common.load_patch_index()["builds"]
            if b.startswith("2.55")]


def agg_cells(prefix_names, subdir="d2"):
    """{cell: aggregated dict} over seeds for results/<subdir>/<cell>_s*.json."""
    out = {}
    for cell in prefix_names:
        rows = []
        for s in common.SEEDS:
            p = os.path.join(R, subdir, f"{cell}_s{s}.json")
            if os.path.exists(p):
                rows.append(jload(subdir, f"{cell}_s{s}.json"))
        if not rows:
            continue
        def m(key, sub=None):
            vals = [(r[key][sub] if sub else r[key]) for r in rows
                    if r.get(key) is not None]
            return (float(np.mean(vals)), float(np.std(vals))) if vals else (None, None)
        out[cell] = {
            "n_seeds": len(rows),
            "val": m("val_acc"), "future": m("test_acc_future"),
            "sizable": m("test_acc_sizable"),
            "sanity28": m("sanity", "passed_28") if rows[0].get("sanity") else (None, None),
            "per_build": {b: float(np.mean([r["test_acc_per_build"][b]["acc"]
                                            for r in rows]))
                          for b in rows[0]["test_acc_per_build"]},
            "epochs": float(np.mean([r["epochs"] for r in rows])),
        }
    return out


def fmt(ms, prec=2):
    mean, std = ms
    if mean is None:
        return "-"
    return f"{mean:.{prec}f} ± {std:.{prec}f}"


# ═══════════════════ W2A ═══════════════════

def depth_analysis(arms, terciles=3):
    """Accuracy by within-build date tercile on the sizable test builds, per
    arm (features pass, model cell). Shows whether causal-local arms improve
    later in a build as local stats accumulate."""
    import torch
    from drift2026.eval_policy_w2b import load_cell_model
    from drift2026.train_drift_wp import enriched_cols
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    builds = common.load_patch_index()["builds"]
    sizable_idx = {builds.index(b): b for b in common.SIZABLE_TEST_BUILDS}
    cols = enriched_cols()

    out = {}
    for feats, cell in arms:
        z = np.load(os.path.join(common.CACHE_DIR, f"features_{feats}.npz"))
        bi = z["build_idx"]
        mask = np.isin(bi, list(sizable_idx))
        X = np.concatenate([z["bases"][mask], z["enricheds"][mask][:, cols]],
                           axis=1)
        y, days, bsel = z["labels"][mask], z["date_days"][mask], bi[mask]
        model = load_cell_model(cell, common.SEED, device)
        with torch.no_grad():
            preds = []
            for i in range(0, len(X), 131072):
                xb = torch.tensor(X[i:i + 131072], dtype=torch.float32,
                                  device=device)
                preds.append((model(xb) > 0.5).float().cpu().numpy())
        correct = (np.concatenate(preds) == y)
        arm = {}
        for gidx, bname in sizable_idx.items():
            m = bsel == gidx
            d = days[m]
            qs = np.quantile(d, np.linspace(0, 1, terciles + 1))
            accs = []
            for t in range(terciles):
                tm = (d >= qs[t]) & (d <= qs[t + 1] if t == terciles - 1
                                     else d < qs[t + 1])
                accs.append(round(float(correct[m][tm].mean() * 100), 2))
            arm[bname] = accs
        out[f"{feats}/{cell}"] = arm
        del z, X
    return out


def w2a(args):
    causal = ["d2c_causal_merge", "d2c_causal_k100", "d2c_causal_k1000"]
    refs = ["d2b_allhist", "d2c_cumprev", "d2c_frozen", "d2c_local"]
    cells = agg_cells(causal + refs)

    lines = [
        "# W2a — within-patch-causal local enrichment (fixes d2c_local leakage)",
        "",
        "Every row's enriched stats = rolling same-build stats from strictly",
        "EARLIER calendar days, blended with the cumulative-to-cutoff prior",
        "(count-weighted shrinkage; kw = prior weight per statistic).",
        "Deployable: test rows never see the row's own day or any post-cutoff",
        "build other than their own build's past. 3 seeds per cell.",
        "",
        "| cell | prior weight | val acc | future acc | sizable | sanity 28 |",
        "|---|---|---|---|---|---|",
    ]
    kw = {"d2c_causal_merge": "true prior counts (~1.8M games)",
          "d2c_causal_k100": "k=100 pseudo-games",
          "d2c_causal_k1000": "k=1000 pseudo-games",
          "d2b_allhist": "(cutoff stats only — no local)",
          "d2c_cumprev": "(cumulative refresh, no within-build)",
          "d2c_frozen": "(paper-1 leaky control)",
          "d2c_local": "(LEAKY within-patch oracle)"}
    for cell in causal + refs:
        if cell not in cells:
            continue
        c = cells[cell]
        lines.append(f"| {cell} | {kw[cell]} | {fmt(c['val'])} | "
                     f"{fmt(c['future'])} | {fmt(c['sizable'])} | "
                     f"{fmt(c['sanity28'], 1)} |")

    verdict = []
    if all(c in cells for c in ("d2c_causal_k100", "d2c_cumprev", "d2b_allhist")):
        base = cells["d2b_allhist"]["future"][0]
        cp = cells["d2c_cumprev"]["future"][0]
        best_causal = max(causal, key=lambda c: cells[c]["future"][0]
                          if c in cells else -1)
        bc = cells[best_causal]["future"][0]
        verdict += [
            "",
            "## Verdict",
            "",
            f"- Best causal-local arm: **{best_causal}** at {bc:.2f}% future acc "
            f"({bc - base:+.2f} vs the cutoff-stats baseline d2b_allhist, "
            f"{bc - cp:+.2f} vs the cumprev refresh champion, vs the leaky "
            f"d2c_local oracle at {cells['d2c_local']['future'][0]:.2f}).",
        ]
    lines += verdict

    if not args.skip_depth:
        arms = [("causal_k100", "d2c_causal_k100"),
                ("causal_merge", "d2c_causal_merge"),
                ("cumulative_prev", "d2c_cumprev"),
                ("cutoff", "d2b_allhist")]
        arms = [(f, c) for f, c in arms
                if os.path.exists(os.path.join(common.MODELS_DIR, f"{c}_s42.pt"))]
        depth = depth_analysis(arms)
        lines += ["", "## Within-build depth (accuracy by date tercile, "
                  "sizable test builds)", "",
                  "Causal-local arms should IMPROVE across terciles (local "
                  "stats accumulate); cumprev/cutoff should stay ~flat.", ""]
        lines.append("| arm | build | early | mid | late | late-early |")
        lines.append("|---|---|---|---|---|---|")
        for arm, per_build in depth.items():
            for b, accs in per_build.items():
                lines.append(f"| {arm} | {b} | " +
                             " | ".join(f"{a:.2f}" for a in accs) +
                             f" | {accs[-1]-accs[0]:+.2f} |")
        common.write_json(os.path.join(R, "w2a_depth.json"), depth)

    path = os.path.join(R, "W2A_SUMMARY.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {path}")
    common.write_json(os.path.join(R, "w2a_summary.json"), cells)


# ═══════════════════ W2B ═══════════════════

W2B_CELLS = ["d2b_allhist", "d2b_win6", "d2b_decay90", "d2b_embed",
             "d2c_cumprev"]


def w2b(args):
    acc = agg_cells(W2B_CELLS)
    pol = {}
    for cell in W2B_CELLS + ["gd"]:
        rows = []
        for s in (common.SEEDS if cell != "gd" else [42]):
            p = os.path.join(R, "w2b", f"{cell}_s{s}.json")
            if os.path.exists(p):
                rows.append(jload("w2b", f"{cell}_s{s}.json"))
        if not rows:
            continue
        def m(key):
            vals = [r[key] for r in rows]
            return (float(np.mean(vals)), float(np.std(vals)))
        pol[cell] = {"n": len(rows), "healer": m("healer_rate"),
                     "degen": m("degen_rate"), "counter": m("counter_future"),
                     "synergy": m("synergy_future"),
                     "entropy": m("entropy"),
                     "distinct": m("distinct_heroes"),
                     "drafts": sum(r["drafts"] for r in rows)}

    lines = [
        "# W2b — policy-level evaluation of the drifted value functions",
        "",
        "Greedy drafting (paper-1 machinery) with each arm's deployment-time",
        "stats, opponents = rerun2026 GD pool (sampled), 3 seeds x 500 drafts",
        "per cell. counter/synergy are scored against FUTURE-PERIOD ground",
        "truth (merged post-cutoff per-build counts). Accuracy/sanity columns",
        "are the wave-1 numbers for the same checkpoints.",
        "",
        "| cell | future acc | sanity 28 | healer % | degen % | counter | "
        "synergy | entropy |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for cell in W2B_CELLS:
        if cell not in pol:
            continue
        a, p = acc.get(cell, {}), pol[cell]
        lines.append(
            f"| {cell} | {fmt(a.get('future', (None, None)))} | "
            f"{fmt(a.get('sanity28', (None, None)), 1)} | {fmt(p['healer'], 1)} | "
            f"{fmt(p['degen'], 1)} | {fmt(p['counter'])} | {fmt(p['synergy'])} | "
            f"{fmt(p['entropy'])} |")
    if "gd" in pol:
        p = pol["gd"]
        lines.append(f"| gd (reference) | - | - | {fmt(p['healer'], 1)} | "
                     f"{fmt(p['degen'], 1)} | {fmt(p['counter'])} | "
                     f"{fmt(p['synergy'])} | {fmt(p['entropy'])} |")

    # spreads + W4 trigger
    have = [c for c in W2B_CELLS if c in pol and c in acc]
    summary = {"cells": pol, "accuracy": {c: acc[c]["future"][0] for c in have}}
    if have:
        acc_spread = max(acc[c]["future"][0] for c in have) - \
            min(acc[c]["future"][0] for c in have)
        degen_spread = max(pol[c]["degen"][0] for c in have) - \
            min(pol[c]["degen"][0] for c in have)
        sanity_spread = max(acc[c]["sanity28"][0] for c in have) - \
            min(acc[c]["sanity28"][0] for c in have)
        healer_spread = max(pol[c]["healer"][0] for c in have) - \
            min(pol[c]["healer"][0] for c in have)
        trigger = degen_spread > 3.0 or sanity_spread >= 3.0
        summary.update({"acc_spread_pp": round(acc_spread, 3),
                        "degen_spread_pp": round(degen_spread, 2),
                        "sanity_spread": round(sanity_spread, 2),
                        "healer_spread_pp": round(healer_spread, 2),
                        "w4_trigger_fired": bool(trigger)})
        lines += [
            "", "## Spreads across regimes (W4 trigger)", "",
            f"- future-accuracy spread: {acc_spread:.2f} pp",
            f"- degen-rate spread: {degen_spread:.2f} pp (trigger: > 3 pp)",
            f"- sanity-28 spread: {sanity_spread:.2f} (trigger: >= 3)",
            f"- healer-rate spread: {healer_spread:.2f} pp",
            f"- **W4 trigger fired: {trigger}** (W4 runs unconditionally per "
            "2026-07-07 decision; the trigger matters for the paper's framing)",
        ]

    path = os.path.join(R, "W2B_SUMMARY.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {path}")
    common.write_json(os.path.join(R, "w2b_summary.json"), summary)


# ═══════════════════ W2C ═══════════════════

def w2c(args):
    builds = builds_255()
    cutoff_pos = builds.index(common.TRAIN_CUTOFF_BUILD)
    last = len(builds) - 1   # 44

    # accuracy matrix: acc[C][N] over positions
    acc, games = {}, {}
    for p in range(C0_POS, last):
        if p == cutoff_pos:
            src = ("d2", "d2c_cumprev_s42.json")
        else:
            src = ("w2c", f"w2c_cut{p:02d}_s42.json")
        if not os.path.exists(os.path.join(R, *src)):
            print(f"  missing {src[1]} — matrix incomplete")
            continue
        r = jload(*src)
        acc[p] = {}
        for b, d in r["test_acc_per_build"].items():
            n = builds.index(b)
            acc[p][n] = d["acc"]
            games[n] = d["n_rows"] // 2
    frozen = jload("w2c", f"w2c_frozenstats_cut{C0_POS:02d}_s42.json")
    frozen_acc = {builds.index(b): d["acc"]
                  for b, d in frozen["test_acc_per_build"].items()}

    Ns = sorted(n for n in games if n > C0_POS)
    w = np.array([games[n] for n in Ns], dtype=float)

    def wavg(curve):
        return float(np.sum(w * np.array([curve[n] for n in Ns])) / w.sum())

    oracle = {n: acc[n - 1][n] for n in Ns}

    policies = {}
    policies["oracle (retrain every build)"] = (oracle, len(Ns) - 1)
    policies["never retrain + frozen stats"] = (frozen_acc, 0)
    policies["never retrain + stats refresh"] = ({n: acc[C0_POS][n] for n in Ns}, 0)
    for K in (1, 3, 6):
        points = list(range(C0_POS, last, K))
        curve = {}
        for n in Ns:
            csel = max(p for p in points if p < n)
            curve[n] = acc[csel][n]
        policies[f"retrain every K={K} builds + refresh"] = (curve, len(
            [p for p in points if C0_POS < p < Ns[-1]]))
    for X in (0.25, 0.5, 1.0):
        cur, peak, curve, retrains = C0_POS, None, {}, 0
        for n in Ns:
            a = acc[cur][n]
            curve[n] = a
            if games[n] >= 5000:      # observable estimate: skip tiny builds
                if peak is not None and peak - a > X:
                    cur, peak = n, None
                    retrains += 1
                else:
                    peak = a if peak is None else max(peak, a)
        policies[f"trigger: acc drop > {X}pp + refresh"] = (curve, retrains)

    o = wavg(oracle)
    table = []
    for name, (curve, n_retrains) in policies.items():
        a = wavg(curve)
        table.append({"policy": name, "retrains": n_retrains,
                      "weighted_acc": round(a, 3),
                      "regret_pp": round(o - a, 3)})

    lines = [
        "# W2c — auto-retrain policy simulation over the build timeline",
        "",
        f"Deployment starts after build {C0_POS} ({builds[C0_POS]}, ~638K games "
        "of history, 2023-07); simulated forward over the remaining "
        f"{len(Ns)} builds (~2.8 years, {int(w.sum()):,} games). All retrain "
        "policies train regime=all on cumulative_prev (causal) features, "
        "seed 42 (wave-1 seed std <= 0.16pp). Stats refresh = per-build-",
        "boundary refresh of the feature aggregates (cumprev). Accuracy is",
        "games-weighted over the simulated builds; regret is vs the",
        "always-retrain oracle.",
        "",
        "| policy | retrains | weighted acc % | regret vs oracle (pp) |",
        "|---|---|---|---|",
    ]
    for row in table:
        lines.append(f"| {row['policy']} | {row['retrains']} | "
                     f"{row['weighted_acc']:.3f} | {row['regret_pp']:+.3f} |")

    # decompose: value of stats refresh vs retraining
    fr = wavg(frozen_acc)
    re = wavg({n: acc[C0_POS][n] for n in Ns})
    lines += [
        "",
        "## Decomposition",
        "",
        f"- Stats refresh alone (no retrain): {re - fr:+.3f} pp over frozen "
        f"stats ({re:.3f} vs {fr:.3f}).",
        f"- Retraining on top of refresh (oracle vs never-retrain+refresh): "
        f"{o - re:+.3f} pp.",
    ]
    common.write_json(os.path.join(R, "w2c_policies.json"),
                      {"oracle_weighted_acc": round(o, 3), "policies": table,
                       "games_per_build": {builds[n]: games[n] for n in Ns},
                       "matrix": {builds[p]: {builds[n]: a for n, a in row.items()}
                                  for p, row in acc.items()}})
    path = os.path.join(R, "W2C_SUMMARY.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {path}")
    for row in table:
        print(f"  {row['policy']:<42} retrains={row['retrains']:>2} "
              f"acc={row['weighted_acc']:.3f} regret={row['regret_pp']:+.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default=None, choices=[None, "a", "b", "c"])
    ap.add_argument("--skip-depth", action="store_true")
    args = ap.parse_args()
    common.setup()
    if args.only in (None, "a"):
        w2a(args)
    if args.only in (None, "b"):
        w2b(args)
    if args.only in (None, "c"):
        w2c(args)


if __name__ == "__main__":
    main()
