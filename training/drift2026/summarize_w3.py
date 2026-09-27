"""
Wave-3 summaries -> results/W3_{HYBRID,EMBED_REFRESH,RETRAIN_SIM}.md.

  W3d  hybrid per-signal features vs the wave-1/2 reference arms
       (accuracy + sanity + W2b-style policy metrics). Q: beats cumprev?
  W3e  embed+refresh stack: (i) retrained arm w3e_embedrefresh (embed regime
       on cumulative_prev features), (ii) eval-only check: wave-1 d2b_embed
       checkpoints scored on cumprev-refreshed test features. Q: do the two
       wave-1/2 winners stack?
  W3g  W2c auto-retrain simulation refreshed with the new arms as policies
       (never-retrain-at-C0 rows + winner K=6 cadence if trained).

Usage:
    python drift2026/summarize_w3.py [--only d|e|g]
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drift2026 import common
from drift2026.summarize_w2 import agg_cells, fmt, builds_255, jload

import numpy as np

R = common.RESULTS_DIR
C0_POS = 8
K6_POSITIONS = [8, 14, 20, 26, 32, 38]

D_REFS = ["d2c_cumprev", "d2b_embed", "d2b_win6", "d2b_allhist"]


def policy_metrics(cells):
    out = {}
    for cell in cells:
        rows = []
        for s in (common.SEEDS if cell != "gd" else [42]):
            p = os.path.join(R, "w2b", f"{cell}_s{s}.json")
            if os.path.exists(p):
                rows.append(jload("w2b", f"{cell}_s{s}.json"))
        if rows:
            def m(key):
                vals = [r[key] for r in rows]
                return (float(np.mean(vals)), float(np.std(vals)))
            out[cell] = {"healer": m("healer_rate"), "degen": m("degen_rate"),
                         "counter": m("counter_future"),
                         "synergy": m("synergy_future"), "entropy": m("entropy")}
    return out


def w3d(args):
    cells = agg_cells(["w3d_hybrid"] + D_REFS)
    pol = policy_metrics(["w3d_hybrid"] + D_REFS + ["gd"])

    window_meta = {}
    p = os.path.join(R, "w3_recovery.json")
    if os.path.exists(p):
        with open(p) as f:
            window_meta["target_games"] = json.load(f)["hybrid_window_games"]
    try:
        import gzip
        b_last = builds_255()[-1]
        with gzip.open(common.stats_path("hybrid", b_last), "rt") as f:
            m = json.load(f)["_meta"]
        window_meta.update(deploy_window_builds=len(m["window_builds"]),
                           deploy_window_games=m["window_games"])
    except Exception:
        pass

    tg = window_meta.get("target_games")
    lines = [
        "# W3(d) — Hybrid per-signal enrichment sourcing",
        "",
        "hero-WR family from a recent window (target "
        f"{tg:,} games" if tg else "hero-WR family from a recent window (?",
        ", from the W3(a) recovery answer; at deployment "
        f"{window_meta.get('deploy_window_builds', '?')} builds / "
        f"{window_meta.get('deploy_window_games', 0):,} games), role-comp WR "
        "from all history, pairwise from cumulative — all strictly causal "
        "per row (cumprev-style). 3 seeds; policy metrics = W2b protocol "
        "(500 greedy drafts/seed vs GD pool, scored vs future truth).",
        "",
        "| cell | val acc | future acc | sizable | sanity 28 | healer % | "
        "degen % | counter | synergy |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for cell in ["w3d_hybrid"] + D_REFS:
        if cell not in cells:
            continue
        c = cells[cell]
        pm = pol.get(cell, {})
        lines.append(
            f"| {cell} | {fmt(c['val'])} | {fmt(c['future'])} | "
            f"{fmt(c['sizable'])} | {fmt(c['sanity28'], 1)} | "
            f"{fmt(pm.get('healer', (None, None)), 1)} | "
            f"{fmt(pm.get('degen', (None, None)), 1)} | "
            f"{fmt(pm.get('counter', (None, None)))} | "
            f"{fmt(pm.get('synergy', (None, None)))} |")

    if "w3d_hybrid" in cells and "d2c_cumprev" in cells:
        h = cells["w3d_hybrid"]["future"][0]
        cp = cells["d2c_cumprev"]["future"][0]
        lines += [
            "", "## Verdict", "",
            f"- w3d_hybrid future acc {h:.2f} vs d2c_cumprev {cp:.2f} "
            f"({h - cp:+.2f} pp): the per-signal window sourcing "
            + ("**beats**" if h > cp else "does **not** beat")
            + " the uniform cumulative refresh.",
        ]
        pb = cells["w3d_hybrid"]["per_build"]
        lines += ["", "Per test build (hybrid): " + ", ".join(
            f"{b}: {a:.2f}" for b, a in sorted(pb.items()))]

    path = os.path.join(R, "W3_HYBRID.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {path}")
    common.write_json(os.path.join(R, "w3d_summary.json"),
                      {"cells": cells, "policy": pol,
                       "window_meta": window_meta})


def eval_embed_on_cumprev():
    """Eval-only stack check: wave-1 d2b_embed checkpoints (trained on
    features_cutoff) scored on cumprev-refreshed future test features."""
    import torch
    from drift2026.eval_policy_w2b import load_cell_model
    from drift2026.train_drift_wp import enriched_cols
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    builds = common.load_patch_index()["builds"]
    cut = builds.index(common.TRAIN_CUTOFF_BUILD)
    z = np.load(os.path.join(common.CACHE_DIR, "features_cumulative_prev.npz"))
    mask = z["build_idx"].astype(np.int64) > cut
    cols = enriched_cols()
    X = np.concatenate([z["bases"][mask], z["enricheds"][mask][:, cols]],
                       axis=1)
    y = z["labels"][mask]
    sizable_idx = [builds.index(b) for b in common.SIZABLE_TEST_BUILDS]
    m_sz = np.isin(z["build_idx"][mask].astype(np.int64), sizable_idx)
    del z
    out = {}
    for seed in common.SEEDS:
        pt = os.path.join(common.MODELS_DIR, f"d2b_embed_s{seed}.pt")
        if not os.path.exists(pt):
            continue
        model = load_cell_model("d2b_embed", seed, device)
        preds = []
        with torch.no_grad():
            for i in range(0, len(X), 131072):
                xb = torch.tensor(X[i:i + 131072], dtype=torch.float32,
                                  device=device)
                preds.append((model(xb) > 0.5).float().cpu().numpy())
        correct = np.concatenate(preds) == y
        out[seed] = {"future": round(float(correct.mean() * 100), 3),
                     "sizable": round(float(correct[m_sz].mean() * 100), 3)}
    return out


def w3e(args):
    cells = agg_cells(["w3e_embedrefresh", "d2b_embed", "d2c_cumprev"])
    pol = policy_metrics(["w3e_embedrefresh", "d2b_embed", "d2c_cumprev"])
    evalonly = eval_embed_on_cumprev()
    eo = ([v["future"] for v in evalonly.values()],
          [v["sizable"] for v in evalonly.values()])

    lines = [
        "# W3(e) — Do the two wave-1 winners stack? (patch embedding x "
        "per-build stats refresh)",
        "",
        "w3e_embedrefresh = embed regime TRAINED on cumulative_prev "
        "(causally refreshed) features; test rows use their build's "
        "prev-cumulative stats + the cutoff-clamped embedding. "
        "'embed eval-only' = wave-1 d2b_embed checkpoints (trained on "
        "cutoff-stats features) scored on cumprev-refreshed test features "
        "without retraining (a deployment shortcut, feature-distribution "
        "mismatched by construction).",
        "",
        "| arm | val acc | future acc | sizable | sanity 28 | degen % |",
        "|---|---|---|---|---|---|",
    ]
    for cell in ("w3e_embedrefresh", "d2b_embed", "d2c_cumprev"):
        if cell not in cells:
            continue
        c = cells[cell]
        pm = pol.get(cell, {})
        lines.append(f"| {cell} | {fmt(c['val'])} | {fmt(c['future'])} | "
                     f"{fmt(c['sizable'])} | {fmt(c['sanity28'], 1)} | "
                     f"{fmt(pm.get('degen', (None, None)), 1)} |")
    if evalonly:
        lines.append(
            f"| d2b_embed eval-only on cumprev feats | - | "
            f"{np.mean(eo[0]):.2f} ± {np.std(eo[0]):.2f} | "
            f"{np.mean(eo[1]):.2f} ± {np.std(eo[1]):.2f} | - | - |")

    if "w3e_embedrefresh" in cells and "d2c_cumprev" in cells:
        s = cells["w3e_embedrefresh"]["future"][0]
        cp = cells["d2c_cumprev"]["future"][0]
        em = cells.get("d2b_embed", {}).get("future", (None,))[0]
        lines += [
            "", "## Verdict", "",
            f"- Stacked arm {s:.2f} vs cumprev alone {cp:.2f} "
            f"({s - cp:+.2f}) and embed alone {em:.2f} ({s - em:+.2f}): "
            + ("the winners **stack**." if s > max(cp, em)
               else "the winners do **not** stack (refresh dominates)."),
        ]
    path = os.path.join(R, "W3_EMBED_REFRESH.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {path}")
    common.write_json(os.path.join(R, "w3e_summary.json"),
                      {"cells": cells, "eval_only": evalonly})


def w3g(args):
    builds = builds_255()
    cutoff_pos = builds.index(common.TRAIN_CUTOFF_BUILD)
    last = len(builds) - 1

    acc, games = {}, {}
    for p in range(C0_POS, last):
        src = (("d2", "d2c_cumprev_s42.json") if p == cutoff_pos
               else ("w2c", f"w2c_cut{p:02d}_s42.json"))
        if not os.path.exists(os.path.join(R, *src)):
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

    def curve_from(subdir, name):
        p = os.path.join(R, subdir, f"{name}.json")
        if not os.path.exists(p):
            return None
        r = jload(subdir, f"{name}.json")
        return {builds.index(b): d["acc"]
                for b, d in r["test_acc_per_build"].items()}

    oracle = {n: acc[n - 1][n] for n in Ns}
    policies = {}
    policies["oracle cumprev (retrain every build)"] = (oracle, len(Ns) - 1)
    policies["never retrain + frozen stats"] = (frozen_acc, 0)
    policies["never retrain + cumprev refresh"] = (
        {n: acc[C0_POS][n] for n in Ns}, 0)
    for K in (3, 6):
        points = list(range(C0_POS, last, K))
        curve = {n: acc[max(p for p in points if p < n)][n] for n in Ns}
        policies[f"retrain every K={K} + cumprev refresh"] = (
            curve, len([p for p in points if C0_POS < p < Ns[-1]]))

    # ── wave-3 additions ──
    added = []
    for arm, label in (("hybrid", "hybrid per-signal refresh"),
                       ("embedrefresh", "embed + cumprev refresh")):
        c = curve_from("w2c", f"w3g_{arm}_cut{C0_POS:02d}_s42")
        if c:
            policies[f"never retrain + {label}"] = (c, 0)
            added.append(arm)
        k6 = {}
        for p in K6_POSITIONS:
            cc = curve_from("w2c", f"w3g_{arm}_cut{p:02d}_s42")
            if cc:
                k6[p] = cc
        if len(k6) == len(K6_POSITIONS):
            curve = {n: k6[max(p for p in k6 if p < n)][n] for n in Ns}
            policies[f"retrain every K=6 + {label}"] = (curve, 5)

    o = wavg(oracle)
    table = []
    for name, (curve, n_retrains) in policies.items():
        missing = [n for n in Ns if n not in curve]
        if missing:
            continue
        a = wavg(curve)
        table.append({"policy": name, "retrains": n_retrains,
                      "weighted_acc": round(a, 3),
                      "regret_pp": round(o - a, 3)})

    best = min(table, key=lambda r: (r["regret_pp"], r["retrains"]))
    cheap = [r for r in table if r["retrains"] <= 12 and r["regret_pp"] < 0.1]
    rec = min(cheap, key=lambda r: (r["regret_pp"], r["retrains"])) if cheap \
        else best

    lines = [
        "# W3(g) — Auto-retrain simulation refreshed with the wave-3 arms",
        "",
        f"Same timeline as W2c (deploy after build {C0_POS} = "
        f"{builds[C0_POS]}, {len(Ns)} simulated builds, "
        f"{int(w.sum()):,} games, games-weighted accuracy; regret vs the "
        "cumprev always-retrain oracle). Wave-3 policies use the (d)/(e) "
        f"arms trained at the same cutoffs (added: {', '.join(added) or 'none'}).",
        "",
        "| policy | retrains | weighted acc % | regret vs oracle (pp) |",
        "|---|---|---|---|",
    ]
    for row in table:
        lines.append(f"| {row['policy']} | {row['retrains']} | "
                     f"{row['weighted_acc']:.3f} | {row['regret_pp']:+.3f} |")
    lines += [
        "", "## Recommended site policy", "",
        f"- **{rec['policy']}** (weighted acc {rec['weighted_acc']:.3f}, "
        f"regret {rec['regret_pp']:+.3f} pp, {rec['retrains']} retrains over "
        "~2.8 years).",
        f"- Best overall: {best['policy']} ({best['weighted_acc']:.3f}).",
    ]
    path = os.path.join(R, "W3_RETRAIN_SIM.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {path}")
    common.write_json(os.path.join(R, "w3g_policies.json"),
                      {"oracle_weighted_acc": round(o, 3), "policies": table,
                       "recommended": rec["policy"]})
    for row in table:
        print(f"  {row['policy']:<48} retrains={row['retrains']:>2} "
              f"acc={row['weighted_acc']:.3f} regret={row['regret_pp']:+.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default=None, choices=[None, "d", "e", "g"])
    args = ap.parse_args()
    common.setup()
    if args.only in (None, "d"):
        w3d(args)
    if args.only in (None, "e"):
        w3e(args)
    if args.only in (None, "g"):
        w3g(args)


if __name__ == "__main__":
    main()
