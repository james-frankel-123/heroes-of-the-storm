"""
Within-snapshot split experiment: tables, paired contrasts, figures.
Input: results/split_search/*.json (+ mcts_selfplay/*.json if present)
Output: results/split_summary.json, figs/split_*.png
"""
import os
import sys
import glob
import json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np

REFS = ["gB", "gB_naive", "RB", "RN", "QM2026"]


def load(pattern="split_search/*.json"):
    runs = {}
    for p in glob.glob(os.path.join(HERE, "results", pattern)):
        r = json.load(open(p))
        runs[os.path.basename(p)[:-5]] = r
    return runs


def row(r):
    s = r["summary"]
    px = r["spec"]["proxies"]
    key = "proxy_ens_mean" if len(px) > 1 else f"proxy:{px[0]}"
    out = {"proxy": s[key] if key in s else [np.mean(r["per_draft"][key]), 0]}
    for k in REFS + ["ensA_std", "ensA_mean", "degen", "healer", "RB_hwr", "RB_syn", "RB_ctr", "RB_comp"]:
        out[k] = s[k]
    out["entropy"] = r["diversity"]["entropy_bits"]
    out["distinct"] = r["diversity"]["distinct"]
    out["cap_hits"] = r.get("cap_hits", 0)
    return out


def paired(a, b, k):
    x = np.array(a["per_draft"][k]) - np.array(b["per_draft"][k])
    return {"diff": float(x.mean()), "se": float(x.std(ddof=1) / np.sqrt(len(x)))}


def main():
    runs = load()
    table = {n: dict(row(r), spec=r["spec"]) for n, r in runs.items()}
    summary = {"runs": table, "contrasts": {}}
    # leak vs oof at matched pressure (bc prior)
    for T in (1.0, 0.0):
        for s in (256, 1024, 4096):
            a = f"pA8_oof__bc__sims{s}__T{T}"
            b = f"pA8_leak__bc__sims{s}__T{T}"
            if a in runs and b in runs:
                summary["contrasts"][f"oof-leak sims{s} T{T}"] = {
                    k: paired(runs[a], runs[b], k) for k in REFS}
    # pressure steps within a proxy
    for px in ("pA8_leak", "pA8_oof"):
        for T in (1.0, 0.0):
            for s0, s1 in ((256, 1024), (1024, 4096)):
                a, b = f"{px}__bc__sims{s1}__T{T}", f"{px}__bc__sims{s0}__T{T}"
                if a in runs and b in runs:
                    summary["contrasts"][f"{px} sims{s1}-sims{s0} T{T}"] = {
                        k: paired(runs[a], runs[b], k) for k in REFS + [f"proxy:{px}"]}
    json.dump(summary, open(os.path.join(HERE, "results", "split_summary.json"), "w"), indent=1)

    def show(names):
        for n in names:
            if n not in table:
                continue
            t = table[n]
            print(f"{n:62s} proxy={t['proxy'][0]:.4f} " + " ".join(
                f"{k}={t[k][0]:.4f}" for k in REFS) +
                f" ensStd={t['ensA_std'][0]:.4f} deg={t['degen'][0]:.3f} H={t['entropy']:.2f}")
    print(json.dumps(summary["contrasts"], indent=0)[:6000])
    for n in sorted(table):
        pass
    show(sorted(table))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
    SIMS = [0, 64, 256, 1024, 4096, 16384]
    # Fig 1: pressure curves, leak vs oof, BC prior, argmax root
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), sharey=False)
    for ax, k, title in zip(axes, ["proxy", "gB", "RB", "RN"],
                            ["proxy (optimized-against)", "judge trained on half B",
                             "realized index, half B", "realized index, post-snapshot"]):
        for px, col in (("pA8_leak", "#d62728"), ("pA8_oof", "#1f77b4")):
            for T, ls in ((0.0, "-"), (1.0, "--")):
                xs, ys, es = [], [], []
                for s in SIMS:
                    n = f"{px}__bc__sims{s}__T{T}"
                    if n in table:
                        xs.append(max(s, 16))
                        ys.append(table[n][k][0])
                        es.append(table[n][k][1])
                if xs:
                    ax.errorbar(xs, ys, yerr=es, ls=ls, marker="o", color=col, capsize=2,
                                label=f"{px.replace('pA8_', '')} proxy, T={T:g}")
        ax.set_xscale("log")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("MCTS sims per move (16 = no search)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("mean P(our team wins)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figs", "split_pressure_curves.png"), dpi=130)

    # Fig 2: proxy vs gold (Gao-style)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, k in zip(axes, ["gB", "RN", "QM2026"]):
        for e, mk in ((1, "v"), (2, "s"), (4, "D"), (8, "o")):
            for mode, col in (("leak", "#d62728"), ("oof", "#1f77b4")):
                px = f"pA{e}_{mode}"
                pts = [(table[n]["proxy"][0], table[n][k][0]) for s in SIMS
                       for n in [f"{px}__bc__sims{s}__T0.0"] if n in table]
                if pts:
                    pts.sort()
                    ax.plot(*zip(*pts), marker=mk, color=col, alpha=0.35 + 0.08 * e,
                            label=f"{mode}, {e}/8 of A")
        lo, hi = ax.get_xlim()
        ax.plot([0.4, 0.9], [0.4, 0.9], color="#999", lw=0.8, ls=":")
        ax.set_xlim(lo, hi)
        ax.set_xlabel("proxy score (what search maximizes)")
        ax.set_ylabel(f"gold: {k}")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figs", "split_proxy_vs_gold.png"), dpi=130)

    # Fig 3: LCB
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, k in zip(axes, ["gB", "RN"]):
        for ens, col in (("pA8_leak+pA8_leak_s1+pA8_leak_s2+pA8_leak_s3", "#d62728"),
                         ("pA8_oof+pA8_oof_s1+pA8_oof_s2+pA8_oof_s3", "#1f77b4")):
            for s, ls in ((256, ":"), (1024, "--"), (4096, "-")):
                xs, ys = [], []
                for lam in (0.0, 1.0, 2.0, 4.0, 8.0):
                    n = f"{ens}__bc__sims{s}__T0.0" + (f"__lcb{lam}" if lam else "")
                    if n in table:
                        xs.append(lam)
                        ys.append(table[n][k][0])
                if xs:
                    ax.plot(xs, ys, marker="o", ls=ls, color=col,
                            label=f"{'leak' if 'leak' in ens else 'oof'} ens, {s} sims")
        ax.set_xlabel("LCB penalty lambda (logit per unit ensemble MAD)")
        ax.set_ylabel(k)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figs", "split_lcb.png"), dpi=130)


if __name__ == "__main__":
    main()
