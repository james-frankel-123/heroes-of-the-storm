"""
Paper-1 agents under gold: per-config means (seed-level SE) for the proxy and
every independent reference, key contrasts, tournament standings, and the
figures. Inputs: results/stage_b.json, stage_b2.json, stage_c.json.
Output: results/paper_gold.json, figs/paper_*.png
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np

REFS = ["proxy", "gN", "gN_naive", "NODRIFT", "BF", "T97", "T17", "QM2026"]
TRAIN_SIMS = [("B_fullwp", 200), ("F_400sim", 400), ("I_600sim", 600), ("J_800sim", 800)]
EPISODES = [("P_100K", 100), ("B_fullwp", 300), ("Q_500K", 500), ("E_1M", 1000)]
OTHERS = ["K_truebase", "M2_relational", "N2_absolute"]


def load():
    b = json.load(open(os.path.join(HERE, "results", "stage_b.json")))
    b2 = json.load(open(os.path.join(HERE, "results", "stage_b2.json")))
    runs = {}
    for name, v in b["dumps"].items():
        per = {k: np.array(v["per_draft"][k]) for k in ("proxy", "NODRIFT", "BF", "T97", "T17", "QM2026")}
        per["gN"] = np.array(b2["dumps"][name]["gN"])
        per["gN_naive"] = np.array(b2["dumps"][name]["gN_naive"])
        runs[name] = {"exp": v["experiment"], "seed": v["seed"], "tag": v["variant"]["tag"],
                      "per": per, "degen": v["degen"], "entropy": v["entropy"]}
    return b, b2, runs


def seed_means(runs, exp):
    rs = [r for r in runs.values() if r["exp"] == exp and r["tag"] == "base" and r["seed"] is not None]
    rs.sort(key=lambda r: r["seed"])
    return {k: np.array([r["per"][k].mean() for r in rs]) for k in REFS}, [r["seed"] for r in rs]


def mse(v):
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))


def contrast(a, b):
    """Welch difference of seed means a - b."""
    d = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return {"diff": float(d), "se": float(se), "z": float(d / se)}


def main():
    b, b2, runs = load()
    out = {"configs": {}, "contrasts": {}, "inference": {}, "tournament": {}}
    exps = sorted({r["exp"] for r in runs.values()})
    SM = {}
    for e in exps:
        m, seeds = seed_means(runs, e)
        if len(seeds) < 2:
            continue
        SM[e] = m
        deg = np.mean([r["degen"] for r in runs.values() if r["exp"] == e and r["tag"] == "base"])
        out["configs"][e] = {k: mse(m[k]) for k in REFS}
        out["configs"][e]["n_seeds"] = len(seeds)
        out["configs"][e]["degen"] = float(deg)
    pairs = [("F_400sim", "B_fullwp"), ("J_800sim", "F_400sim"), ("J_800sim", "B_fullwp"),
             ("I_600sim", "F_400sim"), ("E_1M", "B_fullwp"), ("E_1M", "J_800sim"),
             ("B_fullwp", "K_truebase"), ("J_800sim", "K_truebase"), ("F_400sim", "K_truebase"),
             ("M2_relational", "K_truebase"), ("N2_absolute", "K_truebase")]
    for a, c in pairs:
        out["contrasts"][f"{a} - {c}"] = {k: contrast(SM[a][k], SM[c][k]) for k in REFS}
    # inference-time variants (single seed, 200 paired drafts)
    for s in ("B_fullwp_s0", "J_800sim_s9"):
        base = runs[f"{s}__base"]["per"]
        for tag in ("sims100", "base", "sims400", "sims800", "t0", "t0.5", "t1.5", "dir0.10",
                    "dir0.25", "t1.5_dir0.25"):
            r = runs[f"{s}__{tag}"]
            out["inference"][f"{s}__{tag}"] = {
                k: {"mean": float(r["per"][k].mean()),
                    "paired_diff_vs_base": float((r["per"][k] - base[k]).mean()),
                    "paired_se": float((r["per"][k] - base[k]).std(ddof=1) / np.sqrt(len(base[k])))}
                for k in REFS}
            out["inference"][f"{s}__{tag}"]["degen"] = r["degen"]
            out["inference"][f"{s}__{tag}"]["entropy"] = r["entropy"]
    # tournament standings with gN
    st = {}
    for pair, v in b["tournament"].items():
        a, c = pair.split("__")
        g = b2["tournament"][pair]
        for s, sign in ((a, 1), (c, -1)):
            e = st.setdefault(s, {k: [] for k in ("consensus", "gN", "gN_naive", "NODRIFT", "T17", "QM2026")})
            for k in e:
                val = g[k] if k in g else v[k][0]
                e[k].append(val if sign == 1 else 1 - val)
    out["tournament"] = {s: {k: {"mean": float(np.mean(v)), "wins": int(sum(x > 0.5 for x in v)),
                                 "n": len(v)} for k, v in e.items()} for s, e in st.items()}
    json.dump(out, open(os.path.join(HERE, "results", "paper_gold.json"), "w"), indent=1)

    # print
    print(f"{'config':15s}" + "".join(f"{k:>16s}" for k in REFS) + "  degen")
    for e, v in sorted(out["configs"].items(), key=lambda kv: -kv[1]["proxy"][0]):
        print(f"{e:15s}" + "".join(f"  {v[k][0]:.4f}±{v[k][1]:.4f}" for k in REFS) + f"  {v['degen']:.1f}")
    print()
    for c, v in out["contrasts"].items():
        print(f"{c:28s}" + "".join(f"  {k}={v[k]['diff']:+.4f}({v[k]['z']:+.1f})" for k in REFS))
    print()
    for s, v in sorted(out["tournament"].items(), key=lambda kv: -kv[1]["consensus"]["mean"]):
        print(f"{s:20s}" + "".join(f"  {k}={x['mean']:.3f}[{x['wins']}/{x['n']}]" for k, x in v.items()))

    # figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (axis, lab) in zip(axes, ((TRAIN_SIMS, "training sims per move"),
                                      (EPISODES, "training episodes (K)"))):
        xs = [x for _, x in axis]
        for k, col in (("proxy", "#444"), ("gN", "#1f77b4"), ("NODRIFT", "#2ca02c"),
                       ("T17", "#9467bd"), ("QM2026", "#d62728")):
            m = [out["configs"][e][k][0] for e, _ in axis]
            s = [out["configs"][e][k][1] for e, _ in axis]
            ax.errorbar(xs, np.array(m) - m[0], yerr=s, marker="o", color=col, capsize=3,
                        label=k + (" (optimized-against)" if k == "proxy" else ""))
        ax.axhline(0, color="#aaa", lw=0.8)
        ax.set_xlabel(lab)
        ax.set_ylabel("change in mean P(win) vs first point")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[0].set_title("Paper-1 MCTS: training search depth")
    axes[1].set_title("Paper-1 MCTS: training length")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figs", "paper_pressure_curves.png"), dpi=130)


if __name__ == "__main__":
    main()
