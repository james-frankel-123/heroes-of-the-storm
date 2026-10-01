"""Generate the three paper-2 figures from drift2026 results.

Outputs fig_dose.pdf, fig_vintage.pdf, fig_decay.pdf into paper/drift/overleaf/.
Usage: python3 paper/drift/scripts/gen_figs.py  (from repo root)
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
# v2 (site-tier) rebuild: every figure reads training/drift_v2/results.
# Figures built on MCTS head-to-heads (dose, vintage) fall back to the
# previous build's results until the v2 agents finish.
RES = os.path.join(REPO, "training", "drift_v2", "results")
RB = RES
_RB_PREV = os.path.join(REPO, "training", "drift_rebuild", "results")


def _rb(name):
    p = os.path.join(RB, name)
    return p if os.path.exists(p) else os.path.join(_RB_PREV, name)
TRAINING = os.path.join(REPO, "training")
OUT = os.path.join(REPO, "paper", "drift", "overleaf")

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 200, "axes.spines.top": False, "axes.spines.right": False,
})
BLUE, ORANGE, GREEN, GRAY, RED = "#2166ac", "#e08214", "#1b7837", "#666666", "#b2182b"


def _t_quantile(df, q=0.975):
    """Student-t quantile by bisection on a numerically integrated tail."""
    import math
    import numpy as np

    def sf(x):
        xs = np.linspace(x, x + 400, 200001)
        c = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)
        return float(np.trapezoid(c * (1 + xs ** 2 / df) ** (-(df + 1) / 2), xs))
    lo, hi = 0.0, 50.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if sf(mid) > 1 - q else (lo, mid)
    return (lo + hi) / 2


def fig_dose():
    """Staleness, scored on the W12 out-of-sample builds (T_clean, crossed
    seed random effects), 95% t intervals with each contrast's Satterthwaite
    df (consolidated audit B14). Filled: leak-free agents with era-matched
    opponent models on both sides (U-gc vs S1/S2; 0 months = U-gc vs U-gc,
    drawn at 0). Hollow: the first version (maintained vs leaky unmaintained
    / stale agents, all sharing the paper-1 opponent model)."""
    sc = json.load(open(_rb("r8_scores.json")))

    def pts(keys):
        out = []
        for x, k in keys:
            if k in sc:
                t = sc[k]["by_truth"]["T_clean"]
                h, y = t["future_hero_wr"], t["synergy"]
                out.append((x, h["est"], _t_quantile(h["df"]) * h["se"],
                            y["est"], _t_quantile(y["df"]) * y["se"]))
        return out
    new = [(0, 0.0, 0.0, 0.0, 0.0)] + pts([(12, "Ugc_vs_S1"), (24, "Ugc_vs_S2")])
    old = [(0.8, 0.0, 0.0, 0.0, 0.0)] + pts([(12, "U_vs_S1"), (24, "U_vs_S2")])
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.3), sharex=True)
    for ax, col, ci, title, color in ((axes[0], 1, 2, "hero timing (pp)", BLUE),
                                      (axes[1], 3, 4, "synergy", GREEN)):
        ax.errorbar([p[0] + 0.8 for p in old], [p[col] for p in old],
                    yerr=[p[ci] for p in old], fmt="o--", mfc="white",
                    color="#999999", lw=1.0, ms=4.5, capsize=2.5,
                    label="2026 side on paper-1 opponent model")
        ax.errorbar([p[0] for p in new], [p[col] for p in new],
                    yerr=[p[ci] for p in new], fmt="o-", color=color,
                    lw=1.6, ms=5.5, capsize=3, label="era-matched opponent models")
        ax.axhline(0, color=GRAY, lw=0.8, ls=":")
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel("staleness (months)")
        ax.set_xticks([0, 12, 24])
        ax.set_ylim(-0.6, 2.0)
    axes[0].set_ylabel("cost of staleness")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="upper center", ncol=2, fontsize=6)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(OUT, "fig_dose.pdf"))
    print("fig_dose.pdf (era-matched, t intervals)")


def fig_detector_curves():
    """Rebuilt from detector_curves.json: clean labels, both references."""
    d = json.load(open(os.path.join(RES, "detector_curves.json")))
    ops = d["operating_points"]
    refs = d["reference_rows"]
    paper = [o for o in ops if abs(o["alpha"] - 0.05) < 1e-9][0]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.4))
    # Panel A: precision-recall, labeled sparsely
    rec = [o["recall"] for o in ops]
    pre = [o["precision"] for o in ops]
    a.plot(rec, pre, "o-", color=BLUE, lw=1.8, ms=4)
    a.plot([paper["recall"]], [paper["precision"]], "*", color=ORANGE,
           ms=13, zorder=4)
    for o, dx, dy in ((ops[0], 0.005, -0.06), (paper, 0.006, 0.015),
                      (ops[-1], -0.012, 0.03)):
        a.annotate(f"q={o['alpha']:g}", (o["recall"], o["precision"]),
                   xytext=(o["recall"] + dx, o["precision"] + dy), fontsize=7,
                   color="#444444")
    a.set_xlabel("recall (per-hero changes)")
    a.set_ylabel("precision")
    a.set_ylim(0.4, 0.97)
    # Panel B: regret vs refreshes, references as labeled open markers
    b.plot([o["refreshes"] for o in ops], [o["regret_pp"] for o in ops],
           "o-", color=BLUE, lw=1.8, ms=4, label="detector-triggered sweep")
    b.plot([paper["refreshes"]], [paper["regret_pp"]], "*", color=ORANGE,
           ms=13, zorder=4)
    nv = refs["never_refresh"]
    ev = refs["every_build_refresh"]
    b.plot([nv["refreshes"]], [nv["regret_pp"]], "s", mfc="none",
           color=RED, ms=6)
    b.annotate("never refresh", (nv["refreshes"], nv["regret_pp"]),
               xytext=(1.5, nv["regret_pp"] - 0.012), fontsize=7, color=RED)
    b.plot([ev["refreshes"]], [ev["regret_pp"]], "s", mfc="none",
           color=GREEN, ms=6)
    b.annotate("refresh every build", (ev["refreshes"], ev["regret_pp"]),
               xytext=(20.5, ev["regret_pp"] - 0.035), fontsize=7, color=GREEN)
    b.set_xlabel("stats refreshes over the 2.8-yr replay")
    b.set_ylabel("regret vs oracle (pp)")
    b.set_ylim(0.68, 1.12)
    b.set_xlim(-1.5, 37)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_detector_curves.pdf"))
    print("fig_detector_curves.pdf (rebuilt)")


def fig_vintage():
    v = json.load(open(os.path.join(REPO, "training", "drift2026", "results", "W6_VINTAGE_MATRIX.json")))["judges"]  # placeholders; values come from the c8 file
    qm26 = json.load(open(os.path.join(
        TRAINING, "qm2026", "results", "qm2026_judge.json")))["w6"]
    era = json.load(open(os.path.join(
        TRAINING, "qm2026", "results", "qm_era_judges.json")))
    # vintage date axis positions (years); ranked-2022Q1 from W8d
    ranked = [
        ("ranked 2022Q1", 2022.2, 0.4688, 0.0021),
        ("2022-07", 2022.6, v["2022-07"]["maintained_wp_mean"], v["2022-07"]["se"]),
        ("2023-07", 2023.6, v["2023-07"]["maintained_wp_mean"], v["2023-07"]["se"]),
        ("2024-07", 2024.6, v["2024-07"]["maintained_wp_mean"], v["2024-07"]["se"]),
        ("2025-07", 2025.6, v["2025-07"]["maintained_wp_mean"], v["2025-07"]["se"]),
        ("2026", 2026.1, v["2026-build"]["maintained_wp_mean"], v["2026-build"]["se"]),
    ]
    qm = [
        ("QM 2021", 2021.5, v["QM-2021"]["maintained_wp_mean"], v["QM-2021"]["se"]),
        ("QM 2022", 2022.5, era["QM-2022"]["w6"]["maintained_wp_mean"], era["QM-2022"]["w6"]["se"]),
        ("QM 2024", 2024.5, era["QM-2024"]["w6"]["maintained_wp_mean"], era["QM-2024"]["w6"]["se"]),
        ("QM 2026", 2026.35, qm26["maintained_wp_mean"], qm26["se"]),
    ]
    # Error bars: SE clustered by the 25 seed pairings (audit B18), single
    # judge seed everywhere (the 2026 ranked value is its seed-0 judge).
    # Leak-free M vs U drafts (consolidated audit C8); pairing-clustered SE.
    se = json.load(open(_rb("c8_vintage_leakfree.json")))["leakfree_M_vs_U"]
    key = {"ranked 2022Q1": "2022Q1-ranked", "2022-07": "2022-07",
           "2023-07": "2023-07", "2024-07": "2024-07", "2025-07": "2025-07",
           "2026": "2026-build", "QM 2021": "QM-2021", "QM 2022": "QM-2022",
           "QM 2024": "QM-2024", "QM 2026": "QM-2026"}
    ranked = [(n, x, se[key[n]]["mean"], se[key[n]]["se_pairing"]) for n, x, _, _ in ranked]
    qm = [(n, (2021.95 if n == "QM 2021" else x), se[key[n]]["mean"],
           se[key[n]]["se_pairing"]) for n, x, _, _ in qm]
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    ax.errorbar([p[1] for p in ranked], [p[2] for p in ranked],
                yerr=[1.96 * p[3] for p in ranked], fmt="o-", color=BLUE,
                lw=1.8, ms=4.5, capsize=2.5, label="ranked-play judges")
    ax.errorbar([p[1] for p in qm], [p[2] for p in qm],
                yerr=[1.96 * p[3] for p in qm], fmt="D--", color=RED,
                lw=1.5, ms=5, capsize=2.5,
                label="Quick Match judges\n(draft-free mode)")
    ax.axhline(0.5, color=GRAY, lw=0.8, ls=":")
    ax.annotate("unmaintained\nagent ahead", xy=(2022.75, 0.386), color=GRAY, fontsize=7)
    ax.annotate("maintained agent ahead", xy=(2021.9, 0.512), color=GRAY, fontsize=7)
    ax.set_xlabel("judge training vintage")
    ax.set_ylabel("score for the maintained side\n(same 7,200 drafts)")
    ax.set_ylim(0.38, 0.535)
    ax.set_xlim(2021.7, 2026.6)
    ax.legend(frameon=False, loc="lower right", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_vintage.pdf"))
    print("fig_vintage.pdf")


def fig_decay():
    d = json.load(open(os.path.join(RES, "signal_decay.json")))["by_lag_builds"]
    series = [
        ("comp_wr", "role-composition WR", GREEN),
        ("hero_pickrate", "hero pick rate", ORANGE),
        ("pair_with_raw", "pairwise (raw)", GRAY),
        ("hero_wr", "hero WR", BLUE),
        ("pair_with_net", "pairwise net of heroes (raw r)", RED),
    ]
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    max_lag = 16  # n_pairs >= 12 beyond this the estimates get noisy
    for key, label, color in series:
        if key not in d:
            print(f"  (skip {key}: not in decay json)")
            continue
        lags = sorted(int(k) for k in d[key] if int(k) <= max_lag)
        # attenuation-adjusted r for structured signals; raw r for the net
        # pair signal (its adjusted estimator is degenerate at ~0 reliability)
        field = "r_mean" if key.endswith("_net") else "r_adj_mean"
        ys = [d[key][str(l)][field] for l in lags]
        ax.plot(lags, ys, "-", color=color, lw=1.8, label=label)
    ax.set_xlabel("lag (builds)")
    ax.set_ylabel("self-correlation of per-build\nstatistics (disattenuated r)")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(0.33, 0.42))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_decay.pdf"))
    print("fig_decay.pdf")


def fig_scatter():
    """Leak-free (filled) vs leaky first-version (hollow) regimes, from
    drift_rebuild/results/r12_table1.json (3-seed means)."""
    t = json.load(open(os.path.join(RB, "r12_table1.json")))["rows"]
    names = {"allhist": "all-history", "win3": "window 3", "win6": "window 6",
             "win12": "window 12", "decay90": "sample decay 90d",
             "decay365": "sample decay 365d", "embed": "patch embed"}
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    for key, name in names.items():
        o, l = t[key]["oof"], t[key]["leaky"]
        ax.plot([l["future"][0], o["future"][0]], [l["degen"][0], o["degen"][0]],
                "-", color="#cccccc", lw=0.8, zorder=1)
        ax.scatter([l["future"][0]], [l["degen"][0]], s=22, facecolors="none",
                   edgecolors=GRAY, zorder=2)
        ax.scatter([o["future"][0]], [o["degen"][0]], s=26, color=GRAY, zorder=3)
    m = t["cumprev"]["oof"]
    ax.scatter([m["future"][0]], [m["degen"][0]], s=34, color=BLUE, marker="D",
               zorder=4)
    label_at = {"all-history": (56.40, 20.5), "window 3": (56.22, 54.5),
                "window 6": (56.50, 55.5), "window 12": (56.80, 42.5),
                "sample decay 90d": (56.78, 51.5), "sample decay 365d": (56.62, 27.0),
                "patch embed": (56.93, 36.5)}
    for key, name in names.items():
        o = t[key]["oof"]
        ax.annotate(name, (o["future"][0], o["degen"][0]), xytext=label_at[name],
                    fontsize=6.3, color="#444444",
                    arrowprops=dict(arrowstyle="-", color="#bbbbbb", lw=0.5,
                                    shrinkA=1, shrinkB=2))
    ax.annotate("maintained", (m["future"][0] - 0.22, m["degen"][0] - 5.0),
                fontsize=7, color=BLUE)
    ax.scatter([], [], s=26, color=GRAY, label="leak-free")
    ax.scatter([], [], s=22, facecolors="none", edgecolors=GRAY,
               label="leaky first version")
    ax.legend(frameon=False, loc="lower left", fontsize=6.5)
    ax.set_xlabel("future-window accuracy (%)")
    ax.set_ylabel("degenerate compositions\nunder greedy drafting (%)")
    ax.set_xlim(56.1, 57.25)
    ax.set_ylim(8, 59)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_scatter.pdf"))
    print("fig_scatter.pdf (leak-free vs leaky)")


if __name__ == "__main__":
    fig_dose()
    fig_vintage()
    fig_decay()
    fig_scatter()
    fig_detector_curves()
