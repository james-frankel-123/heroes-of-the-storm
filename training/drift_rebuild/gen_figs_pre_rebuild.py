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
RES = os.path.join(REPO, "training", "drift2026", "results")
TRAINING = os.path.join(REPO, "training")
OUT = os.path.join(REPO, "paper", "drift", "overleaf")

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 200, "axes.spines.top": False, "axes.spines.right": False,
})
BLUE, ORANGE, GREEN, GRAY, RED = "#2166ac", "#e08214", "#1b7837", "#666666", "#b2182b"


def fig_dose():
    """Two panels: what rots (hero timing) vs what keeps (synergy)."""
    x = [0, 12, 24]  # months of staleness beyond the shared training cutoff
    # W12 out-of-sample truth (T_clean), crossed-RE
    w12 = json.load(open(os.path.join(RES, "w12_clean_truth.json")))["files"]
    rows = [w12[k]["by_truth"]["T_clean"] for k in ("_w6", "_stale1yr", "_stale2yr")]
    vm = w12["_w8_volmatch"]["by_truth"]["T_clean"]["future_hero_wr"]
    hero = [r["future_hero_wr"]["est"] for r in rows]
    hero_se = [r["future_hero_wr"]["se"] for r in rows]
    vm_y, vm_se = vm["est"], vm["se"]
    syn = [r["synergy"]["est"] for r in rows]
    syn_se = [r["synergy"]["se"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3), sharex=True)
    a = axes[0]
    a.errorbar(x, hero, yerr=[1.96 * s for s in hero_se], fmt="o-",
               color=BLUE, lw=1.8, ms=5, capsize=3, label="unmaintained deficit")
    a.errorbar([25.5], [vm_y], yerr=[1.96 * vm_se], fmt="s", color=ORANGE,
               ms=5, capsize=3, label="volume-matched control")
    a.axhline(0, color=GRAY, lw=0.8, ls=":")
    a.set_ylabel("hero-WR deficit (pp)")
    a.set_title("hero timing: lost, and more at two years", fontsize=8.5)
    a.set_ylim(-0.3, 1.5)
    a.legend(frameon=False, loc="lower right", fontsize=7)
    b = axes[1]
    b.errorbar(x, syn, yerr=[1.96 * s for s in syn_se], fmt="o-",
               color=GREEN, lw=1.8, ms=5, capsize=3)
    b.axhline(0, color=GRAY, lw=0.8, ls=":")
    b.set_title("synergy: small gap at every staleness", fontsize=8.5)
    b.set_ylabel("synergy deficit")
    b.set_ylim(-0.3, 1.5)
    for ax in axes:
        ax.set_xlabel("staleness beyond the training cutoff (months)")
        ax.set_xticks([0, 12, 24])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_dose.pdf"))
    print("fig_dose.pdf")


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
    v = json.load(open(os.path.join(RES, "W6_VINTAGE_MATRIX.json")))["judges"]
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
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    ax.errorbar([p[1] for p in ranked], [p[2] for p in ranked],
                yerr=[1.96 * p[3] for p in ranked], fmt="o-", color=BLUE,
                lw=1.8, ms=4.5, capsize=2.5, label="ranked-play judges")
    ax.errorbar([p[1] for p in qm], [p[2] for p in qm],
                yerr=[1.96 * p[3] for p in qm], fmt="D--", color=RED,
                lw=1.5, ms=5, capsize=2.5,
                label="Quick Match judges\n(draft-free mode)")
    ax.axhline(0.5, color=GRAY, lw=0.8, ls=":")
    ax.annotate("unmaintained agent\ndeclared winner", xy=(2021.25, 0.4445), color=GRAY, fontsize=7)
    ax.annotate("maintained agent ahead", xy=(2021.3, 0.506), color=GRAY, fontsize=7)
    ax.set_xlabel("judge training vintage")
    ax.set_ylabel("score for the maintained side\n(same 2{,}000 drafts)".replace("{,}", ","))
    ax.set_ylim(0.40, 0.535)
    ax.legend(frameon=False, loc="lower right")
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
        ("pair_with_net", "pairwise (net of heroes)", RED),
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
    ax.set_ylabel("self-correlation of\nper-build statistics")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(0.33, 0.42))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_decay.pdf"))
    print("fig_decay.pdf")


def fig_scatter():
    # (future-window acc, greedy degenerate %) per training regime.
    # W2B_SUMMARY.md + W2B_FILL.md (win3/win12/decay365 filled 2026-07-14).
    fill_path = os.path.join(RES, "w2b_fill.json")
    pts = [
        ("all-history", 56.21, 43.5),
        ("window 6", 56.53, 49.7),
        ("decay 90d", 56.48, 50.1),
        ("patch embed", 56.55, 47.0),
        ("maintained", 57.08, 16.8),
    ]
    if os.path.exists(fill_path):
        fill = json.load(open(fill_path))["cells"]
        label_map = {"d2b_win3": "window 3", "d2b_win12": "window 12",
                     "d2b_decay365": "decay 365d"}
        for cell, name in label_map.items():
            if cell in fill:
                acc = float(fill[cell]["future_acc"][0])
                deg = float(fill[cell]["policy"]["degen"][0])
                pts.append((name, acc, deg))
    # Maintained-family points (stats-refresh axis): W2B_DECAYED.md.
    fam = [("maintained", 57.08, 16.8),
           ("decayed stats", 57.19, 18.9),
           ("decayed + shrink", 57.22, 19.1)]
    # Hand-placed label positions with leader lines: seven regimes cluster in
    # a 0.35pp x 7pp box, so labels ring the cluster instead of sitting on it.
    label_pos = {
        "all-history": (55.98, 40.6),
        "window 3": (56.05, 44.6),
        "decay 365d": (56.05, 49.6),
        "window 6": (56.62, 53.6),
        "window 12": (56.18, 54.0),
        "decay 90d": (56.68, 50.8),
        "patch embed": (56.70, 45.4),
        "maintained": (56.82, 11.6),
        "decayed stats": (56.84, 28.0),
        "decayed + shrink": (57.23, 23.6),
    }
    cluster = [p for p in pts if p[0] != "maintained"]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    # Within-cluster trend: accuracy ranks the standard regimes BACKWARDS.
    import numpy as np
    xs = np.array([p[1] for p in cluster]); ys = np.array([p[2] for p in cluster])
    b1, b0 = np.polyfit(xs, ys, 1)
    tx = np.array([56.16, 56.60])
    ax.plot(tx, b0 + b1 * tx, "--", color="#c08a8a", lw=1.1, zorder=1)
    ax.annotate("within standard regimes:\nmore accurate, more degenerate\n(r = +0.78)",
                (56.28, 30.5), fontsize=6.5, color="#a06060")
    for name, acc, deg in cluster + fam:
        is_fam = name in ("maintained", "decayed stats", "decayed + shrink")
        color = BLUE if is_fam else GRAY
        ax.scatter([acc], [deg], s=30, color=color, zorder=3,
                   marker="o" if name != "maintained" else "D")
    for name, acc, deg in cluster + fam:
        is_fam = name in ("maintained", "decayed stats", "decayed + shrink")
        lx, ly = label_pos.get(name, (acc + 0.03, deg + 1.2))
        ax.annotate(name, (acc, deg), xytext=(lx, ly), fontsize=7,
                    color=BLUE if is_fam else "#444444", zorder=2,
                    arrowprops=dict(arrowstyle="-", color="#bbbbbb", lw=0.6,
                                    shrinkA=1, shrinkB=2))
    ax.set_xlabel("future-window accuracy (%)")
    ax.set_ylabel("degenerate compositions\nunder greedy drafting (%)")
    ax.set_xlim(55.95, 57.5)
    ax.set_ylim(8, 58)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_scatter.pdf"))
    print("fig_scatter.pdf")


if __name__ == "__main__":
    fig_dose()
    fig_vintage()
    fig_decay()
    fig_scatter()
    fig_detector_curves()
