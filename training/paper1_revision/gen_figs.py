"""
Revised Fig. 1 (results at a glance) and Fig. scatter (safety vs synergy) for
draft_revision.tex, from the revision's result files. Same layout and palette
as paper/paper 1/scripts/gen_fig_glance.py and gen_fig_scatter.py; writes NEW
files fig_glance_revision.{pdf,png} and fig_safety_context_revision.{pdf,png}
into paper/paper 1/overleaf/ (the submitted figures are untouched).

Glance columns: tournament consensus WP under the independent judges (mean
of gN, gN_naive, RN, QM2026; composition-corrected v2 judges from
results/comp_rescore.json when present), synergy (external-statistics metric), and
degenerate-team rate, all on the revised round-robin drafts.
Scatter: rich-evaluation rows (unchanged baselines from the submission's
rich_evaluation_results.json; leak-free greedy/MCTS rows from greedy_rich.json)
plus MCTS benchmark configs from SUMMARY.json.
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from paper1_revision import core

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/max/heroes-of-the-storm/paper/paper 1/overleaf/"
matplotlib.rcParams.update({"font.size": 7.0, "font.family": "sans-serif", "axes.linewidth": 0.6})
ACCENT, ACCENT2, NEUTRAL = "#2a78d6", "#8ab6e8", "#b9b7b1"
TEXT, MUTED = "#0b0b0b", "#52514e"
LABELS = {
    "constrained_mcts": "Ours + constraints", "mcts": "Ours (MCTS)",
    "constrained_greedy": "Greedy + constraints", "enriched": "Enriched greedy search",
    "enriched_aug": "Greedy + synthetic data", "k_truebase": "MCTS, no domain features",
    "gourdeau": "WR-estimator search (G&A)", "cql_best": "Conservative RL (CQL, best)",
    "gourdeau_disc": "Prof.-manifold discrim. (G&A)", "gd": "Imitating human drafts (BC)",
    "mcq_t0.5": "Mildly conserv. RL (MCQ)",
}


def glance():
    T = json.load(open(os.path.join(core.RESULTS, "tournament_standings.json")))
    st, dm = T["standings"], T["draft_metrics"]
    # composition-corrected (v2) judges when the rescore exists (comp_rescore.py)
    v2 = os.path.join(core.RESULTS, "comp_rescore.json")
    if os.path.exists(v2):
        st = json.load(open(v2))["tournament"]["v2"]["standings"]
    rows = []
    for s in st:
        rows.append((s, 100 * st[s]["consensus"]["mean"], dm[s]["synergy"], 100 * dm[s]["degen"]))
    by = {r[0]: r for r in rows}
    enr, nai = by.pop("cql_enr_a2.0"), by.pop("cql_naive_a1.0")
    rows = list(by.values()) + [("cql_best", max(enr[1], nai[1]), max(enr[2], nai[2]),
                                 min(enr[3], nai[3]))]
    rows.sort(key=lambda r: -r[1])
    n = len(rows)
    y = np.arange(n)[::-1]
    ours = [s in ("constrained_mcts", "mcts") for s, *_ in rows]
    colors = [ACCENT if s == "constrained_mcts" else ACCENT2 if s == "mcts" else NEUTRAL
              for s, *_ in rows]
    fig = plt.figure(figsize=(3.5, 1.95))
    gs = fig.add_gridspec(1, 3, left=0.345, right=0.975, top=0.85, bottom=0.10, wspace=0.44,
                          width_ratios=[1.0, 0.9, 0.9])
    ax_wp, ax_sy, ax_dg = (fig.add_subplot(gs[i]) for i in range(3))
    BH = 0.62
    wp = [r[1] for r in rows]
    ax_wp.barh(y, wp, height=BH, color=colors, zorder=3)
    ax_wp.axvline(50, color=MUTED, linestyle=(0, (3, 2)), linewidth=0.6, zorder=4)
    ax_wp.set_xlim(0, 80)
    for yi, v, o in zip(y, wp, ours):
        ax_wp.text(v + 2, yi, f"{v:.1f}", va="center", ha="left", fontsize=5.8, zorder=5,
                   color=TEXT if o else MUTED, fontweight="bold" if o else "normal",
                   bbox=dict(facecolor="white", edgecolor="none", pad=0.3))
    ax_wp.set_title("Tournament WP (%),\nindependent judges", fontsize=6.0, color=TEXT, pad=3)
    sy = [r[2] for r in rows]
    ax_sy.barh(y, sy, height=BH, color=colors, zorder=3)
    ax_sy.axvline(0, color=MUTED, linewidth=0.6, zorder=4)
    ax_sy.set_xlim(-2.55, 2.3)
    for yi, v, o in zip(y, sy, ours):
        if v >= 0:
            ax_sy.text(v + 0.09, yi, f"+{v:.2f}", va="center", ha="left", fontsize=5.8,
                       color=TEXT if o else MUTED, fontweight="bold" if o else "normal")
        else:
            ax_sy.text(v - 0.09, yi, f"−{-v:.2f}", va="center", ha="right", fontsize=5.8,
                       color=MUTED)
    ax_sy.set_title("Synergy", fontsize=6.0, color=TEXT, pad=3)
    dg = [r[3] for r in rows]
    dgd = [max(v, 0.9) for v in dg]
    ax_dg.barh(y, dgd, height=BH, color=colors, zorder=3)
    ax_dg.set_xlim(0, 62)
    for yi, v, vd, o in zip(y, dg, dgd, ours):
        ax_dg.text(vd + 1.6, yi, f"{v:.0f}", va="center", ha="left", fontsize=5.8,
                   color=TEXT if o else MUTED, fontweight="bold" if o else "normal")
    ax_dg.set_title("Broken teams %\n(lower is better)", fontsize=6.0, color=TEXT, pad=3)
    for ax in (ax_wp, ax_sy, ax_dg):
        ax.set_ylim(-0.55, n - 0.45)
        ax.set_yticks([])
        ax.set_xticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    for (s, *_), yi, o in zip(rows, y, ours):
        fig.text(0.005, ax_wp.get_position().y0 + (yi + 0.5) / n * ax_wp.get_position().height,
                 LABELS[s], ha="left", va="center", fontsize=5.8, color=TEXT,
                 fontweight="bold" if o else "normal")
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}fig_glance_revision.{ext}", dpi=400)
    plt.close(fig)
    for r in rows:
        print(f"  {LABELS[r[0]]:<32} wp={r[1]:5.1f} syn={r[2]:+.2f} degen={r[3]:.1f}")


FAMILY = {"mcts": dict(marker="p", color="#2a78d6", label="MCTS"),
          "vf": dict(marker="o", color="#1baf7a", label="Value fn. search"),
          "cql": dict(marker="s", color="#4a3aa7", label="CQL / BC-CQL / IQL"),
          "mcq": dict(marker="^", color="#e34948", label="MCQ"),
          "bc": dict(marker="D", color="#eda100", label="Behavioral anchoring")}


def scatter():
    old = json.load(open(os.path.join(core.TRAINING_DIR, "rerun2026", "results",
                                      "rich_evaluation_results.json")))
    pts = []
    fam = {"gourdeau": "vf", "gd": "bc", "gourdeau_discriminator": "bc", "mcq_t0.5": "mcq"}
    for k, v in (old.items() if isinstance(old, dict) else []):
        m = v.get("metrics", v) if isinstance(v, dict) else None
        if not m or "degen_rate" not in m:
            continue
        if k in ("enriched", "enriched_aug"):
            continue                                  # replaced by leak-free rows
        f = fam.get(k, "cql" if ("cql" in k or "iql" in k) else None)
        if f is None:
            continue
        pts.append((k, m["degen_rate"], m["synergy"], m.get("entropy", 4.5), f))
    G = json.load(open(os.path.join(core.RESULTS, "greedy_rich.json")))
    for k, lab, f in (("enriched", "Enriched", "vf"),
                      ("constrained_mcts", "MCTS constr.", "mcts")):
        if k in G:
            pts.append((lab, G[k]["degen"], G[k]["synergy"], G[k]["entropy"], f))
    S = json.load(open(os.path.join(core.RESULTS, "SUMMARY.json")))["mcts"]["configs"]
    for k, lab in (("new:B_oof|T1", "MCTS 200s"), ("new:F_oof|T1", "MCTS 400s"),
                   ("new:J_oof|T1", "MCTS 800s")):
        if k in S:
            v = S[k]
            pts.append((lab, 100 * v["mean"]["degen"], v["mean"]["syn_hp"], v["entropy"], "mcts"))
    fig, ax = plt.subplots(figsize=(3.5, 2.12))
    for name, dg, syn, ent, f in pts:
        st = FAMILY[f]
        ax.scatter(dg, syn, s=max(18, ent ** 2 * 2.6), marker=st["marker"], facecolors=st["color"],
                   edgecolors="white", linewidth=0.7, zorder=3)
        if f in ("mcts", "vf", "mcq") or name == "gourdeau_discriminator":
            lab = {"gourdeau": "Gourdeau est.", "gourdeau_discriminator": "G&A disc.",
                   "mcq_t0.5": "MCQ"}.get(name, name)
            ax.annotate(lab, (dg, syn), xytext=(4, 2), textcoords="offset points", fontsize=5.8,
                        color=TEXT, zorder=4)
    ax.axhline(0, color=MUTED, linestyle=(0, (4, 3)), alpha=0.5, linewidth=0.6)
    ax.set_xlabel("Degenerate composition rate (%)  (lower = safer)", fontsize=7.5, color=TEXT)
    ax.set_ylabel("Synergy  (higher = interaction-aware)", fontsize=6.2, color=TEXT)
    ax.set_xlim(-1.5, 55)
    ax.tick_params(labelsize=7, colors=TEXT)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, linewidth=0.3, alpha=0.25)
    ax.set_axisbelow(True)
    handles = [plt.Line2D([], [], linestyle="", marker=st["marker"], markerfacecolor=st["color"],
                          markeredgecolor="white", markersize=6, label=st["label"])
               for st in FAMILY.values()]
    ax.legend(handles=handles, loc="lower left", fontsize=5.9, frameon=True, framealpha=0.9,
              edgecolor="#d8d7d2")
    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}fig_safety_context_revision.{ext}", dpi=400)
    plt.close(fig)
    for p in pts:
        print(f"  {p[0]:<28} degen={p[1]:5.1f} syn={p[2]:+.2f}")


if __name__ == "__main__":
    glance()
    scatter()
