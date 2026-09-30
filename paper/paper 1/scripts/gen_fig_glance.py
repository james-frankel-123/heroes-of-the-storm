"""Generate Figure 1 "results at a glance" (fig_glance): an aligned bar-table,
one row per tournament strategy, three inline bar columns.

All three columns are measured on the same head-to-head round-robin drafts
(rerun2026 roundrobin_summary.json merged with the 38 constrained pairs of
results/constrained/constrained_summary.json; 11 strategies, 110 ordered
pairs x 200 drafts): consensus tournament WP (mean symmetrized WP over the
four evaluators, both-orderings averaged), synergy exploitation, and
degenerate ("broken") composition rate.

The two CQL variants (cql_naive_a1.0, cql_enr_a2.0) are merged into a single
"Conservative RL (CQL, best)" row, taking the better cell per column; in the
2026 rerun data cql_enr_a2.0 supplies every cell (higher tournament WP,
higher synergy, lower degenerate rate), so the merged row is effectively
cql_enr. 10 rows total summarize the 11-strategy tournament.

Print-first design: drawn at actual IEEE column width (3.5in) so fonts render
at true size. Accent blue for the constrained champion, lighter blue for the
unconstrained MCTS row, neutral for baselines (identity is the only
categorical job here); sign carried by bar direction in the synergy column.
Style follows gen_fig_scatter.py (validated dataviz palette).
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "font.size": 7.0,
    "font.family": "sans-serif",
    "axes.linewidth": 0.6,
})

OUT = "/home/max/heroes-of-the-storm/paper/cql_followup/overleaf/"
RR = ("/home/max/heroes-of-the-storm/training/rerun2026/results/"
      "roundrobin_summary.json")
CS = ("/home/max/heroes-of-the-storm/training/rerun2026/results/"
      "constrained/constrained_summary.json")

EVS = ["naive", "herostrength", "enriched", "augmented"]
LABELS = {
    "constrained_mcts":   "Ours + constraints",
    "mcts":               "Ours (MCTS)",
    "constrained_greedy": "Greedy + constraints",
    "enriched":           "Enriched greedy search",
    "enriched_aug":       "Greedy + synthetic data",
    "gourdeau":           "WR-estimator search (G&A)",
    "cql_best":           "Conservative RL (CQL, best)",
    "gourdeau_disc":      "Prof.-manifold discrim. (G&A)",
    "gd":                 "Imitating human drafts (BC)",
    "mcq_t0.5":           "Mildly conserv. RL (MCQ)",
}

ACCENT = "#2a78d6"    # ours + constraints (validated palette, light mode)
ACCENT2 = "#8ab6e8"   # ours, unconstrained MCTS (lighter blue)
NEUTRAL = "#b9b7b1"   # baselines
TEXT = "#0b0b0b"
MUTED = "#52514e"


def load_rows():
    S = json.load(open(RR))
    C = json.load(open(CS))
    strategies = C["strategies"]                 # all 11
    pair_summaries = dict(S["pair_summaries"])
    pair_summaries.update(C["pairs"])            # 72 + 38 = 110 ordered pairs
    persp = {s: [] for s in strategies}          # consensus WP samples
    degen = {s: [] for s in strategies}
    syn = {s: [] for s in strategies}
    for key, p in pair_summaries.items():
        a, b = key.split("__")
        cons = float(np.mean([p[f"team0_wp_{ev}_sym"] for ev in EVS]))
        persp[a].append(cons)
        persp[b].append(1.0 - cons)
        degen[a].append(p["team0_degen"])
        degen[b].append(p["team1_degen"])
        syn[a].append(p["team0_synergy"])
        syn[b].append(p["team1_synergy"])
    rows = [(s, 100 * float(np.mean(persp[s])), float(np.mean(syn[s])),
             float(np.mean(degen[s]))) for s in strategies]
    # Merge the two CQL variants into one best-per-column row (see module
    # docstring): max WP, max synergy, min degenerate rate. In this data
    # cql_enr_a2.0 is the better cell in every column.
    by_name = {r[0]: r for r in rows}
    enr, nai = by_name.pop("cql_enr_a2.0"), by_name.pop("cql_naive_a1.0")
    print("CQL merge inputs:")
    for r in (enr, nai):
        print(f"  {r[0]:<16} wp={r[1]:5.1f}  syn={r[2]:+5.2f}  degen={r[3]:5.1f}")
    merged = ("cql_best", max(enr[1], nai[1]), max(enr[2], nai[2]),
              min(enr[3], nai[3]))
    rows = list(by_name.values()) + [merged]
    rows.sort(key=lambda r: -r[1])
    return rows


def main():
    rows = load_rows()
    n = len(rows)
    y = np.arange(n)[::-1]  # best on top
    ours = [s in ("constrained_mcts", "mcts") for s, *_ in rows]
    colors = [ACCENT if s == "constrained_mcts"
              else ACCENT2 if s == "mcts" else NEUTRAL
              for s, *_ in rows]

    fig = plt.figure(figsize=(3.5, 1.86))  # page-budget height; fonts true size
    gs = fig.add_gridspec(1, 3, left=0.345, right=0.975, top=0.845,
                          bottom=0.10, wspace=0.44,
                          width_ratios=[1.0, 0.9, 0.9])
    ax_wp = fig.add_subplot(gs[0])
    ax_sy = fig.add_subplot(gs[1])
    ax_dg = fig.add_subplot(gs[2])
    BH = 0.62

    # -- column 1: tournament win rate, 50% reference tick --------------
    wp = [r[1] for r in rows]
    ax_wp.barh(y, wp, height=BH, color=colors, zorder=3)
    ax_wp.axvline(50, color=MUTED, linestyle=(0, (3, 2)), linewidth=0.6,
                  zorder=4)
    ax_wp.set_xlim(0, 88)
    for yi, v, o in zip(y, wp, ours):
        ax_wp.text(v + 2, yi, f"{v:.0f}", va="center", ha="left",
                   fontsize=6.0, color=TEXT if o else MUTED,
                   fontweight="bold" if o else "normal")
    ax_wp.text(50, -0.75, "50", ha="center", va="top", fontsize=5.6,
               color=MUTED)
    ax_wp.set_title("Tournament\nconsensus WP (%)", fontsize=6.2, color=TEXT,
                    pad=3)

    # -- column 2: synergy, diverging from 0 ----------------------------
    sy = [r[2] for r in rows]
    ax_sy.barh(y, sy, height=BH, color=colors, zorder=3)
    ax_sy.axvline(0, color=MUTED, linewidth=0.6, zorder=4)
    ax_sy.set_xlim(-2.55, 2.3)
    for yi, v, o in zip(y, sy, ours):
        if v >= 0:
            ax_sy.text(v + 0.09, yi, f"+{v:.2f}", va="center", ha="left",
                       fontsize=6.0, color=TEXT if o else MUTED,
                       fontweight="bold" if o else "normal")
        else:
            ax_sy.text(v - 0.09, yi, f"−{-v:.2f}", va="center",
                       ha="right", fontsize=6.0, color=MUTED)
    ax_sy.set_title("Synergy", fontsize=6.2, color=TEXT, pad=3)

    # -- column 3: broken-team rate, lower = better ----------------------
    # Zero/near-zero rates get a minimum hairline stub so every row shows
    # its bar (and its row color) in this column; labels carry the true
    # value.
    dg = [r[3] for r in rows]
    dg_draw = [max(v, 0.9) for v in dg]
    ax_dg.barh(y, dg_draw, height=BH, color=colors, zorder=3)
    ax_dg.set_xlim(0, 62)
    for yi, v, vd, o in zip(y, dg, dg_draw, ours):
        ax_dg.text(vd + 1.6, yi, f"{v:.0f}", va="center", ha="left",
                   fontsize=6.0, color=TEXT if o else MUTED,
                   fontweight="bold" if o else "normal")
    ax_dg.set_title("Broken teams %\n(lower is better)", fontsize=6.2,
                    color=TEXT, pad=3)

    # -- shared row labels + cleanup -------------------------------------
    for ax in (ax_wp, ax_sy, ax_dg):
        ax.set_ylim(-0.55, n - 0.45)
        ax.set_yticks([])
        ax.set_xticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    for (s, *_), yi, o in zip(rows, y, ours):
        fig.text(0.005, ax_wp.get_position().y0
                 + (yi + 0.5) / n * (ax_wp.get_position().height),
                 LABELS[s], ha="left", va="center", fontsize=6.0,
                 color=TEXT, fontweight="bold" if o else "normal")

    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}fig_glance.{ext}", dpi=400)
    print("Saved fig_glance.pdf/.png")
    for s, w, sv, d in rows:
        print(f"  {LABELS[s]:<32} wp={w:5.1f}  syn={sv:+5.2f}  degen={d:5.1f}")


if __name__ == "__main__":
    main()
