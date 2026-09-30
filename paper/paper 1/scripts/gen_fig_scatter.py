"""Generate safety vs. context-awareness scatter plot (Fig. safety_context).

Print-first design: figure is drawn at actual IEEE column width (3.5in) so
fonts render at true size. Marker shape + validated categorical palette
(dataviz six-checks: worst adjacent CVD dE 47.2) carry family identity;
every mark is direct-labeled (relief rule for low-contrast slots).

Data: Table tab:rich (rerun2026 rich_evaluation_results.json, 5x1000 drafts)
+ rerun2026 mcts_experiment_results.json aggregates (B_fullwp, H_augmented, J_800sim)
+ rerun2026 results/constrained/constrained_summary.json rich_eval (constrained_mcts).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.size": 7.5,
    "font.family": "sans-serif",
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
})

OUT = "/home/max/heroes-of-the-storm/paper/cql_followup/overleaf/"

# name, degen%, synergy, counter, entropy, family
POINTS = [
    ("Gourdeau est.",      41.1, +0.31, -0.05, 5.75, "vf"),
    ("GD",              1.2, -0.20, +0.06, 4.37, "bc"),
    ("G&A disc.",      0.0, +0.09, +0.07, 3.96, "bc"),
    ("CQL .5 naive",    3.6, -0.19, +0.07, 4.52, "cql"),
    ("CQL 1.0 naive",   3.0, -0.16, +0.07, 4.52, "cql"),
    ("CQL .5 enr.",     2.3, -0.16, +0.05, 4.29, "cql"),
    ("CQL 2.0 enr.",    2.5, -0.20, +0.06, 4.45, "cql"),
    ("MCQ",            45.4, -1.60, +0.01, 2.87, "mcq"),
    ("BC-CQL",          2.2, -0.16, +0.06, 4.50, "cql"),
    ("Enriched",       18.5, +0.85, +0.25, 6.08, "vf"),
    ("Enr.+aug",       18.4, +0.76, +0.21, 6.11, "vf"),
    ("MCTS 200s",      11.0, +1.20, +0.028, 4.8, "mcts"),
    ("MCTS aug",        8.3, +1.10, +0.019, 4.7, "mcts"),
    ("MCTS 800s",       7.5, +1.44, +0.074, 5.0, "mcts"),
    ("MCTS constr.",    0.0, +0.95, +0.20, 3.84, "mcts"),
]

# Validated palette (dataviz reference, light mode): CVD-safe as a set.
FAMILY = {
    "mcts": dict(marker="p", color="#2a78d6", label="MCTS"),
    "vf":   dict(marker="o", color="#1baf7a", label="Value fn. search"),
    "cql":  dict(marker="s", color="#4a3aa7", label="CQL / BC-CQL"),
    "mcq":  dict(marker="^", color="#e34948", label="MCQ"),
    "bc":   dict(marker="D", color="#eda100", label="Behavioral anchoring"),
}

TEXT = "#0b0b0b"
MUTED = "#52514e"

# Direct-label placement in data coords: (x, y, ha).
# CQL cluster points share one bracket label; others labeled individually.
LABELS_SYNERGY = {
    "Gourdeau est.":     (41.1, 0.50, "right"),
    "G&A disc.":   (2.2, 0.18, "left"),
    "MCQ":          (43.5, -1.60, "right"),
    "Enriched":     (20.2, 0.88, "left"),
    "Enr.+aug":     (20.1, 0.70, "left"),
    "MCTS 200s":    (13.0, 1.19, "left"),
    "MCTS aug":     (10.3, 1.03, "left"),
    "MCTS 800s":    (9.5, 1.47, "left"),
    "MCTS constr.": (1.5, 0.80, "left"),
}


def make_plot(y_idx, y_label, stem):
    fig, ax = plt.subplots(figsize=(3.5, 2.12))  # tightened for page budget

    for name, degen, syn, ctr, ent, fam in POINTS:
        st = FAMILY[fam]
        y = (syn, ctr)[y_idx - 2] if False else (syn if y_idx == 2 else ctr)
        size = max(18, (ent ** 2) * 2.6)
        ax.scatter(degen, y, s=size, marker=st["marker"], facecolors=st["color"],
                   edgecolors="white", linewidth=0.7, zorder=3)

    if y_idx == 2:
        # Individual labels
        for name, degen, syn, ctr, ent, fam in POINTS:
            if name not in LABELS_SYNERGY:
                continue
            lx, ly, ha = LABELS_SYNERGY[name]
            ax.annotate(name, (degen, syn), xytext=(lx, ly), ha=ha,
                        va="center", fontsize=6.2, color=TEXT, zorder=4)
        # One bracket label for the overlapping CQL/BC-CQL cluster
        ax.annotate("GD + CQL / BC-CQL (6 variants):\nsafe but not interaction-aware",
                    xy=(4.2, -0.20),
                    xytext=(13.5, -0.36), fontsize=6.2, color=TEXT,
                    ha="left", va="center", zorder=4,
                    arrowprops=dict(arrowstyle="-", lw=0.5, color=MUTED,
                                    shrinkA=0, shrinkB=2))

        # Region annotation (muted, italic)
        ax.text(34, 1.30, "interaction-aware,\nsafe only with search", fontsize=6.0,
                style="italic", color=MUTED, ha="center", va="center")

    ax.axhline(y=0, color=MUTED, linestyle=(0, (4, 3)), alpha=0.5, linewidth=0.6)
    ax.set_xlabel("Degenerate composition rate (%)  (lower = safer)",
                  fontsize=7.5, color=TEXT)
    ax.set_ylabel(y_label, fontsize=6.2, color=TEXT)  # small so the long label fits the tightened height
    ax.set_xlim(-1.5, 55)
    if y_idx == 2:
        ax.set_ylim(-1.80, 1.62)
    ax.tick_params(labelsize=7, colors=TEXT)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.grid(True, linewidth=0.3, alpha=0.25)
    ax.set_axisbelow(True)

    handles = [plt.Line2D([], [], linestyle="", marker=st["marker"],
                          markerfacecolor=st["color"], markeredgecolor="white",
                          markeredgewidth=0.5, markersize=6, label=st["label"])
               for st in FAMILY.values()]
    ax.legend(handles=handles, loc="lower left", fontsize=5.9,
              frameon=True, framealpha=0.9, edgecolor="#d8d7d2",
              borderpad=0.4, handletextpad=0.4, labelspacing=0.28)

    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}{stem}.{ext}", dpi=400)
    print(f"Saved {stem}.pdf/.png")
    plt.close(fig)


make_plot(2, "Synergy  (higher = interaction-aware)", "fig_safety_context")
make_plot(3, "Counter responsiveness", "fig_safety_context_counter")
