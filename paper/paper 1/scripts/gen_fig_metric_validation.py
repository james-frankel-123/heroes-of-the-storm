"""Generate metric-validation figure (fig_metric_validation) for the supplementary.

Question: do the interaction metrics (team synergy delta, team counter delta)
carry real outcome signal? On the held-out test split we bin per-team drafts
by metric decile and plot the ACTUAL mean win rate (ground truth outcomes,
95% binomial CI) per decile.

Finding: the raw pooled curve is NOT monotone, and the enriched WP model's
predictions track the raw zigzag closely (systematic, not noise). Cause: by
construction the deltas subtract an independence expectation built from
individual hero win rates, so they anti-correlate strongly with team hero
strength (corr(synergy delta, team mean hero WR) = -0.87): teams of
individually strong heroes regress toward the mean (negative delta) yet win
more. The figure therefore shows both the raw pooled curve and a
strength-adjusted curve (deciles within tier x own/opp team mean-hero-WR
quintile strata, count-weighted standardization); the adjusted relationship
is the validity test for the interaction component the metrics are designed
to isolate.

Data pinning (identical to rerun2026):
  - replay snapshot 2026-05-22, patch-2.55 filtered, replay-level split
    (seed 42, test_frac 0.02) -> ~39K held-out test replays, never used in
    training any model or statistic;
  - frozen HeroesProfile stats 2026-05-19 (StatsCache);
  - metrics are the exact functions used throughout the paper:
    experiment_rich_evaluation.synergy_exploitation (mean pairwise synergy
    delta over the C(5,2) teammate pairs) and .counter_responsiveness (mean
    pairwise counter delta over the 5x5 cross-team pairs), which wrap
    experiment_draft_quality.synergy_delta / counter_delta.
  Each test replay contributes two observations (one per side), outcome 1 if
  that side won. WP reference: wp_enriched_256.pt, team-swap symmetrized.

Style per gen_fig_scatter.py (IEEE column width, validated palette).

Usage:
  python3 gen_fig_metric_validation.py            # compute (slow: loads 2.9GB snapshot) + plot
  python3 gen_fig_metric_validation.py --plot-only  # replot from metric_validation_cache.npz
"""
import os
import sys
import argparse

HERE = "/home/max/heroes-of-the-storm/paper/cql_followup/overleaf"
TRAINING = "/home/max/heroes-of-the-storm/training"
sys.path.insert(0, TRAINING)

CACHE = os.path.join(HERE, "metric_validation_cache.npz")
N_BINS = 10

import numpy as np


def compute():
    from rerun2026 import common
    common.setup()
    import torch
    from experiment_rich_evaluation import (counter_responsiveness,
                                            synergy_exploitation)
    from sweep_enriched_wp import (WinProbEnrichedModel, FEATURE_GROUPS,
                                   compute_group_indices, extract_features)
    from experiment_synthetic_augmentation import ENRICHED_GROUPS

    stats = common.stats_cache()
    _, test = common.load_split()

    gi = compute_group_indices()
    cols = []
    for g in ENRICHED_GROUPS:
        s, e = gi[g]
        cols.extend(range(s, e))
    all_mask = [True] * len(FEATURE_GROUPS)
    model = WinProbEnrichedModel(197 + len(cols), [256, 128], dropout=0.3)
    model.load_state_dict(torch.load(
        os.path.join(common.MODELS_DIR, "wp_enriched_256.pt"),
        weights_only=True, map_location="cpu"))
    model.eval()

    def wp_raw(d):
        base, enr = extract_features(d, stats, all_mask)
        x = np.concatenate([base, enr[cols]])
        with torch.no_grad():
            return model(torch.tensor(x, dtype=torch.float32).unsqueeze(0)).item()

    from shared import SKILL_TIERS

    def team_wr(heroes, tier):
        return float(np.mean([stats.get_hero_wr(h, tier) for h in heroes]))

    syn, ctr, win, wp, twr, owr, tid = [], [], [], [], [], [], []
    for i, r in enumerate(test):
        t0, t1 = r["team0_heroes"], r["team1_heroes"]
        if len(t0) != 5 or len(t1) != 5:
            continue
        tier, gmap, winner = r["skill_tier"], r["game_map"], r["winner"]
        # symmetrized P(team0 wins)
        p0 = wp_raw({"team0_heroes": t0, "team1_heroes": t1, "game_map": gmap,
                     "skill_tier": tier, "winner": 0})
        p0s = wp_raw({"team0_heroes": t1, "team1_heroes": t0, "game_map": gmap,
                      "skill_tier": tier, "winner": 0})
        p_team0 = (p0 + (1.0 - p0s)) / 2.0
        for side, (a, b) in enumerate(((t0, t1), (t1, t0))):
            syn.append(synergy_exploitation(a, stats, tier))
            ctr.append(counter_responsiveness(a, b, stats, tier))
            win.append(1.0 if winner == side else 0.0)
            wp.append(p_team0 if side == 0 else 1.0 - p_team0)
            twr.append(team_wr(a, tier))
            owr.append(team_wr(b, tier))
            tid.append(SKILL_TIERS.index(tier))
        if (i + 1) % 5000 == 0:
            print(f"  {i+1}/{len(test)} test replays", flush=True)

    arrs = {k: np.asarray(v, dtype=np.float64)
            for k, v in (("syn", syn), ("ctr", ctr), ("win", win), ("wp", wp),
                         ("twr", twr), ("owr", owr), ("tid", tid))}
    np.savez_compressed(CACHE, **arrs)
    print(f"Cached {len(win)} team-observations -> {CACHE}")
    return arrs


def qbin(x, n):
    qs = np.quantile(x, np.linspace(0, 1, n + 1))
    qs[0] -= 1e-9
    return np.clip(np.searchsorted(qs, x, side="left") - 1, 0, n - 1)


def curve_raw(metric, win, n_bins=N_BINS):
    """Pooled decile curve: (win%, ci%) per decile."""
    idx = qbin(metric, n_bins)
    y, ci = [], []
    for b in range(n_bins):
        m = idx == b
        p = win[m].mean()
        y.append(p * 100)
        ci.append(1.96 * np.sqrt(p * (1 - p) / m.sum()) * 100)
    return np.array(y), np.array(ci)


def curve_adjusted(metric, win, tid, twr, owr, n_bins=N_BINS):
    """Strength-adjusted decile curve: deciles computed WITHIN strata of
    (tier x own-team mean hero WR quintile x opposing-team mean hero WR
    quintile), then directly standardized (count-weighted) across strata.
    Controls for the individual-strength component that the deltas subtract
    by construction (corr(synergy delta, team mean hero WR) = -0.87)."""
    key = tid * 100 + qbin(twr, 5) * 10 + qbin(owr, 5)
    wins = np.zeros(n_bins)
    ns = np.zeros(n_bins)
    for s in np.unique(key):
        m = key == s
        if m.sum() < 200:
            continue
        idx = qbin(metric[m], n_bins)
        for b in range(n_bins):
            mb = idx == b
            wins[b] += win[m][mb].sum()
            ns[b] += mb.sum()
    p = wins / ns
    return p * 100, 1.96 * np.sqrt(p * (1 - p) / ns) * 100


def _ranks(a):
    """Average ranks (ties -> mean rank), numpy only."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(x, y):
    rx, ry = _ranks(x), _ranks(y)
    rx -= rx.mean()
    ry -= ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))


def plot(arrs):
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
    TEXT = "#0b0b0b"
    MUTED = "#52514e"
    C_ACTUAL = "#2a78d6"   # blue  (validated categorical palette, slot 1)
    C_PRED = "#eda100"     # amber (slot: behavioral anchoring in fig_scatter)

    tid, twr, owr, win = arrs["tid"], arrs["twr"], arrs["owr"], arrs["win"]
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.9), sharey=True)
    specs = [("syn", "Team synergy delta decile"),
             ("ctr", "Team counter delta decile")]
    x = np.arange(1, N_BINS + 1)
    for ax, (key, xlabel) in zip(axes, specs):
        y_raw, _ = curve_raw(arrs[key], win)
        y_adj, ci = curve_adjusted(arrs[key], win, tid, twr, owr)
        ax.axhline(50, color=MUTED, linestyle=(0, (4, 3)), alpha=0.5,
                   linewidth=0.6)
        ax.plot(x, y_raw, color=MUTED, linewidth=0.9, linestyle=(0, (5, 2)),
                marker="s", markersize=2.0, markeredgecolor="white",
                markeredgewidth=0.4, alpha=0.75, zorder=2,
                label="Raw (pooled)")
        ax.fill_between(x, y_adj - ci, y_adj + ci, color=C_ACTUAL,
                        alpha=0.18, linewidth=0)
        ax.plot(x, y_adj, color=C_ACTUAL, linewidth=1.2, marker="o",
                markersize=2.6, markeredgecolor="white", markeredgewidth=0.4,
                zorder=3, label="Strength-adjusted")
        ax.set_xlabel(xlabel, fontsize=7.2, color=TEXT)
        ax.set_xticks([1, 4, 7, 10])
        ax.tick_params(labelsize=6.8, colors=TEXT)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
        ax.grid(True, linewidth=0.3, alpha=0.25)
        ax.set_axisbelow(True)
        ax.text(0.04, 0.96, f"{y_adj[-1] - y_adj[0]:+.1f}pp",
                transform=ax.transAxes, fontsize=6.4, color=C_ACTUAL,
                ha="left", va="top")

    axes[0].set_ylabel("Actual win rate (%)", fontsize=7.5, color=TEXT)
    axes[0].legend(loc="lower right", fontsize=5.8, frameon=True,
                   framealpha=0.9, edgecolor="#d8d7d2", borderpad=0.4,
                   handletextpad=0.4, labelspacing=0.3)
    fig.tight_layout(pad=0.4, w_pad=0.8)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"fig_metric_validation.{ext}"), dpi=400)
    print("Saved fig_metric_validation.pdf/.png")

    # Console summary for the supplementary text
    for key, name in (("syn", "synergy"), ("ctr", "counter")):
        y_raw, ci_raw = curve_raw(arrs[key], win)
        y_adj, ci = curve_adjusted(arrs[key], win, tid, twr, owr)
        rho = spearman(arrs[key], win)
        print(f"{name}: raw decile spread {y_raw[-1] - y_raw[0]:+.1f}pp "
              f"({sum(1 for d in np.diff(y_raw) if d > 0)}/9 increasing, "
              f"rho_s={rho:.3f}); adjusted spread {y_adj[-1] - y_adj[0]:+.1f}pp "
              f"({sum(1 for d in np.diff(y_adj) if d > 0)}/9 increasing), "
              f"bottom {y_adj[0]:.1f}±{ci[0]:.1f}%, top {y_adj[-1]:.1f}±{ci[-1]:.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    if args.plot_only and os.path.exists(CACHE):
        arrs = dict(np.load(CACHE))
    else:
        arrs = compute()
    plot(arrs)


if __name__ == "__main__":
    main()
