"""
P3 shrinkage study: forward-game scores and smooth calibration curves for
rich-CF arms (results/P3_SHRINKAGE.md).

For each arm (results/<study>/rcf/p3c/<arm>__merged.npz, written by
p3_rcf.py online):
  game level  V2 and OOT: log-loss gain over population WP (x1000), accuracy,
              both paired against a reference arm (SE by game)
  calibration slots of V2 + OOT. The local slope of the realized residual
              r = y - WP on the slot prediction (offset + skill estimate) as a
              smooth function of
                games on the hero before the game (log scale, full history),
                the hero's share of the player's games,
                days since the player's last game on the hero (log scale),
              by kernel-weighted least squares (Gaussian kernel on the
              transformed axis) on fine bins; 95% bands by a player-cluster
              Poisson bootstrap. Slope 1 is calibrated; above 1 means the
              records are shrunk harder than the data supports.
  flatness    slot-weighted RMS of (slope - 1) over each axis, and the slope at
              the heavy end (300+ games, share 50%+, 365+ days)

Usage (from training/; study env from remote/rcf_env.sh):
  python3 personalization/p3_rcf_calib.py --arms full_ms,hl365,... --ref full_ms --name study [--plot a,b,c]
Output: $P3_RESULTS/rcf/calib_<name>.json, figure results/fig_shrink_<name>.png
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import p3_rcf as R

AXES = {
    "games": {"lo": 0.0, "hi": np.log1p(2000), "bw": 0.30, "label": "games on the hero before the game"},
    "share": {"lo": 0.0, "hi": 1.0, "bw": 0.05, "label": "hero share of the player's games"},
    "recency": {"lo": 0.0, "hi": np.log1p(1500), "bw": 0.35, "label": "days since the last game on the hero"},
}
NB = 240
NBOOT = 80


def ll_of(p, y, eps=1e-7):
    return -(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))


def load(name):
    z = np.load(os.path.join(R.RCF_RES, "p3c", f"{name}__merged.npz"))
    return {k: z[k] for k in z.files}


def axis_values(A):
    nh = A["s_nh"].astype(float)
    npl = A["s_np"].astype(float)
    gap = A["s_gap"].astype(float)
    return {"games": (np.log1p(nh), nh >= 1),
            "share": (nh / np.maximum(npl, 1), (nh >= 1) & (npl >= 20)),
            "recency": (np.log1p(np.maximum(gap, 0)), gap >= 0)}


def curve(x, ok, p, r, pid, ax, rng):
    a = AXES[ax]
    m = ok & np.isfinite(x)
    x, p, r, pid = x[m], p[m], r[m], pid[m]
    b = np.clip(((x - a["lo"]) / (a["hi"] - a["lo"]) * NB).astype(int), 0, NB - 1)
    centers = a["lo"] + (np.arange(NB) + 0.5) / NB * (a["hi"] - a["lo"])
    grid = np.linspace(a["lo"], a["hi"], 41)
    Kw = np.exp(-0.5 * ((grid[:, None] - centers[None, :]) / a["bw"]) ** 2)
    up, pinv = np.unique(pid, return_inverse=True)

    def sums(w):
        S = [np.bincount(b, weights=w * v, minlength=NB) for v in (np.ones(len(p)), p, r, p * p, p * r)]
        return [Kw @ s for s in S]

    def slope(S):
        n, sp, sr, spp, spr = S
        return (spr - sp * sr / n) / np.maximum(spp - sp * sp / n, 1e-12), (sr - sp) / np.maximum(n, 1e-12), n
    s0, gap0, n0 = slope(sums(np.ones(len(p))))
    bs = []
    for _ in range(NBOOT):
        w = rng.poisson(1.0, len(up)).astype(float)[pinv]
        bs.append(slope(sums(w))[0])
    bs = np.array(bs)
    dens = np.bincount(b, minlength=NB).astype(float)
    dgrid = Kw @ dens
    keep = dgrid >= 2000
    rms = float(np.sqrt(np.average((s0[keep] - 1) ** 2, weights=dgrid[keep])))
    return {"x": grid.tolist(), "slope": s0.tolist(), "lo": np.percentile(bs, 2.5, 0).tolist(),
            "hi": np.percentile(bs, 97.5, 0).tolist(), "level_gap": gap0.tolist(), "eff_n": dgrid.tolist(),
            "rms_slope_minus_1": rms, "slots": int(len(p))}


def heavy_end(x, ok, p, r, ax):
    lim = {"games": np.log1p(300), "share": 0.5, "recency": np.log1p(365)}[ax]
    m = ok & (x >= lim)
    pm, rm = p[m], r[m]
    c = np.cov(pm, rm)
    return {"slots": int(m.sum()), "slope": float(c[0, 1] / c[0, 0])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--plot", help="arms to draw (<= 3), in order")
    ap.add_argument("--labels", help="display names for --plot arms")
    a = ap.parse_args()
    names = a.arms.split(",")
    ref = load(a.ref)
    y = ref["g_y"]
    win = ref["g_win"]
    LL0 = ll_of(ref["g_p0"], y)
    LLr = ll_of(ref["g_p"], y)
    out = {"ref": a.ref, "arms": {}}
    rng = np.random.default_rng(0)
    for nm in names:
        A = load(nm)
        assert np.array_equal(A["g_replay_id"], ref["g_replay_id"])
        assert np.array_equal(A["s_replay_id"], ref["s_replay_id"])
        LL = ll_of(A["g_p"], y)
        row = {"meta": json.loads(str(A["meta"]))}
        for wk, wn in ((1, "V1"), (2, "V2"), (3, "OOT")):
            m = win == wk
            g = LL0[m] - LL[m]
            dd = LLr[m] - LL[m]
            row[wn] = {"games": int(m.sum()), "gain": 1000 * float(g.mean()),
                       "vs_ref": 1000 * float(dd.mean()), "vs_ref_se": 1000 * float(dd.std(ddof=1) / np.sqrt(m.sum())),
                       "acc": float(((A["g_p"][m] > .5) == (y[m] == 1)).mean()),
                       "acc_vs_ref": float((((A["g_p"][m] > .5) == (y[m] == 1)).astype(float)
                                            - ((ref["g_p"][m] > .5) == (y[m] == 1)).astype(float)).mean())}
        s = A["s_win"] >= 2
        p = A["s_pred"][s].astype(float)
        r = A["s_r"][s].astype(float)
        pid = A["s_pid"][s] if "s_pid" in A else A["s_key"][s]
        av = {k: (v[0][s], v[1][s]) for k, v in axis_values(A).items()}
        row["curves"] = {ax: curve(av[ax][0], av[ax][1], p, r, pid, ax, rng) for ax in AXES}
        row["heavy_end"] = {ax: heavy_end(av[ax][0], av[ax][1], p, r, ax) for ax in AXES}
        row["slope_all"] = float(np.cov(p, r)[0, 1] / np.var(p, ddof=1))
        out["arms"][nm] = row
        print(f"[{nm}] V2 {row['V2']['gain']:+.2f} ({row['V2']['vs_ref']:+.2f}, {row['V2']['vs_ref_se']:.2f}) "
              f"OOT {row['OOT']['gain']:+.2f} ({row['OOT']['vs_ref']:+.2f}, {row['OOT']['vs_ref_se']:.2f}) "
              f"acc V2 {100 * row['V2']['acc']:.2f} OOT {100 * row['OOT']['acc']:.2f}; slope all {row['slope_all']:.3f}; "
              + "; ".join(f"{ax} rms {row['curves'][ax]['rms_slope_minus_1']:.3f} heavy {row['heavy_end'][ax]['slope']:.2f}"
                          for ax in AXES), flush=True)
    p = os.path.join(R.RCF_RES, f"calib_{a.name}.json")
    with open(p, "w") as f:
        json.dump(out, f)
    if a.plot:
        plot(out, a.plot.split(","), (a.labels or a.plot).split(","), a.name)


def plot(out, arms, labels, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    surf, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    cols = ["#2a78d6", "#eb6834", "#1baf7a"]
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.2), facecolor=surf)
    for k, ax in enumerate(AXES):
        A = axs[k]
        A.set_facecolor(surf)
        for j, (nm, lb) in enumerate(zip(arms, labels)):
            c = out["arms"][nm]["curves"][ax]
            x = np.array(c["x"])
            keep = np.array(c["eff_n"]) >= 2000
            xs = np.expm1(x) if ax in ("games", "recency") else 100 * x
            A.fill_between(xs[keep], np.array(c["lo"])[keep], np.array(c["hi"])[keep], color=cols[j], alpha=0.15, lw=0)
            A.plot(xs[keep], np.array(c["slope"])[keep], color=cols[j], lw=2, label=lb)

        A.axhline(1.0, color=ink2, lw=1, ls=(0, (4, 3)))
        if ax in ("games", "recency"):
            A.set_xscale("symlog", linthresh=1)
            A.set_xlim(left=0)
        A.set_xlabel(AXES[ax]["label"] + (" (%)" if ax == "share" else ""), color=ink2, fontsize=9)
        if k == 0:
            A.set_ylabel("calibration slope (1 = calibrated)", color=ink2, fontsize=9)
        A.grid(True, color=grid, lw=0.6)
        for sp in A.spines.values():
            sp.set_color(grid)
        A.tick_params(colors=ink2, labelsize=8)
    axs[0].legend(frameon=False, fontsize=8, labelcolor=ink, loc="lower left")
    fig.suptitle("Calibration of the slot prediction on held-out V2 + OOT games (95% player-bootstrap bands)",
                 color=ink, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    here = os.path.dirname(os.path.abspath(__file__))
    fp = os.path.join(here, "results", f"fig_shrink_{name}.png")
    fig.savefig(fp, dpi=150, facecolor=surf)
    print(f"wrote {fp}")


if __name__ == "__main__":
    main()
