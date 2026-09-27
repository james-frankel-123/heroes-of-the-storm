"""
Q11 — DETECTOR OPERATING CURVES: threshold sweep of the W3(b) outcome-shift
detector + W2c detector-triggered replay at every operating point.

The paper's single numbers — detector precision 0.79 (wr_bh at BH FDR
q=0.05) and the detector-triggered arm at 14 refreshes / +0.743 pp regret —
become two curves:

  1. P/R curve: the BH FDR level q (the detection threshold on the WR
     two-proportion tests) is swept over ~8 operating points; at each q the
     per-boundary detected-hero sets are recomputed from the daily cache
     (identical primitives: w3_changepoints.two_prop / bh_flags, MIN_G=200)
     and scored against the patch-note ground truth, micro-averaged over the
     validatable boundaries — exactly the W3_CHANGEPOINTS validation
     protocol.
  2. Regret frontier: at each q, the fire schedule (boundaries with >= 1
     BH-surviving hero after C0) is replayed through the W2c
     detector-triggered arm (w2c_detector_arm machinery reused as imports:
     plan / extract_cells / npz_correct, fixed C0 model w2c_cut08_s42,
     honest first-detection z_naive latency per fire) -> (n refreshes,
     games-weighted acc, regret vs oracle).

VERIFICATION GATES (all hard asserts):
  - q=0.05 detected sets == results/w3_changepoints.json `wr_bh` per
    boundary, and micro P/R reproduces TP=46 FP=12 FN=687 (precision 0.793).
  - q=0.05 fire schedule (14 fires, per-fire first/median z_naive latencies)
    == results/w2c_detector_arm.json `fires`.
  - replayed cumprev per-build accuracies reproduce the stored W2c matrix
    row (< 0.01 pp).
  - q=0.05 replay reproduces weighted acc 56.068 / regret +0.743 exactly
    (3 decimals).

Existing results files are not modified. Outputs:
  results/detector_curves.json + results/DETECTOR_CURVES.md
  paper/drift/overleaf/fig_detector_curves.pdf (2 panels: P/R tradeoff,
  regret-vs-refresh frontier; paper operating point marked on both)

Usage:
    CUDA_VISIBLE_DEVICES=0 python drift2026/detector_curves.py [--preview]
"""
import os
import sys
import json
import gzip
import pickle
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drift2026 import common
from drift2026.w3_recovery import DAILY_CACHE
from drift2026.w3_changepoints import (two_prop, bh_flags, build_totals,
                                       latency_walk, load_ground_truth, MIN_G)
from drift2026.w2c_detector_arm import (plan, extract_cells, load_model,
                                        npz_correct, correctness,
                                        canonical_ranks, b255_list,
                                        C0_POS, C0_BUILD, MEDIAN_TP_LATENCY)

import numpy as np

R = common.RESULTS_DIR
ALPHAS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50]
PAPER_ALPHA = 0.05
FIG_PDF = os.path.join(os.path.dirname(common.TRAINING_DIR),
                       "paper", "drift", "overleaf", "fig_detector_curves.pdf")


# ── phase 1: boundary tests + threshold sweep (CPU) ──

def compute_boundaries():
    """Recompute per-boundary WR p-values + truth from the daily cache
    (same code path as w3_changepoints.main, WR family only)."""
    with gzip.open(DAILY_CACHE, "rb") as f:
        cache = pickle.load(f)
    builds, per_build, build_games = (cache["builds"], cache["per_build"],
                                      cache["build_games"])
    order = sorted(per_build)
    sizable = [bi for bi in order if build_games[bi] >= common.SIZABLE_GAMES]
    truth_by_build, gt_meta = load_ground_truth(set(builds))
    assert truth_by_build is not None, gt_meta
    totals = {bi: build_totals(per_build[bi]) for bi in order}

    boundaries = []
    for k in range(1, len(sizable)):
        prev, new = sizable[k - 1], sizable[k]
        inter = [bi for bi in order if prev < bi <= new]
        n0, hero0, _ = totals[prev]
        n1, hero1, _ = totals[new]
        p_wr = {}
        for h in set(hero0) | set(hero1):
            g0, w0 = hero0.get(h, (0, 0))
            g1, w1 = hero1.get(h, (0, 0))
            if g0 >= MIN_G and g1 >= MIN_G:
                _, _, p = two_prop(w0, g0, w1, g1)
                p_wr[h] = p
        t, known = set(), []
        for bi in inter:
            b = builds[bi]
            if b in truth_by_build:
                t |= truth_by_build[b]
                known.append(b)
        boundaries.append({
            "new_build": builds[new], "new_bi": new, "prev_bi": prev,
            "p_wr": p_wr, "hero0": hero0, "prev_games": n0,
            "truth": sorted(t), "validatable": len(known) == len(inter),
        })
    return boundaries, per_build, gt_meta


def sweep_detected(boundaries):
    """{alpha: [detected hero set per boundary]}."""
    return {a: [bh_flags(b["p_wr"], q=a) for b in boundaries] for a in ALPHAS}


def micro_pr(boundaries, det_sets):
    tp = fp = fn = 0
    for b, d in zip(boundaries, det_sets):
        if not b["validatable"]:
            continue
        t = set(b["truth"])
        tp += len(d & t)
        fp += len(d - t)
        fn += len(t - d)
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = (2 * prec * rec / (prec + rec)) if prec and rec else None
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 3) if prec is not None else None,
            "recall": round(rec, 3) if rec is not None else None,
            "f1": round(f1, 3) if f1 is not None else None}


def compute_latencies(boundaries, per_build, detected):
    """z_naive latency walk for every hero ever detected at any swept alpha
    (superset of the stored wr_p05 walk; identical latency_walk)."""
    need = defaultdict(set)
    for det_sets in detected.values():
        for i, d in enumerate(det_sets):
            need[i] |= d
    lat = {}
    for i, heroes in need.items():
        b = boundaries[i]
        out = {}
        for h in sorted(heroes):
            g0, w0 = b["hero0"].get(h, (0, 0))
            if g0 == 0:
                continue
            walk = latency_walk(per_build[b["new_bi"]], h, g0, w0,
                                b["prev_games"])
            if "z_naive" in walk:
                out[h] = walk["z_naive"][0]   # build games
        lat[i] = out
    return lat


def fire_schedule(boundaries, det_sets, lat, b255):
    fires = []
    for i, (b, d) in enumerate(zip(boundaries, det_sets)):
        if not d:
            continue
        pos = b255.index(b["new_build"])
        if pos <= C0_POS:
            continue
        lats = sorted(lat[i][h] for h in d if h in lat[i])
        fires.append({
            "pos": pos, "build": b["new_build"],
            "n_detections": len(d), "detected_heroes": sorted(d),
            "latencies_z_naive": lats,
            "lat_first": (lats[0] if lats else MEDIAN_TP_LATENCY),
            "lat_median": (float(np.median(lats)) if lats
                           else MEDIAN_TP_LATENCY),
        })
    return sorted(fires, key=lambda f: f["pos"])


# ── verification against stored results ──

def verify_alpha05(boundaries, det05, pr05, fires05):
    with open(os.path.join(R, "w3_changepoints.json")) as f:
        cp = json.load(f)
    stored = {b["new_build"]: set(b["detected"]["wr_bh"])
              for b in cp["boundaries"]}
    for b, d in zip(boundaries, det05):
        assert d == stored[b["new_build"]], \
            (b["new_build"], sorted(d), sorted(stored[b["new_build"]]))
    sv = cp["validation"]["variants"]["wr_bh"]
    assert (pr05["tp"], pr05["fp"], pr05["fn"]) == (sv["tp"], sv["fp"],
                                                    sv["fn"]), (pr05, sv)
    assert pr05["precision"] == sv["precision"], (pr05, sv)
    print(f"verify 1 (q=0.05 detected sets + P/R vs w3_changepoints.json): "
          f"OK — TP={pr05['tp']} FP={pr05['fp']} FN={pr05['fn']} "
          f"precision={pr05['precision']} recall={pr05['recall']}")

    with open(os.path.join(R, "w2c_detector_arm.json")) as f:
        da = json.load(f)
    assert len(fires05) == len(da["fires"]), (len(fires05), len(da["fires"]))
    for a, b in zip(fires05, da["fires"]):
        assert a["build"] == b["build"] and a["pos"] == b["pos"]
        assert a["n_detections"] == b["n_detections"], a["build"]
        assert a["lat_first"] == b["lat_first"], a["build"]
        assert a["lat_median"] == b["lat_median"], a["build"]
    print(f"verify 2 (q=0.05 fire schedule vs w2c_detector_arm.json): OK — "
          f"{len(fires05)} fires, latencies exact")
    return da


# ── phase 2: replay (w2c_detector_arm machinery) ──

def replay_all(fires_by_alpha, num_workers):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    b255 = b255_list()
    with open(os.path.join(R, "w2c", "w2c_cut08_s42.json")) as f:
        cut08 = json.load(f)
    with open(os.path.join(R, "w2c_policies.json")) as f:
        policies = json.load(f)
    Ns = sorted(b255.index(b) for b in cut08["test_acc_per_build"])
    games = {b255.index(b): d["n_rows"] // 2
             for b, d in cut08["test_acc_per_build"].items()}
    matrix_c0 = {b255.index(b): d["acc"]
                 for b, d in cut08["test_acc_per_build"].items()}
    w = np.array([games[n] for n in Ns], dtype=float)

    def wavg(curve):
        return float(np.sum(w * np.array([curve[n] for n in Ns])) / w.sum())

    oracle = policies["oracle_weighted_acc"]

    # union of custom feature cells across alphas (full build wins over cap)
    plans, union = {}, {}
    for a, fires in fires_by_alpha.items():
        span_v, pre_v, post_v, cells = plan(fires, Ns, b255)
        plans[a] = (span_v, pre_v, {f["pos"]: f for f in fires})
        for key, cap in cells.items():
            if key in union:
                union[key] = (None if union[key] is None or cap is None
                              else max(union[key], cap))
            else:
                union[key] = cap
    n_replays_est = sum(1 for c in union.values())
    print(f"union: {len(union)} custom (vintage, build) cells across "
          f"{len(fires_by_alpha)} operating points")
    for (v, _p) in union:
        assert os.path.exists(common.stats_path("cumulative", v)), v

    # CPU extraction BEFORE any CUDA init (fork safety), then forward passes
    custom_raw = extract_cells(union, b255, num_workers)
    model = load_model(device)
    cum = npz_correct("cumulative_prev", model, Ns, b255, device)
    frz = npz_correct(f"cutoff_{C0_BUILD}", model, Ns, b255, device)
    diffs = [abs(float(cum[n][2].mean() * 100) - matrix_c0[n]) for n in Ns]
    print(f"verify 3 (cumprev row vs stored W2c matrix): max |diff| = "
          f"{max(diffs):.4f} pp")
    assert max(diffs) < 0.01, max(diffs)
    custom = {}
    for key, (rids, labels, X) in custom_raw.items():
        custom[key] = (rids, correctness(model, X, labels, device))
    del custom_raw

    all_fire_pos = {p for _, (_, _, fp) in plans.items() for p in fp}
    ranks = {n: canonical_ranks(cum[n][0], cum[n][1]) for n in all_fire_pos}

    def rows_from(src_rids, src_corr, n, lo, hi):
        rk = ranks[n]
        r = np.array([rk.get(int(x), -1) for x in src_rids])
        m = (r >= lo) & (r < hi)
        return src_corr[m]

    def source(n, vintage):
        if vintage == C0_BUILD:
            rids, _, corr = frz[n]
            return rids, corr
        return custom[(vintage, n)]

    out = {}
    for a, (span_v, pre_v, fire_pos) in plans.items():
        curve = {}
        for n in Ns:
            if n in fire_pos:
                L = int(fire_pos[n]["lat_first"])   # honest per-detection lag
                pre_rids, pre_corr = source(n, pre_v[n])
                pre = rows_from(pre_rids, pre_corr, n, 0, L)
                post = rows_from(cum[n][0], cum[n][2], n, L, 10**9)
                curve[n] = float(np.concatenate([pre, post]).mean() * 100)
            else:
                _, corr = source(n, span_v[n])
                curve[n] = float(corr.mean() * 100)
        acc = wavg(curve)
        out[a] = {"refreshes": len(fire_pos), "weighted_acc": round(acc, 3),
                  "regret_pp": round(oracle - acc, 3)}
        print(f"  q={a:<6} refreshes={len(fire_pos):>2} acc={acc:.3f} "
              f"regret={oracle-acc:+.3f}")

    # reference rows: taken verbatim from the stored W2c policy table so the
    # numbers match the published W2C_SUMMARY/W2C_DETECTOR_ARM rows exactly
    # (avoids re-rounding drift); the never-refresh acc is recomputed and
    # asserted against it.
    eb_row = next(p for p in policies["policies"]
                  if p["policy"] == "never retrain + stats refresh")
    nr_acc = round(wavg({n: float(source(n, C0_BUILD)[1].mean() * 100)
                         for n in Ns}), 3)
    assert abs(nr_acc - 55.748) < 0.01, nr_acc
    refs = {
        "oracle_weighted_acc": oracle,
        "never_refresh": {"refreshes": 0, "weighted_acc": nr_acc,
                          "regret_pp": round(oracle - nr_acc, 3)},
        "every_build_refresh": {"refreshes": len(Ns) - 1,
                                "weighted_acc": eb_row["weighted_acc"],
                                "regret_pp": eb_row["regret_pp"]},
    }
    return out, refs


# ── figure ──

def make_figure(rows, refs, paper_row):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE, RED, GRAY, INK = "#2a78d6", "#e34948", "#8a8988", "#52514e"
    plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False,
                         "axes.spines.right": False,
                         "axes.linewidth": 0.8, "pdf.fonttype": 42})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.4))

    swept = [r for r in rows if r["precision"] is not None]
    rec = [r["recall"] for r in swept]
    prec = [r["precision"] for r in swept]
    ax1.plot(rec, prec, "-o", color=BLUE, lw=1.4, ms=3.5, zorder=3)
    # hand-placed offsets: (dx, dy) points, keyed by alpha
    OFF1 = {0.001: (-16, 9), 0.005: (7, 3), 0.01: (3, -12), 0.02: (5, 4),
            0.1: (5, 3), 0.2: (0, -12), 0.35: (2, 6), 0.5: (-8, -12)}
    for r in swept:
        if r["alpha"] == PAPER_ALPHA:
            continue
        dx, dy = OFF1[r["alpha"]]
        ax1.annotate(f"q={r['alpha']:g}", (r["recall"], r["precision"]),
                     textcoords="offset points", xytext=(dx, dy),
                     fontsize=7, color=INK)
    pr = next(r for r in rows if r["alpha"] == PAPER_ALPHA)
    ax1.plot([pr["recall"]], [pr["precision"]], marker="*", ms=11,
             color=RED, mec="white", mew=0.5, zorder=4)
    ax1.annotate("q=0.05 (paper)", (pr["recall"], pr["precision"]),
                 textcoords="offset points", xytext=(6, 7), fontsize=7,
                 color=RED)
    ax1.set_xlabel("Recall (vs patch-note ground truth)")
    ax1.set_ylabel("Precision")
    ax1.set_ylim(0, 1.0)
    ax1.set_xlim(left=0)
    ax1.grid(True, lw=0.3, alpha=0.4)
    ax1.set_title("Detector operating curve (BH FDR level q)", fontsize=8.5)

    nr, eb = refs["never_refresh"], refs["every_build_refresh"]
    xs = [r["refreshes"] for r in rows]
    ys = [r["regret_pp"] for r in rows]
    ax2.plot(xs, ys, "-o", color=BLUE, lw=1.4, ms=3.5, zorder=3)
    OFF2 = {0.001: (2, -12), 0.5: (-6, -12)}
    for r in rows:
        if r["alpha"] in OFF2:
            dx, dy = OFF2[r["alpha"]]
            ax2.annotate(f"q={r['alpha']:g}", (r["refreshes"],
                                               r["regret_pp"]),
                         textcoords="offset points", xytext=(dx, dy),
                         fontsize=7, color=INK)
    # zoomed to the frontier band; never-refresh (+1.063) is off scale by
    # a factor ~40x the frontier spread — shown as an arrow annotation.
    ax2.plot([eb["refreshes"]], [eb["regret_pp"]], "s", color=GRAY, ms=4.5,
             zorder=3)
    ax2.annotate("refresh every build", (eb["refreshes"], eb["regret_pp"]),
                 textcoords="offset points", xytext=(-2, 6), fontsize=7,
                 color=INK, ha="right")
    ax2.axhline(eb["regret_pp"], color=GRAY, lw=0.6, ls=":", zorder=1)
    ax2.annotate(f"never refresh: {nr['regret_pp']:+.2f} (off scale)",
                 xy=(0, 0.7775), xytext=(1.5, 0.7695),
                 arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.8),
                 fontsize=7, color=INK)
    ax2.plot([paper_row["refreshes"]], [paper_row["regret_pp"]], marker="*",
             ms=11, color=RED, mec="white", mew=0.5, zorder=4)
    ax2.annotate("q=0.05 (paper)",
                 (paper_row["refreshes"], paper_row["regret_pp"]),
                 textcoords="offset points", xytext=(-24, 9), fontsize=7,
                 color=RED)
    ax2.set_xlabel("Stats refreshes over deployment window")
    ax2.set_ylabel("Regret vs oracle (pp)")
    ax2.set_xlim(-1.5, 37)
    ax2.set_ylim(0.722, 0.78)
    ax2.grid(True, lw=0.3, alpha=0.4)
    ax2.set_title("Regret vs refresh count (first-detection latency)",
                  fontsize=8.5)

    fig.tight_layout(pad=0.4)
    os.makedirs(os.path.dirname(FIG_PDF), exist_ok=True)
    fig.savefig(FIG_PDF, bbox_inches="tight")
    print(f"Wrote {FIG_PDF}")


# ── main ──

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preview", action="store_true",
                    help="schedules + P/R only, no replay")
    ap.add_argument("--fig-only", action="store_true",
                    help="re-render the figure from results/detector_curves"
                         ".json")
    ap.add_argument("--num-workers", type=int, default=48)
    args = ap.parse_args()
    common.setup()

    if args.fig_only:
        with open(os.path.join(R, "detector_curves.json")) as f:
            saved = json.load(f)
        rows = saved["operating_points"]
        make_figure(rows, saved["reference_rows"],
                    next(r for r in rows if r["alpha"] == PAPER_ALPHA))
        return

    b255 = b255_list()
    boundaries, per_build, gt_meta = compute_boundaries()
    print(f"{len(boundaries)} boundaries "
          f"({sum(b['validatable'] for b in boundaries)} validatable)")
    detected = sweep_detected(boundaries)
    lat = compute_latencies(boundaries, per_build, detected)

    pr = {a: micro_pr(boundaries, detected[a]) for a in ALPHAS}
    fires = {a: fire_schedule(boundaries, detected[a], lat, b255)
             for a in ALPHAS}
    stored_da = verify_alpha05(boundaries, detected[PAPER_ALPHA],
                               pr[PAPER_ALPHA], fires[PAPER_ALPHA])

    print("\n=== operating points ===")
    for a in ALPHAS:
        print(f"  q={a:<6} P={pr[a]['precision']} R={pr[a]['recall']} "
              f"TP/FP/FN={pr[a]['tp']}/{pr[a]['fp']}/{pr[a]['fn']} "
              f"fires={len(fires[a])}")
    if args.preview:
        return

    replay, refs = replay_all(fires, args.num_workers)

    # verify 4: q=0.05 replay reproduces the stored detector-arm row exactly
    stored_row = next(r for r in stored_da["detector_rows"]
                      if "first-detection" in r["policy"])
    got = replay[PAPER_ALPHA]
    assert got["refreshes"] == stored_row["refreshes"], (got, stored_row)
    assert got["weighted_acc"] == stored_row["weighted_acc"], (got, stored_row)
    assert got["regret_pp"] == stored_row["regret_pp"], (got, stored_row)
    print(f"verify 4 (q=0.05 replay vs stored detector arm): OK — "
          f"{got['refreshes']} refreshes, acc {got['weighted_acc']}, "
          f"regret +{got['regret_pp']}")

    rows = []
    for a in ALPHAS:
        rows.append({"alpha": a, **pr[a], "n_fires": len(fires[a]),
                     **replay[a]})

    payload = {
        "_meta": {
            "phase": "Q11 detector operating curves",
            "detector": "W3b wr_bh family: two-proportion WR z-test per hero "
                        f"(>= {MIN_G} games/side), BH FDR within boundary at "
                        "level q (swept); latency = z_naive sequential test "
                        "at day boundaries (fixed across the sweep)",
            "replay": "W2c detector-triggered arm, fixed C0 model "
                      "w2c_cut08_s42, first-detection latency, refresh = "
                      "stats cumulative through fired build's predecessor",
            "ground_truth": gt_meta,
            "paper_operating_point": {"alpha": PAPER_ALPHA,
                                      **pr[PAPER_ALPHA],
                                      **replay[PAPER_ALPHA]},
        },
        "operating_points": rows,
        "reference_rows": refs,
        "fire_schedules": {str(a): [{k: f[k] for k in
                                     ("build", "pos", "n_detections",
                                      "lat_first", "lat_median")}
                                    for f in fires[a]] for a in ALPHAS},
    }
    common.write_json(os.path.join(R, "detector_curves.json"), payload)

    paper_row = next(r for r in rows if r["alpha"] == PAPER_ALPHA)
    make_figure(rows, refs, paper_row)

    nr, eb = refs["never_refresh"], refs["every_build_refresh"]
    lines = [
        "# Detector operating curves (Q11)",
        "",
        "The W3(b) outcome-shift detector's decision rule — per-hero "
        "two-proportion WR",
        f"z-tests (>= {MIN_G} games/side) at each sizable-build boundary, "
        "Benjamini-Hochberg",
        "FDR within boundary — swept over the FDR level q. At each operating "
        "point:",
        "precision/recall vs the official-patch-note ground truth "
        "(micro-averaged over",
        f"the {sum(b['validatable'] for b in boundaries)} validatable "
        "boundaries, W3_CHANGEPOINTS protocol), and the W2c",
        "detector-triggered deployment replay (fixed C0 model w2c_cut08_s42, "
        "refresh",
        "fires with honest per-boundary first-detection z_naive latency; "
        "W2C_DETECTOR_ARM",
        "protocol). The paper's operating point is q=0.05. Figure: "
        "fig_detector_curves.pdf.",
        "",
        "| FDR q | TP | FP | FN | precision | recall | F1 | refreshes | "
        "weighted acc % | regret vs oracle (pp) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        star = " **(paper)**" if r["alpha"] == PAPER_ALPHA else ""
        lines.append(
            f"| {r['alpha']:g}{star} | {r['tp']} | {r['fp']} | {r['fn']} | "
            f"{r['precision']} | {r['recall']} | {r['f1']} | "
            f"{r['refreshes']} | {r['weighted_acc']:.3f} | "
            f"{r['regret_pp']:+.3f} |")
    lines += [
        "",
        "Reference rows (same fixed C0 model): never refresh 0 refreshes / "
        f"{nr['weighted_acc']:.3f}% / {nr['regret_pp']:+.3f} pp; refresh "
        f"every build {eb['refreshes']} refreshes / {eb['weighted_acc']:.3f}%"
        f" / {eb['regret_pp']:+.3f} pp; oracle (retrain every build) "
        f"{refs['oracle_weighted_acc']:.3f}%.",
        "",
        "## Verification",
        "",
        "- q=0.05 detected sets, P/R (0.793 / 0.063), fire schedule (14 "
        "fires, per-fire latencies) and replay row (56.068 / +0.743) all "
        "reproduce the stored W3_CHANGEPOINTS / W2C_DETECTOR_ARM numbers "
        "exactly; the replayed cumprev accuracies reproduce the stored W2c "
        "matrix row to < 0.01 pp.",
        "",
        "## Reading",
        "",
    ]
    best = min(rows, key=lambda r: r["regret_pp"])
    lines += [
        f"- P/R tradeoff: precision falls from "
        f"{rows[0]['precision']} at q={rows[0]['alpha']:g} to "
        f"{rows[-1]['precision']} at q={rows[-1]['alpha']:g} while recall "
        f"rises from {rows[0]['recall']} to {rows[-1]['recall']} — the "
        "detector is precision-limited by design (patch notes are a lower "
        "bound; see W3_CHANGEPOINTS caveats).",
        f"- Regret frontier: every operating point sits between never-refresh "
        f"({nr['regret_pp']:+.3f}) and every-build refresh "
        f"({eb['regret_pp']:+.3f}); the best swept point is q="
        f"{best['alpha']:g} at {best['regret_pp']:+.3f} pp with "
        f"{best['refreshes']} refreshes (paper point: q=0.05, "
        f"{paper_row['regret_pp']:+.3f} pp at {paper_row['refreshes']} "
        "refreshes).",
    ]
    md = os.path.join(R, "DETECTOR_CURVES.md")
    with open(md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {md}")


if __name__ == "__main__":
    main()
