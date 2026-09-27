"""
Blind patch-timing inference, v2: TUNED MULTI-SCALE detector.

Follow-up to blind_patch_inference.py (single-scale probe: P 0.23 / R 0.26).
Design changes:
  - Six window scales (2, 3, 5, 7, 10, 14 days): sharp steps light up short
    scales at exact dates; slow-burn shifts need long ones. A real patch is
    scale-CONSISTENT; single-scale noise is not, so scales are combined by
    mean of per-scale normalized scores.
  - Three signal families: pick rate, ban rate (bans react to patch
    perception immediately), win rate.
  - Per-(hero, scale) self-normalization by that hero's own median contrast
    (kills free-rotation oscillators).
  - HONEST TUNING: config grid (top-K, family weights, scale combiner,
    threshold) selected by F1 on TRAIN boundaries (pre-2025-01-01, n=16)
    only; headline numbers are held-out P/R on TEST boundaries (2025+,
    n=11). Detections are assigned to the split of the period they fall in.

Validation identical to v1: +/-3-day tolerance, greedy 1:1 matching against
sizable-build release dates.

Outputs: results/blind_patch_multiscale.json (+ appends summary to
BLIND_PATCH_INFERENCE.md by hand afterwards).

Usage:
    REPLAY_SNAPSHOT=1 python drift2026/blind_patch_multiscale.py
"""
import datetime
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drift2026 import common

SCALES = [2, 3, 5, 7, 10, 14]
MIN_SEP = 10
TOL = 3
SPLIT_DATE = datetime.date(2025, 1, 1)
THRESHOLDS = [1.2, 1.4, 1.6, 1.8, 2.0, 2.4, 2.8, 3.2, 4.0]
TOP_KS = [3, 5, 10]
FAMILY_SETS = {
    "pick": ("pick",), "ban": ("ban",), "wr": ("wr",),
    "pick+ban": ("pick", "ban"), "all": ("pick", "ban", "wr"),
}
SCALE_COMBINERS = ["mean", "max"]
MIN_WR_G = 150


def build_daily():
    rows, _ = common.load_data_with_patches()
    day_min = min(r["date_days"] for r in rows)
    day_max = max(r["date_days"] for r in rows)
    n_days = day_max - day_min + 1
    heroes = sorted({h for r in rows for h in r["team0_heroes"] + r["team1_heroes"]})
    hidx = {h: i for i, h in enumerate(heroes)}
    H = len(heroes)
    games = np.zeros(n_days)
    picks = np.zeros((n_days, H))
    wins = np.zeros((n_days, H))
    bans = np.zeros((n_days, H))
    for r in rows:
        d = r["date_days"] - day_min
        games[d] += 1
        for h in set(r["team0_bans"] + r["team1_bans"]):
            j = hidx.get(h)
            if j is not None:
                bans[d, j] += 1
        for ti, team in enumerate((r["team0_heroes"], r["team1_heroes"])):
            won = 1.0 if r["winner"] == ti else 0.0
            for h in team:
                j = hidx[h]
                picks[d, j] += 1
                wins[d, j] += won
    return day_min, games, picks, wins, bans


def contrasts(games, picks, wins, bans):
    """per (scale, family) -> (n_days, H) self-normalized contrast arrays."""
    n_days, H = picks.shape
    cg = np.concatenate([[0], np.cumsum(games)])
    cp = np.vstack([np.zeros(H), np.cumsum(picks, axis=0)])
    cw = np.vstack([np.zeros(H), np.cumsum(wins, axis=0)])
    cb = np.vstack([np.zeros(H), np.cumsum(bans, axis=0)])
    out = {}
    for W in SCALES:
        c_pick = np.zeros((n_days, H))
        c_ban = np.zeros((n_days, H))
        c_wr = np.zeros((n_days, H))
        for t in range(W, n_days - W):
            g1 = cg[t] - cg[t - W]
            g2 = cg[t + W] - cg[t]
            if g1 < 300 * W or g2 < 300 * W:
                continue
            p1 = cp[t] - cp[t - W]; p2 = cp[t + W] - cp[t]
            w1 = cw[t] - cw[t - W]; w2 = cw[t + W] - cw[t]
            b1 = cb[t] - cb[t - W]; b2 = cb[t + W] - cb[t]
            c_pick[t] = np.abs(p2 / (g2 * 10) - p1 / (g1 * 10)) * 100
            c_ban[t] = np.abs(b2 / g2 - b1 / g1) * 100
            ok = (p1 >= MIN_WR_G) & (p2 >= MIN_WR_G)
            with np.errstate(divide="ignore", invalid="ignore"):
                wr1 = np.where(ok, w1 / np.maximum(p1, 1), 0.0)
                wr2 = np.where(ok, w2 / np.maximum(p2, 1), 0.0)
            c_wr[t] = np.where(ok, np.abs(wr2 - wr1) * 100, 0.0)
        for fam, c in (("pick", c_pick), ("ban", c_ban), ("wr", c_wr)):
            z = np.zeros_like(c)
            for j in range(H):
                col = c[:, j]
                pos = col[col > 0]
                med = np.median(pos) if len(pos) else 0
                if med > 0:
                    z[:, j] = col / med
            out[(W, fam)] = z
    return out


def day_score(con, fams, top_k, combiner):
    """(n_days,) combined score for one config."""
    per_scale = []
    for W in SCALES:
        fam_scores = []
        for fam in fams:
            z = con[(W, fam)]
            fam_scores.append(np.mean(np.sort(z, axis=1)[:, -top_k:], axis=1))
        per_scale.append(np.mean(fam_scores, axis=0))
    per_scale = np.stack(per_scale)  # (n_scales, n_days)
    med = np.median(per_scale[per_scale > 0])
    per_scale = per_scale / med if med > 0 else per_scale
    return per_scale.mean(axis=0) if combiner == "mean" else per_scale.max(axis=0)


def find_peaks(score, thresh):
    cand = [t for t in range(1, len(score) - 1)
            if score[t] >= thresh and score[t] >= score[t - 1] and score[t] >= score[t + 1]]
    cand.sort(key=lambda t: -score[t])
    kept = []
    for t in cand:
        if all(abs(t - u) >= MIN_SEP for u in kept):
            kept.append(t)
    return sorted(kept)


def pr_for(detected, truth_days):
    used = set()
    tps = []
    for d in detected:
        best = None
        for i, td in enumerate(truth_days):
            if i in used or abs(d - td) > TOL:
                continue
            if best is None or abs(d - td) < abs(d - truth_days[best]):
                best = i
        if best is not None:
            used.add(best)
            tps.append((d, truth_days[best]))
    return tps, len(detected) - len(tps), len(truth_days) - len(tps)


def main():
    common.setup()
    idx = common.load_patch_index()
    builds, meta = idx["builds"], idx["build_meta"]
    epoch = datetime.date(1970, 1, 1)
    sizable = [b for b in builds if meta[b]["n_db"] >= 20000]
    truth = [(b, (datetime.date.fromisoformat(meta[b]["min_date"]) - epoch).days)
             for b in sizable[1:]]
    split_day = (SPLIT_DATE - epoch).days

    day_min, games, picks, wins, bans = build_daily()
    con = contrasts(games, picks, wins, bans)
    truth_train = [td - day_min for _, td in truth if td < split_day]
    truth_test = [td - day_min for _, td in truth if td >= split_day]
    split_t = split_day - day_min
    print(f"{len(truth_train)} train boundaries (pre-2025), {len(truth_test)} test")

    def f1(p, r):
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    best = None
    grid = list(itertools.product(FAMILY_SETS.items(), TOP_KS, SCALE_COMBINERS, THRESHOLDS))
    for (fname, fams), top_k, combiner, th in grid:
        score = day_score(con, fams, top_k, combiner)
        det = find_peaks(score, th)
        det_train = [t for t in det if t < split_t]
        tps, fp, fn = pr_for(det_train, truth_train)
        p = len(tps) / max(len(det_train), 1)
        r = len(tps) / max(len(truth_train), 1)
        f = f1(p, r)
        if best is None or f > best["f1_train"]:
            best = {"family": fname, "top_k": top_k, "combiner": combiner,
                    "threshold": th, "f1_train": f, "p_train": p, "r_train": r}
    print("best train config:", best)

    # Held-out evaluation of the tuned config
    fams = FAMILY_SETS[best["family"]]
    score = day_score(con, fams, best["top_k"], best["combiner"])
    det = find_peaks(score, best["threshold"])
    det_test = [t for t in det if t >= split_t]
    tps, fp, fn = pr_for(det_test, truth_test)
    p_test = len(tps) / max(len(det_test), 1)
    r_test = len(tps) / max(len(truth_test), 1)
    errs = [d - t for d, t in tps]
    print(f"HELD-OUT: {len(det_test)} detected, TP={len(tps)} FP={fp} FN={fn} "
          f"-> P={p_test:.2f} R={r_test:.2f}, timing errs {errs}")

    # held-out curve for context (same tuned config, threshold swept)
    test_curve = []
    for th in THRESHOLDS:
        d2 = [t for t in find_peaks(score, th) if t >= split_t]
        tp2, fp2, fn2 = pr_for(d2, truth_test)
        test_curve.append({"threshold": th, "n": len(d2),
                           "precision": round(len(tp2) / max(len(d2), 1), 3),
                           "recall": round(len(tp2) / max(len(truth_test), 1), 3)})
        print(f"  test th={th}: n={len(d2)} P={test_curve[-1]['precision']} R={test_curve[-1]['recall']}")

    # per-test-boundary score (visibility profile under the tuned detector)
    per_test = []
    for b, td in truth:
        t = td - day_min
        if t < split_t:
            continue
        lo, hi = max(0, t - TOL), min(len(score), t + TOL + 1)
        per_test.append({"build": b, "date": meta[b]["min_date"],
                         "peak_score": round(float(score[lo:hi].max()), 2)})

    out = {"config": {"scales": SCALES, "tol": TOL, "min_sep": MIN_SEP,
                      "split": str(SPLIT_DATE), "grid_size": len(grid)},
           "tuned": best,
           "held_out": {"n_detected": len(det_test), "tp": len(tps), "fp": fp,
                        "fn": fn, "precision": round(p_test, 3),
                        "recall": round(r_test, 3), "timing_err_days": errs},
           "held_out_curve": test_curve,
           "per_test_boundary": per_test}
    path = os.path.join(common.RESULTS_DIR, "blind_patch_multiscale.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
