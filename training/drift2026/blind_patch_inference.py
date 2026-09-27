"""
Blind patch-timing inference from match data alone (transfer-paper probe).

Question: without version labels, can the match stream tell us WHEN the
environment changed? The existing W3(b) detector answers "did stats change
across this KNOWN boundary"; every other ACA domain we are considering for
the transfer paper (NBA lineups, MLB, Best Ball) has no boundary labels, so
the transferable primitive is blind changepoint detection. This domain has
ground truth (patch_index build dates) to validate it.

Method (deliberately label-free):
  1. Daily per-hero (games, wins, picks, bans) from the pinned snapshot
     (dates via the patch sidecar; build ids NEVER consulted).
  2. For each candidate day t, split a +/- WINDOW-day neighborhood into
     before/after halves and compute per-hero two-proportion z on
     (a) pick rate (against total team slots) and (b) win rate (>= MIN_G
     games per side). Aggregate D(t) = mean of top-K |z| across heroes,
     per family; combined score = max of the two families' normalized D.
  3. Changepoints = local maxima of D(t) above threshold with >= MIN_SEP
     days separation. Threshold swept -> P/R curve (no tuning on truth).
  4. Validate against sizable-build release dates (>= 20K games, n=27
     excluding the corpus-start build): a detection within +/- TOL days of
     a true date is a TP (greedy 1:1 matching); report P/R per threshold,
     timing error distribution, and peak height vs patch size (games until
     next build as a crude size proxy, plus patch-note truth counts where
     available).

Outputs: results/blind_patch_inference.json + BLIND_PATCH_INFERENCE.md

Usage:
    python drift2026/blind_patch_inference.py
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drift2026 import common

WINDOW = 7        # days on each side of the candidate point
MIN_G = 200       # min games per side for a hero WR test
TOP_K = 10        # aggregate the K most-shifted heroes (patches touch a few)
MIN_SEP = 10      # days between detected changepoints
TOL = 3           # +/- days for a TP match
THRESHOLDS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]


def build_daily():
    rows, builds = common.load_data_with_patches()
    day_min = min(r["date_days"] for r in rows)
    day_max = max(r["date_days"] for r in rows)
    n_days = day_max - day_min + 1
    heroes = sorted({h for r in rows for h in r["team0_heroes"] + r["team1_heroes"]})
    hidx = {h: i for i, h in enumerate(heroes)}
    H = len(heroes)
    games = np.zeros(n_days)                # total games per day
    picks = np.zeros((n_days, H))           # hero picks per day
    wins = np.zeros((n_days, H))            # hero wins per day
    for r in rows:
        d = r["date_days"] - day_min
        games[d] += 1
        for ti, team in enumerate((r["team0_heroes"], r["team1_heroes"])):
            won = 1.0 if r["winner"] == ti else 0.0
            for h in team:
                j = hidx[h]
                picks[d, j] += 1
                wins[d, j] += won
    return day_min, games, picks, wins, heroes


def two_prop_z(x1, n1, x2, n2):
    if n1 <= 0 or n2 <= 0:
        return 0.0
    p = (x1 + x2) / (n1 + n2)
    if p <= 0 or p >= 1:
        return 0.0
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return abs(x1 / n1 - x2 / n2) / se if se > 0 else 0.0


def drift_series(games, picks, wins):
    """D_pick(t), D_wr(t): mean top-K per-hero |z| between the WINDOW days
    before t and the WINDOW days from t on."""
    n_days, H = picks.shape
    d_pick = np.zeros(n_days)
    d_wr = np.zeros(n_days)
    sp = np.zeros((n_days, H))  # per-hero pick-rate shift at each day
    sw = np.zeros((n_days, H))  # per-hero WR shift at each day
    cg = np.concatenate([[0], np.cumsum(games)])
    cp = np.vstack([np.zeros(H), np.cumsum(picks, axis=0)])
    cw = np.vstack([np.zeros(H), np.cumsum(wins, axis=0)])
    for t in range(WINDOW, n_days - WINDOW):
        g1 = cg[t] - cg[t - WINDOW]
        g2 = cg[t + WINDOW] - cg[t]
        if g1 < 2000 or g2 < 2000:  # thin days (early corpus): skip
            continue
        p1 = cp[t] - cp[t - WINDOW]
        p2 = cp[t + WINDOW] - cp[t]
        w1 = cw[t] - cw[t - WINDOW]
        w2 = cw[t + WINDOW] - cw[t]
        # Effect sizes, not z: at these window sizes two-proportion z is
        # "significant" for smooth week-to-week drift everywhere, drowning
        # the step changes patches create.
        for j in range(H):
            sp[t, j] = abs(p2[j] / (g2 * 10) - p1[j] / (g1 * 10)) * 100  # pp of slots
            if p1[j] >= MIN_G and p2[j] >= MIN_G:
                sw[t, j] = abs(w2[j] / p2[j] - w1[j] / p1[j]) * 100      # pp WR

    # SELF-NORMALIZE per hero: divide each hero's shift series by its own
    # median shift. Heroes with chronically oscillating pick rates (free
    # rotation, events) stop masquerading as patch signal; a patch is a
    # hero shifting abnormally RELATIVE TO ITS OWN HISTORY.
    def self_norm(s):
        z = np.zeros_like(s)
        for j in range(s.shape[1]):
            col = s[:, j]
            med = np.median(col[col > 0])
            if med and med > 0:
                z[:, j] = col / med
        return z

    zp_all, zw_all = self_norm(sp), self_norm(sw)
    for t in range(WINDOW, n_days - WINDOW):
        d_pick[t] = np.mean(np.sort(zp_all[t])[-TOP_K:])
        d_wr[t] = np.mean(np.sort(zw_all[t])[-TOP_K:])
    return d_pick, d_wr


def find_peaks(score, thresh):
    """Local maxima above thresh with >= MIN_SEP separation (greedy by height)."""
    cand = [t for t in range(1, len(score) - 1)
            if score[t] >= thresh and score[t] >= score[t - 1] and score[t] >= score[t + 1]]
    cand.sort(key=lambda t: -score[t])
    kept = []
    for t in cand:
        if all(abs(t - u) >= MIN_SEP for u in kept):
            kept.append(t)
    return sorted(kept)


def match_truth(detected_days, truth_days):
    """Greedy 1:1 matching within TOL days; returns (tp_pairs, fp, fn)."""
    used = set()
    tps = []
    for d in detected_days:
        best = None
        for i, td in enumerate(truth_days):
            if i in used or abs(d - td) > TOL:
                continue
            if best is None or abs(d - td) < abs(d - truth_days[best]):
                best = i
        if best is not None:
            used.add(best)
            tps.append((d, truth_days[best]))
    fp = len(detected_days) - len(tps)
    fn = len(truth_days) - len(tps)
    return tps, fp, fn


def main():
    common.setup()
    idx = common.load_patch_index()
    builds, meta = idx["builds"], idx["build_meta"]
    import datetime
    epoch = datetime.date(1970, 1, 1)
    truth = []
    sizable = [b for b in builds if meta[b]["n_db"] >= 20000]
    for b in sizable:
        if b == sizable[0]:  # corpus-start build: no before-window, unobservable
            continue
        if True:
            # sidecar date_days = days since 1970-01-01; match that basis
            truth.append((b, (datetime.date.fromisoformat(meta[b]["min_date"]) - epoch).days))
    print(f"{len(truth)} sizable ground-truth boundaries")

    day_min, games, picks, wins, heroes = build_daily()
    d_pick, d_wr = drift_series(games, picks, wins)
    # normalize each family by its own median-positive level, combine by max
    def norm(x):
        pos = x[x > 0]
        return x / np.median(pos) if len(pos) else x
    score = np.maximum(norm(d_pick), norm(d_wr))
    truth_days = [td - day_min for _, td in truth]

    curve = []
    for th in THRESHOLDS:
        det = find_peaks(score, th)
        tps, fp, fn = match_truth(det, truth_days)
        prec = len(tps) / max(len(det), 1)
        rec = len(tps) / max(len(truth_days), 1)
        errs = [d - t for d, t in tps]
        curve.append({"threshold": th, "n_detected": len(det),
                      "tp": len(tps), "fp": fp, "fn": fn,
                      "precision": round(prec, 3), "recall": round(rec, 3),
                      "timing_err_days": errs})
        print(f"th={th}: {len(det)} detected, P={prec:.2f} R={rec:.2f}")

    # peak height at each true boundary (visible-vs-invisible patches)
    per_truth = []
    for (b, td) in truth:
        t = td - day_min
        lo, hi = max(0, t - TOL), min(len(score), t + TOL + 1)
        per_truth.append({"build": b, "date": meta[b]["min_date"],
                          "peak_score": round(float(score[lo:hi].max()), 2),
                          "n_games_build": meta[b]["n_db"]})

    out = {"config": {"window": WINDOW, "top_k": TOP_K, "min_sep": MIN_SEP,
                      "tol_days": TOL, "min_wr_games": MIN_G},
           "curve": curve, "per_truth_boundary": per_truth}
    path = os.path.join(common.RESULTS_DIR, "blind_patch_inference.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
