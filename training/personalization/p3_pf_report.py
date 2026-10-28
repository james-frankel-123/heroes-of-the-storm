"""
P3 post-freeze confirmation report: the definitive build's coefficients scored
once, with no refit, on 2.57 games from 2026-09-28 (window 4 of the
post-freeze build, remote/pf_score.sh).

For today's kernel, the previous main model and the chosen model:
  - log-loss gain over population WP (x1000), accuracy;
  - chosen - previous and chosen - today: game SE, week-block and 3-day-block
    95% bootstrap CIs (the window has about four weeks);
  - game-level calibration: logistic slope of y on logit p (1 = calibrated);
  - slot-level calibration by games on the hero: slope of the realized
    residual r = y - WP on the slot prediction, smooth curve and bins
    (p3_rcf_calib.curve), player-bootstrap bands;
  - the pre-freeze windows reproduce the frozen build (V1, V2, OOT gains).

Usage (from training/, env of the post-freeze build):
  python3 personalization/p3_pf_report.py <D> <B>
Output: results/def_<D>/pf_report.json and P3_POSTFREEZE_<D>.md
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = (("today", "today's kernel"), ("full_ms_layer", "previous main model"), ("chosen", "chosen model"))
BINS = ((1, 9), (10, 29), (30, 99), (100, 299), (300, 10 ** 9))


def ll_of(p, y):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def block_ci(d, block, rng, n=2000):
    u, inv = np.unique(block, return_inverse=True)
    S = np.bincount(inv, d)
    N = np.bincount(inv).astype(float)
    bs = [S[i].sum() / N[i].sum() for i in (rng.integers(0, len(u), len(u)) for _ in range(n))]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], int(len(u))


def logit_slope(p, y):
    lp = np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))
    import p3_hs_core as C
    w = C.fit_logistic(lp[:, None], y)
    return float(w[1]), float(w[0])


def main():
    D, B = sys.argv[1], sys.argv[2]
    import p3_rcf_calib as K
    rd = os.path.join(HERE, "results", f"def_{D}", "rcf", "p3c")
    rb = os.path.join(HERE, "results", f"def_{B}", "rcf", "p3c")
    A = {k: dict(np.load(os.path.join(rd, f"{k}__merged.npz"))) for k, _ in ARMS}
    F = {k: dict(np.load(os.path.join(rb, f"{k}__merged.npz"))) for k, _ in ARMS}
    ref = A["today"]
    for k in A:
        assert np.array_equal(A[k]["g_replay_id"], ref["g_replay_id"]), k
    y = ref["g_y"].astype(float)
    win = ref["g_win"]
    m = win == 4
    day = ref["g_day"]
    rng = np.random.default_rng(0)
    LL0 = ll_of(ref["g_p0"], y)
    out = {"build": D, "frozen_from": B, "games": int(m.sum()),
           "days": [int(day[m].min()), int(day[m].max())], "arms": {}, "contrasts": {}, "prefreeze": {}}
    L = [f"# P3 post-freeze confirmation ({D}, coefficients frozen from {B})", ""]
    for k, nm in ARMS:
        a = A[k]
        ll = ll_of(a["g_p"], y)
        g = 1000 * (LL0[m] - ll[m])
        acc = float(((a["g_p"][m] > .5) == (y[m] == 1)).mean())
        sl, ic = logit_slope(a["g_p"][m], y[m])
        out["arms"][k] = {"gain": float(g.mean()), "gain_se": float(g.std(ddof=1) / np.sqrt(m.sum())), "acc": acc,
                          "game_calibration_slope": sl, "game_calibration_intercept": ic}
        # the pre-freeze windows of this build against the frozen build itself
        fz = F[k]
        pre = {}
        for wk, wn in ((1, "V1"), (2, "V2"), (3, "OOT")):
            mw = a["g_win"] == wk
            fw = fz["g_win"] == wk
            pre[wn] = {"this": float(1000 * (ll_of(a["g_p0"][mw], a["g_y"][mw]) - ll_of(a["g_p"][mw], a["g_y"][mw])).mean()),
                       "frozen_build": float(1000 * (ll_of(fz["g_p0"][fw], fz["g_y"][fw]) - ll_of(fz["g_p"][fw], fz["g_y"][fw])).mean()),
                       "games": [int(mw.sum()), int(fw.sum())]}
        out["prefreeze"][k] = pre
    for hi, lo in (("chosen", "full_ms_layer"), ("chosen", "today"), ("full_ms_layer", "today")):
        d = 1000 * (ll_of(A[lo]["g_p"][m], y[m]) - ll_of(A[hi]["g_p"][m], y[m]))
        wci, nw = block_ci(d, day[m] // 7, rng)
        dci, nd = block_ci(d, day[m] // 3, rng)
        out["contrasts"][f"{hi}-{lo}"] = {"gain": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d))),
                                          "week_ci": wci, "weeks": nw, "day3_ci": dci, "blocks3": nd,
                                          "acc_diff": float((((A[hi]["g_p"][m] > .5) == (y[m] == 1)).astype(float)
                                                             - ((A[lo]["g_p"][m] > .5) == (y[m] == 1)).astype(float)).mean())}
    # slot-level calibration by games on the hero (window 4 slots)
    for k, _ in ARMS:
        a = A[k]
        s = a["s_win"] == 4
        p = a["s_pred"][s].astype(float)
        r = a["s_r"][s].astype(float)
        av = {kk: (v[0][s], v[1][s]) for kk, v in K.axis_values(a).items()}
        c = K.curve(av["games"][0], av["games"][1], p, r, a["s_pid"][s], "games", rng)
        x = np.array(c["x"])
        pts = {}
        for n in (0, 3, 10, 30, 100, 300, 1000):
            i = int(np.argmin(np.abs(x - np.log1p(n))))
            pts[str(n)] = [round(c["slope"][i], 3), round(c["lo"][i], 3), round(c["hi"][i], 3)]
        nh = a["s_nh"][s].astype(float)
        bins = {}
        for lo_, hi_ in BINS:
            mm = (nh >= lo_) & (nh <= hi_)
            cv = np.cov(p[mm], r[mm])
            bins[f"{lo_}-{hi_}"] = {"slots": int(mm.sum()), "slope": float(cv[0, 1] / cv[0, 0])}
        cv = np.cov(p, r)
        out["arms"][k]["slot_calibration"] = {"overall_slope": float(cv[0, 1] / cv[0, 0]), "curve_points": pts,
                                              "rms_slope_minus_1": c["rms_slope_minus_1"], "bins": bins,
                                              "slots": int(s.sum())}
    L += [f"Games: {out['games']:,} complete 2.57 Storm League games (all ten players in the v1 hero set), "
          f"days {out['days'][0]}..{out['days'][1]}.", "",
          "| model | gain over population WP x1000 (SE) | accuracy | game calibration slope | slot calibration slope |",
          "|---|---|---|---|---|"]
    for k, nm in ARMS:
        r = out["arms"][k]
        L.append(f"| {nm} | {r['gain']:+.2f} ({r['gain_se']:.2f}) | {100 * r['acc']:.2f}% | "
                 f"{r['game_calibration_slope']:.3f} | {r['slot_calibration']['overall_slope']:.3f} |")
    L += ["", "| contrast | gain x1000 (game SE) | week-block 95% CI | 3-day-block 95% CI | accuracy |", "|---|---|---|---|---|"]
    for kk, c in out["contrasts"].items():
        L.append(f"| {kk} | {c['gain']:+.2f} ({c['se']:.2f}) | [{c['week_ci'][0]:+.2f}, {c['week_ci'][1]:+.2f}] "
                 f"({c['weeks']} weeks) | [{c['day3_ci'][0]:+.2f}, {c['day3_ci'][1]:+.2f}] | {100 * c['acc_diff']:+.2f}pp |")
    L += ["", "Slot calibration slope by games on the hero (smooth curve, 95% player-bootstrap band):", "",
          "| games | " + " | ".join(nm for _, nm in ARMS) + " |", "|---|" + "---|" * len(ARMS)]
    for n in (0, 3, 10, 30, 100, 300, 1000):
        L.append(f"| {n} | " + " | ".join(
            "{:.2f} ({:.2f}-{:.2f})".format(*out["arms"][k]["slot_calibration"]["curve_points"][str(n)]) for k, _ in ARMS) + " |")
    L += ["", "Pre-freeze windows of this build vs the frozen build (gain x1000):", ""]
    for k, nm in ARMS:
        L.append(f"- {nm}: " + ", ".join(f"{wn} {v['this']:+.3f} vs {v['frozen_build']:+.3f}" for wn, v in out["prefreeze"][k].items()))
    with open(os.path.join(HERE, "results", f"def_{D}", "pf_report.json"), "w") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(HERE, "results", f"def_{D}", f"P3_POSTFREEZE_{D}.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
