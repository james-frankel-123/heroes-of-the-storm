"""
R12 — assemble the rebuilt Table I (training regimes): leak-free (out-of-fold)
and leaky (paper) versions side by side, 3 seeds each, plus calibration
slopes from r3_eval_vf.json and the accuracy-degeneracy correlations.

Usage: python3 drift_rebuild/r12_table1.py
Output: drift_rebuild/results/r12_table1.json, R12_TABLE1.md
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rb_common as rb  # noqa: E402

import numpy as np  # noqa: E402

D2R = os.path.join(rb.TRAINING_DIR, "drift2026", "results")
REGS = ["allhist", "win3", "win6", "win12", "decay90", "decay365", "embed"]
SEEDS = [42, 123, 777]


def j(*p):
    return json.load(open(os.path.join(*p)))


def ms(v):
    v = [x for x in v if x is not None]
    return (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, len(v))


def main():
    r3 = j(rb.RESULTS_DIR, "r3_eval_vf.json")
    rows = {}
    for reg in REGS + ["cumprev"]:
        row = {}
        for tag in ("leaky", "oof"):
            if reg == "cumprev":
                tr = [j(D2R, "d2", f"d2c_cumprev_s{s}.json") for s in SEEDS]
                w2 = [j(D2R, "w2b", f"d2c_cumprev_s{s}.json") for s in SEEDS]
                cal = [r3.get(f"maintained/cumprev/s{s}") for s in SEEDS]
            elif tag == "leaky":
                tr = [j(D2R, "d2", f"d2b_{reg}_s{s}.json") for s in SEEDS]
                w2 = [j(D2R, "w2b", f"d2b_{reg}_s{s}.json") for s in SEEDS]
                cal = [r3.get(f"leaky/{reg}/s{s}") for s in SEEDS]
            else:
                tr = [j(rb.RESULTS_DIR, "d2", f"r2_{reg}_oof_s{s}.json") for s in SEEDS]
                w2 = [j(rb.RESULTS_DIR, "r4_w2b", f"r2_{reg}_oof_s{s}.json")
                      for s in SEEDS
                      if os.path.exists(os.path.join(rb.RESULTS_DIR, "r4_w2b",
                                                     f"r2_{reg}_oof_s{s}.json"))]
                cal = [r3.get(f"oof/{reg}/s{s}") for s in SEEDS]
            row[tag] = {
                "val": ms([t["val_acc"] for t in tr]),
                "future": ms([t["test_acc_future"] for t in tr]),
                "degen": ms([w["degen_rate"] for w in w2]),
                "healer": ms([w["healer_rate"] for w in w2]),
                "future_slope": ms([c["future"]["calib_slope"] for c in cal if c]),
                "future_logloss": ms([c["future"]["logloss"] for c in cal if c]),
            }
        rows[reg] = row
    out = {"rows": rows}
    for tag in ("leaky", "oof"):
        x = np.array([rows[r][tag]["future"][0] for r in REGS])
        y = np.array([rows[r][tag]["degen"][0] for r in REGS])
        out[f"corr_{tag}"] = float(np.corrcoef(x, y)[0, 1])
        out[f"slope_{tag}"] = float(np.polyfit(x, y, 1)[0])
    json.dump(out, open(os.path.join(rb.RESULTS_DIR, "r12_table1.json"), "w"), indent=1)
    L = ["# R12 Table I, leak-free vs leaky", "",
         "| regime | val OOF | future OOF | degen OOF | slope OOF | val leaky | future leaky | degen leaky | slope leaky |",
         "|---|---|---|---|---|---|---|---|---|"]
    for reg in REGS + ["cumprev"]:
        o, l = rows[reg]["oof"], rows[reg]["leaky"]
        f = lambda t, p=2: f"{t[0]:.{p}f} ± {t[1]:.{p}f}"
        L.append(f"| {reg} | {f(o['val'])} | {f(o['future'])} | {f(o['degen'],1)} (n={o['degen'][2]}) | "
                 f"{f(o['future_slope'])} | {f(l['val'])} | {f(l['future'])} | {f(l['degen'],1)} | "
                 f"{f(l['future_slope'])} |")
    L += ["", f"corr(future acc, degen) across the 7 regimes: leaky {out['corr_leaky']:+.2f} "
          f"(slope {out['slope_leaky']:.1f}), leak-free {out['corr_oof']:+.2f} "
          f"(slope {out['slope_oof']:.1f})"]
    open(os.path.join(rb.RESULTS_DIR, "R12_TABLE1.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
