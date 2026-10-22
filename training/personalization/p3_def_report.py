"""
P3 definitive rerun: the numbers page (results/def_<D>/P3_DEFINITIVE_<D>.md)
from the collected reports of one build:
  calib_def.json        p3_rcf_calib.py (V1 / V2 / OOT lifts, smooth calibration curves)
  p3_c_tests_def.json   p3_c_tests.py report (subsets, R1 calibration by record) against today's kernel
  band_def_chosen.json  p3_rcf_band_def.py (display band checks)
  manifest.json         p3_manifest.py

Usage (main box): /usr/bin/python3 personalization/p3_def_report.py <D>
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    D = sys.argv[1]
    R = os.path.join(HERE, "results", f"def_{D}")
    cal = json.load(open(os.path.join(R, "calib_def.json")))
    tests = json.load(open(os.path.join(R, "p3_c_tests_def.json"))) if os.path.exists(os.path.join(R, "p3_c_tests_def.json")) else None
    band = json.load(open(os.path.join(R, "band_def_chosen.json")))
    man = json.load(open(os.path.join(R, "manifest.json")))
    A = cal["arms"]
    L = [f"# P3 definitive rerun, build {D}", "",
         "Every input was built off the 3090: exports and the merged store on the main box, the A1 cache, merged rows, "
         "scoreboard card, model caches and walks on the 3080, fits on the 5090. "
         f"The pinned manifest (`results/def_{D}/manifest.json`) lists {len(man['files'])} files with sha256 "
         f"({sum(e['role'] == 'personalization' for e in man['files'])} personalization, "
         f"{sum(e['role'] == 'upstream' for e in man['files'])} upstream, "
         f"{sum(e['role'] == 'tournament' for e in man['files'])} tournament).", "",
         "## Forward games (log-loss gain over population WP, x1000; paired against the previous main model)", "",
         "| model | V1 | V2 | V2 vs previous (SE) | OOT | OOT vs previous (SE) | acc V2 | acc OOT |",
         "|---|---|---|---|---|---|---|---|"]
    names = {"today": "today's kernel (A2S-equivalent)", "full_ms_layer": "previous main model (full_ms + layer)",
             "chosen": "**chosen shrinkage model**", "chosen_flip": "chosen model, judge half (full_ms_flip)"}
    for k, nm in names.items():
        if k not in A:
            continue
        a = A[k]
        L.append(f"| {nm} | {a['V1']['gain']:+.2f} | {a['V2']['gain']:+.2f} | {a['V2']['vs_ref']:+.2f} ({a['V2']['vs_ref_se']:.2f}) | "
                 f"{a['OOT']['gain']:+.2f} | {a['OOT']['vs_ref']:+.2f} ({a['OOT']['vs_ref_se']:.2f}) | "
                 f"{100 * a['V2']['acc']:.2f}% | {100 * a['OOT']['acc']:.2f}% |")
    if tests:
        L += ["", "## Subsets (against today's kernel; gain x1000, SE)", "",
              "| set | games | today | previous | chosen | chosen − today (SE) |", "|---|---|---|---|---|---|"]
        for sn, v in tests["sets"].items():
            a = v["arms"]
            p = v["paired"].get("chosen-today")
            if p is None:
                continue
            L.append(f"| {sn} | {v['games']:,} | {1000 * a['today']['gain']:+.2f} | {1000 * a['full_ms_layer']['gain']:+.2f} | "
                     f"{1000 * a['chosen']['gain']:+.2f} | {1000 * p['ll_gain']:+.2f} ({1000 * p['se']:.2f}) |")
    L += ["", "## Calibration of the chosen model (slope of realized on predicted, V2 + OOT, smooth curve)", "",
          "| axis | point | previous main | chosen |", "|---|---|---|---|"]
    pts = {"games": [0, 3, 10, 30, 100, 300, 1000], "share": [0.05, 0.2, 0.4, 0.6, 0.8, 0.95],
           "recency": [0, 10, 30, 100, 365, 1000]}
    for ax, ps in pts.items():
        for p in ps:
            row = []
            for arm in ("full_ms_layer", "chosen"):
                c = A[arm]["curves"][ax]
                x = np.array(c["x"])
                tx = np.log1p(p) if ax in ("games", "recency") else p
                row.append(float(np.interp(tx, x, np.array(c["slope"]))))
            L.append(f"| {ax} | {p} | {row[0]:.2f} | {row[1]:.2f} |")
    L += ["", f"RMS of (slope − 1), slot-weighted: previous "
          + ", ".join(f"{ax} {A['full_ms_layer']['curves'][ax]['rms_slope_minus_1']:.3f}" for ax in pts)
          + "; chosen " + ", ".join(f"{ax} {A['chosen']['curves'][ax]['rms_slope_minus_1']:.3f}" for ax in pts) + ".", "",
          "## Display band (D = s2 (1 + exp(-games / 10)))", "",
          "| window | cells | realized / claimed (95% CI) | 80% coverage |", "|---|---|---|---|"]
    for wn in ("V1", "V2", "OOT"):
        b = band[wn]
        L.append(f"| {wn} | {b['cells']:,} | {b['ratio']:.2f} ({b['ratio_ci'][0]:.2f}–{b['ratio_ci'][1]:.2f}) | {b['cover80']:.3f} |")
    L += ["", "Median display sd by games on the hero: "
          + ", ".join(f"{k}: {v:.1f}pp" for k, v in band["display_sd_pp_by_games"].items()) + ".", ""]
    out = os.path.join(R, f"P3_DEFINITIVE_{D}.md")
    with open(out, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
