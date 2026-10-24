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
    tests = json.load(open(os.path.join(R, "p3_c_tests_def.json")))
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
    miss = [k for k in names if k not in A]
    assert not miss, f"calib_def.json lacks arms {miss}"
    for k, nm in names.items():
        a = A[k]
        L.append(f"| {nm} | {a['V1']['gain']:+.2f} | {a['V2']['gain']:+.2f} | {a['V2']['vs_ref']:+.2f} ({a['V2']['vs_ref_se']:.2f}) | "
                 f"{a['OOT']['gain']:+.2f} | {a['OOT']['vs_ref']:+.2f} ({a['OOT']['vs_ref_se']:.2f}) | "
                 f"{100 * a['V2']['acc']:.2f}% | {100 * a['OOT']['acc']:.2f}% |")
    need = ["V2 all", "OOT all", "OOT before 2026-09-12", "OOT from 2026-09-12",
            "OOT without 2.55.17.98025", "OOT 2.55.17.98025 only"]
    miss = [k for k in need if k not in tests["sets"]]
    assert not miss, f"p3_c_tests_def.json lacks subsets {miss}"
    for sn, v in tests["sets"].items():
        for pk in ("chosen-today", "chosen-full_ms_layer"):
            assert pk in v["paired"], f"p3_c_tests_def.json: subset {sn} lacks the {pk} contrast"
    fmt = lambda p: (f"{1000 * p['ll_gain']:+.2f} ({1000 * p['se']:.2f}) "
                     f"[{1000 * p['week_ci'][0]:+.2f}, {1000 * p['week_ci'][1]:+.2f}]")
    L += ["", "## Primary contrasts with week-block 95% CIs (log-loss gain x1000; game SE in parentheses)", "",
          "V2 and OOT were both read during model development (shrinkage study, team shares, role weights), "
          "so neither is an untouched holdout. OOT also contains the formerly sealed build 2.55.17.98025 "
          f"({tests.get('oot_sealed_build_games', 0):,} games), shown separately below; the 5,000 sealed replays "
          "were never read. The previous main model is the pre-stated comparator.", "",
          "| set | games | chosen − previous | chosen − today |", "|---|---|---|---|"]
    for sn in need:
        v = tests["sets"][sn]
        L.append(f"| {sn} | {v['games']:,} | {fmt(v['paired']['chosen-full_ms_layer'])} | {fmt(v['paired']['chosen-today'])} |")
    L += ["", "## Subsets (against today's kernel; gain x1000, SE, week-block 95% CI)", "",
          "| set | games | today | previous | chosen | chosen − today |", "|---|---|---|---|---|---|"]
    for sn, v in tests["sets"].items():
        a = v["arms"]
        L.append(f"| {sn} | {v['games']:,} | {1000 * a['today']['gain']:+.2f} | {1000 * a['full_ms_layer']['gain']:+.2f} | "
                 f"{1000 * a['chosen']['gain']:+.2f} | {fmt(v['paired']['chosen-today'])} |")
    L += ["", "## Disclosures", "",
          "- The drafter fit (full_ms, chosen) and the judge fit (full_ms_flip, chosen_flip) use disjoint player halves "
          "but the same V1 games: the shrinkage function, direct channel, role weights, calibration and layer of each "
          "arm are fit on V1 slot outcomes, and the two halves share those games' outcomes (teammates and opponents). "
          "Their coefficients are fit separately. A game-disjoint variant is a planned sensitivity after this run.",
          "- Filtered states (the judge window, split halves, exclusions) carry no temporal layer, whose residuals "
          "come from unfiltered predictions; a poison check of every filter form gates each answer run.",
          "- The tournament reads the base posterior sd (`sd`); the display band (`sd_display`) is for the site only.", ""]
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
