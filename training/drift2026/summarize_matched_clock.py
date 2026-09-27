"""
Q12 — MATCHED-CLOCK summary: per-signal-class decayed stats (hero family at
its own fast measured clock, pair/comp at the slow one) vs uniform decay and
cumulative. Aggregates results/q12/ (2 new cells x 3 seeds, D2 protocol) with
the existing reference rows (results/d2 d2c_cumprev, results/q7 decayed
cells; those files are read, never modified).

Outputs: results/MATCHED_CLOCK.md + results/matched_clock.json.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drift2026 import common

import numpy as np

R = common.RESULTS_DIR
ROWS = [
    # (cell, subdir, label)
    ("d2c_cumprev", "d2", "cumulative (d2c_cumprev, reference)"),
    ("q7_decayed90k100", "q7",
     "uniform decayed 90d + k=100 shrink (Q7 champion, reference)"),
    ("q7_decayed90", "q7", "uniform decayed 90d (reference)"),
    ("q7_decayed365", "q7", "uniform decayed 365d (reference)"),
    ("q12_hero90pair365", "q12",
     "MATCHED CLOCK: hero family 90d / pair+comp 365d **(new)**"),
    ("q12_hero365pair90", "q12",
     "reverse control: hero family 365d / pair+comp 90d **(new)**"),
]


def agg(cell, subdir):
    rows = []
    for s in common.SEEDS:
        p = os.path.join(R, subdir, f"{cell}_s{s}.json")
        if not os.path.exists(p):
            return None
        with open(p) as f:
            rows.append(json.load(f))

    def m(get):
        v = [get(r) for r in rows]
        return float(np.mean(v)), float(np.std(v))
    per_sizable = {b: round(float(np.mean(
        [r["test_acc_per_build"][b]["acc"] for r in rows])), 2)
        for b in common.SIZABLE_TEST_BUILDS}
    return {
        "seeds": [r["seed"] for r in rows],
        "n_train": rows[0]["n_train"],
        "val": m(lambda r: r["val_acc"]),
        "future": m(lambda r: r["test_acc_future"]),
        "future_per_seed": {r["seed"]: r["test_acc_future"] for r in rows},
        "sizable": m(lambda r: r["test_acc_sizable"]),
        "per_sizable_build": per_sizable,
        "sanity28": m(lambda r: r["sanity"]["passed_28"]),
        "sanity21": m(lambda r: r["sanity"]["passed_21"]),
        "epochs": m(lambda r: r["epochs"]),
    }


def fmt(ms, prec=2):
    return f"{ms[0]:.{prec}f} ± {ms[1]:.{prec}f}"


def main():
    common.setup()
    cells = {}
    for cell, subdir, label in ROWS:
        a = agg(cell, subdir)
        assert a is not None, f"{cell}: incomplete seeds in results/{subdir}"
        cells[cell] = {"label": label, **a}

    lines = [
        "# Q12 — Matched-clock per-signal decay (reviewer follow-up)",
        "",
        "SIGNAL_DECAY/PARTIAL_REFRESH finding: hero-WR is the fastest-decaying",
        "signal but pairwise/comp carries refresh value -> decay each signal",
        "class at its own measured rate. Stats built by",
        "build_matched_clock_stats.py: hero-family aggregates (hero WR,",
        "pick+ban, hero-map) from the Q7 decayed-90d files, pair/comp",
        "(pairwise counter/synergy, comp WR) from the decayed-365d files —",
        "composed at the exact W3(d)/Q5 attribute split; reverse assignment",
        "as the control. Feature passes decayedmc_prev / decayedmcrev_prev",
        "(strictly-causal `_prev` convention, row-aligned with",
        "features_cumulative_prev.npz, 3,898,174 rows, 0 failed). Training =",
        "train_drift_wp.py verbatim (D2 protocol, regime=all, 3 seeds,",
        "n_train 3,538,596 — identical to Q7). counter/synergy features are",
        "genuine hybrids by construction (pairwise WRs normalized by the",
        "other clock's hero WRs), verified column-exact against the parent",
        "passes for the pure groups.",
        "",
        "| cell | val acc | future acc | sizable acc | sanity 28 (21) | "
        "epochs |",
        "|---|---|---|---|---|---|",
    ]
    for cell, _, label in ROWS:
        c = cells[cell]
        lines.append(
            f"| {label} | {fmt(c['val'])} | {fmt(c['future'])} | "
            f"{fmt(c['sizable'])} | {c['sanity28'][0]:.1f} "
            f"({c['sanity21'][0]:.1f}) | {c['epochs'][0]:.0f} |")

    lines += ["", "Per-seed future acc (42/123/777):"]
    for cell, _, label in ROWS:
        ps = cells[cell]["future_per_seed"]
        lines.append(f"- {cell}: " + " / ".join(
            f"{ps[s]:.3f}" for s in common.SEEDS))

    mc = cells["q12_hero90pair365"]
    rev = cells["q12_hero365pair90"]
    cum = cells["d2c_cumprev"]
    uni = cells["q7_decayed90k100"]
    d90 = cells["q7_decayed90"]
    verdict_beats_uniform = mc["future"][0] > uni["future"][0]
    mc_min = min(mc["future_per_seed"].values())
    d90_max = max(d90["future_per_seed"].values())
    lines += [
        "",
        "## Verdict — does per-class clocking beat uniform decay?",
        "",
        f"- matched clock {mc['future'][0]:.2f} vs cumulative "
        f"{cum['future'][0]:.2f} ({mc['future'][0]-cum['future'][0]:+.2f} pp),"
        f" vs uniform decayed90k100 {uni['future'][0]:.2f} "
        f"({mc['future'][0]-uni['future'][0]:+.2f} pp), vs uniform decayed90 "
        f"{d90['future'][0]:.2f} ({mc['future'][0]-d90['future'][0]:+.2f} "
        "pp).",
        f"- reverse control {rev['future'][0]:.2f} "
        f"({rev['future'][0]-mc['future'][0]:+.2f} pp vs matched): "
        + ("the assignment direction matters — hero-fast/pair-slow is the "
           "right way round."
           if mc["future"][0] - rev["future"][0] > 0.03 else
           ("the REVERSE assignment (hero slow / pair fast) is the better "
            "of the two — the matched-clock hypothesis fails in its own "
            "direction."
            if rev["future"][0] - mc["future"][0] > 0.03 else
            "the assignment direction barely matters at seed noise.")),
        "",
        "**Verdict: NO — per-signal-class clocking does not beat uniform "
        "decay.** Both mixed cells sit below uniform decayed90 "
        f"({d90['future'][0]:.2f}) and the decayed90k100 champion "
        f"({uni['future'][0]:.2f}).",
        "",
        "Reading: each cell's future accuracy tracks its PAIR/COMP clock, "
        "not its hero clock — matched (pair 365d) "
        f"{mc['future'][0]:.2f} ~= uniform decayed365 "
        f"{cells['q7_decayed365']['future'][0]:.2f}, reverse (pair 90d) "
        f"{rev['future'][0]:.2f} sits near uniform decayed90 "
        f"{d90['future'][0]:.2f}. The training-time recency gain of "
        "DECAYED_AGGREGATES is therefore carried at least as much by the "
        "pair/comp features as by the hero family. That is consistent with "
        "PARTIAL_REFRESH (pairwise/comp carries the refresh value at "
        "inference) but contrary to the naive recipe \"decay each class at "
        "its measured signal half-life\": hero-WR decaying fastest as a "
        "SIGNAL does not mean the hero features are where the fast CLOCK "
        "pays. Recommendation for the paper: keep the uniform-90d (k=100 "
        "shrunk) champion; report this 2-cell control as the reviewer "
        "follow-up.",
    ]
    payload = {
        "_meta": {
            "phase": "Q12 matched-clock per-signal decay",
            "stats": "build_matched_clock_stats.py (decayedmc = hero90/"
                     "pair365, decayedmcrev = hero365/pair90)",
            "protocol": "train_drift_wp.py, D2, regime=all, 3 seeds, "
                        "results-subdir q12",
        },
        "cells": cells,
        "verdict": {
            "matched_beats_uniform_decayed90k100": bool(verdict_beats_uniform),
            "matched_minus_cumulative_pp": round(
                mc["future"][0] - cum["future"][0], 3),
            "matched_minus_decayed90k100_pp": round(
                mc["future"][0] - uni["future"][0], 3),
            "matched_minus_decayed90_pp": round(
                mc["future"][0] - d90["future"][0], 3),
            "matched_minus_reverse_pp": round(
                mc["future"][0] - rev["future"][0], 3),
            "seed_clean_vs_decayed90": bool(mc_min > d90_max),
        },
    }
    common.write_json(os.path.join(R, "matched_clock.json"), payload)
    md = os.path.join(R, "MATCHED_CLOCK.md")
    with open(md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {md}")
    print(json.dumps(payload["verdict"], indent=2))


if __name__ == "__main__":
    main()
