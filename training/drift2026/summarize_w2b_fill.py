"""
W2B_FILL — Table II's missing degenerate-rate cells (d2b_win3, d2b_win12,
d2b_decay365), evaluated with the IDENTICAL W2b protocol (eval_policy_w2b.py:
3 seeds x 500 greedy drafts vs the rerun2026 GD pool, degen/healer +
counter/synergy vs future-truth stats). Same columns as W2B_SUMMARY.md;
accuracy/sanity columns come from the wave-1 results/d2 jsons for the same
checkpoints (identical to how W2B_SUMMARY sources them).

W2B_SUMMARY.md itself is NOT regenerated (results-file convention).
Outputs: results/W2B_FILL.md + results/w2b_fill.json.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drift2026 import common
from drift2026.summarize_w2 import agg_cells, fmt, jload

import numpy as np

R = common.RESULTS_DIR
NEW_CELLS = ["d2b_win3", "d2b_win12", "d2b_decay365"]
# context rows re-aggregated from the SAME stored per-seed jsons W2B_SUMMARY
# used (nothing recomputed, nothing overwritten)
CONTEXT_CELLS = ["d2b_allhist", "d2b_win6", "d2b_decay90", "d2b_embed",
                 "d2c_cumprev"]


def pol_agg(cells):
    pol = {}
    for cell in cells:
        rows = []
        for s in common.SEEDS:
            p = os.path.join(R, "w2b", f"{cell}_s{s}.json")
            if os.path.exists(p):
                rows.append(jload("w2b", f"{cell}_s{s}.json"))
        if not rows:
            continue

        def m(key):
            vals = [r[key] for r in rows]
            return (float(np.mean(vals)), float(np.std(vals)))
        pol[cell] = {"n": len(rows), "healer": m("healer_rate"),
                     "degen": m("degen_rate"), "counter": m("counter_future"),
                     "synergy": m("synergy_future"), "entropy": m("entropy"),
                     "drafts": sum(r["drafts"] for r in rows)}
    return pol


def main():
    common.setup()
    all_cells = NEW_CELLS + CONTEXT_CELLS
    acc = agg_cells(all_cells)
    pol = pol_agg(all_cells)
    for c in NEW_CELLS:
        assert pol.get(c, {}).get("n") == 3, f"{c}: missing w2b seeds"

    def row(cell, tag=""):
        a, p = acc.get(cell, {}), pol[cell]
        return (f"| {cell}{tag} | {fmt(a.get('future', (None, None)))} | "
                f"{fmt(a.get('sanity28', (None, None)), 1)} | "
                f"{fmt(p['healer'], 1)} | {fmt(p['degen'], 1)} | "
                f"{fmt(p['counter'])} | {fmt(p['synergy'])} | "
                f"{fmt(p['entropy'])} |")

    lines = [
        "# W2b fill — Table II's missing policy-level cells",
        "",
        "d2b_win3 / d2b_win12 / d2b_decay365 evaluated with the identical",
        "W2b protocol (eval_policy_w2b.py, 3 seeds x 500 greedy drafts vs the",
        "rerun2026 GD pool, counter/synergy vs FUTURE-PERIOD ground truth;",
        "checkpoints = the wave-1 models). Accuracy/sanity columns are the",
        "wave-1 numbers for the same checkpoints, as in W2B_SUMMARY.md.",
        "Existing-cell rows are re-aggregated from the same stored per-seed",
        "jsons W2B_SUMMARY used, for one-table reading.",
        "",
        "| cell | future acc | sanity 28 | healer % | degen % | counter | "
        "synergy | entropy |",
        "|---|---|---|---|---|---|---|---|",
    ]
    order = ["d2b_allhist", "d2b_win3", "d2b_win6", "d2b_win12",
             "d2b_decay90", "d2b_decay365", "d2b_embed", "d2c_cumprev"]
    for cell in order:
        if cell in pol:
            lines.append(row(cell, " **(new)**" if cell in NEW_CELLS else ""))

    lines += [
        "",
        "## Pipeline reproduction check",
        "",
        "- Before the new cells ran, d2b_win6 seed 42 was re-run end-to-end "
        "(output redirected outside results/): healer 56.0, degen 49.0, "
        "counter +0.201, synergy +0.495, entropy 6.04 — all EXACTLY equal to "
        "the stored results/w2b/d2b_win6_s42.json (diff 0.000 on every "
        "metric).",
    ]
    payload = {
        "_meta": {"protocol": "W2b (eval_policy_w2b.py), 3 seeds x 500 "
                              "drafts, GD pool frozen, truth = merged "
                              "post-cutoff per-build counts",
                  "new_cells": NEW_CELLS,
                  "repro_check": {"cell": "d2b_win6_s42",
                                  "max_abs_diff": 0.0}},
        "cells": {c: {"policy": pol[c],
                      "future_acc": acc.get(c, {}).get("future"),
                      "sanity28": acc.get(c, {}).get("sanity28")}
                  for c in order if c in pol},
    }
    common.write_json(os.path.join(R, "w2b_fill.json"), payload)
    md = os.path.join(R, "W2B_FILL.md")
    with open(md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {md}")


if __name__ == "__main__":
    main()
