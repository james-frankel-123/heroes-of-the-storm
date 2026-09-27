"""Aggregate results/d2/*.json into per-cell (mean +/- sd over seeds) tables:
results/d2_summary.json + results/D2_SUMMARY.md."""
import os
import sys
import json
import glob
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drift2026 import common

import numpy as np

CELL_ORDER = ["d2b_allhist", "d2b_win3", "d2b_win6", "d2b_win12",
              "d2b_decay90", "d2b_decay365", "d2b_embed",
              "d2c_local", "d2c_cumprev", "d2c_frozen"]


def main():
    common.setup()
    cells = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(common.RESULTS_DIR, "d2", "*.json"))):
        with open(path) as f:
            r = json.load(f)
        cells[r["name"].rsplit("_s", 1)[0]].append(r)

    def ms(vals):
        return f"{np.mean(vals):.2f} ± {np.std(vals):.2f}"

    summary = {}
    lines = ["# D2 regime / enrichment-sourcing summary (3 seeds per cell)", "",
             "Future = all builds after 2.55.14.95918; sizable = the 4 headline",
             "test builds. Val = train-period held-out slice. Sanity = paper-1",
             "28-test suite (21-test subset in parens).", "",
             "| cell | n_train | val acc | future acc | sizable acc | " +
             " | ".join(b.split(".", 2)[-1] for b in common.SIZABLE_TEST_BUILDS) +
             " | sanity 28 (21) | epochs |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for cell in CELL_ORDER:
        rs = cells.get(cell)
        if not rs:
            continue
        per_build = {b: [r["test_acc_per_build"][b]["acc"] for r in rs
                         if b in r["test_acc_per_build"]]
                     for b in common.SIZABLE_TEST_BUILDS}
        summary[cell] = {
            "seeds": [r["seed"] for r in rs],
            "n_train": rs[0]["n_train"],
            "val_acc": [r["val_acc"] for r in rs],
            "test_acc_future": [r["test_acc_future"] for r in rs],
            "test_acc_sizable": [r["test_acc_sizable"] for r in rs],
            "per_sizable_build": per_build,
            "sanity_28": [r["sanity"]["passed_28"] for r in rs],
            "sanity_21": [r["sanity"]["passed_21"] for r in rs],
            "epochs": [r["epochs"] for r in rs],
        }
        s = summary[cell]
        lines.append(
            f"| {cell} | {s['n_train']:,} | {ms(s['val_acc'])} | "
            f"{ms(s['test_acc_future'])} | {ms(s['test_acc_sizable'])} | "
            + " | ".join(f"{np.mean(per_build[b]):.2f}" for b in
                         common.SIZABLE_TEST_BUILDS)
            + f" | {np.mean(s['sanity_28']):.1f} ({np.mean(s['sanity_21']):.1f})"
            + f" | {np.mean(s['epochs']):.0f} |")

    common.write_json(os.path.join(common.RESULTS_DIR, "d2_summary.json"), summary)
    md = os.path.join(common.RESULTS_DIR, "D2_SUMMARY.md")
    with open(md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {md}\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
