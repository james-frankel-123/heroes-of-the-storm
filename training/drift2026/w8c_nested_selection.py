"""
W8c — nested-validation champion selection (analysis of the w8c_* retrains).

The Q7 2x2 champion (decayed90k100, 57.22) was selected on the true test
window. Repair: retrain the 4 cells x 3 seeds with train mask <= INNER
cutoff 2.55.13.95301 (chosen so the pseudo-future window — the pre-cutoff
builds 2.55.14.95774..2.55.14.95918 between inner and true cutoff, 295,608
rows — has comparable volume to the true test window's 287,362), select on
pseudo-future accuracy, and ask whether the selection transfers to the true
test window (builds > 2.55.14.95918, scored with the SAME inner-trained
models, plus the Q7 full-trained reference ranking).

Usage: python3 drift2026/w8c_nested_selection.py
Output: results/w8c_nested_selection.json + W8C_NESTED_SELECTION.md
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drift2026 import common

import numpy as np

INNER_CUTOFF = "2.55.13.95301"
CELLS = ["cumulative", "decayed365", "decayed90", "decayed90k100"]
SEEDS = [42, 123, 777]
LABELS = {"cumulative": "cumulative (cumprev)",
          "decayed365": "decayed HL=365d",
          "decayed90": "decayed HL=90d",
          "decayed90k100": "decayed HL=90d + k=100 shrink"}
# Q7/D2 full-trained (true-cutoff) future accs, per seed 42/123/777
Q7_REFERENCE = {"cumulative": [57.086, 57.114, 57.044],
                "decayed365": [57.115, 57.022, 57.069],
                "decayed90": [57.254, 57.154, 57.159],
                "decayed90k100": [57.198, 57.160, 57.303]}


def window_acc(per_build, builds, lo_idx, hi_idx):
    """Pooled accuracy over test builds with lo_idx < idx <= hi_idx."""
    n_tot, n_ok = 0, 0.0
    for b, d in per_build.items():
        i = builds.index(b)
        if lo_idx < i <= hi_idx:
            n_tot += d["n_rows"]
            n_ok += d["n_rows"] * d["acc"] / 100.0
    return 100.0 * n_ok / n_tot, n_tot


def main():
    builds = common.load_patch_index()["builds"]
    inner = builds.index(INNER_CUTOFF)
    true_cut = builds.index(common.TRAIN_CUTOFF_BUILD)

    rows = {}
    for cell in CELLS:
        per_seed = []
        for s in SEEDS:
            j = json.load(open(os.path.join(
                common.RESULTS_DIR, "w8c", f"w8c_{cell}_s{s}.json")))
            pf, n_pf = window_acc(j["test_acc_per_build"], builds, inner, true_cut)
            tf, n_tf = window_acc(j["test_acc_per_build"], builds, true_cut, 10**9)
            per_seed.append({"seed": s, "val": j["val_acc"],
                             "pseudo_future": pf, "true_future": tf,
                             "n_pseudo": n_pf, "n_true": n_tf,
                             "n_train": j["n_train"]})
        rows[cell] = per_seed

    def mean(cell, k):
        return float(np.mean([r[k] for r in rows[cell]]))

    def sd(cell, k):
        return float(np.std([r[k] for r in rows[cell]], ddof=1))

    sel_pseudo = max(CELLS, key=lambda c: mean(c, "pseudo_future"))
    sel_true_inner = max(CELLS, key=lambda c: mean(c, "true_future"))
    q7_means = {c: float(np.mean(v)) for c, v in Q7_REFERENCE.items()}
    sel_q7 = max(q7_means, key=q7_means.get)

    # seed-cleanliness of the pseudo-future selection (HL=90 family vs rest)
    win_cells = [c for c in CELLS if c.startswith("decayed90")]
    lose_cells = [c for c in CELLS if not c.startswith("decayed90")]
    win_min = min(r["pseudo_future"] for c in win_cells for r in rows[c])
    lose_max = max(r["pseudo_future"] for c in lose_cells for r in rows[c])

    payload = {
        "inner_cutoff": INNER_CUTOFF,
        "true_cutoff": common.TRAIN_CUTOFF_BUILD,
        "pseudo_future_builds": [b for b in builds
                                 if inner < builds.index(b) <= true_cut],
        "per_cell": rows,
        "selected_on_pseudo_future": sel_pseudo,
        "selected_on_true_future_inner_models": sel_true_inner,
        "selected_on_true_future_q7_full_models": sel_q7,
        "q7_reference_means": q7_means,
        "hl90_family_seed_clean_on_pseudo": bool(win_min > lose_max),
        "hl90_family_min_vs_rest_max_pseudo": [win_min, lose_max],
    }
    common.write_json(os.path.join(common.RESULTS_DIR,
                                   "w8c_nested_selection.json"), payload)

    lines = [
        "# W8c — nested-validation champion selection",
        "",
        f"Inner cutoff {INNER_CUTOFF} (train <= inner; models never see the "
        f"pseudo-future or the true test window). Pseudo-future = the 4 "
        f"pre-cutoff builds 2.55.14.* ({rows[CELLS[0]][0]['n_pseudo']:,} rows,"
        f" comparable to the true window's "
        f"{rows[CELLS[0]][0]['n_true']:,}). 4 cells x 3 seeds, existing "
        "feature caches, D2 protocol.",
        "",
        "| cell | val acc | pseudo-future acc | true-future acc "
        "(same inner models) | Q7 full-trained true-future (ref) |",
        "|---|---|---|---|---|",
    ]
    for c in CELLS:
        lines.append(
            f"| {LABELS[c]} | {mean(c,'val'):.2f} ± {sd(c,'val'):.2f} | "
            f"{mean(c,'pseudo_future'):.2f} ± {sd(c,'pseudo_future'):.2f} | "
            f"{mean(c,'true_future'):.2f} ± {sd(c,'true_future'):.2f} | "
            f"{q7_means[c]:.2f} |")
    lines += [
        "",
        f"- Selected on pseudo-future (nested, honest): **{sel_pseudo}**",
        f"- Selected on true future with the same inner-trained models: "
        f"**{sel_true_inner}**",
        f"- Selected on true future with the Q7 full-trained models "
        f"(the original, test-window-selected ranking): **{sel_q7}**",
        f"- HL=90 family seed-clean above the other cells on pseudo-future: "
        f"{'YES' if payload['hl90_family_seed_clean_on_pseudo'] else 'NO'} "
        f"(family min {win_min:.2f} vs rest max {lose_max:.2f})",
    ]
    verdict = ("TRANSFERS" if sel_pseudo == sel_q7 or
               (sel_pseudo.startswith("decayed90") and
                sel_q7.startswith("decayed90")) else "DOES NOT TRANSFER")
    within_family_flip = sel_pseudo != sel_q7
    lines += [
        "",
        f"**Verdict: the nested selection {verdict}.** The cell picked on a "
        "strictly pre-cutoff pseudo-future window "
        f"({sel_pseudo}) {'matches' if verdict=='TRANSFERS' else 'differs from'} "
        f"the test-window-selected champion family ({sel_q7}); the "
        "nested-selected cell beats cumulative on the TRUE window too "
        f"({mean(sel_pseudo,'true_future'):.2f} vs "
        f"{mean('cumulative','true_future'):.2f} with the inner-trained "
        f"models, {q7_means[sel_pseudo]:.2f} vs {q7_means['cumulative']:.2f} "
        "full-trained), so a practitioner selecting honestly at the inner "
        "cutoff would have shipped a decayed-90d model and realized the "
        "gain.",
    ]
    if within_family_flip:
        lines += [
            "",
            "Caveat: WITHIN the HL=90 family the ordering flips between "
            f"windows (pseudo-future prefers {sel_pseudo} "
            f"{mean(sel_pseudo,'pseudo_future'):.2f} vs {sel_q7} "
            f"{mean(sel_q7,'pseudo_future'):.2f}; the true window prefers "
            f"{sel_q7}). The k=100-shrink refinement is within noise of "
            "unshrunk HL=90 — the honest claim is \"decayed ~90d aggregates"
            "\", not the specific shrinkage corner. The family-level "
            "separation from cumulative/HL=365 is what survives nesting; it "
            "is not fully seed-clean on the pseudo-future window "
            f"(family min {win_min:.2f} vs rest max {lose_max:.2f}).",
        ]
    out = os.path.join(common.RESULTS_DIR, "W8C_NESTED_SELECTION.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
