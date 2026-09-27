"""
W8 — assemble the W8a / W8b deliverables from the head-to-head + rescoring +
inference outputs.

Usage:
  python3 drift2026/w8_assemble.py --item a   # W8A_VOLMATCH.{md,json}
  python3 drift2026/w8_assemble.py --item b   # W8B_CHAMPION.{md,json}
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drift2026 import common

import numpy as np

R = common.RESULTS_DIR


def j(path):
    with open(os.path.join(R, path)) as f:
        return json.load(f)


def h2h_block(suffix):
    h = j(f"w6_head2head{suffix}.json")
    jf = j(f"w6_judgefree{suffix}.json")
    inf = j(f"w8/inference{suffix}.json")
    return {"h2h": h, "jf": jf, "inf": inf}


def fmt_h2h_rows(name, blk):
    h, jf, inf = blk["h2h"], blk["jf"], blk["inf"]
    c = inf["continuous"]
    b = inf["binary"]
    g = inf["judge_free_crossed"]
    arms = h["arms"]
    return (
        f"| {name} | {h['n_drafts']} | {c['est']:.4f} ± {c['se']:.4f} "
        f"(z {c['z_vs_half']:.2f}) | {b['est']:.3f} ± {b['se']:.3f} | "
        f"{g['future_hero_wr']['est']:+.2f} ± {g['future_hero_wr']['se']:.2f} "
        f"(z {g['future_hero_wr']['z']:.1f}) | "
        f"{g['synergy']['est']:+.2f} ± {g['synergy']['se']:.2f} "
        f"(z {g['synergy']['z']:.1f}) | "
        f"{100*g['degen']['est']:+.1f} ± {100*g['degen']['se']:.1f} "
        f"(z {g['degen']['z']:.1f}) | "
        f"{h['side_rates'][arms['maintained']]['degen']:.1f} / "
        f"{h['side_rates'][arms['frozen']]['degen']:.1f} |")


def vintage_table(blocks):
    names = [n for n, _ in blocks]
    judges = None
    for _, blk in blocks:
        vm = blk["h2h"].get("vintage_matrix")
        if vm:
            judges = list(vm.keys())
            break
    if judges is None:
        return ["(vintage rescoring not present)"]
    lines = ["| judge | " + " | ".join(names) + " |",
             "|---|" + "---|" * len(names)]
    for jd in judges:
        cells = []
        for _, blk in blocks:
            vm = blk["h2h"].get("vintage_matrix", {})
            cells.append(f"{vm[jd]['maintained_wp']:.4f}" if jd in vm else "—")
        lines.append(f"| {jd} | " + " | ".join(cells) + " |")
    return lines


def item_a():
    vols = {}
    vols["maintained (cutoff 2.55.14.95918, 2026-02)"] = \
        j("d2/d2c_cumprev_s42.json")["n_train"]
    vols["stale-1yr (cutoff 2.55.9.93613, 2025-02)"] = \
        j("w7/w7_stale_2.55.9.93613_s42.json")["n_train"]
    vols["stale-2yr (cutoff 2.55.4.91418, 2024-02)"] = \
        j("w7/w7_stale_2.55.4.91418_s42.json")["n_train"]
    vm_meta = {s: j(f"w8/w8_volmatch_s{s}.json") for s in [42, 123, 777]}
    vols["volume-matched maintained (subsample)"] = vm_meta[42]["n_train"]

    blk_vm = h2h_block("_w8_volmatch")
    blk_2yr = h2h_block("_stale2yr")

    vm_acc = [m["test_acc_future"] for m in vm_meta.values()]
    full_acc = [j(f"d2/d2c_cumprev_s{s}.json")["test_acc_future"]
                for s in [42, 123, 777]]
    stale_acc = [j(f"w7/w7_stale_2.55.4.91418_s{s}.json")["test_acc_future"]
                 for s in [42, 123, 777]]

    g_vm = blk_vm["inf"]["judge_free_crossed"]
    g_2yr = blk_2yr["inf"]["judge_free_crossed"]
    hw_vm, hw_2yr = g_vm["future_hero_wr"], g_2yr["future_hero_wr"]
    retained = hw_vm["est"] / hw_2yr["est"] if hw_2yr["est"] else float("nan")

    payload = {
        "training_volumes_rows": vols,
        "vf_future_acc": {"volmatch": vm_acc, "full_maintained": full_acc,
                          "stale2yr": stale_acc},
        "h2h_volmatch_vs_stale2yr": {k: blk_vm[k] for k in ("jf", "inf")},
        "reference_maintained_vs_stale2yr": {k: blk_2yr[k] for k in ("jf", "inf")},
        "volmatch_consensus": blk_vm["h2h"]["maintained_consensus_wp"],
        "edge_retained_hero_wr_frac": retained,
    }
    common.write_json(os.path.join(R, "w8a_volmatch.json"), payload)

    lines = [
        "# W8a — volume-matched staleness-gradient control",
        "",
        "The corpus grows over time, so the stale-2yr arm trained on fewer "
        "rows than the maintained arm; the W7 \"+1.16pp at 2yr\" could "
        "conflate staleness with training-set size. Control: the maintained "
        "config (cumulative_prev features, regime=all) trained on a seeded "
        "random subsample matched to stale-2yr's n_train, then the full W4 "
        "MCTS recipe (5 seeds) and the W6 head-to-head protocol vs the same "
        "stale-2yr agent.",
        "",
        "## Per-cutoff training volumes (rows; every VF sees 2x-augmented "
        "rows of its era's replays)",
        "",
        "| arm | train rows |",
        "|---|---:|",
    ]
    for k, v in vols.items():
        lines.append(f"| {k} | {v:,} |")
    lines += [
        "",
        "## Value-function future accuracy (3 seeds)",
        "",
        f"- maintained, full volume: {np.mean(full_acc):.2f} ± "
        f"{np.std(full_acc, ddof=1):.2f}",
        f"- maintained config, volume-matched to 2yr: {np.mean(vm_acc):.2f} ± "
        f"{np.std(vm_acc, ddof=1):.2f}",
        f"- stale-2yr: {np.mean(stale_acc):.2f} ± "
        f"{np.std(stale_acc, ddof=1):.2f}",
        "",
        "## Head-to-head vs stale-2yr (crossed-RE over 5x5 seed pairings)",
        "",
        "| maintained side | n | consensus WP (z vs .5) | binary share | "
        "Δ future hero WR pp | Δ synergy | Δ degen pp | degen % (m/f) |",
        "|---|---|---|---|---|---|---|---|",
        fmt_h2h_rows("volume-matched", blk_vm),
        fmt_h2h_rows("full-volume (W7 ref)", blk_2yr),
        "",
        "## Vintage matrix",
        "",
    ]
    lines += vintage_table([("volmatch vs 2yr", blk_vm),
                            ("full vs 2yr (ref)", blk_2yr)])
    lines += [
        "",
        "## Verdict",
        "",
        f"The volume-matched maintained agent retains "
        f"{100*retained:.0f}% of the full-volume agent's judge-free edge "
        f"over stale-2yr ({hw_vm['est']:+.2f} ± {hw_vm['se']:.2f} vs "
        f"{hw_2yr['est']:+.2f} ± {hw_2yr['se']:.2f} pp future-hero-WR; "
        f"z {hw_vm['z']:.1f} vs {hw_2yr['z']:.1f}).",
    ]
    out = os.path.join(R, "W8A_VOLMATCH.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def degen_trend(blocks_with_staleness):
    """Seed-level degen rates on the STALE side vs staleness (years); OLS."""
    xs, ys, arm_rows = [], [], []
    for name, years, blk in blocks_with_staleness:
        recs = blk["h2h"]["records"]
        by_seed = {}
        for r in recs:
            by_seed.setdefault(r["sb"], []).append(
                float(__import__("shared").is_degenerate(r["frozen"])))
        rates = {sb: float(np.mean(v)) for sb, v in sorted(by_seed.items())}
        for sb, rate in rates.items():
            xs.append(years)
            ys.append(rate)
        arm_rows.append((name, years, float(np.mean(list(rates.values()))),
                         len(rates)))
    x, y = np.array(xs), np.array(ys)
    n = len(x)
    slope = float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1))
    icpt = float(y.mean() - slope * x.mean())
    resid = y - (icpt + slope * x)
    se = float(np.sqrt((resid @ resid) / (n - 2) / np.sum((x - x.mean()) ** 2)))
    return {"slope_per_year": slope, "se": se, "z": slope / se, "n_seeds": n,
            "arm_rates": arm_rows}


def item_b():
    blocks = [("champion vs stale-4mo (15x15, 10/cell, staggered)",
               1 / 3, h2h_block("_w8_champ4mo")),
              ("champion vs stale-1yr (5x5, 40/cell)", 1.0,
               h2h_block("_w8_champ1yr")),
              ("champion vs stale-2yr (5x5, 40/cell)", 2.0,
               h2h_block("_w8_champ2yr"))]
    trend = degen_trend([(n, yr, b) for n, yr, b in blocks])

    c4 = blocks[0][2]["inf"]
    degen_z = c4["judge_free_crossed"]["degen"]["z"]
    w6_ref = j("w6_inference.json")

    payload = {
        "arms": {n: {"jf": b["jf"], "inf": b["inf"],
                     "consensus": b["h2h"]["maintained_consensus_wp"],
                     "side_rates": b["h2h"]["side_rates"],
                     "vintage": b["h2h"].get("vintage_matrix", {})}
                 for n, _, b in blocks},
        "degen_trend_stale_side": trend,
        "champ4mo_degen_delta_z": degen_z,
        "w6_reference_degen_z": w6_ref["judge_free_crossed"]["degen_delta"]["z"],
    }
    common.write_json(os.path.join(R, "w8b_champion.json"), payload)

    lines = [
        "# W8b — champion-config maintained agent + seed power",
        "",
        "Maintained side = the RECOMMENDED policy (Q7 champion decayed90k100"
        ", 57.22 future acc; VF q7_decayed90k100_s777, deploy stats "
        "decayed90k100 @ last completed build), MCTS x15 seeds. Stale-4mo "
        "(d2b_allhist) extended 5 -> 15 seeds (w4_ + w8_ runs pooled). All "
        "inference = crossed-RE over (seed_maintained, seed_stale).",
        "",
        "| head-to-head | n | consensus WP (z vs .5) | binary share | "
        "Δ future hero WR pp | Δ synergy | Δ degen pp | degen % (champ/stale) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, _, blk in blocks:
        lines.append(fmt_h2h_rows(name, blk))
    lines += ["", "## Vintage matrix (maintained WP by judge era)", ""]
    lines += vintage_table([(n, b) for n, _, b in blocks])
    tr = trend
    lines += [
        "",
        "## Degenerate-rate trend across the gradient (stale side, "
        "seed-level OLS)",
        "",
        "| arm | staleness (yr) | stale-side degen % | n stale seeds |",
        "|---|---|---|---|",
    ]
    for name, yr, rate, k in tr["arm_rates"]:
        lines.append(f"| {name} | {yr:.2f} | {100*rate:.1f} | {k} |")
    lines += [
        "",
        f"OLS slope: {100*tr['slope_per_year']:+.2f} pp/year "
        f"(SE {100*tr['se']:.2f}, z {tr['z']:.2f}, {tr['n_seeds']} seed-level "
        "points).",
        "",
        "## The z=1.8 degen question",
        "",
        f"W6 (cumprev vs 4mo, 5x5): degen delta z = "
        f"{w6_ref['judge_free_crossed']['degen_delta']['z']:.2f} (n.s.). "
        f"W8b (champion vs 4mo, 15x15): degen delta z = {degen_z:.2f} — "
        + ("RESOLVED (now individually significant)." if abs(degen_z) >= 1.96
           else "still not individually significant at 15x15."),
    ]
    out = os.path.join(R, "W8B_CHAMPION.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--item", required=True, choices=["a", "b"])
    args = ap.parse_args()
    (item_a if args.item == "a" else item_b)()


if __name__ == "__main__":
    main()
