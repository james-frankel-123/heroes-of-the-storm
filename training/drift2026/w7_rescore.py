"""
W7 — vintage-matrix rescoring of a staleness-gradient head-to-head file
using the saved Q2 vintage judges + the QM-2021 judge, plus assembly of the
dose-response summary table across all gradient points.

Usage:
  python3 drift2026/w7_rescore.py --suffix _stale1yr
  python3 drift2026/w7_rescore.py --suffix _stale2yr
  python3 drift2026/w7_rescore.py --assemble   # writes W7_STALENESS_GRADIENT.md
"""
import os
import sys
import json
import glob
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drift2026 import common

common.setup()

import numpy as np
import torch

from qm2026.train_qm_wp import MLP, sym_wp

HERE = os.path.dirname(os.path.abspath(__file__))
JUDGE_DIR = os.path.join(HERE, "models", "vintage_judges")
QM_JUDGE = os.path.join(HERE, "..", "qm2026", "results", "qm_wp_v0.pt")


def load_judges(device):
    judges = {}
    for p in sorted(glob.glob(os.path.join(JUDGE_DIR, "wp_*.pt"))):
        name = os.path.basename(p)[3:-3]
        m = MLP().to(device)
        m.load_state_dict(torch.load(p, weights_only=True, map_location=device))
        m.eval()
        judges[name] = m
    qm = MLP().to(device)
    qm.load_state_dict(torch.load(QM_JUDGE, weights_only=True, map_location=device))
    qm.eval()
    judges["QM-2021"] = qm
    return judges


def rescore(suffix):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = os.path.join(HERE, "results", f"w6_head2head{suffix}.json")
    res = json.load(open(path))
    judges = load_judges(device)
    out = {}
    for name, m in judges.items():
        wps = [sym_wp(m, device, r["maintained"], r["frozen"],
                      r["game_map"], r["tier"]) for r in res["records"]]
        arr = np.array(wps)
        out[name] = {"maintained_wp": float(arr.mean()),
                     "se": float(arr.std(ddof=1) / np.sqrt(len(arr))),
                     "win_share": float((arr > 0.5).mean())}
        print(f"{suffix} judge {name}: {arr.mean():.4f} ± "
              f"{arr.std(ddof=1)/np.sqrt(len(arr)):.4f}")
    res["vintage_matrix"] = out
    json.dump(res, open(path, "w"), indent=2)


def assemble():
    rows = []
    points = [("", "d2b_allhist (~4mo stale)"),
              ("_stale1yr", "stale1yr (~1yr, cutoff 2025-02)"),
              ("_stale2yr", "stale2yr (~2yr, cutoff 2024-02)")]
    for suffix, label in points:
        h2h = json.load(open(os.path.join(HERE, "results",
                                          f"w6_head2head{suffix}.json")))
        jf_path = os.path.join(HERE, "results", f"w6_judgefree{suffix}.json")
        jf = json.load(open(jf_path)) if os.path.exists(jf_path) else None
        row = {"label": label,
               "consensus_wp": h2h["maintained_consensus_wp"],
               "degen_maintained": h2h["side_rates"]["d2c_cumprev"]["degen"]
               if "d2c_cumprev" in h2h["side_rates"]
               else list(h2h["side_rates"].values())[0]["degen"],
               "degen_stale": [v["degen"] for k, v in h2h["side_rates"].items()
                               if k != "d2c_cumprev"][0]}
        if jf:
            row["future_hero_wr_delta"] = jf["paired_deltas"]["future_hero_wr"]["mean"]
            row["future_hero_wr_se"] = jf["paired_deltas"]["future_hero_wr"]["se"]
            row["synergy_delta"] = jf["paired_deltas"]["synergy"]["mean"]
        row["vintage"] = h2h.get("vintage_matrix", {})
        rows.append(row)
    lines = ["# W7 — staleness gradient (dose-response)",
             "",
             "Maintained (d2c_cumprev MCTS) head-to-head vs agents of "
             "increasing staleness; 2,000 drafts per point.",
             "",
             "| staleness | consensus WP (maintained) | Δ future hero WR (pp) | "
             "Δ synergy | stale-side degen % |",
             "|---|---|---|---|---|"]
    for r in rows:
        fh = (f"{r['future_hero_wr_delta']:+.2f} ± {r['future_hero_wr_se']:.2f}"
              if "future_hero_wr_delta" in r else "—")
        sy = f"{r['synergy_delta']:+.2f}" if "synergy_delta" in r else "—"
        lines.append(f"| {r['label']} | {r['consensus_wp']:.4f} | {fh} | {sy} | "
                     f"{r['degen_stale']:.1f} |")
    if all(r.get("vintage") for r in rows):
        lines += ["", "## Vintage matrix across gradient points "
                  "(maintained WP by judge era)", "",
                  "| judge | " + " | ".join(r["label"] for r in rows) + " |",
                  "|---|" + "---|" * len(rows)]
        for judge in rows[0]["vintage"]:
            lines.append(f"| {judge} | " + " | ".join(
                f"{r['vintage'][judge]['maintained_wp']:.4f}" for r in rows) + " |")
    out = os.path.join(HERE, "results", "W7_STALENESS_GRADIENT.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default=None)
    ap.add_argument("--assemble", action="store_true")
    args = ap.parse_args()
    if args.assemble:
        assemble()
    elif args.suffix is not None:
        rescore(args.suffix)
    else:
        ap.error("need --suffix or --assemble")


if __name__ == "__main__":
    main()
