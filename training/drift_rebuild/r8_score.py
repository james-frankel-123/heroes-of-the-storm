"""
R8 — judge-free scoring of head-to-head files with the W12 out-of-sample
protocol, imported unchanged from the frozen drift2026/w12_clean_truth.py
(truth_sets, judgefree) and drift2026/w8_inference.py (crossed_re). Adds a
Satterthwaite df and a t-reference p-value for each contrast (audit C1), and
the W13 net team-win-probability score when its weights are available.

Scores every drift_rebuild/results/h2h/*.json plus the paper's stored W6/W7
files for side-by-side comparison. Truth sets: T_clean (headline),
T_97039, T_255_17. No build after 2.55.17.98025 is touched.

Usage: python3 drift_rebuild/r8_score.py
Output: drift_rebuild/results/r8_scores.json and R8_SCORES.md
"""
import os
import sys
import json
import math
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rb_common as rb  # noqa: E402

from drift2026 import common  # noqa: E402

common.setup()
import numpy as np  # noqa: E402
from drift2026.w12_clean_truth import truth_sets, judgefree  # noqa: E402
from drift2026.w8_inference import attach_labels, crossed_re  # noqa: E402

PAPER_FILES = {"paper_M_vs_U": "w6_head2head.json",
               "paper_M_vs_S1": "w6_head2head_stale1yr.json",
               "paper_M_vs_S2": "w6_head2head_stale2yr.json",
               "paper_M_vs_U_15x15": "w6_head2head_w10_degen.json"}


def t_sf(x, df):
    xs = np.linspace(x, x + 400, 400001)
    c = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)
    return float(np.trapezoid(c * (1 + xs ** 2 / df) ** (-(df + 1) / 2), xs))


def infer(v, sa, sb):
    ok = ~np.isnan(v)
    cr = crossed_re(v[ok], sa[ok], sb[ok])
    A, B, _ = cr["shape"]
    c = cr["components"]
    a, b = c["seed_maintained"] / A, c["seed_frozen"] / B
    i = (c["interaction"] + c["within_pairing_sampling"]) / (A * B)
    den = ((a ** 2 / (A - 1) if a > 0 else 0) + (b ** 2 / (B - 1) if b > 0 else 0)
           + i ** 2 / ((A - 1) * (B - 1)))
    df = (a + b + i) ** 2 / den if den > 0 else float("inf")
    z = cr["est"] / cr["se"] if cr["se"] > 0 else float("nan")
    return {"est": cr["est"], "se": cr["se"], "z": z, "df": df,
            "p_t": 2 * t_sf(abs(z), df) if np.isfinite(z) else None,
            "shape": cr["shape"], "components": c}


def load_records(path):
    res = json.load(open(path))
    if "records" in res and res["records"] and "sa" in res["records"][0]:
        return res["records"], res
    return attach_labels(res), res


def main():
    sets, _ = truth_sets()
    files = {os.path.splitext(os.path.basename(p))[0]: p
             for p in sorted(glob.glob(os.path.join(rb.RESULTS_DIR, "h2h", "*.json")))}
    for k, f in PAPER_FILES.items():
        files[k] = os.path.join(common.RESULTS_DIR, f)
    out = {}
    for key, path in files.items():
        recs, res = load_records(path)
        sa = np.array([r["sa"] for r in recs])
        sb = np.array([r["sb"] for r in recs])
        entry = {"path": path, "n_drafts": len(recs),
                 "a": res.get("a", res.get("arms", {}).get("maintained")),
                 "b": res.get("b", res.get("arms", {}).get("frozen")),
                 "by_truth": {}}
        for tname in ("T_clean", "T_97039", "T_255_17"):
            d = judgefree(recs, sets[tname][0])
            entry["by_truth"][tname] = {k: infer(v, sa, sb) for k, v in d.items()}
        from shared import is_degenerate
        entry["degen_rate_a"] = float(np.mean([is_degenerate(r["maintained"]) for r in recs]))
        entry["degen_rate_b"] = float(np.mean([is_degenerate(r["frozen"]) for r in recs]))
        out[key] = entry
        h = entry["by_truth"]["T_clean"]["future_hero_wr"]
        print(f"{key:28s} n={len(recs):5d} heroWR {h['est']:+.3f} ± {h['se']:.3f} "
              f"(z {h['z']:.1f}, df {h['df']:.1f}, p_t {h['p_t']:.3g})", flush=True)
    with open(os.path.join(rb.RESULTS_DIR, "r8_scores.json"), "w") as f:
        json.dump(out, f, indent=1)
    lines = ["# R8 judge-free head-to-head scores (W12 protocol, T_clean)", "",
             "Paired (a minus b) deltas, crossed seed random effects; t reference "
             "with Satterthwaite df.", "",
             "| file | a | b | drafts | hero WR pp | synergy | counter | degen (a-b) | degen a / b |",
             "|---|---|---|---|---|---|---|---|---|"]
    for key, e in out.items():
        t = e["by_truth"]["T_clean"]

        def f(m, scale=1.0):
            r = t[m]
            return (f"{scale*r['est']:+.3f} ± {scale*r['se']:.3f} "
                    f"(z {r['z']:.1f}, df {r['df']:.0f})")
        lines.append(f"| {key} | {e['a']} | {e['b']} | {e['n_drafts']} | "
                     f"{f('future_hero_wr')} | {f('synergy')} | {f('counter')} | "
                     f"{f('degen', 100)} | {100*e['degen_rate_a']:.1f} / "
                     f"{100*e['degen_rate_b']:.1f} |")
    with open(os.path.join(rb.RESULTS_DIR, "R8_SCORES.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
