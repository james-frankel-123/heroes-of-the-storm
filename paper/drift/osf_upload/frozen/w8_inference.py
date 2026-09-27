"""
W8 — crossed-random-effects inference for head-to-head files, generalized to
non-5x5 seed grids (the Q3/W6 method-of-moments ANOVA, reimplemented as a
reusable script; `--check-w6` verifies exact reproduction of the stored
results/w6_inference.json numbers).

Estimands per head-to-head file:
  - continuous consensus score, crossed RE over (seed_a, seed_b)
  - binary win share (consensus > 0.5), crossed RE
  - conservative disjoint-pairings estimate (diagonal cells)
  - judge-free paired per-draft deltas (future hero WR pp / synergy / degen
    fraction), recomputed from the terminal records against future-truth
    stats, each under the same crossed-RE decomposition

Method (balanced two-way crossed random effects, method of moments):
  cell means M[a,b] over the n_cell drafts of pairing (a,b) (both orderings);
  v_s   = pooled within-cell variance / n_cell        (sampling var of a mean)
  MSI'  = interaction mean square of M                 -> sigma2_int = MSI'-v_s
  MSA'  = B*sum_a(rowmean-grand)^2/(A-1)               -> sigma2_a = (MSA'-MSI')/B
  MSB'  analogously                                    -> sigma2_b = (MSB'-MSI')/A
  SE^2(grand) = sigma2_a/A + sigma2_b/B + MSI'/(A*B)
(components clipped at 0; MSI' term keeps sampling + interaction together).

Usage:
  python3 drift2026/w8_inference.py --check-w6
  python3 drift2026/w8_inference.py --suffix _w8_volmatch
Output: results/w8/inference<suffix>.json (+ stdout table)
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drift2026 import common

common.setup()

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def attach_labels(res):
    """Ensure every record carries (sa, sb); reconstruct from the
    deterministic loop order (sa, sb, order, i) when absent, and verify the
    reconstruction against the stored pairing_means."""
    records = res["records"]
    if "sa" in records[0]:
        return records
    n_pair = res["n_pairings"]
    na = nb = int(round(np.sqrt(n_pair)))
    assert na * nb == n_pair, "non-square grid without stored labels"
    d = len(records) // (na * nb * 2)
    assert na * nb * 2 * d == len(records)
    k = 0
    for sa in range(na):
        for sb in range(nb):
            for order in (0, 1):
                for _ in range(d):
                    records[k]["sa"], records[k]["sb"] = sa, sb
                    records[k]["order"] = order
                    k += 1
    # verify against stored pairing means
    for key, stored in res["pairing_means"].items():
        sa, sb = (int(x) for x in key.split("_"))
        vals = [r["consensus_maintained"] for r in records
                if r["sa"] == sa and r["sb"] == sb]
        assert abs(np.mean(vals) - stored) < 1e-9, key
    return records


def crossed_re(values, sa, sb):
    """values: per-draft array; sa/sb: seed labels. Returns dict."""
    values, sa, sb = np.asarray(values, float), np.asarray(sa), np.asarray(sb)
    A, B = sa.max() + 1, sb.max() + 1
    M = np.zeros((A, B))
    cell_vars, cell_ns = [], []
    for a in range(A):
        for b in range(B):
            v = values[(sa == a) & (sb == b)]
            M[a, b] = v.mean()
            cell_vars.append(v.var(ddof=1))
            cell_ns.append(len(v))
    assert len(set(cell_ns)) == 1, "unbalanced cells"
    n_cell = cell_ns[0]
    v_s = float(np.mean(cell_vars) / n_cell)
    grand = float(M.mean())
    row, col = M.mean(axis=1), M.mean(axis=0)
    msa = B * np.sum((row - grand) ** 2) / (A - 1)
    msb = A * np.sum((col - grand) ** 2) / (B - 1)
    msi = float(np.sum((M - row[:, None] - col[None, :] + grand) ** 2)
                / ((A - 1) * (B - 1)))
    s2_int = max(msi - v_s, 0.0)
    s2_a = max((msa - msi) / B, 0.0)
    s2_b = max((msb - msi) / A, 0.0)
    se = float(np.sqrt(s2_a / A + s2_b / B + (s2_int + v_s) / (A * B)))
    diag = np.array([M[i, i] for i in range(min(A, B))])
    return {
        "est": grand, "se": se,
        "components": {"seed_maintained": s2_a, "seed_frozen": s2_b,
                       "interaction": s2_int, "within_pairing_sampling": v_s},
        "disjoint_diag": {"est": float(diag.mean()),
                          "se": float(diag.std(ddof=1) / np.sqrt(len(diag)))},
        "shape": [int(A), int(B), int(n_cell)],
    }


def judgefree_deltas(records):
    """Per-draft paired (maintained - frozen) deltas vs future-truth stats:
    future hero WR (pp), synergy, degenerate indicator (fraction).
    Mirrors w6_judgefree.py exactly."""
    from shared import is_degenerate, HERO_ROLE_FINE  # noqa: F401
    from drift2026.phase_w4_mcts import future_truth_stats
    truth = future_truth_stats()

    def ctr_d(ha, hb, tier):
        r = truth.get_counter(ha, hb, tier)
        if r is None:
            return None
        return r - (truth.get_hero_wr(ha, tier)
                    + (100 - truth.get_hero_wr(hb, tier)) - 50)

    def syn_d(ha, hb, tier):
        r = truth.get_synergy(ha, hb, tier)
        if r is None:
            return None
        return r - (50 + (truth.get_hero_wr(ha, tier) - 50)
                    + (truth.get_hero_wr(hb, tier) - 50))

    def side_metrics(own, opp, tier):
        cd = [x for o in opp for h in own
              for x in [ctr_d(h, o, tier)] if x is not None]
        sy = [x for j, h1 in enumerate(own) for h2 in own[j + 1:]
              for x in [syn_d(h1, h2, tier)] if x is not None]
        hw = [truth.get_hero_wr(h, tier) for h in own]
        hw = [v for v in hw if v is not None]
        return (np.mean(cd) if cd else 0.0, np.mean(sy) if sy else 0.0,
                np.mean(hw) if hw else np.nan, float(is_degenerate(own)))

    out = {"counter": [], "synergy": [], "future_hero_wr": [], "degen": []}
    for r in records:
        m = side_metrics(r["maintained"], r["frozen"], r["tier"])
        f = side_metrics(r["frozen"], r["maintained"], r["tier"])
        out["counter"].append(m[0] - f[0])
        out["synergy"].append(m[1] - f[1])
        out["future_hero_wr"].append(m[2] - f[2])
        out["degen"].append(m[3] - f[3])
    return {k: np.array(v) for k, v in out.items()}


def analyze(path):
    res = json.load(open(path))
    records = attach_labels(res)
    sa = [r["sa"] for r in records]
    sb = [r["sb"] for r in records]
    cons = np.array([r["consensus_maintained"] for r in records])

    result = {"in": os.path.basename(path),
              "arms": res["arms"], "n_drafts": len(records)}
    result["continuous"] = crossed_re(cons, sa, sb)
    result["binary"] = crossed_re((cons > 0.5).astype(float), sa, sb)
    for k in ("continuous", "binary"):
        result[k]["z_vs_half"] = (result[k]["est"] - 0.5) / result[k]["se"]

    deltas = judgefree_deltas(records)
    jf = {}
    for name, vals in deltas.items():
        ok = ~np.isnan(vals)
        r = crossed_re(vals[ok], np.asarray(sa)[ok], np.asarray(sb)[ok])
        r["z"] = r["est"] / r["se"] if r["se"] > 0 else float("nan")
        jf[name] = r
    result["judge_free_crossed"] = jf
    return result


def check_w6():
    stored = json.load(open(os.path.join(HERE, "results", "w6_inference.json")))
    got = analyze(os.path.join(HERE, "results", "w6_head2head.json"))

    def close(a, b, tol=1e-9):
        assert abs(a - b) < tol, (a, b)

    close(got["continuous"]["est"], stored["continuous"]["est"])
    close(got["continuous"]["se"], stored["continuous"]["se"])
    close(got["binary"]["est"], stored["binary"]["est"])
    close(got["binary"]["se"], stored["binary"]["se"])
    close(got["continuous"]["disjoint_diag"]["est"], stored["disjoint5"]["est"])
    close(got["continuous"]["disjoint_diag"]["se"], stored["disjoint5"]["se"])
    for k, v in stored["variance_components"].items():
        close(got["continuous"]["components"][k], v)
    key_map = {"future_hero_wr_delta_pp": "future_hero_wr",
               "synergy_delta": "synergy", "degen_delta": "degen"}
    for sk, gk in key_map.items():
        close(got["judge_free_crossed"][gk]["est"],
              stored["judge_free_crossed"][sk]["est"], 1e-6)
        close(got["judge_free_crossed"][gk]["se"],
              stored["judge_free_crossed"][sk]["se"], 1e-6)
    print("CHECK-W6: all stored w6_inference.json numbers reproduced "
          "(continuous/binary/disjoint/components/judge-free).")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suffix", default=None)
    ap.add_argument("--check-w6", action="store_true")
    args = ap.parse_args()
    if args.check_w6:
        check_w6()
        return
    assert args.suffix is not None
    path = os.path.join(HERE, "results", f"w6_head2head{args.suffix}.json")
    result = analyze(path)
    out = os.path.join(HERE, "results", "w8", f"inference{args.suffix}.json")
    common.write_json(out, result)
    c, b = result["continuous"], result["binary"]
    print(f"consensus crossed-RE: {c['est']:.4f} ± {c['se']:.4f} "
          f"(z vs .5 = {c['z_vs_half']:.2f}); binary {b['est']:.4f} ± "
          f"{b['se']:.4f} (z {b['z_vs_half']:.2f}); "
          f"diag {c['disjoint_diag']['est']:.4f} ± {c['disjoint_diag']['se']:.4f}")
    for k, r in result["judge_free_crossed"].items():
        print(f"  {k}: {r['est']:+.4f} ± {r['se']:.4f} (z {r['z']:.2f})")


if __name__ == "__main__":
    main()
