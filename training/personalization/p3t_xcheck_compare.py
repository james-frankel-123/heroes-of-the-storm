"""
P3 tournament cross-check (2026-10-21), local comparison step. Compares
results computed on the 5090 or 3080 with those computed on the 3090:
  judge   judge.json coefficients (fit on each host from the same judgefit.npz)
  h2h     every head-to-head in analysis.json (J, CONS, judge-free columns)
  drafts  re-drafted pairs (tourney_xchk) against the 3090's pair files
  scores  per-lobby score arrays of the re-drafted pairs
Usage: python3 p3t_xcheck_compare.py <dir with the other host's files> <dir with the 3090's files>
Each dir may hold judge.json, analysis.json, tourney/*.npz, scores/*.npz.
Writes xcheck_compare.json into the first dir and prints a summary.
"""
import glob
import json
import os
import sys

import numpy as np


def main():
    a_dir, b_dir = sys.argv[1], sys.argv[2]
    out = {}
    ja, jb = os.path.join(a_dir, "judge.json"), os.path.join(b_dir, "judge.json")
    if os.path.exists(ja) and os.path.exists(jb):
        A, B = json.load(open(ja)), json.load(open(jb))
        num = {k: (A[k], B[k]) for k in A if isinstance(A.get(k), (int, float)) and isinstance(B.get(k), (int, float))}
        out["judge"] = {"max_abs_diff": max(abs(x - y) for x, y in num.values()) if num else None,
                        "fields": {k: v for k, v in num.items()}}
    aa, ab = os.path.join(a_dir, "analysis.json"), os.path.join(b_dir, "analysis.json")
    if os.path.exists(aa) and os.path.exists(ab):
        A, B = json.load(open(aa))["h2h"], json.load(open(ab))["h2h"]
        common = sorted(set(A) & set(B))
        worst, diffs = {}, {}
        for m in ("judge_personal", "consensus", "diff_rec_mean", "diff_truth_hero"):
            d = [abs(A[k][m][0] - B[k][m][0]) for k in common if m in A[k] and m in B[k]]
            diffs[m] = {"pairs": len(d), "max_abs_diff_point": max(d) if d else None,
                        "identical_points": sum(x == 0 for x in d)}
            dc = [max(abs(A[k][m][1] - B[k][m][1]), abs(A[k][m][2] - B[k][m][2])) for k in common if m in A[k] and m in B[k]]
            diffs[m]["max_abs_diff_ci"] = max(dc) if dc else None
        out["h2h"] = {"pairs_common": len(common), "only_a": len(set(A) - set(B)), "only_b": len(set(B) - set(A)),
                      "metrics": diffs}
    da, db = os.path.join(a_dir, "tourney"), os.path.join(b_dir, "tourney")
    if os.path.isdir(da):
        res = {}
        for f in sorted(glob.glob(os.path.join(da, "*__*.npz"))):
            g = os.path.join(db, os.path.basename(f))
            if not os.path.exists(g):
                continue
            x, y = np.load(f)["acts"], np.load(g)["acts"]
            diff = (x != y).any(1)
            first = np.where(diff, (x != y).argmax(1), -1)
            res[os.path.basename(f)[:-4]] = {"drafts_differing": int(diff.sum()), "of": int(len(x)),
                                             "first_diff_step_hist": np.bincount(first[diff], minlength=16).tolist()
                                             if diff.any() else []}
        out["drafts"] = res
    sa, sb = os.path.join(a_dir, "scores"), os.path.join(b_dir, "scores")
    if os.path.isdir(sa):
        res = {}
        for f in sorted(glob.glob(os.path.join(sa, "*.npz"))):
            g = os.path.join(sb, os.path.basename(f))
            if not os.path.exists(g):
                continue
            x, y = np.load(f), np.load(g)
            r = {}
            for k in x.files:
                if k in y.files and x[k].dtype.kind == "f":
                    a_, b_ = x[k].astype(np.float64), y[k].astype(np.float64)
                    m = np.isfinite(a_) & np.isfinite(b_)
                    r[k] = float(np.abs(a_[m] - b_[m]).max()) if m.any() else 0.0
            res[os.path.basename(f)[:-4]] = r
        out["scores"] = res
    json.dump(out, open(os.path.join(a_dir, "xcheck_compare.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k in ("judge", "h2h")}, indent=1))
    if "drafts" in out:
        n = sum(v["drafts_differing"] for v in out["drafts"].values())
        t = sum(v["of"] for v in out["drafts"].values())
        print(f"drafts: {len(out['drafts'])} pairs, {n} of {t} drafts differ")
        for k, v in out["drafts"].items():
            if v["drafts_differing"]:
                print("  ", k, v["drafts_differing"], v["first_diff_step_hist"])


if __name__ == "__main__":
    main()
