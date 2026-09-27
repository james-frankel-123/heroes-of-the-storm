"""
PHASE W3 orchestrator (wave 3 GPU jobs, self-driving through (d)(e)(g)):

  W3d  hybrid per-signal features arm: w3d_hybrid x 3 seeds
       (features=hybrid from w3_build_hybrid.py, regime=all, full sanity).
  W3e  embed+refresh arm: w3e_embedrefresh x 3 seeds (features=
       cumulative_prev, regime=embed) — the wave-1 regime winner stacked on
       the wave-1 sourcing winner.
  W3b' W2b-style greedy policy eval for both new arms (3 seeds x 500 drafts;
       reference cells already exist in results/w2b/).
  W3g  W2c retrain-simulation additions:
         - never-retrain-at-C0 arms for BOTH new regimes
           (w3g_{hybrid,embedrefresh}_cut08_s42)
         - K=6 cadence trains (cutoff positions 14..38) for the WINNER of
           (d)/(e) IF its mean future acc beats d2c_cumprev; decided from
           the (d)/(e) results after they finish.

Stages run sequentially in-process; every job is output-idempotent, so the
script can be re-run. Use --num-gpus to share the pool politely.

Usage:
    python drift2026/phase_w3.py [--dry-run] [--num-gpus 2] [--skip-w2b]
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drift2026 import common

TRAIN = os.path.join(common.DRIFT_DIR, "train_drift_wp.py")
EVAL_W2B = os.path.join(common.DRIFT_DIR, "eval_policy_w2b.py")

ARMS = {   # cell -> (features, regime)
    "w3d_hybrid": ("hybrid", "all"),
    "w3e_embedrefresh": ("cumulative_prev", "embed"),
}
C0_POS = 8   # keep in sync with phase_w2.py
K6_POSITIONS = [14, 20, 26, 32, 38]
CUMPREV_REF_CELL = "d2c_cumprev"


def d2_result(cell, seed):
    return os.path.join(common.RESULTS_DIR, "d2", f"{cell}_s{seed}.json")


def mean_future(cell):
    vals = []
    for s in common.SEEDS:
        p = d2_result(cell, s)
        if os.path.exists(p):
            with open(p) as f:
                vals.append(json.load(f)["test_acc_future"])
    return sum(vals) / len(vals) if vals else None


def train_job(name, features, regime, seed, cutoff=None, subdir="d2",
              skip_sanity=False):
    argv = [TRAIN, "--name", name, "--features", features,
            "--regime", regime, "--seed", str(seed)]
    if cutoff:
        argv += ["--cutoff-build", cutoff]
    if skip_sanity:
        argv += ["--skip-sanity"]
    argv += ["--results-subdir", subdir]
    return common.Job(name, argv,
                      [os.path.join(common.RESULTS_DIR, subdir,
                                    f"{name}.json")],
                      weight="light")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--num-gpus", type=int, default=None)
    ap.add_argument("--skip-w2b", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    common.setup()

    builds_255 = [b for b in common.load_patch_index()["builds"]
                  if b.startswith("2.55")]

    hybrid_npz = os.path.join(common.CACHE_DIR, "features_hybrid.npz")
    if not os.path.exists(hybrid_npz) and not args.dry_run:
        print(f"FATAL: {hybrid_npz} missing — run w3_build_hybrid.py first")
        sys.exit(1)

    # ── stage 1: (d)+(e) trains ──
    jobs = [train_job(f"{cell}_s{seed}", feats, regime, seed)
            for cell, (feats, regime) in ARMS.items()
            for seed in common.SEEDS]
    print(f"Stage 1 (W3 d/e trains): {len(jobs)} jobs")
    failed = common.run_pool(jobs, num_gpus=args.num_gpus,
                             dry_run=args.dry_run, force=args.force)
    if failed:
        print(f"FATAL: stage 1 failures: {[j.name for j in failed]}")
        sys.exit(1)

    # ── stage 2: policy evals ──
    if not args.skip_w2b:
        jobs = []
        for cell in ARMS:
            for seed in common.SEEDS:
                out = os.path.join(common.RESULTS_DIR, "w2b",
                                   f"{cell}_s{seed}.json")
                jobs.append(common.Job(
                    f"w2b_{cell}_s{seed}",
                    [EVAL_W2B, "--cell", cell, "--seed", str(seed),
                     "--drafts", "500"],
                    [out], weight="light"))
        print(f"Stage 2 (W2b-style policy evals): {len(jobs)} jobs")
        failed = common.run_pool(jobs, num_gpus=args.num_gpus,
                                 dry_run=args.dry_run, force=args.force)
        if failed:
            print(f"WARNING: policy eval failures: {[j.name for j in failed]}")

    # ── stage 3: (g) retrain-simulation additions ──
    if args.dry_run:
        print("Stage 3 (dry run): would train never-retrain-at-C0 arms + "
              "winner K=6 cadence")
        return

    jobs = []
    for cell, (feats, regime) in ARMS.items():
        arm = cell.split("_", 1)[1]
        jobs.append(train_job(f"w3g_{arm}_cut{C0_POS:02d}_s42", feats, regime,
                              42, cutoff=builds_255[C0_POS], subdir="w2c",
                              skip_sanity=True))

    scores = {cell: mean_future(cell) for cell in ARMS}
    ref = mean_future(CUMPREV_REF_CELL)
    winner = max(scores, key=lambda c: scores[c] or -1)
    print(f"(d)/(e) future acc: {scores}; cumprev ref: {ref}")
    if scores[winner] is not None and ref is not None and scores[winner] > ref:
        feats, regime = ARMS[winner]
        arm = winner.split("_", 1)[1]
        print(f"winner {winner} beats cumprev -> K=6 cadence trains")
        for p in K6_POSITIONS:
            jobs.append(train_job(f"w3g_{arm}_cut{p:02d}_s42", feats, regime,
                                  42, cutoff=builds_255[p], subdir="w2c",
                                  skip_sanity=True))
    else:
        print(f"winner {winner} does NOT beat cumprev ({ref}) — skipping "
              "K=6 cadence trains (never-retrain rows still added)")

    print(f"Stage 3 (W3g W2c additions): {len(jobs)} jobs")
    failed = common.run_pool(jobs, num_gpus=args.num_gpus, force=args.force)
    if failed:
        print(f"WARNING: stage 3 failures: {[j.name for j in failed]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
