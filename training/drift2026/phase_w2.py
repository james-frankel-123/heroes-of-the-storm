"""
PHASE W2 orchestrator (wave 2, all GPU jobs on the rerun2026 pool):

  W2a  within-patch-causal local enrichment (fixes d2c_local self-leakage):
       d2c_causal_{merge,k100,k1000} x 3 seeds, trained on the
       build_causal_local_features.py passes (rolling same-build daily stats
       blended with cumulative-to-cutoff priors, count-weighted shrinkage).
  W2b  policy-level evaluation of the wave-1 regime cells (eval_policy_w2b.py):
       {allhist, win6, decay90, embed, cumprev} x 3 seeds x 500 greedy drafts
       vs GD opponents + a GD reference arm. Metrics scored against
       future-period ground-truth stats.
  W2c  auto-retrain timeline: one WP model per cutoff position 8..43 (seed 42,
       cumulative_prev features, regime=all, sanity skipped) -> full
       acc[cutoff][future_build] matrix consumed by summarize_w2.py policy
       simulation. Position 37 (the wave-1 cutoff) reuses d2c_cumprev_s42.
       Plus one 'frozen stats at C0' arm (features_cutoff_<C0>.npz).

Jobs whose feature cache is missing are deferred with a warning (re-run this
script after build_causal_local_features.py / build_drift_features.py
--cutoff-at complete); jobs whose outputs exist are skipped, so the script is
idempotent.

Usage:
    python drift2026/phase_w2.py --dry-run
    python drift2026/phase_w2.py [--only w2b] [--force]
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drift2026 import common

TRAIN = os.path.join(common.DRIFT_DIR, "train_drift_wp.py")
EVAL_W2B = os.path.join(common.DRIFT_DIR, "eval_policy_w2b.py")

CAUSAL_PASSES = ["causal_merge", "causal_k100", "causal_k1000"]
W2B_CELLS = ["d2b_allhist", "d2b_win6", "d2b_decay90", "d2b_embed",
             "d2c_cumprev"]
W2B_DRAFTS = 500

# W2c timeline: C0 = first deployment cutoff (position within the ordered
# 2.55 builds). Position 8 = 2.55.3.89754 (~638K games / 1.28M rows of
# training data, ends 2023-07-24) -> 36 simulated builds over ~2.8 years.
C0_POS = 8
C0_BUILD = "2.55.3.89754"


def feature_ok(pass_name):
    return os.path.exists(os.path.join(common.CACHE_DIR,
                                       f"features_{pass_name}.npz"))


def build_jobs():
    builds_255 = [b for b in common.load_patch_index()["builds"]
                  if b.startswith("2.55")]
    assert builds_255[C0_POS] == C0_BUILD
    cutoff_pos = builds_255.index(common.TRAIN_CUTOFF_BUILD)
    assert cutoff_pos == 37, cutoff_pos

    jobs, deferred = [], []

    # W2b first in the queue (longest individual jobs)
    for cell in W2B_CELLS:
        for seed in common.SEEDS:
            out = os.path.join(common.RESULTS_DIR, "w2b", f"{cell}_s{seed}.json")
            jobs.append(common.Job(
                f"w2b_{cell}_s{seed}",
                [EVAL_W2B, "--cell", cell, "--seed", str(seed),
                 "--drafts", str(W2B_DRAFTS)],
                [out], weight="light"))
    jobs.append(common.Job(
        "w2b_gd_s42",
        [EVAL_W2B, "--cell", "gd", "--seed", "42", "--drafts", "1500"],
        [os.path.join(common.RESULTS_DIR, "w2b", "gd_s42.json")],
        weight="light"))

    # W2c timeline (cumulative_prev features exist since wave 1)
    for p in range(C0_POS, cutoff_pos + 7):   # 8..43 inclusive (44 = last)
        if p == cutoff_pos:
            continue   # reuse d2c_cumprev_s42 (identical cell)
        if p >= len(builds_255) - 1:
            break
        name = f"w2c_cut{p:02d}_s42"
        out = os.path.join(common.RESULTS_DIR, "w2c", f"{name}.json")
        jobs.append(common.Job(
            name,
            [TRAIN, "--name", name, "--features", "cumulative_prev",
             "--regime", "all", "--seed", "42",
             "--cutoff-build", builds_255[p],
             "--skip-sanity", "--results-subdir", "w2c"],
            [out], weight="light"))

    # W2c policy (i): never retrain + stats frozen at C0
    frozen_pass = f"cutoff_{C0_BUILD}"
    name = f"w2c_frozenstats_cut{C0_POS:02d}_s42"
    job = common.Job(
        name,
        [TRAIN, "--name", name, "--features", frozen_pass,
         "--regime", "all", "--seed", "42", "--cutoff-build", C0_BUILD,
         "--skip-sanity", "--results-subdir", "w2c"],
        [os.path.join(common.RESULTS_DIR, "w2c", f"{name}.json")],
        weight="light")
    (jobs if feature_ok(frozen_pass) else deferred).append(job)

    # W2a training cells
    for pass_name in CAUSAL_PASSES:
        for seed in common.SEEDS:
            name = f"d2c_{pass_name}_s{seed}"
            job = common.Job(
                name,
                [TRAIN, "--name", name, "--features", pass_name,
                 "--regime", "all", "--seed", str(seed)],
                [os.path.join(common.RESULTS_DIR, "d2", f"{name}.json")],
                weight="light")
            (jobs if feature_ok(pass_name) else deferred).append(job)

    return jobs, deferred


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default=None)
    ap.add_argument("--num-gpus", type=int, default=None)
    args = ap.parse_args()

    common.setup()
    jobs, deferred = build_jobs()
    print(f"Phase W2: {len(jobs)} jobs runnable, {len(deferred)} deferred "
          f"(feature cache not built yet)")
    for j in deferred:
        print(f"  DEFER {j.name} (missing features npz)")

    failed = common.run_pool(jobs, num_gpus=args.num_gpus, dry_run=args.dry_run,
                             force=args.force, only=args.only)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
