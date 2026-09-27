"""
PHASE D2 orchestrator: regime comparison (b) + patch-local enrichment (c) on
the rerun2026 GPU pool (logs -> drift2026/logs/).

Grid (3 seeds each = 30 jobs):
  D2(b) — features=cutoff (cumulative-through-cutoff stats everywhere):
    d2b_allhist        all-history
    d2b_win{3,6,12}    recency windows of 3/6/12 builds ending at the cutoff
    d2b_decay{90,365}  time-decay weighting, half-life 90/365 days
    d2b_embed          learned per-build embedding (test clamped to cutoff)
  D2(c) — regime=all, feature-stats sourcing varies:
    d2c_local          per-patch ("patch-local") stats
    d2c_cumprev        cumulative-through-previous-build stats (strictly causal)
    d2c_frozen         frozen end-of-time stats (paper-1 control)
    (the cumulative-to-cutoff arm of (c) IS d2b_allhist — shared, not re-run)

Usage:
    python drift2026/phase_d2.py --dry-run
    python drift2026/phase_d2.py
    python drift2026/phase_d2.py --only d2b_win --force
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drift2026 import common

RUNNER = os.path.join(common.DRIFT_DIR, "train_drift_wp.py")

WINDOWS = [3, 6, 12]          # builds
HALF_LIVES = [90, 365]        # days


def build_jobs():
    jobs = []

    def add(name, extra):
        out = os.path.join(common.RESULTS_DIR, "d2", f"{name}.json")
        argv = [RUNNER, "--name", name] + extra
        # WP trainings are "light" in rerun2026's classification, so the
        # thermally-throttled GPU (SLOW_GPU) participates too.
        jobs.append(common.Job(name, argv, [out], weight="light"))

    for seed in common.SEEDS:
        s = ["--seed", str(seed)]
        add(f"d2b_allhist_s{seed}",
            ["--features", "cutoff", "--regime", "all"] + s)
        for w in WINDOWS:
            add(f"d2b_win{w}_s{seed}",
                ["--features", "cutoff", "--regime", "window",
                 "--window", str(w)] + s)
        for hl in HALF_LIVES:
            add(f"d2b_decay{hl}_s{seed}",
                ["--features", "cutoff", "--regime", "decay",
                 "--half-life", str(hl)] + s)
        add(f"d2b_embed_s{seed}",
            ["--features", "cutoff", "--regime", "embed"] + s)
        add(f"d2c_local_s{seed}",
            ["--features", "local", "--regime", "all"] + s)
        add(f"d2c_cumprev_s{seed}",
            ["--features", "cumulative_prev", "--regime", "all"] + s)
        add(f"d2c_frozen_s{seed}",
            ["--features", "frozen", "--regime", "all"] + s)
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default=None)
    ap.add_argument("--num-gpus", type=int, default=None)
    args = ap.parse_args()

    common.setup()
    jobs = build_jobs()
    print(f"Phase D2: {len(jobs)} jobs defined")

    if not args.dry_run:
        missing = [p for p in ["cutoff", "local", "cumulative_prev", "frozen"]
                   if not os.path.exists(os.path.join(
                       common.CACHE_DIR, f"features_{p}.npz"))]
        if missing:
            print(f"feature caches missing ({missing}) — run "
                  "build_drift_features.py first")
            sys.exit(1)

    failed = common.run_pool(jobs, num_gpus=args.num_gpus, dry_run=args.dry_run,
                             force=args.force, only=args.only)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
