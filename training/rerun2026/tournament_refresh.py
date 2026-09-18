"""
Tournament refresh driver: retrain every strategy in the paper's head-to-head
tournament on a newer snapshot and replay the tournament, inside an isolated
rerun2026 namespace (RERUN_NS), so the expert-rating item pool can be rebuilt
on the current patch without touching the paper's July artifacts.

Stages (each skip-if-done through the underlying script's own checks):
  snapshot   assert the pinned snapshot + stats files exist (make_snapshot.py)
  phase0     feature/transition caches: full, cql, gd (step caches skipped —
             nothing in the tournament needs partial_wp)
  phase1     the phase1 jobs the tournament strategies depend on
  disc       Gourdeau discriminator (cache + train stages)
  ens        20-member WP ensemble (rating_items_ood.py OOD covariate)
  phase4     MCTS J_800sim seed 9 (the tournament's MCTS policy)
  phase3b    9-strategy round robin (72 ordered pairs x 200 drafts)
  constrained constrained-search strategies: rich-eval + pairs vs the 9

Usage (from training/):
    set -a && source ../.env && set +a
    python rerun2026/tournament_refresh.py --ns sept2026 --cutoff 2026-09-01 \
        --gpus 1,2 [--stage phase1] [--dry-run]
"""
import os
import sys
import glob
import time
import argparse
import subprocess

TRAINING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RERUN = os.path.join(TRAINING_DIR, "rerun2026")

PHASE1_JOBS = [
    "wp_naive", "wp_herostrength", "wp_enriched_256", "wp_aug_v2_512",
    "gd_0", "gd_1", "gd_2", "gd_3", "gd_4",
    "cql_naive_a1.0", "cql_enr_a2.0", "mcq_t0.5", "gourdeau_wp",
]
MCTS_RUN = "J_800sim_s9"
STAGES = ["snapshot", "phase0", "phase1", "disc", "ens", "phase4", "phase3b", "constrained"]


def sh(argv, env, log_path, dry_run):
    print(f"$ {' '.join(argv)}  (log: {os.path.relpath(log_path, TRAINING_DIR)})", flush=True)
    if dry_run:
        return
    t0 = time.time()
    with open(log_path, "a") as lf:
        lf.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(argv)}\n")
        lf.flush()
        rc = subprocess.call([sys.executable, "-u"] + argv, cwd=TRAINING_DIR, env=env,
                             stdout=lf, stderr=subprocess.STDOUT)
    print(f"  -> rc={rc} in {(time.time() - t0) / 3600:.2f} h", flush=True)
    if rc != 0:
        sys.exit(f"stage failed (rc={rc}); see {log_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", required=True)
    ap.add_argument("--cutoff", required=True)
    ap.add_argument("--gpus", default="1,2", help="physical GPU ids for the pools")
    ap.add_argument("--stage", default=None, help="run one stage only")
    ap.add_argument("--from-stage", default=None, help="run this stage and all later ones")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    snaps = sorted(glob.glob(os.path.join(TRAINING_DIR, "snapshots",
                                          f"replay_snapshot_{args.cutoff}_*.json")))
    stats = os.path.join(TRAINING_DIR, "snapshots", f"stats_decayed90_{args.cutoff}.json")
    if not snaps or not os.path.exists(stats):
        sys.exit(f"snapshot/stats for cutoff {args.cutoff} missing — run make_snapshot.py first")
    ns_dir = os.path.join(RERUN, "ns", args.ns)
    os.makedirs(ns_dir, exist_ok=True)
    log_path = os.path.join(ns_dir, "refresh.log")

    env = os.environ.copy()
    env.update({
        "RERUN_NS": args.ns,
        "REPLAY_SNAPSHOT_PATH": snaps[-1],
        "WP_STATS_PATH": stats,
        "RERUN_GPU_IDS": args.gpus,
        "RERUN_SLOW_GPU": "-1",
        "NUM_GPUS": str(len(args.gpus.split(","))),
    })
    gpu0 = args.gpus.split(",")[0]
    dry = ["--dry-run"] if args.dry_run else []

    stages = STAGES
    if args.stage:
        stages = [args.stage]
    elif args.from_stage:
        stages = STAGES[STAGES.index(args.from_stage):]
    print(f"namespace {args.ns}: snapshot={os.path.basename(snaps[-1])} stats={os.path.basename(stats)} "
          f"gpus={args.gpus} stages={stages}", flush=True)

    for stage in stages:
        print(f"\n### {stage} — {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        if stage == "snapshot":
            continue
        if stage == "phase0":
            for task in ("full", "cql", "gd"):
                sh([f"{RERUN}/phase0_features.py", "--only", task] + dry, env, log_path, False)
        elif stage == "phase1":
            # phase1_models.main() insists on the step caches; run its job list
            # through the pool directly, filtered to the tournament's needs.
            runner = (
                "import sys; sys.path.insert(0, %r); "
                "from rerun2026 import common; from rerun2026 import phase1_models as p1; "
                "common.setup(); jobs=[j for j in p1.build_jobs() if j.name in %r]; "
                "print('phase1 subset:', [j.name for j in jobs]); "
                "failed=common.run_pool(jobs, dry_run=%r); sys.exit(1 if failed else 0)"
                % (TRAINING_DIR, PHASE1_JOBS, args.dry_run))
            sh(["-c", runner], env, log_path, False)
        elif stage == "disc":
            denv = dict(env, CUDA_VISIBLE_DEVICES=gpu0)
            for st in ("cache", "train"):
                sh([f"{RERUN}/train_gourdeau_discriminator.py", "--stage", st] + dry,
                   denv, log_path, False)
        elif stage == "ens":
            sh([f"{RERUN}/ensemble_uncertainty.py", "--task", "pool"] + dry, env, log_path, False)
        elif stage == "phase4":
            sh([f"{RERUN}/phase4_mcts.py", "--only", MCTS_RUN] + dry, env, log_path, False)
        elif stage == "phase3b":
            sh([f"{RERUN}/phase3b_roundrobin.py", "--mcts-run", MCTS_RUN] + dry, env, log_path, False)
        elif stage == "constrained":
            sh([f"{RERUN}/constrained_search.py", "--mcts-run", MCTS_RUN] + dry, env, log_path, False)
        else:
            sys.exit(f"unknown stage {stage}")
    print(f"\nall stages done — {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
