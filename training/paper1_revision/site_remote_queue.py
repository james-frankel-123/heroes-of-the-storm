"""
Pause-safe queues of the p1site baseline trainings for a remote worker
(training/remote_workers; launched as plain hotsjob jobs).

Each queue runs its jobs one after another. A job counts as finished when
its completion marker exists (the meta/result JSON the trainer writes at the
very end, never a checkpoint written during training), so the queue is safe
to rerun after a pause or kill: finished jobs are skipped, and the job in
progress restarts.
  * GD: epoch-level resume (GD_RESUME_PATH, train_generic_draft.py), one
    epoch is about 6 min, so a pause loses at most one epoch.
  * CQL / MCQ / BC-CQL / IQL / discriminator: restart from scratch (about
    1.5-2.5 h each on the main box).

Queues (balanced for concurrent use of one 24 GB GPU):
  gd      generic_draft_0..4, then the discriminator (needs the GD pool)
  gd0..gd4  one GD seed each (run concurrently); disc  the discriminator,
          which waits for all five GD seeds
  cqlA    MCQ tau x5, BC-CQL beta x4
  cqlB    IQL 6 cells, CQL grid (alpha 1) taus at 512x256x128
  cqlC    CQL grid at 1024x512x256 and 256x128x64

Usage (worker, from ~/hots/repo/training): python paper1_revision/site_remote_queue.py <queue>
"""
import os
import sys
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
SITE = os.path.join(HERE, "site")
NS = os.path.join(TRAINING_DIR, "rerun2026", "ns", "p1site")
META = os.path.join(NS, "models", "meta")
SNAP = os.path.join(TRAINING_DIR, "snapshots", "replay_snapshot_2026-05-22_1956753_p1site.json")
ENV = {
    "RERUN_NS": "p1site", "RERUN_SPLIT": "p1val", "REPLAY_SNAPSHOT_PATH": SNAP,
    "WP_STATS_PATH": os.path.join(SITE, "cache", "stats", "deploy.json"),
    "P1R_COMP_PATH": os.path.join(SITE, "cache", "stats", "deploy_compositions.json"),
    "PYTHONPATH": os.path.join(HERE, "oct2026_site"), "RERUN_SLOW_GPU": "-1",
    "REPLAY_SNAPSHOT": "1", "PYTHONUNBUFFERED": "1", "WANDB_MODE": "disabled",
}
TJ = os.path.join(TRAINING_DIR, "rerun2026", "train_jobs.py")


def m(name):
    return os.path.join(META, f"{name}.json")


def queues():
    Q = {"gd": [], "cqlA": [], "cqlB": [], "cqlC": []}
    for i in range(5):
        Q["gd"].append((f"gd_{i}", [TJ, "gd", "--variant", str(i)], m(f"gd_{i}"),
                        {"GD_RESUME_PATH": os.path.join(NS, "models", f"generic_draft_{i}.resume.pt")}))
    disc = os.path.join(TRAINING_DIR, "rerun2026", "train_gourdeau_discriminator.py")
    for st in ("cache", "train", "their-eval"):
        Q["gd"].append((f"disc_{st}", [disc, "--stage", st],
                        os.path.join(NS, "results", "gourdeau_discriminator.json") if st == "their-eval"
                        else None, {}))
    for t in (0.5, 0.4, 0.6, 0.7, 0.8):
        Q["cqlA"].append((f"mcq_t{t}", [TJ, "mcq", "--threshold", str(t)], m(f"mcq_t{t}"), {}))
    for b in (1.0, 0.1, 0.5, 2.0):
        Q["cqlA"].append((f"bccql_b{b}", [TJ, "bccql", "--bc-weight", str(b)], m(f"bccql_b{b}"), {}))
    iql = os.path.join(TRAINING_DIR, "rerun2026", "experiment_iql_draft.py")
    for tau, beta in ((0.9, 3.0), (0.7, 1.0), (0.7, 3.0), (0.8, 1.0), (0.8, 3.0), (0.9, 1.0)):
        Q["cqlB"].append((f"iql_t{tau}_b{beta}", [iql, "--task", "train", "--tau", str(tau), "--beta", str(beta)],
                          m(f"iql_t{tau}_b{beta}"), {}))
    for arch, q in (("512,256,128", "cqlB"), ("1024,512,256", "cqlC"), ("256,128,64", "cqlC")):
        a_s = arch.replace(",", "x")
        for tau in (0.005, 0.001, 0.01, 0.05, 0.1):
            Q[q].append((f"cql_hp_t{tau}_{a_s}", [TJ, "cql_hp", "--alpha", "1.0", "--tau", str(tau), "--arch", arch],
                         m(f"cql_hp_a1.0_t{tau}_{a_s}"), {}))
    # single-job queues, so the GD seeds can train concurrently
    for i in range(5):
        Q[f"gd{i}"] = [Q["gd"][i]]
    Q["disc"] = Q["gd"][5:]
    return Q


def main():
    import time
    q = sys.argv[1]
    env = dict(os.environ)
    env.update(ENV)
    # phase-0 caches are built by a separate job (site_phase0.py); wait for them
    need = os.path.join(NS, "feature_cache", "gd_test" if q.startswith("gd") or q == "disc"
                        else "cql_naive_test", "meta.json")
    while not os.path.exists(need):
        time.sleep(30)
    if q == "disc":     # the discriminator needs the whole GD pool
        while not all(os.path.exists(m(f"gd_{i}")) for i in range(5)):
            time.sleep(60)
    for name, argv, marker, extra in queues()[q]:
        if marker and os.path.exists(marker):
            print(f"skip {name} (done)", flush=True)
            continue
        print(f"=== {name}", flush=True)
        e = dict(env)
        e.update(extra)
        rc = subprocess.call([sys.executable, "-u"] + argv, cwd=TRAINING_DIR, env=e)
        if rc != 0:
            print(f"FAILED {name} rc={rc}", flush=True)
            sys.exit(rc)
        if marker and not os.path.exists(marker):
            print(f"FAILED {name}: no marker {marker}", flush=True)
            sys.exit(1)
    print(f"QUEUE {q} DONE", flush=True)


if __name__ == "__main__":
    main()
