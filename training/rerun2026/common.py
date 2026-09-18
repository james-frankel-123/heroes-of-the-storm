"""
Shared utilities for the rerun2026 orchestration suite.

Everything in this package standardizes on:
  - Pinned replay snapshot: training/snapshots/replay_snapshot_2026-05-22_1956753.json
    (loaded via shared.load_replay_data, which defaults to REPLAY_SNAPSHOT=1),
    filtered to patch 2.55 via rerun2026/pre255_exclude_ids.json
    (dataset of record: 1,949,102 replays).
  - Frozen pairwise/hero stats: training/frozen_stats_2026-05-19.json
    (StatsCache in sweep_enriched_wp.py picks this file up automatically via
    its FROZEN_STATS_PATH constant whenever the file exists — no env var).
  - Replay-level train/test splitting (shared.split_data operates on replay
    rows, so splitting BEFORE any per-step/per-swap expansion is replay-level
    by construction; team-swap augmentation always happens after the split).

Env knobs:
  NUM_GPUS            number of GPUs for pools (default 4; GPU NUM_GPUS-1 —
                      i.e. GPU 3 — is deprioritized: heavy jobs avoid it)
  RERUN_REPLAY_LIMIT  cap replay count (smoke tests only)
  RERUN_ALLOW_UNFILTERED=1  proceed without the 2.55 exclusion list
"""
import os
import sys
import json
import time
import subprocess

TRAINING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RERUN_DIR = os.path.join(TRAINING_DIR, "rerun2026")
# RERUN_NS=<name> runs the whole suite in an isolated namespace
# (rerun2026/ns/<name>/{models,logs,results,feature_cache,mcts_runs}) against
# a different snapshot (REPLAY_SNAPSHOT_PATH) and stats file (WP_STATS_PATH),
# leaving the paper's July artifacts untouched. Unset = the paper rerun.
RERUN_NS = os.environ.get("RERUN_NS", "")
NS_DIR = os.path.join(RERUN_DIR, "ns", RERUN_NS) if RERUN_NS else RERUN_DIR
MODELS_DIR = os.path.join(NS_DIR, "models")
META_DIR = os.path.join(MODELS_DIR, "meta")
LOGS_DIR = os.path.join(NS_DIR, "logs")
RESULTS_DIR = os.path.join(NS_DIR, "results")
CACHE_DIR = os.path.join(NS_DIR, "feature_cache")
MCTS_RUNS_DIR = os.path.join(NS_DIR, "mcts_runs")

SNAPSHOT_PATH = os.environ.get("REPLAY_SNAPSHOT_PATH") or os.path.join(
    TRAINING_DIR, "snapshots", "replay_snapshot_2026-05-22_1956753.json")
FROZEN_STATS_PATH = os.environ.get("WP_STATS_PATH") or os.path.join(
    TRAINING_DIR, "frozen_stats_2026-05-19.json")
EXCLUDE_IDS_PATH = os.path.join(RERUN_DIR, "pre255_exclude_ids.json")

# All pre-2.55 replays have replay_id <= this (verified against DB); the
# exclusion list only needs to cover ids at or below it.
PRE255_MAX_REPLAY_ID = 63653039

# RERUN_GPU_IDS="1,2" restricts pools to those physical GPUs (pool slot i ->
# physical id GPU_IDS[i]); default = 0..NUM_GPUS-1.
if os.environ.get("RERUN_GPU_IDS"):
    GPU_IDS = [int(x) for x in os.environ["RERUN_GPU_IDS"].split(",") if x.strip()]
    NUM_GPUS = len(GPU_IDS)
else:
    NUM_GPUS = int(os.environ.get("NUM_GPUS", "4"))
    GPU_IDS = list(range(NUM_GPUS))
# GPU to deprioritize (thermal throttling). Heavy jobs avoid it. Pool-slot
# index; RERUN_SLOW_GPU=-1 disables (no throttled GPU in the pool).
SLOW_GPU = int(os.environ.get("RERUN_SLOW_GPU", NUM_GPUS - 1))
REPLAY_LIMIT = int(os.environ.get("RERUN_REPLAY_LIMIT", "0")) or None

SEED = 42
FULL_TEST_FRAC = 0.02   # full-draft WP / GD / CQL scripts use split_data() default
STEP_TEST_FRAC = 0.15   # step-conditioned models historically use 15% (train_partial_wp.py)

# Cache file names (phase0 outputs)
FULL_TRAIN_NPZ = os.path.join(CACHE_DIR, "full_train_features.npz")
FULL_TEST_NPZ = os.path.join(CACHE_DIR, "full_test_features.npz")
STEP_NPZ = {  # (split, mode) -> path
    ("train", "partial"): os.path.join(CACHE_DIR, "step_train_partial.npz"),
    ("test", "partial"): os.path.join(CACHE_DIR, "step_test_partial.npz"),
    ("train", "base"): os.path.join(CACHE_DIR, "step_train_base.npz"),
    ("test", "base"): os.path.join(CACHE_DIR, "step_test_base.npz"),
}
CQL_NAIVE_TRAIN = os.path.join(CACHE_DIR, "cql_naive_train")
CQL_NAIVE_TEST = os.path.join(CACHE_DIR, "cql_naive_test")
CQL_ENR_TRAIN = os.path.join(CACHE_DIR, "cql_enriched_train")
CQL_ENR_TEST = os.path.join(CACHE_DIR, "cql_enriched_test")
GD_TRAIN = os.path.join(CACHE_DIR, "gd_train")
GD_TEST = os.path.join(CACHE_DIR, "gd_test")
SPLIT_META_PATH = os.path.join(CACHE_DIR, "split_meta.json")


def setup():
    """Create dirs, pin env, put training/ on sys.path. Call first in every entry point."""
    os.environ.setdefault("REPLAY_SNAPSHOT", "1")
    for d in (MODELS_DIR, META_DIR, LOGS_DIR, RESULTS_DIR, CACHE_DIR, MCTS_RUNS_DIR):
        os.makedirs(d, exist_ok=True)
    if TRAINING_DIR not in sys.path:
        sys.path.insert(0, TRAINING_DIR)
    if not os.path.exists(FROZEN_STATS_PATH):
        raise RuntimeError(f"Frozen stats snapshot missing: {FROZEN_STATS_PATH} — "
                           "StatsCache would silently fall back to the live DB.")
    if not os.path.exists(SNAPSHOT_PATH):
        raise RuntimeError(f"Pinned replay snapshot missing: {SNAPSHOT_PATH}")


def load_exclude_ids():
    """Set of pre-2.55 replay_ids to drop from the pinned snapshot."""
    if os.path.exists(EXCLUDE_IDS_PATH):
        with open(EXCLUDE_IDS_PATH) as f:
            return set(json.load(f))
    if os.environ.get("RERUN_ALLOW_UNFILTERED") == "1":
        print("WARNING: pre255_exclude_ids.json missing — running UNFILTERED "
              "(includes 7,651 pre-2.55 replays).")
        return set()
    raise RuntimeError(
        f"{EXCLUDE_IDS_PATH} not found. Run "
        "'python rerun2026/phase0_features.py --fetch-exclusions' once "
        "(requires DATABASE_URL), or set RERUN_ALLOW_UNFILTERED=1.")


def fetch_exclusions():
    """Query DB for pre-2.55 replay_ids and write pre255_exclude_ids.json."""
    import psycopg2
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL required to fetch the pre-2.55 exclusion list")
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    cur.execute(
        "SELECT replay_id FROM replay_draft_data "
        "WHERE replay_id <= %s AND game_version NOT LIKE '2.55%%'",
        (PRE255_MAX_REPLAY_ID,))
    ids = sorted(r[0] for r in cur.fetchall())
    cur.close()
    conn.close()
    with open(EXCLUDE_IDS_PATH, "w") as f:
        json.dump(ids, f)
    print(f"Wrote {len(ids)} pre-2.55 replay_ids to {EXCLUDE_IDS_PATH}")
    return ids


def load_data():
    """Load the pinned snapshot, filtered to patch 2.55. Single source of
    truth for every phase."""
    from shared import load_replay_data
    rows = load_replay_data(limit=REPLAY_LIMIT)
    exclude = load_exclude_ids()
    if exclude:
        before = len(rows)
        rows = [r for r in rows if r.get("replay_id") not in exclude]
        print(f"Filtered to patch 2.55: {before} -> {len(rows)} replays "
              f"({before - len(rows)} excluded)")
    return rows


def load_split(test_frac=FULL_TEST_FRAC, seed=SEED):
    """Replay-level train/test split of the filtered snapshot."""
    from shared import split_data
    data = load_data()
    train, test = split_data(data, test_frac=test_frac, seed=seed)
    print(f"Split (test_frac={test_frac}, seed={seed}): "
          f"train={len(train)}, test={len(test)} replays")
    return train, test


def stats_cache():
    """StatsCache pinned to frozen_stats_2026-05-19.json (asserted in setup())."""
    from sweep_enriched_wp import StatsCache
    return StatsCache()


def stats_as_dict(stats):
    """Serialize a StatsCache for mp workers (same shape existing scripts use)."""
    return {
        "hero_wr": stats.hero_wr,
        "hero_meta": stats.hero_meta,
        "hero_map_wr": stats.hero_map_wr,
        "pairwise": stats.pairwise,
        "comp_data": stats.comp_data,
    }


def write_meta(name, payload):
    path = os.path.join(META_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"Meta written: {path}")


# ── Memmap caches (phase0 writes, phase1 reads) ──

def memmap_write(out_dir, chunks_iter, state_dim, fields):
    """Stream (states, *extra) numpy chunks into raw .dat files under out_dir.
    fields: list of (name, dtype) for the extra arrays, e.g.
    [("actions", "int64"), ("outcomes", "float32")]."""
    import numpy as np
    os.makedirs(out_dir, exist_ok=True)
    files = {"states": open(os.path.join(out_dir, "states.dat"), "wb")}
    for fname, _ in fields:
        files[fname] = open(os.path.join(out_dir, f"{fname}.dat"), "wb")
    n = 0
    for chunk in chunks_iter:
        states = chunk[0]
        if len(states) == 0:
            continue
        np.ascontiguousarray(states, dtype=np.float32).tofile(files["states"])
        for (fname, dtype), arr in zip(fields, chunk[1:]):
            np.ascontiguousarray(arr, dtype=dtype).tofile(files[fname])
        n += len(states)
    for f in files.values():
        f.close()
    meta = {"n": n, "state_dim": state_dim, "state_dtype": "float32",
            "fields": [[fname, dtype] for fname, dtype in fields]}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f)
    print(f"  {out_dir}: {n:,} rows x {state_dim} dims")
    return n


def memmap_meta(cache_dir):
    with open(os.path.join(cache_dir, "meta.json")) as f:
        return json.load(f)


def memmap_open(cache_dir):
    """Return dict of lazily-opened numpy memmaps for a phase0 cache dir."""
    import numpy as np
    meta = memmap_meta(cache_dir)
    out = {"meta": meta}
    out["states"] = np.memmap(os.path.join(cache_dir, "states.dat"),
                              dtype=np.float32, mode="r",
                              shape=(meta["n"], meta["state_dim"]))
    for fname, dtype in meta["fields"]:
        out[fname] = np.memmap(os.path.join(cache_dir, f"{fname}.dat"),
                               dtype=dtype, mode="r", shape=(meta["n"],))
    return out


# ── GPU pool ──

class Job:
    """A pool job = one subprocess pinned to one GPU.

    weight: "heavy" jobs are never scheduled on the deprioritized GPU
    (SLOW_GPU, thermal throttling) unless nothing else remains in the queue
    and only that GPU is free."""

    def __init__(self, name, argv, outputs=(), weight="heavy", env=None):
        self.name = name
        self.argv = list(argv)
        self.outputs = list(outputs)
        self.weight = weight
        self.env = env or {}

    def done(self):
        return bool(self.outputs) and all(os.path.exists(p) for p in self.outputs)


def run_pool(jobs, num_gpus=None, dry_run=False, force=False, only=None):
    """Simple GPU pool: one subprocess per job, one GPU each, refill on
    completion. Mirrors run_all_experiments.py but parameterizes NUM_GPUS via
    env (default 4) and deprioritizes the throttling GPU for heavy jobs."""
    num_gpus = num_gpus or NUM_GPUS
    if only:
        jobs = [j for j in jobs if only in j.name]
    queue, skipped = [], []
    for j in jobs:
        if not force and j.done():
            skipped.append(j)
        else:
            queue.append(j)

    print(f"Pool: {len(queue)} to run, {len(skipped)} skipped (outputs exist), "
          f"{num_gpus} GPUs (GPU {SLOW_GPU} light-jobs-only)")
    for j in skipped:
        print(f"  SKIP  {j.name}")
    for j in queue:
        print(f"  QUEUE {j.name} [{j.weight}]  ->  {' '.join(j.argv)}")
    if dry_run:
        print("(dry run: nothing executed)")
        return []

    active = {}   # gpu_id -> (job, proc, logfile)
    failed, completed = [], []

    def pick_job(gpu_id):
        if not queue:
            return None
        if gpu_id == SLOW_GPU:
            for i, j in enumerate(queue):
                if j.weight == "light":
                    return queue.pop(i)
            # only heavy jobs left: take one only if no other GPU is free
            if all(g in active for g in range(num_gpus) if g != SLOW_GPU):
                return queue.pop(0)
            return None
        return queue.pop(0)

    def start(gpu_id):
        job = pick_job(gpu_id)
        if job is None:
            return False
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(GPU_IDS[gpu_id])
        env.update(job.env)
        log_path = os.path.join(LOGS_DIR, f"{job.name}.log")
        lf = open(log_path, "w")
        proc = subprocess.Popen([sys.executable, "-u"] + job.argv,
                                cwd=TRAINING_DIR, env=env,
                                stdout=lf, stderr=subprocess.STDOUT)
        active[gpu_id] = (job, proc, lf)
        print(f"  START {job.name} on GPU {GPU_IDS[gpu_id]} (pid {proc.pid}) "
              f"[{len(queue)} queued]", flush=True)
        return True

    for g in range(num_gpus):
        start(g)
    while active:
        time.sleep(5)
        for g in list(active.keys()):
            job, proc, lf = active[g]
            if proc.poll() is None:
                continue
            lf.close()
            ok = proc.returncode == 0
            (completed if ok else failed).append(job)
            print(f"  DONE  {job.name}: {'OK' if ok else f'FAIL rc={proc.returncode}'} "
                  f"(log: logs/{job.name}.log)", flush=True)
            del active[g]
            start(g)
        # a light-only GPU may idle while heavy jobs wait; retry it
        for g in range(num_gpus):
            if g not in active and queue:
                start(g)

    print(f"Pool finished: {len(completed)} ok, {len(failed)} failed")
    for j in failed:
        print(f"  FAILED: {j.name}")
    return failed
