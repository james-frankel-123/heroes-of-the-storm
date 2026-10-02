"""
Phase 0: regenerate every feature/transition cache from the pinned 2026-05-22
snapshot (patch-2.55 filtered) + frozen 2026-05-19 stats.

All splits are REPLAY-LEVEL: we split the replay list first (shared.split_data
with a deterministic seed) and only then expand replays into per-step samples
and team-swap augmented pairs. This is the fix for the sample-level leakage
bug the paper documents (Section VI-C): the historical step caches
(feature_cache/{partial,base}_step_features.npz, built by train_wp_512.py)
were extracted from train+test combined and split per SAMPLE. Here the step
caches are extracted separately from the train and the test replay sets, so
leakage is impossible by construction.

Outputs (rerun2026/feature_cache/):
  full_train_features.npz / full_test_features.npz   full-draft WP features
      (bases 197d, enricheds 160d, labels; swap-augmented AFTER the split,
       inside sweep_enriched_wp._extract_chunk)
  step_{train,test}_{partial,base}.npz               step-conditioned WP features
      (features 283d/197d, steps, labels; 15% replay-level test split)
  cql_naive_{train,test}/                            289d CQL transitions (memmap)
  cql_enriched_{train,test}/                         375d CQL transitions (memmap)
  gd_{train,test}/                                   289d GD next-pick samples (memmap)
  split_meta.json                                    provenance record

Valid masks are NOT stored for CQL/GD caches: they are derivable as
1 - clip(team0 + team1 + bans) from state columns 0:270 (train_jobs.py does
this at batch time; identical to the masks the original scripts stored).

Usage:
    set -a && source .env && set +a          # only needed for --fetch-exclusions
    python rerun2026/phase0_features.py --fetch-exclusions   # once
    python rerun2026/phase0_features.py --dry-run
    python rerun2026/phase0_features.py
    python rerun2026/phase0_features.py --force --only cql
"""
import os
import sys
import json
import time
import argparse
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rerun2026 import common
from rerun2026.common import (
    CACHE_DIR, FULL_TRAIN_NPZ, FULL_TEST_NPZ, STEP_NPZ,
    CQL_NAIVE_TRAIN, CQL_NAIVE_TEST, CQL_ENR_TRAIN, CQL_ENR_TEST,
    GD_TRAIN, GD_TEST, SPLIT_META_PATH,
    FULL_TEST_FRAC, STEP_TEST_FRAC, SEED,
)

NUM_WORKERS = min(mp.cpu_count(), 56)


# ── mp workers (module-level for pickling) ──

def _cql_naive_chunk(chunk):
    import numpy as np
    from experiment_cql_draft import replay_to_transitions
    states, actions, outcomes = [], [], []
    for r in chunk:
        for t in replay_to_transitions(r):
            states.append(t["state"])
            actions.append(t["action"])
            outcomes.append(t["outcome"])
    if not states:
        return (np.zeros((0, 289), dtype=np.float32),
                np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32))
    return (np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(outcomes, dtype=np.float32))


def _cql_enriched_chunk(args):
    import numpy as np
    from sweep_enriched_wp import StatsCache, FEATURE_GROUPS, compute_group_indices
    from experiment_cql_enriched import replay_to_enriched_transitions, get_enriched_cols
    chunk, stats_data = args
    stats = object.__new__(StatsCache)
    for k, v in stats_data.items():
        setattr(stats, k, v)
    gi = compute_group_indices()
    cols = get_enriched_cols(gi)
    all_mask = [True] * len(FEATURE_GROUPS)
    dim = 289 + len(cols)
    states, actions, outcomes = [], [], []
    for r in chunk:
        for t in replay_to_enriched_transitions(r, stats, gi, cols, all_mask):
            states.append(t["state"])
            actions.append(t["action"])
            outcomes.append(t["outcome"])
    if not states:
        return (np.zeros((0, dim), dtype=np.float32),
                np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32))
    return (np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(outcomes, dtype=np.float32))


def _gd_chunk(chunk):
    import numpy as np
    from train_generic_draft import replay_to_training_samples
    xs, ys = [], []
    for r in chunk:
        for x, y, _mask in replay_to_training_samples(r):
            xs.append(x)
            ys.append(y)
    if not xs:
        return (np.zeros((0, 289), dtype=np.float32), np.zeros(0, dtype=np.int64))
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.int64)


def _chunks(data, n_workers=NUM_WORKERS, per_chunk=None):
    size = per_chunk or max(1, len(data) // (n_workers * 4))
    return [data[i:i + size] for i in range(0, len(data), size)]


def _pool_stream(worker, args_list, label):
    """Yield results from an mp pool in order (imap keeps memory bounded)."""
    t0 = time.time()
    with mp.Pool(NUM_WORKERS) as pool:
        for i, res in enumerate(pool.imap(worker, args_list)):
            if (i + 1) % 20 == 0:
                print(f"  {label}: chunk {i+1}/{len(args_list)} "
                      f"({time.time()-t0:.0f}s)", flush=True)
            yield res


# ── Cache builders ──

def build_full(train_data, test_data, stats, force):
    from sweep_enriched_wp import precompute_all_features
    for path, data in ((FULL_TRAIN_NPZ, train_data), (FULL_TEST_NPZ, test_data)):
        if os.path.exists(path) and force:
            os.remove(path)
        if os.path.exists(path):
            print(f"  exists, skipping: {path}")
            continue
        # swap-augmentation happens inside _extract_chunk, i.e. AFTER the split
        precompute_all_features(data, stats, cache_path=path,
                                num_workers=NUM_WORKERS)


def build_steps(stats, force):
    import numpy as np
    from train_wp_512 import _extract_partial_chunk
    stats_data = common.stats_as_dict(stats)
    train_data, test_data = common.load_split(test_frac=STEP_TEST_FRAC, seed=SEED)
    for split, data in (("train", train_data), ("test", test_data)):
        for mode in ("partial", "base"):
            path = STEP_NPZ[(split, mode)]
            if os.path.exists(path) and force:
                os.remove(path)
            if os.path.exists(path):
                print(f"  exists, skipping: {path}")
                continue
            print(f"  extracting step features: {split}/{mode} ({len(data)} replays)")
            feats, steps, labels = [], [], []
            args = [(c, stats_data, mode) for c in _chunks(data)]
            for f, s, l in _pool_stream(_extract_partial_chunk, args,
                                        f"step_{split}_{mode}"):
                if len(f):
                    feats.append(f)
                    steps.append(s)
                    labels.append(l)
            feats = np.concatenate(feats)
            steps = np.concatenate(steps)
            labels = np.concatenate(labels)
            np.savez(path, features=feats, steps=steps, labels=labels)
            print(f"  {path}: {feats.shape}")


def build_cql(train_data, test_data, stats, force):
    from sweep_enriched_wp import compute_group_indices
    from experiment_cql_enriched import get_enriched_cols
    stats_data = common.stats_as_dict(stats)
    enr_dim = 289 + len(get_enriched_cols(compute_group_indices()))
    fields = [("actions", "int64"), ("outcomes", "float32")]
    plans = [
        (CQL_NAIVE_TRAIN, _cql_naive_chunk, train_data, 289, False),
        (CQL_NAIVE_TEST, _cql_naive_chunk, test_data, 289, False),
        (CQL_ENR_TRAIN, _cql_enriched_chunk, train_data, enr_dim, True),
        (CQL_ENR_TEST, _cql_enriched_chunk, test_data, enr_dim, True),
    ]
    for out_dir, worker, data, dim, needs_stats in plans:
        if os.path.isdir(out_dir) and force:
            import shutil
            shutil.rmtree(out_dir)
        if os.path.exists(os.path.join(out_dir, "meta.json")):
            print(f"  exists, skipping: {out_dir}")
            continue
        print(f"  building {out_dir} ({len(data)} replays, {dim}d states)")
        chunks = _chunks(data)
        args = [(c, stats_data) for c in chunks] if needs_stats else chunks
        common.memmap_write(out_dir, _pool_stream(worker, args, os.path.basename(out_dir)),
                            dim, fields)


def build_gd(train_data, test_data, force):
    fields = [("actions", "int64")]
    for out_dir, data in ((GD_TRAIN, train_data), (GD_TEST, test_data)):
        if os.path.isdir(out_dir) and force:
            import shutil
            shutil.rmtree(out_dir)
        if os.path.exists(os.path.join(out_dir, "meta.json")):
            print(f"  exists, skipping: {out_dir}")
            continue
        print(f"  building {out_dir} ({len(data)} replays)")
        common.memmap_write(out_dir, _pool_stream(_gd_chunk, _chunks(data),
                                                  os.path.basename(out_dir)),
                            289, fields)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only", default=None,
                        help="substring filter: full|steps|cql|gd")
    parser.add_argument("--fetch-exclusions", action="store_true",
                        help="query DB for pre-2.55 replay_ids, write exclusion file, exit")
    parser.add_argument("--limit", type=int, default=None,
                        help="replay cap for smoke tests (overrides RERUN_REPLAY_LIMIT)")
    args = parser.parse_args()

    if args.fetch_exclusions:
        os.makedirs(common.RERUN_DIR, exist_ok=True)
        common.fetch_exclusions()
        return

    common.setup()
    if args.limit:
        common.REPLAY_LIMIT = args.limit

    tasks = ["full", "steps", "cql", "gd"]
    if args.only:
        tasks = [t for t in tasks if args.only in t]

    if args.dry_run:
        print(f"Phase 0 dry run. Snapshot: {common.SNAPSHOT_PATH}")
        print(f"Frozen stats: {common.FROZEN_STATS_PATH}")
        print(f"Exclusion list present: {os.path.exists(common.EXCLUDE_IDS_PATH)}")
        print(f"Tasks: {tasks}  (force={args.force}, workers={NUM_WORKERS})")
        outputs = []
        if "full" in tasks:
            outputs += [FULL_TRAIN_NPZ, FULL_TEST_NPZ]
        if "steps" in tasks:
            outputs += list(STEP_NPZ.values())
        if "cql" in tasks:
            outputs += [CQL_NAIVE_TRAIN, CQL_NAIVE_TEST, CQL_ENR_TRAIN, CQL_ENR_TEST]
        if "gd" in tasks:
            outputs += [GD_TRAIN, GD_TEST]
        for o in outputs:
            state = "EXISTS" if os.path.exists(o) or os.path.exists(
                os.path.join(o, "meta.json")) else "missing"
            print(f"  [{state}] {o}")
        return

    t0 = time.time()
    stats = common.stats_cache()

    # Full-draft split (2%) — used by full/cql/gd caches, matching split_data()
    # defaults in sweep_enriched_wp / experiment_cql_* / train_generic_draft.
    train_data, test_data = common.load_split(test_frac=FULL_TEST_FRAC, seed=SEED)

    with open(SPLIT_META_PATH, "w") as f:
        json.dump({
            "snapshot": os.path.basename(common.SNAPSHOT_PATH),
            "frozen_stats": os.path.basename(common.FROZEN_STATS_PATH),
            "patch_filter": "2.55 (pre255_exclude_ids.json)",
            "n_replays": len(train_data) + len(test_data),
            "full_split": {"test_frac": FULL_TEST_FRAC, "seed": SEED,
                           "n_train": len(train_data), "n_test": len(test_data)},
            "step_split": {"test_frac": STEP_TEST_FRAC, "seed": SEED},
            "splitting": "replay-level (split before per-step expansion and team-swap augmentation)",
        }, f, indent=2)

    if "full" in tasks:
        print("\n[full] full-draft WP feature caches")
        build_full(train_data, test_data, stats, args.force)
    if "steps" in tasks:
        print("\n[steps] step-conditioned WP feature caches (replay-level 15% split)")
        build_steps(stats, args.force)
    if "cql" in tasks:
        print("\n[cql] CQL transition memmaps")
        build_cql(train_data, test_data, stats, args.force)
    if "gd" in tasks:
        print("\n[gd] Generic Draft sample memmaps")
        build_gd(train_data, test_data, args.force)

    print(f"\nPhase 0 complete in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
