"""
Phase-0 caches for the p1site rerun2026 namespace: Generic Draft samples and
naive CQL transitions only (the leak-free enriched CQL transitions come from
paper1_revision/deferred.py cql_build; full-draft WP features come from
paper1_revision/core.py). Uses rerun2026.phase0_features' own chunk workers
with a 6-process pool. Run with the p1site environment (site_rebuild.py):
RERUN_NS=p1site, RERUN_SPLIT=p1val, REPLAY_SNAPSHOT_PATH=<p1site snapshot>.

Usage (from training/): python3 paper1_revision/site_phase0.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)


def main():
    assert os.environ.get("RERUN_NS") == "p1site" and os.environ.get("RERUN_SPLIT") == "p1val"
    from rerun2026 import common, phase0_features as p0
    common.setup()
    p0.NUM_WORKERS = 6
    train, val = common.load_split()
    fields_gd = [("actions", "int64")]
    fields_cql = [("actions", "int64"), ("outcomes", "float32")]
    plans = [(common.GD_TRAIN, p0._gd_chunk, train, fields_gd),
             (common.GD_TEST, p0._gd_chunk, val, fields_gd),
             (common.CQL_NAIVE_TRAIN, p0._cql_naive_chunk, train, fields_cql),
             (common.CQL_NAIVE_TEST, p0._cql_naive_chunk, val, fields_cql)]
    for out_dir, worker, data, fields in plans:
        if os.path.exists(os.path.join(out_dir, "meta.json")):
            print("exists", out_dir, flush=True)
            continue
        print(f"building {out_dir} ({len(data)} replays)", flush=True)
        common.memmap_write(out_dir, p0._pool_stream(worker, p0._chunks(data, n_workers=6),
                                                     os.path.basename(out_dir)), 289, fields)


if __name__ == "__main__":
    main()
