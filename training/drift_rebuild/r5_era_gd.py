"""
R5 — era-matched opponent models (audit A6). The drift paper's MCTS agents
were all bootstrapped from, and trained against, the paper-1 full-snapshot
generic-draft model (rerun2026/models/generic_draft_0.pt), which saw the test
period. Here a GenericDraftModel is trained on replays at or before a given
cutoff build only, with the unchanged W5 protocol (w5_gd_drift.py:
replay_to_training_samples, 98/2 replay split seed 42, train_single_model
variant 0). The 2026 cutoff already has one (drift2026/models/gd_cutoff).

Also writes an MCTS value-pretraining exclusion list for the era: the
rerun2026 pre-2.55 exclusions plus every replay after the cutoff build.

Usage:
  python3 drift_rebuild/r5_era_gd.py --cutoff 2.55.4.91418 --stage cache
  CUDA_VISIBLE_DEVICES=1 python3 drift_rebuild/r5_era_gd.py --cutoff 2.55.4.91418 --stage train
Outputs: drift_rebuild/models/gd_era_<cutoff>/generic_draft_0.pt,
         drift_rebuild/models/exclude_after_<cutoff>.json
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rb_common as rb  # noqa: E402

from drift2026 import common  # noqa: E402


def cache_dirs(cutoff):
    base = os.path.join(rb.CACHE_DIR, f"gd_era_{cutoff}")
    return base + "_train", base + "_val"


def stage_cache(cutoff):
    from shared import split_data
    from rerun2026 import common as rerun_common
    from rerun2026.phase0_features import _pool_stream, _chunks
    from drift2026.w5_gd_drift import _gd_rows_chunk
    common.setup()
    rows, builds = common.load_data_with_patches()
    cut = builds.index(cutoff)
    past = [r for r in rows if r["build_idx"] <= cut]
    print(f"era {cutoff}: {len(past):,} replays at or before the cutoff")
    tr, va = split_data(past, test_frac=0.02, seed=42)
    for d, data, name in zip(cache_dirs(cutoff), (tr, va), ("train", "val")):
        if os.path.exists(os.path.join(d, "meta.json")):
            continue
        rerun_common.memmap_write(
            d, _pool_stream(_gd_rows_chunk, _chunks(data), name), 289,
            [("actions", "int64"), ("builds", "int64")])
    # value-pretraining exclusion list for this era
    from rerun2026 import common as rc
    with open(rc.EXCLUDE_IDS_PATH) as f:
        ids = set(json.load(f))
    rids, bidx, _, blist = common.load_sidecar()
    ids.update(int(r) for r in rids[bidx > blist.index(cutoff)])
    out = os.path.join(rb.MODELS_DIR, f"exclude_after_{cutoff}.json")
    with open(out, "w") as f:
        json.dump(sorted(ids), f)
    print(f"wrote {out} ({len(ids):,} ids)")


def stage_train(cutoff):
    import torch
    import train_generic_draft as tgd
    from rerun2026.train_jobs import MemmapGDDataset
    common.setup()
    out_dir = os.path.join(rb.MODELS_DIR, f"gd_era_{cutoff}")
    os.makedirs(out_dir, exist_ok=True)
    tgd.__file__ = os.path.join(out_dir, "x.py")
    trd, vad = cache_dirs(cutoff)
    train_ds, val_ds = MemmapGDDataset(trd), MemmapGDDataset(vad)
    print(f"gd_era_{cutoff}: {len(train_ds):,} train / {len(val_ds):,} val samples")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss = tgd.train_single_model(0, tgd.MODEL_VARIANTS[0], train_ds, val_ds, device)
    print(f"gd_era_{cutoff}: best val loss {loss:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", required=True)
    ap.add_argument("--stage", required=True, choices=["cache", "train"])
    a = ap.parse_args()
    stage_cache(a.cutoff) if a.stage == "cache" else stage_train(a.cutoff)
