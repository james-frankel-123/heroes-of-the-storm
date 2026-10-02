"""
GD 0-4 (oct2026) held-out metrics on VALID rows only, and a count of invalid
rows in the train and test memmaps.

A row is invalid when its target hero is already picked or banned in the
state (the GD mask sets that logit to -1e9, so one such row adds ~1e9 to the
summed loss: 18 held-out rows explain the reported test loss of ~29954).
Mask = 1 - clip(sum of the three 90-wide pick/ban blocks), exactly as
rerun2026.train_jobs.MemmapGDDataset builds it.

  python3 paper1_revision/oct2026_gd_valid_eval.py   (RERUN_NS=oct2026 env; CPU)
Output: results/expert_v6/gd_valid_eval.json
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)

import numpy as np
import torch
import torch.nn.functional as F

from rerun2026 import common
from train_generic_draft import GenericDraftModel, MODEL_VARIANTS

OUT = os.path.join(HERE, "results", "expert_v6", "gd_valid_eval.json")
CHUNK = 1 << 18


def invalid_rows(cache_dir):
    mm = common.memmap_open(cache_dir)
    X, y = mm["states"], mm["actions"]
    bad = []
    for s in range(0, len(y), CHUNK):
        x = np.asarray(X[s:s + CHUNK, :270]).reshape(-1, 3, 90).sum(1)
        t = np.asarray(y[s:s + CHUNK]).astype(np.int64)
        taken = x[np.arange(len(t)), t] > 0
        bad.extend((np.nonzero(taken)[0] + s).tolist())
    return len(y), bad


def main():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    out = {"note": __doc__.strip().splitlines()[0]}
    for name, d in (("train", common.GD_TRAIN), ("test", common.GD_TEST)):
        n, bad = invalid_rows(d)
        out[f"{name}_rows"] = n
        out[f"{name}_invalid_rows"] = len(bad)
        if name == "test":
            out["test_invalid_row_indices"] = bad
        print(f"{name}: {n:,} rows, {len(bad)} invalid", flush=True)
    mm = common.memmap_open(common.GD_TEST)
    X = torch.from_numpy(np.asarray(mm["states"]))
    y = torch.from_numpy(np.asarray(mm["actions"]).astype(np.int64))
    occ = X[:, :270].reshape(-1, 3, 90).sum(1)
    mask = 1.0 - occ.clamp(0, 1)
    valid = mask[torch.arange(len(y)), y] > 0
    out["models"] = {}
    for v in range(5):
        path = os.path.join(common.MODELS_DIR, f"generic_draft_{v}.pt")
        var = MODEL_VARIANTS[v]
        m = GenericDraftModel(dropout1=var["dropout1"], dropout2=var["dropout2"])
        m.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
        m.eval()
        ce_all, ce_valid, top1, top5, nv = 0.0, 0.0, 0, 0, 0
        with torch.no_grad():
            for s in range(0, len(y), CHUNK):
                lg = m(X[s:s + CHUNK], mask[s:s + CHUNK])
                t, ok = y[s:s + CHUNK], valid[s:s + CHUNK]
                ce = F.cross_entropy(lg, t, reduction="none")
                ce_all += float(ce.sum())
                ce_valid += float(ce[ok].sum())
                top1 += int((lg[ok].argmax(1) == t[ok]).sum())
                top5 += int((lg[ok].topk(5, 1).indices == t[ok, None]).any(1).sum())
                nv += int(ok.sum())
        r = {"file": os.path.relpath(path, TRAINING_DIR), "test_ce_all_rows": ce_all / len(y),
             "test_ce_valid_rows": ce_valid / nv, "test_top1_valid": top1 / nv,
             "test_top5_valid": top5 / nv, "valid_rows": nv}
        out["models"][f"gd_{v}"] = r
        print(f"gd_{v}: CE all {r['test_ce_all_rows']:.2f} | valid CE {r['test_ce_valid_rows']:.4f} "
              f"top1 {r['test_top1_valid']:.4f} top5 {r['test_top5_valid']:.4f}", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    common.setup()
    main()
