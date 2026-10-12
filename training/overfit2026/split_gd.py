"""
Generic Draft (behavior cloning of human picks) trained on half A only, for
the within-snapshot split experiment under P1_TIERS=site. The paper's GD pool
(rerun2026/ns/p1site) trained on every non-test replay, so half B reached the
split self-play through the GD bootstrap/opponent and the search's GD
rollouts, and reached the BC search prior through distillation. This model
replaces all three in the site rerun (train_mcts_split.py, stage_e.py,
stage_f_selfplay.py, bc_prior.py).

Rows: the p1site snapshot (full file; the GD samples need draft_order),
streamed object by object so the 2.9 GB JSON is never parsed whole, filtered
exactly like rerun2026.common's p1val split (pre-2.55 excluded, the paper's
test replays excluded, unknown tier dropped) and then to half A
(split.side_of(replay_id) == "A"). Early-stopping set = half-A replays of the
p1val validation hash (paper1_revision.core.is_val); train = the rest of half
A. Same sample builder (rerun2026.phase0_features._gd_chunk) and the same
trainer (train_generic_draft.train_single_model, variant 0 hyperparameters)
as the p1site pool; the browser ONNX optimize/quantize step is skipped (it
reloads the whole snapshot). Row order differs from p1site (snapshot order,
not the split permutation); the DataLoader shuffles, so it does not matter.

Outputs (site namespace only)
  site/cache/gd_splitA/{gd_train,gd_test}/   memmaps (about 14 GB on disk)
  site/models/gd_splitA/generic_draft_<v>.pt (+ .onnx, resume state)
  site/models/gd_splitA/gd_<v>.json          completion marker

Usage (P1_TIERS=site, from training/):
  python3 overfit2026/split_gd.py cache
  python3 overfit2026/split_gd.py train [--variant 0]
"""
import os
import sys
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)

from overfit2026 import data

VARIANTS = (0,)        # one GD (the p1site pool's generic_draft_0 hyperparameters)
SNAP = os.path.join(TRAINING_DIR, "snapshots", "replay_snapshot_2026-05-22_1956753_p1site.json")


def _require_site():
    if data.TIER_SCHEME != "site":
        raise SystemExit("split_gd is the P1_TIERS=site half-A GD; set P1_TIERS=site")


def cache_dir(which):
    _require_site()
    return data.art(HERE, "cache", "gd_splitA", which)


def model_dir():
    _require_site()
    return data.art(HERE, "models", "gd_splitA")


def model_path(v=0):
    return os.path.join(model_dir(), f"generic_draft_{v}.pt")


def marker(v=0):
    return os.path.join(model_dir(), f"gd_{v}.json")


def model_paths():
    """The half-A GD pool used as opponent / rollout policy / bootstrap."""
    return [model_path(v) for v in VARIANTS]


def iter_rows(path, block=1 << 24):
    """Yield the objects of a top-level JSON array one at a time."""
    dec = json.JSONDecoder()
    with open(path) as f:
        buf = f.read(block)
        pos = buf.index("[") + 1
        while True:
            while True:
                while pos < len(buf) and buf[pos] in " \t\r\n,":
                    pos += 1
                if pos < len(buf):
                    break
                more = f.read(block)
                if not more:
                    return
                buf, pos = more, 0
            if buf[pos] == "]":
                return
            try:
                obj, end = dec.raw_decode(buf, pos)
            except json.JSONDecodeError:
                more = f.read(block)
                if not more:
                    raise
                buf, pos = buf[pos:] + more, 0
                continue
            yield obj
            pos = end


def cmd_cache(a):
    import numpy as np
    import multiprocessing as mp
    from overfit2026.split import side_of
    from paper1_revision import core
    from rerun2026.phase0_features import _gd_chunk
    pre255 = set(json.load(open(os.path.join(TRAINING_DIR, "rerun2026", "pre255_exclude_ids.json"))))
    if not os.path.exists(core.TEST_IDS):     # never rebuilt here (p1val env would change it)
        raise SystemExit(f"missing {core.TEST_IDS} (sync paper1_revision/site/cache/paper_test_ids.json)")
    test = core.test_ids()
    outs = {k: cache_dir(k) for k in ("gd_train", "gd_test")}
    if all(os.path.exists(os.path.join(d, "meta.json")) for d in outs.values()):
        print("exists", outs, flush=True)
        return
    files = {k: (open(os.path.join(d, "states.dat"), "wb"), open(os.path.join(d, "actions.dat"), "wb"))
             for k, d in outs.items()}
    n_rows = {k: 0 for k in outs}
    n_rep = {k: 0 for k in outs}
    seen = 0
    keep = ("replay_id", "game_map", "skill_tier", "draft_order")
    pending = {k: [] for k in outs}
    nproc = int(os.environ.get("SPLIT_GD_NPROC", "6"))
    ctx = mp.get_context("fork")

    def flush(k, pool):
        rows = pending[k]
        if not rows:
            return
        parts = [rows[i:i + 5000] for i in range(0, len(rows), 5000)]
        for xs, ys in pool.map(_gd_chunk, parts):
            if len(xs):
                np.ascontiguousarray(xs, dtype=np.float32).tofile(files[k][0])
                np.ascontiguousarray(ys, dtype=np.int64).tofile(files[k][1])
                n_rows[k] += len(xs)
        pending[k] = []

    with ctx.Pool(nproc) as pool:
        for r in iter_rows(SNAP):
            seen += 1
            rid = int(r["replay_id"])
            if (rid in pre255 or rid in test or r.get("skill_tier") not in ("low", "mid", "high")
                    or side_of(rid) != "A"):
                continue
            k = "gd_test" if core.is_val(rid) else "gd_train"
            pending[k].append({f: r.get(f) for f in keep})
            n_rep[k] += 1
            if len(pending[k]) >= 100000:
                flush(k, pool)
                print(f"  {seen:,} rows read; train {n_rep['gd_train']:,} / val {n_rep['gd_test']:,} "
                      f"replays", flush=True)
        for k in outs:
            flush(k, pool)
    for k, (fs, fa) in files.items():
        fs.close()
        fa.close()
        meta = {"n": n_rows[k], "state_dim": 289, "state_dtype": "float32",
                "fields": [["actions", "int64"]], "replays": n_rep[k],
                "filter": "p1site snapshot, p1val minus paper test, known tier, half A"}
        json.dump(meta, open(os.path.join(outs[k], "meta.json"), "w"))
        print(f"  {outs[k]}: {n_rep[k]:,} replays, {n_rows[k]:,} samples", flush=True)


def _memmap_gd_dataset(path):
    """rerun2026.train_jobs.MemmapGDDataset (same rows, same masks), defined
    here because importing train_jobs runs rerun2026's namespace setup."""
    import numpy as np
    import torch
    from torch.utils.data import Dataset
    from rerun2026 import common

    class _DS(Dataset):
        def __init__(self, d):
            self._d = d
            self._open()

        def _open(self):
            mm = common.memmap_open(self._d)
            self.X, self.y = mm["states"], mm["actions"]

        def __getstate__(self):
            return {"_d": self._d}

        def __setstate__(self, st):
            self._d = st["_d"]
            self._open()

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            x = np.array(self.X[i], dtype=np.float32)
            mask = (1.0 - np.clip(x[:270].reshape(3, 90).sum(axis=0), 0.0, 1.0)).astype(np.float32)
            return torch.from_numpy(x), torch.tensor(int(self.y[i]), dtype=torch.long), torch.from_numpy(mask)
    return _DS(path)


def cmd_train(a):
    import torch
    import train_generic_draft as tgd
    v = a.variant
    if os.path.exists(marker(v)):
        print("exists", marker(v))
        return
    tgd.__file__ = os.path.join(model_dir(), "train_generic_draft.py")
    # skip the browser export's optimize/quantize/verify (variant 0 only): it
    # reloads the whole snapshot and its outputs are never read here
    tgd.optimize_onnx = lambda *x, **k: None
    tgd.quantize_onnx = lambda *x, **k: None
    tgd.verify_quantized_model = lambda *x, **k: None
    tgd.load_replay_data = lambda *x, **k: []
    os.environ.setdefault("GD_RESUME_PATH", os.path.join(model_dir(), f"generic_draft_{v}.resume.pt"))
    train_ds = _memmap_gd_dataset(cache_dir("gd_train"))
    test_ds = _memmap_gd_dataset(cache_dir("gd_test"))
    print(f"half-A gd variant {v}: {len(train_ds):,} train / {len(test_ds):,} val samples", flush=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss = tgd.train_single_model(v, tgd.MODEL_VARIANTS[v], train_ds, test_ds, dev)
    json.dump({"kind": "gd_splitA", "variant": tgd.MODEL_VARIANTS[v], "best_val_loss": loss,
               "out": model_path(v), "train_samples": len(train_ds), "val_samples": len(test_ds)},
              open(marker(v), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cache", "train"])
    ap.add_argument("--variant", type=int, default=0)
    a = ap.parse_args()
    _require_site()
    {"cache": cmd_cache, "train": cmd_train}[a.cmd](a)
