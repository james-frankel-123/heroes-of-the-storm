"""
Run the p1site Q-learning baselines (CQL, BC-CQL, TCQL, MCQ, IQL, the CQL
size grid, enriched CQL) with the transition cache held in memory instead of
streamed through a 4-worker DataLoader. The training functions run unchanged:
the dataset class train_jobs hands them and the DataLoader name in each trainer
module are swapped for an in-memory equivalent (shuffled batches of the same
size; the same (state, action, outcome, mask) tensors).

Naive caches (289-d, every column binary except the step fraction) are
bit-packed once (cache_dir/packed) and each process holds a ~2 GB GPU copy;
wider caches (enriched) sit in host RAM and are gathered per batch.

Usage (from training/, p1site env):
  python3 paper1_revision/fastq.py cql_naive_a1.0 bccql_b1.0 iql_t0.9_b3.0 mcq_real_l0.9 ...
  python3 paper1_revision/fastq.py cql_enr_a2.0        # enriched (deferred.job_cql_train)
Each job is skipped when its completion marker exists.
"""
import os
import re
import sys
import json
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)
from rerun2026 import common

common.setup()

import numpy as np
import torch

STEP_COL = 287
_DATA = {}


def _pack(cache_dir):
    """Bit-pack a naive cache once (288 binary columns -> 36 bytes, step index as
    uint8) so a full copy fits on the GPU per worker (~2 GB instead of 14 GB)."""
    out = os.path.join(cache_dir, "packed")
    if os.path.exists(os.path.join(out, "done.json")):
        return out
    t0 = time.time()
    mm = common.memmap_open(cache_dir)
    n = mm["meta"]["n"]
    os.makedirs(out, exist_ok=True)
    bits = np.lib.format.open_memmap(os.path.join(out, "bits.npy"), "w+", np.uint8, (n, 36))
    step = np.lib.format.open_memmap(os.path.join(out, "step.npy"), "w+", np.uint8, (n,))
    CH = 1_000_000
    for i in range(0, n, CH):
        x = np.asarray(mm["states"][i:i + CH])
        s_ = np.rint(x[:, STEP_COL] * 15)
        b = np.delete(x, STEP_COL, 1)
        assert np.isin(b, (0, 1)).all() and np.isin(s_, range(16)).all()
        bits[i:i + len(x)] = np.packbits(b.astype(bool), axis=1)
        step[i:i + len(x)] = s_.astype(np.uint8)
    acts = np.asarray(mm["actions"][:])
    outs = np.asarray(mm["outcomes"][:])
    assert acts.min() >= 0 and acts.max() < 256 and np.isin(outs, (0.0, 1.0)).all()
    np.save(os.path.join(out, "actions.npy"), acts.astype(np.uint8))
    np.save(os.path.join(out, "outcomes.npy"), outs.astype(np.uint8))
    bits.flush(); step.flush()
    with open(os.path.join(out, "done.json"), "w") as f:
        json.dump({"n": int(n), "from": cache_dir}, f)
    print(f"[fastq] packed {cache_dir} in {time.time() - t0:.0f}s", flush=True)
    return out


_SHIFTS = None


def _unpack(bits, step):
    global _SHIFTS
    if _SHIFTS is None or _SHIFTS.device != bits.device:
        _SHIFTS = torch.arange(7, -1, -1, device=bits.device, dtype=torch.uint8)
    b = ((bits.unsqueeze(-1) >> _SHIFTS) & 1).reshape(len(bits), 288).float()
    return torch.cat([b[:, :STEP_COL], (step.float() / 15.0).unsqueeze(1), b[:, STEP_COL:]], 1)


def _load(cache_dir):
    if cache_dir in _DATA:
        return _DATA[cache_dir]
    t0 = time.time()
    mm = common.memmap_open(cache_dir)
    n, dim = mm["meta"]["n"], mm["meta"]["state_dim"]
    if dim == 289:
        assert os.path.exists(os.path.join(cache_dir, "swapfix.json")), f"{cache_dir} predates the swap fix"
        d = _pack(cache_dir)
        dev = torch.device("cuda")
        L = lambda f: torch.from_numpy(np.load(os.path.join(d, f), mmap_mode="r")[:].copy()).to(dev)
        bits, step = L("bits.npy"), L("step.npy")
        acts, outs = L("actions.npy").long(), L("outcomes.npy").float()
        # spot-check the packed copy against the source cache
        idx = np.random.default_rng(0).choice(n, 4096, replace=False)
        idx.sort()
        ref = torch.from_numpy(np.asarray(mm["states"][idx])).to(dev)
        assert torch.allclose(_unpack(bits[idx], step[idx]), ref, atol=1e-6), "packed cache mismatch"
        print(f"[fastq] loaded {cache_dir} n={n:,} packed on {dev} in {time.time() - t0:.0f}s", flush=True)
        _DATA[cache_dir] = (bits, step, None, acts, outs)
        return _DATA[cache_dir]
    base = torch.empty((n, 289), dtype=torch.uint8)
    extra = torch.empty((n, dim - 289), dtype=torch.float32)
    CH = 1_000_000
    for i in range(0, n, CH):
        x = np.asarray(mm["states"][i:i + CH])
        b = x[:, :289].copy()
        b[:, STEP_COL] = np.rint(b[:, STEP_COL] * 15)
        base[i:i + len(x)] = torch.from_numpy(b.astype(np.uint8))
        extra[i:i + len(x)] = torch.from_numpy(x[:, 289:])
    acts = torch.from_numpy(np.asarray(mm["actions"][:]).astype(np.int64))
    outs = torch.from_numpy(np.asarray(mm["outcomes"][:]).astype(np.float32))
    print(f"[fastq] loaded {cache_dir} n={n:,} dim={dim} on cpu in {time.time() - t0:.0f}s", flush=True)
    _DATA[cache_dir] = (base, None, extra, acts, outs)
    return _DATA[cache_dir]


class InMemoryCQLDataset:
    """Stands in for train_jobs.MemmapCQLDataset."""

    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        self.bits, self.step, self.extra, self.actions, self.outcomes = _load(cache_dir)
        self.device = self.bits.device if self.extra is None else torch.device("cuda")

    def __len__(self):
        return len(self.actions)

    def batch(self, idx):
        if self.extra is None:
            b = _unpack(self.bits[idx], self.step[idx])
            a, o = self.actions[idx], self.outcomes[idx]
        else:
            i = idx.cpu()
            b = self.bits[i].to(self.device).float()
            b[:, STEP_COL] /= 15.0
            b = torch.cat([b, self.extra[i].to(self.device)], 1)
            a, o = self.actions[i].to(self.device), self.outcomes[i].to(self.device)
        occ = b[:, :270].reshape(-1, 3, 90).sum(1)
        mask = 1.0 - occ.clamp(0.0, 1.0)
        return b, a, o, mask


class _Loader:
    def __init__(self, ds, batch_size, shuffle):
        self.ds, self.bs, self.shuffle = ds, batch_size, shuffle

    def __len__(self):
        return (len(self.ds) + self.bs - 1) // self.bs

    def __iter__(self):
        n = len(self.ds)
        dev = self.ds.device
        order = torch.randperm(n, device=dev) if self.shuffle else torch.arange(n, device=dev)
        for i in range(0, n, self.bs):
            yield self.ds.batch(order[i:i + self.bs])


def fast_loader(ds, batch_size=1, shuffle=False, **kw):
    if isinstance(ds, InMemoryCQLDataset):
        return _Loader(ds, batch_size, shuffle)
    from torch.utils.data import DataLoader
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, **kw)


def install():
    from rerun2026 import train_jobs
    train_jobs.MemmapCQLDataset = InMemoryCQLDataset
    import experiment_cql_draft, experiment_mcq_draft, experiment_cql_hyperparams, experiment_cql_enriched
    from rerun2026 import experiment_iql_draft
    from paper1_revision import mcq_real
    for m in (experiment_cql_draft, experiment_mcq_draft, experiment_cql_hyperparams,
              experiment_cql_enriched, experiment_iql_draft, mcq_real):
        m.DataLoader = fast_loader
    return train_jobs


def meta(name):
    return os.path.join(common.META_DIR, f"{name}.json")


def run(job, tj):
    NS = argparse.Namespace
    if m := re.fullmatch(r"cql_naive_a([\d.]+)", job):
        done = meta(job)
        fn = lambda: tj.cmd_cql_naive(NS(alpha=float(m[1]), epochs=50))
    elif m := re.fullmatch(r"mcq_t([\d.]+)", job):
        done = meta(job)
        fn = lambda: tj.cmd_mcq(NS(threshold=float(m[1]), epochs=50))
    elif m := re.fullmatch(r"bccql_b([\d.]+)", job):
        done = meta(job)
        fn = lambda: tj.cmd_bccql(NS(bc_weight=float(m[1]), epochs=50))
    elif m := re.fullmatch(r"cql_hp_t([\d.]+)_(\d+)x(\d+)x(\d+)", job):
        done = meta(f"cql_hp_a1.0_t{m[1]}_{m[2]}x{m[3]}x{m[4]}")
        fn = lambda: tj.cmd_cql_hp(NS(alpha=1.0, tau=float(m[1]), arch=f"{m[2]},{m[3]},{m[4]}", epochs=50))
    elif m := re.fullmatch(r"iql_t([\d.]+)_b([\d.]+)", job):
        done = meta(job)
        from rerun2026 import experiment_iql_draft as iql
        fn = lambda: iql.train_iql(float(m[1]), float(m[2]))
    elif m := re.fullmatch(r"mcq_real_l([\d.]+)", job):
        done = meta(job)
        from paper1_revision import mcq_real
        fn = lambda: mcq_real.train(float(m[1]))
    elif m := re.fullmatch(r"cql_enr_a([\d.]+)", job):
        from paper1_revision import deferred
        done = os.path.join(deferred.OUT, f"cql_enriched_train_a{float(m[1])}.json")
        fn = lambda: deferred.job_cql_train(float(m[1]))
    else:
        raise SystemExit(f"unknown job {job}")
    if os.path.exists(done):
        print(f"[fastq] skip {job} (done)", flush=True)
        return
    t0 = time.time()
    print(f"[fastq] start {job}", flush=True)
    fn()
    print(f"[fastq] done {job} in {(time.time() - t0) / 60:.1f} min", flush=True)


def _worker(wid, q):
    torch.cuda.set_per_process_memory_fraction(0.22)
    tj = install()
    while True:
        j = q.get()
        if j is None:
            return
        try:
            run(j, tj)
        except Exception as e:  # keep the worker alive for the next job
            import traceback
            traceback.print_exc()
            print(f"[fastq] FAILED {j}: {e}", flush=True)


def main():
    args = sys.argv[1:]
    workers = 1
    if args and args[0].startswith("--workers="):
        workers = int(args.pop(0).split("=")[1])
    jobs = args
    if workers == 1:
        tj = install()
        for j in jobs:
            run(j, tj)
        return
    # pack the naive caches once; each worker then holds its own packed GPU copy (~2 GB)
    import torch.multiprocessing as tmp
    ctx = tmp.get_context("spawn")
    for p in (common.CQL_NAIVE_TRAIN, common.CQL_NAIVE_TEST):
        _pack(p)
    q = ctx.Queue()
    for j in jobs:
        q.put(j)
    for _ in range(workers):
        q.put(None)
    ps = [ctx.Process(target=_worker, args=(i, q)) for i in range(workers)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    print("[fastq] all workers done", flush=True)


if __name__ == "__main__":
    main()
