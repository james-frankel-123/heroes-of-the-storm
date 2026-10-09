"""
Per-engine CUDA streams: correctness and throughput checks.

  record      run a fixed batch of episodes and save the outputs (one kernel build
              per process, so the old and new builds are recorded separately and
              compared with `compare`)
  compare     exact equality of two recordings
  throughput  K engines in one process, each in its own thread, each running
              B batches of 128 episodes; reports total episodes/s

Usage (from training/, worker env):
  python cuda_mcts/stream_test.py record --so <kernel.so> --out a.npz
  python cuda_mcts/stream_test.py compare a.npz b.npz
  python cuda_mcts/stream_test.py throughput --so <kernel.so> --engines 2 --batches 6
"""
import argparse
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import x2_common as X

SIMS = 400
C_PUCT = 2.0
CHANCE = 1


def record(a):
    k = X.load_kernel(a.so)
    eng = X.make_engine(k, a.policy)
    res = eng.run_episodes(X.configs(a.n, seed=7), SIMS, C_PUCT, 123, 1.0, 0.3, 0.0, CHANCE)
    np.savez(a.out, **X.pack(res))
    print(f"recorded {len(res)} episodes -> {a.out}")


def compare(a):
    x, y = np.load(a.a), np.load(a.b)
    bad = [k for k in x.files if not np.array_equal(x[k], y[k])]
    print("IDENTICAL" if not bad else f"DIFFERENT: {bad}")
    sys.exit(1 if bad else 0)


def throughput(a):
    k = X.load_kernel(a.so)
    engines = [X.make_engine(k, a.policy, gd_idx=i) for i in range(a.engines)]
    cfgs = [X.configs(128, seed=100 + i) for i in range(a.engines)]
    for e, c in zip(engines, cfgs):  # warm-up (arena allocation, first launch)
        e.run_episodes(c[:8], SIMS, C_PUCT, 1, 1.0, 0.3, 0.0, CHANCE)
    done = [0] * a.engines

    def work(i):
        for b in range(a.batches):
            engines[i].run_episodes(cfgs[i], SIMS, C_PUCT, 1000 * i + b, 1.0, 0.3, 0.0, CHANCE)
            done[i] += 128

    t0 = time.time()
    th = [threading.Thread(target=work, args=(i,)) for i in range(a.engines)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    dt = time.time() - t0
    print(f"engines={a.engines} episodes={sum(done)} seconds={dt:.1f} "
          f"total={sum(done) / dt:.2f} ep/s per_engine={sum(done) / dt / a.engines:.2f} ep/s")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--so", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--n", type=int, default=64)
    r.add_argument("--policy", default="new:F_oof_s0")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    t = sub.add_parser("throughput")
    t.add_argument("--so", required=True)
    t.add_argument("--engines", type=int, default=1)
    t.add_argument("--batches", type=int, default=6)
    t.add_argument("--policy", default="new:F_oof_s0")
    a = ap.parse_args()
    {"record": record, "compare": compare, "throughput": throughput}[a.cmd](a)


if __name__ == "__main__":
    main()
