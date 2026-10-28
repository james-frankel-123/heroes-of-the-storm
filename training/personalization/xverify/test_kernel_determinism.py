"""
Regression test (GPU): every MCTS kernel must give byte-identical outputs when
the same search runs twice with the same inputs and seed, in the modes that
skip main-loop steps (decide-only prefix replay, forced bans), and must never
produce a draft with a repeated hero.

Guards the skip-path race found by the independent verification (2026-10-19,
results/P3_MCTS_VERIFICATION.md): the P3 main loop read the shared skip flag
and `continue`d without a barrier, so thread 0 could rewrite the flags and the
draft state of the next step before the other warps had read them. Before the
fix about 3-4% of decide-only episodes with a prefix changed between
identical runs, and about 1 in 1000 forced-ban drafts repeated a hero.

Kernels: P3 personal / prior / pvnet / explore from XV_BUILD (default the
repo's personalization/ builds) and the paper-1 kernel from XV_PAPER1_DIR
(default training/cuda_mcts). One process loads one build of each module.

Usage (from training/, a GPU host):
  python personalization/xverify/test_kernel_determinism.py      (or pytest)
  XV_BUILD=~/hots/xv/src_patched XV_SAVE=out.npz python ...       (other builds; save raw outputs)
Exit 1 on any failure.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING = os.environ.get("XV_TRAINING") or os.path.dirname(os.path.dirname(HERE))
os.environ.setdefault("XV_BUILD", os.path.join(TRAINING, "personalization"))
os.environ.setdefault("XV_PAPER1_DIR", os.path.join(TRAINING, "cuda_mcts"))
sys.path.insert(0, HERE)
import numpy as np

import xv_cases as XC
import xv_gpu as G

G.MODS["paper1"] = (os.environ["XV_PAPER1_DIR"], "cuda_mcts_kernel")

N = int(os.environ.get("XV_DET_N", "1024"))
RUNS = 3
SAVE = {}
FAIL = []


def _same(a, b):
    return np.asarray(a).tobytes() == np.asarray(b).tobytes()


def _forced_cfg(n, seed, selfplay):
    rng = np.random.default_rng(seed)
    cfg = np.full((n, 59), -1, np.int32)
    cfg[:, 0], cfg[:, 1], cfg[:, 2] = rng.integers(0, 14, n), rng.integers(0, 3, n), rng.integers(0, 2, n)
    cfg[:, 3:8] = 0
    cfg[:, 3] = selfplay
    cfg[:, 4] = -1
    cfg[:, 5] = 1
    cfg[:, 24:40] = XC.slot_of_default()
    cfg[:, 56:59] = 0
    cfg[:, 58] = 1
    for i in range(n):
        cfg[i, 40:44] = rng.choice(90, 4, replace=False)
        cfg[i, 49:51] = rng.choice(90, 2, replace=False)
    return cfg


def check_p3(fl):
    vnet = XC.vnet_weights(7)
    e = G.engine(fl, N)
    # decide-only at every step (prefix replayed through the main loop's skip path)
    z = XC.make_cases(fl, N, seed=G.FSEED[fl] + 5)
    z["cfg"][:, 40:56] = -1
    runs = [G.run_p3(fl, z, XC.COEFS[fl], vnet, 400, eng=e) for _ in range(RUNS)]
    differ = np.zeros(N, bool)
    for r in runs[1:]:
        for j in (5, 6):   # root visits, root Q (terminal outputs are not written in decide-only mode)
            differ |= np.array([not _same(np.asarray(runs[0][j])[i], np.asarray(r[j])[i]) for i in range(N)])
    SAVE[f"decide_{fl}_pol"] = np.stack([np.asarray(r[5])[:, 0] for r in runs])
    SAVE[f"decide_{fl}_q"] = np.stack([np.asarray(r[6])[:, 0] for r in runs])
    SAVE[f"decide_{fl}_cfg"] = z["cfg"]
    SAVE[f"decide_{fl}_pools"] = z["pools"]
    print(f"{fl} decide-only: {int(differ.sum())} of {N} episodes differ between identical runs", flush=True)
    if differ.any():
        FAIL.append(f"{fl} decide-only nondeterministic ({int(differ.sum())}/{N})")
    # full drafts with forced bans, opponent and self-play variants
    for sp in (0, 1):
        zz = XC.make_cases(fl, N, seed=31)
        zz["cfg"] = _forced_cfg(N, 9 + sp, sp)
        terms = [np.asarray(G.run_p3(fl, zz, XC.COEFS[fl], vnet, 50, eng=e)[3]) for _ in range(RUNS)]
        differ = np.zeros(N, bool)
        for t in terms[1:]:
            differ |= (t != terms[0]).any(1)
        dup = sum(int(sum(len(set(t[i].tolist())) != 16 for i in range(N))) for t in terms)
        notforced = sum(int((t[:, :4] != zz["cfg"][:, 40:44]).any(1).sum()) for t in terms)
        SAVE[f"forced{sp}_{fl}_term"] = np.stack(terms)
        print(f"{fl} forced bans (selfplay {sp}): {int(differ.sum())} differ, {dup} drafts with a repeated hero, "
              f"{notforced} with a forced ban not applied", flush=True)
        if differ.any() or dup or notforced:
            FAIL.append(f"{fl} forced/selfplay {sp}: differ {int(differ.sum())}, dup {dup}, notforced {notforced}")
    del e


def check_paper1():
    e = G.engine("paper1", 256)
    cfg = G.paper1_cfg(256, 3)
    flat = lambda res: [np.asarray(r[2], np.float32).tobytes() + np.float32(r[0]).tobytes()
                        + b"".join(np.asarray(x[1], np.float32).tobytes() for x in r[1]) for r in res]
    runs = [flat(e.run_episodes(np.ascontiguousarray(cfg), 200, 2.0, G.BASE_SEED, 1.0, 0.3, 0.25, 1))
            for _ in range(RUNS)]
    differ = sum(any(r[i] != runs[0][i] for r in runs[1:]) for i in range(256))
    print(f"paper1 full drafts: {differ} of 256 differ", flush=True)
    if differ:
        FAIL.append(f"paper1 nondeterministic ({differ}/256)")


def test_p3_kernels_deterministic():
    for fl in XC.FLAVORS:
        check_p3(fl)
    assert not FAIL, FAIL


def test_paper1_deterministic():
    check_paper1()
    assert not FAIL, FAIL


if __name__ == "__main__":
    for fl in XC.FLAVORS:
        check_p3(fl)
    check_paper1()
    if os.environ.get("XV_SAVE"):
        np.savez(os.environ["XV_SAVE"], **SAVE)
    print("FAILURES:" if FAIL else "ALL DETERMINISTIC", FAIL, flush=True)
    sys.exit(1 if FAIL else 0)
