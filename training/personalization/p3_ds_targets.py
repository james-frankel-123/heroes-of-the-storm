"""
P3 distillation, stage 1: training targets.

Personalized MCTS (personal kernel, BC prior, 400 sims, decide-only) at every
real pick state of V1 lobbies (2026-02-10 .. 2026-04-01) whose ten players
all have >= 50 earlier games. V1 lobbies are disjoint from every V2 lobby
used to evaluate drafters. Targets are the root visit distributions.

Run (from training/):
  CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=2 nice -n 19 taskset -c 48-63 \
    python3 personalization/p3_ds_targets.py [--lobbies 6000] [--sims 400]
Output: cache/ds_targets_s<sims>.pkl.gz
"""
import os
import sys
import gzip
import time
import pickle
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from p3_heroes import NUM_HEROES
import p3_hs_core as C
import p3_mcts_core as M
from p3_mcts_drafter import Setup


def v1_lobbies(S, n, seed=11):
    d = S.d
    days = d["day"]
    post = ~d["in_sample"]
    _, first_row = np.unique(d["g"][post], return_index=True)
    med = np.median(days[post][first_row])
    L = S.L
    prow = L["steps"][:, :, 3]
    picks = prow[prow >= 0].reshape(len(L["replay_id"]), 10)
    npos = np.vectorize(lambda r: S.T["pos"][int(r)])(picks)
    est = (S.T["n_p"][npos] >= 50).all(1)
    cand = np.flatnonzero((L["day"] < med) & est)
    rng = np.random.RandomState(seed)
    return rng.choice(cand, min(n, len(cand)), replace=False), len(cand)


def extend_ipers(S, rows):
    I = S.I
    rows = np.array(sorted(set(int(r) for r in rows) - set(S.ipers)))
    if len(rows) == 0:
        return
    rec = I.recency_features(S.d, rows)
    fake = {"row": rows, "recpos": np.arange(len(rows)), "lp": np.zeros((len(rows), NUM_HEROES), np.float32)}
    Xp = I.feature_tensor(fake, S.T, rec, S.meta)
    ip = (Xp[:, :, 1:] * S.iw[1:]).sum(-1)
    for i, r in enumerate(rows):
        S.ipers[int(r)] = ip[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lobbies", type=int, default=6000)
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--search-mode", required=True, choices=["chance", "rollfwd"])
    a = ap.parse_args()
    M.set_search_mode(a.search_mode)
    torch.set_num_threads(2)
    t0 = time.time()
    S = Setup()
    gis, n_elig = v1_lobbies(S, a.lobbies)
    assert not set(gis.tolist()) & set(np.r_[S.full_set, S.extra_set].tolist())
    extend_ipers(S, [r for gi in gis for r in S.L["steps"][gi][:, 3] if r >= 0])
    print(f"V1 eligible lobbies {n_elig:,}; using {len(gis):,} ({time.time() - t0:.0f}s)", flush=True)
    jobs, meta_, lob_meta = [], [], []
    for gi in gis:
        lb = S.lobby(gi)
        lob_meta.append({"gi": lb["gi"], "first": lb["first"], "rows": lb["rows"], "slot_of": lb["slot_of"],
                         "acts_real": lb["acts"], "map": lb["map"], "tier": lb["tier"]})
        for k in range(16):
            if M.IS_PICK[k]:
                jobs.append((lb, S.cfg_row(lb, M.DRAFT_TEAM[k], 1, decide=k)))
                meta_.append((len(lob_meta) - 1, k))
    rr = S.run(jobs, a.sims, 9000)
    out = {"sims": a.sims, "lobbies": lob_meta,
           "decisions": [(m[0], m[1], r["pol"][0], r["q"][0]) for m, r in zip(meta_, rr)]}
    M.save_pickle(out, f"ds_targets_s{a.sims}.pkl.gz")
    print(f"done: {len(meta_):,} searches ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
