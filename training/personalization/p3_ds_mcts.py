"""
P3 distillation, stage 3: personalized MCTS with the distilled prior.

Kernel: cuda_prior/prior_kernel (personal kernel + per-slot prior term).
Verification (stage verify):
  - prior term off (cfg[56] = 0): identical outputs to the personal kernel
    (decide-only and full drafts, same seeds)
  - root priors reported by the kernel vs Python: BC prior alone (alpha 1,
    zero term) and the distilled prior (alpha, g), max |difference|
Runs (same lobbies, seeds, GD cycling and sims as p3_mcts_drafter; only the
personalized drafter changes its prior):
  full      a-gd, a-imit, b (personalized, distilled prior)
  real      decide-only at the real pick states of the first N lobbies
  collapse  300 players x 30 contexts (same contexts as the BC-prior run)

--assign {slot,team}: see p3_mcts_drafter (default slot).

Run (from training/):
  CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=2 nice -n 19 taskset -c 48-63 \
    python3 personalization/p3_ds_mcts.py --stage verify|full|real|collapse --sims 400
Output: cache/dsmcts_<stage>_s<sims>.pkl.gz, results/p3_ds_verify.json
"""
import os
import sys
import gzip
import json
import time
import pickle
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from p3_heroes import NUM_HEROES
import p3_hs_core as C
import p3_mcts_core as M
import p3_ds_common as DS
from p3_mcts_drafter import Setup, BATCH
from p3_ds_targets import extend_ipers


def make_prior_engine(kmod, W, key, gd_i, max_conc, device):
    (wf, wo, lut), _ = W.wp_cfg(key)
    pf, po = W.policy
    gf, go = W.gds[gd_i % len(W.gds)]
    return kmod.PriorEngine(pf, gf, wf, po, go, wo, lut, max_concurrent=max_conc, device_id=device)


class PriorSetup(Setup):
    def __init__(self):
        super().__init__()
        self.KP = M.load_module("prior_kernel", os.path.join(M.HERE, "cuda_prior"))
        self.model, ck = DS.load_model()
        self.alpha = float(ck["alpha"])
        self._bias = {}

    def bias_for(self, lb):
        key = tuple(int(r) for r in lb["rows"])
        if key not in self._bias:
            self._bias[key] = DS.slot_bias(self.model, self, lb["rows"], self.to_sh)
        return self._bias[key]

    def precompute_bias(self, lbs):
        rows = np.unique(np.concatenate([lb["rows"] for lb in lbs]))
        F = DS.player_features(self, rows)
        with torch.no_grad():
            b = self.model.bias(torch.tensor(F)).numpy()
        pos = {int(r): i for i, r in enumerate(rows)}
        for lb in lbs:
            out = np.zeros((10, NUM_HEROES), np.float32)
            for s in range(10):
                out[s, self.to_sh] = b[pos[int(lb["rows"][s])]]
            self._bias[tuple(int(r) for r in lb["rows"])] = out

    def run_prior(self, jobs, sims, seed, use_prior=1, alpha=None, zero_bias=False, kernel="prior"):
        out = [None] * len(jobs)
        by_key = {}
        for i, (lb, c) in enumerate(jobs):
            by_key.setdefault(lb["key"], []).append(i)
        a = self.alpha if alpha is None else alpha
        coefs = np.r_[self.coefs, np.float32(a)].astype(np.float32)
        for key, idx in by_key.items():
            for bs in range(0, len(idx), BATCH):
                ii = idx[bs:bs + BATCH]
                cfg = np.stack([jobs[i][1] for i in ii]).astype(np.int32)
                cfg[:, 56] = use_prior
                s = np.stack([jobs[i][0]["s"] for i in ii])
                off = np.stack([jobs[i][0]["off"] for i in ii])
                imit = np.stack([jobs[i][0]["imit"] for i in ii])
                pool = np.stack([jobs[i][0]["pool"] for i in ii])
                bias = np.zeros((len(ii), 10, NUM_HEROES), np.float32) if zero_bias else \
                    np.stack([self.bias_for(jobs[i][0]) for i in ii])
                gdi = (bs // BATCH) % 5
                if kernel == "prior":
                    eng = make_prior_engine(self.KP, self.W, key, gdi, len(ii), 0)
                    r = eng.run(cfg, s, off, imit, pool, coefs, bias, sims, 2.0, seed + bs, 0.0, 0.3, 0.0, 1,
                    search_mode=M.search_mode())
                else:
                    eng = M.make_engine(self.K, self.W, key, gdi, len(ii), 0)
                    cfg[:, 56] = 0
                    r = eng.run(cfg, s, off, imit, pool, self.coefs, sims, 2.0, seed + bs,
                                0.0, 0.3, 0.0, 1,
                    search_mode=M.search_mode())
                del eng
                for j, i in enumerate(ii):
                    nt = max(int(r[4][j]), 1)
                    out[i] = {"win": float(r[0][j]), "wp_t0": float(r[1][j]), "v_t0": float(r[2][j]),
                              "acts": r[3][j].copy(), "turns": int(r[4][j]), "pol": r[5][j, :nt].copy(),
                              "q": r[6][j, :nt].copy(), "max_nodes": int(r[9][j]), "cap_hits": int(r[10][j]),
                              "prior": r[13][j, :nt].copy() if kernel == "prior" else None}
                    if cfg[j, M.CFG_ASSIGN] == 1 and cfg[j, 4] < 0:
                        out[i]["assign"] = self.assignment(jobs[i][0], out[i]["acts"])
        return out


def verify(S):
    import search as SR
    net = SR.policy_net("path:" + __import__("p3_dr_core").bc_prior_path())
    out = {}
    gis = S.full_set[:64]
    lbs = [S.lobby(gi) for gi in gis]
    S.precompute_bias(lbs)
    kc = [S.ctrl[lb["gi"]] ^ lb["first"] for lb in lbs]
    # 1. prior term off == personal kernel (full drafts and decide-only)
    ident = {}
    for label, mk in (("full draft", lambda lb, k: S.cfg_row(lb, k, 1)),
                      ("decide-only", lambda lb, k: S.cfg_row(lb, M.DRAFT_TEAM[6], 1, decide=6))):
        jobs = [(lb, mk(lb, k)) for lb, k in zip(lbs, kc)]
        a = S.run_prior(jobs, 200, 11, use_prior=0)
        b = S.run_prior(jobs, 200, 11, kernel="personal")
        ident[label] = {"episodes": len(jobs),
                        "identical_win_prob": all(x["win"] == y["win"] for x, y in zip(a, b)),
                        "identical_actions": all(np.array_equal(x["acts"], y["acts"]) for x, y in zip(a, b)),
                        "identical_root_visits": all(np.array_equal(x["pol"], y["pol"]) for x, y in zip(a, b))}
    out["prior_off_vs_personal_kernel"] = ident
    # 2. root priors vs Python (decide-only at step 6 of each lobby)
    ks = 6
    X = np.zeros((len(lbs), 290), np.float32)
    Mk = np.zeros((len(lbs), NUM_HEROES), bool)
    Bias = np.zeros((len(lbs), NUM_HEROES), np.float32)
    for i, lb in enumerate(lbs):
        lob = {"acts_real": lb["acts"], "map": lb["map"], "tier": lb["tier"]}
        sl = lb["slot_of"][ks]
        X[i], Mk[i] = DS.state_and_mask(lob, ks, S.T["pool"][S.T["pos"][int(lb["rows"][sl])]], S.to_sh)
        Bias[i] = S.bias_for(lb)[sl]
    bl = DS.bc_logp(net, X, Mk)
    ref_bc = np.where(Mk, np.exp(bl), 0)
    u = S.alpha * bl + Bias
    u = np.where(Mk, u, -1e9)
    ref_ds = np.exp(u - u.max(1, keepdims=True))
    ref_ds /= ref_ds.sum(1, keepdims=True)
    jobs = [(lb, S.cfg_row(lb, M.DRAFT_TEAM[ks], 1, decide=ks)) for lb in lbs]
    kbc = S.run_prior(jobs, 1, 3, use_prior=1, alpha=1.0, zero_bias=True)
    kds = S.run_prior(jobs, 1, 3, use_prior=1)
    out["root_prior_vs_python"] = {
        "BC prior (alpha 1, zero term) max |diff|": float(max(np.abs(r["prior"][0] - ref_bc[i]).max() for i, r in enumerate(kbc))),
        "distilled prior max |diff|": float(max(np.abs(r["prior"][0] - ref_ds[i]).max() for i, r in enumerate(kds))),
        "states": len(lbs)}
    print(json.dumps(out, indent=1), flush=True)
    with open(os.path.join(C.RESULTS, "p3_ds_verify.json"), "w") as f:
        json.dump(out, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["verify", "full", "real", "collapse"])
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--nlobbies", type=int, default=2000)
    ap.add_argument("--contexts", type=int, default=30)
    ap.add_argument("--search-mode", required=True, choices=["chance", "rollfwd"])
    ap.add_argument("--assign", default="slot", choices=["slot", "team"])
    a = ap.parse_args()
    M.set_search_mode(a.search_mode)
    M.set_assign_mode(a.assign)
    torch.set_num_threads(2)
    t0 = time.time()
    S = PriorSetup()
    if a.stage == "verify":
        verify(S)
        return
    res = {"sims": a.sims, "stage": a.stage, "alpha": S.alpha, "assign": a.assign}
    if a.stage == "full":
        lbs = [S.lobby(gi) for gi in S.full_set]
        S.precompute_bias(lbs)
        kctrl = [S.ctrl[lb["gi"]] ^ lb["first"] for lb in lbs]
        for nm, (sp, opp) in {"a-gd|pers": (0, 0), "a-imit|pers": (0, 1), "b|pers": (1, 0)}.items():
            jobs = [(lb, S.cfg_row(lb, kc, 1, selfplay=sp, opp=opp)) for lb, kc in zip(lbs, kctrl)]
            res[nm] = S.run_prior(jobs, a.sims, 1000 + (7 if sp else 0))
            print(f"{nm}: {time.time() - t0:.0f}s", flush=True)
        res["lobbies"] = [{"gi": lb["gi"], "first": lb["first"], "rows": lb["rows"], "slot_of": lb["slot_of"],
                           "kctrl": kc, "acts_real": lb["acts"]} for lb, kc in zip(lbs, kctrl)]
    elif a.stage == "real":
        gis = np.r_[S.full_set, S.extra_set][:a.nlobbies]
        lbs = [S.lobby(gi) for gi in gis]
        S.precompute_bias(lbs)
        jobs, meta_, lob_meta = [], [], []
        for lb in lbs:
            lob_meta.append({"gi": lb["gi"], "first": lb["first"], "rows": lb["rows"], "slot_of": lb["slot_of"],
                             "acts_real": lb["acts"]})
            for k in range(16):
                if M.IS_PICK[k]:
                    jobs.append((lb, S.cfg_row(lb, M.DRAFT_TEAM[k], 1, decide=k)))
                    meta_.append((len(lob_meta) - 1, k, 1))
        rr = S.run_prior(jobs, a.sims, 5000)
        res["decisions"] = [(m[0], m[1], m[2], r["pol"][0], r["q"][0]) for m, r in zip(meta_, rr)]
        res["lobbies"] = lob_meta
    else:
        rng = np.random.RandomState(9)
        jobs, meta_ = [], []
        for pr in S.coll_players:
            for j in range(a.contexts):
                gi = int(rng.choice(S.full_set))
                st = S.L["steps"][gi]
                ks = [k for k in range(16) if st[k][1] == 1]
                k = int(rng.choice(ks))
                lb = S.lobby(gi, sub=(k, int(pr)))
                jobs.append((lb, S.cfg_row(lb, M.DRAFT_TEAM[k], 1, decide=k)))
                meta_.append((int(pr), k, 1))
        extend_ipers(S, [])
        S.precompute_bias([j[0] for j in jobs])
        rr = S.run_prior(jobs, a.sims, 7000)
        res["decisions"] = [(m[0], m[1], m[2], int(np.argmax(r["pol"][0]))) for m, r in zip(meta_, rr)]
    M.save_pickle(res, f"dsmcts_{a.stage}_s{a.sims}.pkl.gz")
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
