"""
P3 personalized MCTS drafter runs (search-only, behavioral prior).

Kernel: cuda_personal/personal_kernel (verified in p3_mcts_verify). Prior:
the outcome-free BC prior distilled from GD (overfit2026/models/bc_prior.pt).
In-tree opponent and rollouts: GD (one of the 5 paper-1 GD models per
launch, cycled; the personalized and population runs of a batch share the
GD model and the seed). Root: argmax of visits. Pools restrict every pick
to the acting player's pool. No real bans are forced (default, matches the
one-step drafter): our bans are searched, the opponent's come from GD;
Setup.force_real_bans = True restores forcing real bans where still free.
Valuation: personal (V) or population (WP).
--assign team (default slot): assign mode (p3_mcts_core docstring): team
picks from the union of its players' pools, personal terms from each team's
best hero -> player assignment; full-draft results carry that assignment
("assign": per team, the slot of its j-th pick). Outputs get _assign-team.

Lobbies and players are those of p3_dr_drafter (same 1,000 full lobbies,
same 5,724 realized lobbies, same 300 collapse players; controlled team as
in that run).

Stages
  full      a-gd, a-imit (controlled team personalized or population vs GD
            or imitation opponent), b (self-play, both teams), and paired
            decisions: decide-only personalized and population searches at
            the states of the personalized a-gd trajectory
  curve     a-gd only (predicted-gain curve)
  real      decide-only at every real pick state, both valuations
            (realized-agreement check)
  collapse  decide-only for 300 players x 30 contexts, both valuations

Run (from training/):
  CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=2 nice -n 19 taskset -c 48-63 \
    python3 personalization/p3_mcts_drafter.py --stage full --sims 400
Output: cache/mcts_<stage>_s<sims>.pkl.gz
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

BATCH = 256


class Setup:
    force_real_bans = False   # True: real bans forced where still free (old protocol)

    def __init__(self):
        import p3_dr_core as D
        import p3_dr_imitation as I
        from shared import MAPS, SKILL_TIERS
        self.D, self.I = D, I
        self.MAPS, self.TIERS = MAPS, SKILL_TIERS
        self.d = C.load_slots()
        self.meta = C.hero_meta(self.d["hero_names"])
        self.L = D.load_lobbies()
        self.T = D.load_personal()
        self.W = M.Weights()
        self.K = M.personal_kernel()
        gd = D.GDPolicy(self.d["hero_names"])
        self.to_sh = gd.to_shared
        self.from_sh = np.argsort(self.to_sh)
        b = D.combiner()
        self.iw = np.load(I.OUT)["w"]
        self.coefs = np.array([0.0, b[1], b[2], b[3], self.iw[0]], np.float32)
        w = np.load(C.WP)
        self.bidx_all = w["build_idx"][np.argsort(w["replay_ids"])]
        z = pickle.load(gzip.open(os.path.join(C.CACHE, "dr_runs.pkl.gz")))
        self.full_set = z["full_set"]
        self.extra_set = z["extra_set"]
        self.coll_players = z["coll_players"]
        self.ctrl = {l["gi"]: l["ctrl"] for l in z["lobbies"]}
        # imitation personal part for all needed rows
        rows = set()
        for gi in np.r_[self.full_set, self.extra_set]:
            rows |= set(int(r) for r in self.L["steps"][gi][:, 3] if r >= 0)
        rows |= set(int(r) for r in self.coll_players)
        rows = np.array(sorted(rows))
        rec = I.recency_features(self.d, rows)
        fake = {"row": rows, "recpos": np.arange(len(rows)), "lp": np.zeros((len(rows), NUM_HEROES), np.float32)}
        Xp = I.feature_tensor(fake, self.T, rec, self.meta)
        ip = (Xp[:, :, 1:] * self.iw[1:]).sum(-1)
        self.ipers = {int(r): ip[i] for i, r in enumerate(rows)}

    def lobby(self, gi, sub=None):
        """Kernel-labeled tensors for lobby gi. sub = (step k, row) replaces
        the player at pick step k (collapse contexts)."""
        st = self.L["steps"][gi]
        T = self.T
        first = st[[k for k in range(16) if st[k][1] == 1][0]][2]
        acts = np.zeros(16, np.int32)
        slot_of = np.full(16, -1, np.int32)
        forced = np.full(16, -1, np.int32)
        rows = np.zeros(10, np.int64)
        s = np.zeros((10, NUM_HEROES), np.float32)
        off = np.zeros((10, NUM_HEROES), np.float32)
        imit = np.zeros((10, NUM_HEROES), np.float32)
        pool = np.zeros((10, 3), np.uint32)
        cnt = [0, 0]
        for k in range(16):
            h, ty, tm, row = st[k]
            kt = tm ^ first
            assert kt == M.DRAFT_TEAM[k] and ty == M.IS_PICK[k]
            acts[k] = self.to_sh[h]
            if ty == 0:
                if self.force_real_bans:
                    forced[k] = self.to_sh[h]
                continue
            if sub is not None and sub[0] == k:
                row = sub[1]
            sl = 5 * kt + cnt[kt]
            cnt[kt] += 1
            slot_of[k] = sl
            rows[sl] = row
            p = T["pos"][int(row)]
            s[sl, self.to_sh] = T["s"][p]
            off[sl, self.to_sh] = T["off"][p].astype(np.float32)
            imit[sl, self.to_sh] = self.ipers[int(row)]
            pm = np.zeros(NUM_HEROES, bool)
            pm[self.to_sh] = T["pool"][p]
            pool[sl] = M.pool_bits(pm)
        return {"gi": int(gi), "first": int(first), "acts": acts, "slot_of": slot_of, "forced": forced,
                "rows": rows, "s": s, "off": off, "imit": imit, "pool": pool,
                "map": self.MAPS.index(str(self.L["map"][gi])), "tier": self.TIERS.index(str(self.L["tier"][gi])),
                "key": self.W.stats_key(self.bidx_all[self.L["g"][gi]])}

    def cfg_row(self, lb, our_team, personal, selfplay=0, decide=-1, opp=0):
        c = np.full(M.CFG_LEN, -1, np.int32)
        c[0], c[1], c[2] = lb["map"], lb["tier"], our_team
        c[3], c[4], c[5], c[6], c[7] = selfplay, decide, 1, opp, personal
        c[8:24] = lb["acts"] if decide >= 0 else -1
        c[24:40] = lb["slot_of"]
        c[40:56] = lb["forced"]
        c[56:M.CFG_LEN] = 0
        c[M.CFG_ASSIGN] = M.assign_mode()
        return c

    def assignment(self, lb, acts):
        """Assign mode: best assignment of a complete draft (host recompute,
        float64; the kernel uses the same rule in float32)."""
        return M.assignment_terms(lb["s"], lb["off"], acts, float(self.coefs[2]), float(self.coefs[3]))[2]

    def run(self, jobs, sims, seed, device=0):
        """jobs: list of (lobby dict, cfg row). Batched by stats key; the
        batch index sets the GD model. Returns results aligned with jobs."""
        out = [None] * len(jobs)
        by_key = {}
        for i, (lb, c) in enumerate(jobs):
            by_key.setdefault(lb["key"], []).append(i)
        bc = M.BatchCache(f"{getattr(self, 'tag', 'mcts')}_s{sims}_seed{seed}_n{len(jobs)}")
        for key, idx in by_key.items():
            for bs in range(0, len(idx), BATCH):
                ii = idx[bs:bs + BATCH]
                got = bc.get((key, bs))
                if got is not None:
                    for i, v in zip(ii, got):
                        out[i] = v
                    continue
                cfg = np.stack([jobs[i][1] for i in ii])
                s = np.stack([jobs[i][0]["s"] for i in ii])
                off = np.stack([jobs[i][0]["off"] for i in ii])
                imit = np.stack([jobs[i][0]["imit"] for i in ii])
                pool = np.stack([jobs[i][0]["pool"] for i in ii])
                gdi = (bs // BATCH) % 5
                eng = M.make_engine(self.K, self.W, key, gdi, len(ii), device)
                r = eng.run(cfg, s, off, imit, pool, self.coefs, sims, 2.0, seed + bs, 0.0, 0.3, 0.0, 1,
                    search_mode=M.search_mode())
                del eng
                for j, i in enumerate(ii):
                    out[i] = {"win": float(r[0][j]), "wp_t0": float(r[1][j]), "v_t0": float(r[2][j]),
                              "acts": r[3][j].copy(), "turns": int(r[4][j]),
                              "pol": r[5][j, :max(int(r[4][j]), 1)].copy(), "q": r[6][j, :max(int(r[4][j]), 1)].copy(),
                              "step": r[7][j, :max(int(r[4][j]), 1)].copy(), "team": r[8][j, :max(int(r[4][j]), 1)].copy(),
                              "max_nodes": int(r[9][j]), "cap_hits": int(r[10][j])}
                    if cfg[j, M.CFG_ASSIGN] == 1 and cfg[j, 4] < 0:
                        out[i]["assign"] = self.assignment(jobs[i][0], out[i]["acts"])
                bc.put((key, bs), [out[i] for i in ii])
        bc.flush()
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["full", "curve", "real", "collapse"])
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--nlobbies", type=int, default=0, help="real stage: first N lobbies (0 = all)")
    ap.add_argument("--contexts", type=int, default=30)
    ap.add_argument("--search-mode", required=True, choices=["chance", "rollfwd"])
    ap.add_argument("--assign", default="slot", choices=["slot", "team"])
    a = ap.parse_args()
    M.set_search_mode(a.search_mode)
    M.set_assign_mode(a.assign)
    torch.set_num_threads(2)
    t0 = time.time()
    S = Setup()
    S.tag = f"mcts_{a.stage}"
    res = {"sims": a.sims, "stage": a.stage, "assign": a.assign}
    if a.stage in ("full", "curve"):
        lbs = [S.lobby(gi) for gi in S.full_set]
        kctrl = [S.ctrl[lb["gi"]] ^ lb["first"] for lb in lbs]
        runs = {"a-gd|pers": (1, 0, 0), "a-gd|pop": (0, 0, 0)}
        if a.stage == "full":
            runs.update({"a-imit|pers": (1, 0, 1), "a-imit|pop": (0, 0, 1), "b|pers": (1, 1, 0), "b|pop": (0, 1, 0)})
        for nm, (pers, sp, opp) in runs.items():
            jobs = [(lb, S.cfg_row(lb, kc, pers, selfplay=sp, opp=opp)) for lb, kc in zip(lbs, kctrl)]
            res[nm] = S.run(jobs, a.sims, 1000 + (7 if sp else 0))
            print(f"{nm}: {time.time() - t0:.0f}s", flush=True)
        if a.stage == "full":
            # paired decisions at the states of the personalized a-gd trajectory
            jobs, meta_ = [], []
            for li, (lb, kc) in enumerate(zip(lbs, kctrl)):
                acts = res["a-gd|pers"][li]["acts"]
                for k in range(16):
                    if M.IS_PICK[k] and M.DRAFT_TEAM[k] == kc:
                        lb2 = dict(lb)
                        lb2["acts"] = acts.astype(np.int32)
                        for pers in (1, 0):
                            jobs.append((lb2, S.cfg_row(lb2, kc, pers, decide=k)))
                            meta_.append((li, k, pers))
            rr = S.run(jobs, a.sims, 3000)
            res["paired"] = [(m[0], m[1], m[2], r["pol"][0], r["q"][0]) for m, r in zip(meta_, rr)]
            print(f"paired: {time.time() - t0:.0f}s", flush=True)
        res["lobbies"] = [{"gi": lb["gi"], "first": lb["first"], "rows": lb["rows"], "slot_of": lb["slot_of"],
                           "kctrl": kc, "acts_real": lb["acts"]} for lb, kc in zip(lbs, kctrl)]
    elif a.stage == "real":
        gis = np.r_[S.full_set, S.extra_set]
        if a.nlobbies:
            gis = gis[:a.nlobbies]
        jobs, meta_ = [], []
        lob_meta = []
        for gi in gis:
            lb = S.lobby(gi)
            lob_meta.append({"gi": lb["gi"], "first": lb["first"], "rows": lb["rows"], "slot_of": lb["slot_of"],
                             "acts_real": lb["acts"]})
            for k in range(16):
                if M.IS_PICK[k]:
                    for pers in (1, 0):
                        jobs.append((lb, S.cfg_row(lb, M.DRAFT_TEAM[k], pers, decide=k)))
                        meta_.append((len(lob_meta) - 1, k, pers))
        rr = S.run(jobs, a.sims, 5000)
        res["decisions"] = [(m[0], m[1], m[2], r["pol"][0], r["q"][0]) for m, r in zip(meta_, rr)]
        res["lobbies"] = lob_meta
    else:
        rng = np.random.RandomState(9)
        jobs, meta_ = [], []
        for pr in S.coll_players:
            for j in range(a.contexts):
                gi = int(rng.choice(S.full_set))
                st = S.L["steps"][gi]
                first = st[[k for k in range(16) if st[k][1] == 1][0]][2]
                ks = [k for k in range(16) if st[k][1] == 1]
                k = int(rng.choice(ks))
                lb = S.lobby(gi, sub=(k, int(pr)))
                for pers in (1, 0):
                    jobs.append((lb, S.cfg_row(lb, M.DRAFT_TEAM[k], pers, decide=k)))
                    meta_.append((int(pr), k, pers))
        rr = S.run(jobs, a.sims, 7000)
        res["decisions"] = [(m[0], m[1], m[2], int(np.argmax(r["pol"][0]))) for m, r in zip(meta_, rr)]
    res["max_nodes"] = int(max(r["max_nodes"] for k, v in res.items() if isinstance(v, list) and v and isinstance(v[0], dict) and "max_nodes" in v[0] for r in v)) if a.stage in ("full", "curve") else None
    M.save_pickle(res, f"mcts_{a.stage}_s{a.sims}.pkl.gz")
    M.clear_partials(S.tag)
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
