"""
P3 personalized GD, stage 4: the personalized GD as the opponent model of the
personalized MCTS (kernel cuda_pgd).

Protocol (audit fixes): pools are the heroes a player played before the game
day (the real hero is not added); no real bans are forced into the search
(bans are searched for our team and drawn from the GD for the other team).
Configurations, all with argmax root, 400 sims, c_puct 2:
  R1 pers-GD      personalized value, BC prior, paper-1 GD everywhere (the
                  previous model under this protocol)
  R2 pers-PGD     personalized value, BC prior, personalized GD for the
                  opponent, the in-tree opponent and all rollouts
  R3 pers-PGD+P   R2 with our own PUCT priors from the personalized GD
  P1 pop-GD       population value, paper-1 GD
  P2 pop-PGD      population value, personalized GD
Runs:
  verify  flags off = personal kernel (bitwise); kernel personalized-GD
          probabilities vs Python
  real    decide-only at every real pick and ban state of the first 2,000
          drafter lobbies, all five configurations
  full    1,000 lobbies: controlled team vs the configuration's opponent
          model, and self-play
  analyze tables
--assign {slot,team}: see p3_mcts_drafter (default slot). verify step 2
(kernel personalized-GD vs Python) masks with the acting slot's pool; in
assign mode it masks with the team's pool union.
Run (from training/):
  CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=2 nice -n 19 taskset -c 48-63 \
    python3 personalization/p3_pgd_mcts.py --stage verify|real|full|analyze
Outputs: cache/pgdmcts_*.pkl.gz, results/p3_pgd_mcts.json
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

from p3_heroes import NUM_HEROES, HKEY
import p3_hs_core as C
import p3_mcts_core as M
import p3_pgd_common as P
import p3_pgd_feats as PF
from p3_mcts_drafter import Setup, BATCH
from p3_pgd_model import Head, BAN_FEATS

CFGS = {"R1 pers-GD": (1, 0, 0), "R2 pers-PGD": (1, 1, 0), "R3 pers-PGD+P": (1, 1, 1),
        "P1 pop-GD": (0, 0, 0), "P2 pop-PGD": (0, 1, 0)}   # personal valuation, use_pgd, pgd_prior


def metagd_kernel_weights(mg, to_sh):
    """Meta GD in the kernel's GenericDraftModel layout (shared hero order);
    returns (flat, offsets) and the meta part of the first layer (my order)."""
    from train_generic_draft import GenericDraftModel
    from extract_weights import extract_gd_weights
    W1 = mg.net[0].weight.detach().numpy()
    b1 = mg.net[0].bias.detach().numpy()
    W2, b2 = mg.net[3].weight.detach().numpy(), mg.net[3].bias.detach().numpy()
    W3, b3 = mg.net[6].weight.detach().numpy(), mg.net[6].bias.detach().numpy()
    W1s = np.zeros((256, 3 * NUM_HEROES + 19), np.float32)
    for blk in range(3):
        W1s[:, blk * NUM_HEROES + to_sh] = W1[:, blk * NUM_HEROES:(blk + 1) * NUM_HEROES]
    W1s[:, 3 * NUM_HEROES:3 * NUM_HEROES + 19] = W1[:, 3 * NUM_HEROES:3 * NUM_HEROES + 19]
    W3s = np.zeros_like(W3)
    b3s = np.zeros_like(b3)
    W3s[to_sh] = W3
    b3s[to_sh] = b3
    gm = GenericDraftModel()
    sd = gm.state_dict()
    sd["net.0.weight"] = torch.tensor(W1s)
    sd["net.0.bias"] = torch.tensor(b1)
    sd["net.3.weight"] = torch.tensor(W2)
    sd["net.3.bias"] = torch.tensor(b2)
    sd["net.6.weight"] = torch.tensor(W3s)
    sd["net.6.bias"] = torch.tensor(b3s)
    gm.load_state_dict(sd)
    return extract_gd_weights(gm), W1[:, 289:469]


class PGDSetup(Setup):
    def __init__(self):
        super().__init__()
        import p3_x_common as X
        self.KG = M.load_module("pgd_kernel", os.path.join(M.HERE, "cuda_pgd"))
        self.games = P.load_games()
        self.mg = P.load_metagd()
        self.meta_flat, self.W1meta = metagd_kernel_weights(self.mg, self.to_sh)
        ck = torch.load(os.path.join(C.CACHE, "pgd_personal.pt"), weights_only=False)
        self.hp, self.hb = Head(len(PF.FEATS)), Head(len(BAN_FEATS))
        self.hp.load_state_dict(ck["pick"])
        self.hb.load_state_dict(ck["ban"])
        self.hp.eval()
        self.hb.eval()
        self.dx = X.load_ext()
        # hs_slots row -> extended-table row, by (replay_id, hero)
        kx = self.dx["replay_id"] * HKEY + self.dx["hero"]
        ox = np.argsort(kx)
        kh = self.d["replay_id"] * HKEY + self.d["hero"]
        j = np.searchsorted(kx[ox], kh)
        assert np.array_equal(kx[ox][j], kh)
        self.hs2ext = ox[j]
        assert np.array_equal(self.dx["pid"][self.hs2ext], self.d["pid"])
        self.ctx = {}

    def prepare_context(self, rows):
        rows = np.unique(np.asarray(rows))
        rows = rows[[int(r) not in self.ctx for r in rows]]
        if len(rows) == 0:
            return
        hist = PF.walk(self.dx, self.hs2ext[rows])
        F, aux = PF.context(hist, self.meta["fine"])
        with torch.no_grad():
            pb = self.hp.bias(torch.tensor(F)).numpy()
        for i, r in enumerate(rows):
            self.ctx[int(r)] = {"F": F[i], "pick_bias": pb[i], "sshare": aux["sshare"][i],
                                "main": aux["is_main"][i] * aux["main_share"][i], "tot": aux["tot"][i]}

    def ban_bias(self, opp_rows, own_rows):
        G = np.zeros((NUM_HEROES, len(BAN_FEATS)), np.float32)
        co = [self.ctx[int(r)] for r in opp_rows]
        cw = [self.ctx[int(r)] for r in own_rows]
        so = np.stack([c["sshare"] for c in co])
        G[:, 0], G[:, 1] = so.max(0), so.sum(0)
        G[:, 2] = np.stack([c["main"] for c in co]).max(0)
        G[:, 3] = (np.stack([c["F"][:, 10] for c in co]) * so).max(0)
        G[:, 4] = np.stack([c["F"][:, 5] for c in co]).max(0)
        G[:, 5] = (np.expm1(np.stack([c["F"][:, 3] for c in co])) >= 10).sum(0) / 5.0
        G[:, 6] = np.stack([c["F"][:, 2] for c in co]).max(0)
        if cw:
            sw = np.stack([c["sshare"] for c in cw])
            G[:, 7], G[:, 9] = sw.max(0), sw.sum(0)
            G[:, 8] = np.stack([c["main"] for c in cw]).max(0)
        G[:, 10] = len(co) / 5.0
        with torch.no_grad():
            return self.hb.bias(torch.tensor(G[None]))[0].numpy()

    def pgd_tensors(self, lb):
        """(14, 90) personal-GD terms (shared order) and the meta GD offset."""
        out = np.zeros((14, NUM_HEROES), np.float32)
        for sl in range(10):
            out[sl, self.to_sh] = self.ctx[int(lb["rows"][sl])]["pick_bias"]
        slot_step = {int(lb["slot_of"][k]): k for k in range(16) if M.IS_PICK[k]}
        for phase, steps in ((0, (0, 1, 2, 3)), (1, (9, 10))):
            for t in (0, 1):
                k = [s for s in steps if M.DRAFT_TEAM[s] == t][0]
                opp = [lb["rows"][sl] for sl in range(10) if (sl < 5) != (t == 0) and slot_step[sl] > k]
                own = [lb["rows"][sl] for sl in range(10) if (sl < 5) == (t == 0) and slot_step[sl] > k]
                out[10 + 2 * phase + t, self.to_sh] = self.ban_bias(opp, own)
        gi = lb["gi"]
        day = int(self.L["day"][gi]) - int(self.games["day0"])
        tier = int(lb["tier"])
        meta = np.r_[self.games["meta_pick"][day, tier], self.games["meta_ban"][day, tier]]
        b1 = (self.W1meta @ meta).astype(np.float32)
        return out, b1

    def lobby_pgd(self, gi):
        lb = self.lobby(gi)
        # audit fix: pools from pre-day history only
        for sl in range(10):
            pm = np.zeros(NUM_HEROES, bool)
            pm[self.to_sh] = self.T["n"][self.T["pos"][int(lb["rows"][sl])]] > 0
            lb["pool"][sl] = M.pool_bits(pm)
        lb["forced"] = np.full(16, -1, np.int32)
        return lb

    def cfg_row58(self, lb, our_team, personal, use_pgd, pgd_prior, selfplay=0, decide=-1):
        """cfg row with the pgd flags [56] use_pgd, [57] pgd_prior."""
        c = self.cfg_row(lb, our_team, personal, selfplay=selfplay, decide=decide, opp=0)
        c[56], c[57] = use_pgd, pgd_prior
        return c

    def engine(self, key, gdi, n, use_pgd):
        (wf, wo, lut), _ = self.W.wp_cfg(key)
        pf, po = self.W.policy
        gf, go = self.meta_flat if use_pgd else self.W.gds[gdi % len(self.W.gds)]
        return self.KG.PgdEngine(pf, gf, wf, po, go, wo, lut, max_concurrent=n, device_id=0)

    def run_pgd(self, jobs, sims, seed, use_pgd):
        """jobs: (lobby, cfg58, pgd (14,90), b1 (256))."""
        out = [None] * len(jobs)
        by_key = {}
        for i, j in enumerate(jobs):
            by_key.setdefault(j[0]["key"], []).append(i)
        coefs = np.r_[self.coefs, np.float32(self.hp.alpha.item()), np.float32(self.hb.alpha.item())].astype(np.float32)
        for key, idx in by_key.items():
            for bs in range(0, len(idx), BATCH):
                ii = idx[bs:bs + BATCH]
                cfg = np.stack([jobs[i][1] for i in ii])
                s = np.stack([jobs[i][0]["s"] for i in ii])
                off = np.stack([jobs[i][0]["off"] for i in ii])
                imit = np.stack([jobs[i][0]["imit"] for i in ii])
                pool = np.stack([jobs[i][0]["pool"] for i in ii])
                pg = np.stack([jobs[i][2] for i in ii])
                b1 = np.stack([jobs[i][3] for i in ii])
                eng = self.engine(key, (bs // BATCH) % 5, len(ii), use_pgd)
                r = eng.run(cfg, s, off, imit, pool, coefs, pg, b1, sims, 2.0, seed + bs, 0.0, 0.3, 0.0, 1,
                    search_mode=M.search_mode())
                del eng
                for j, i in enumerate(ii):
                    nt = max(int(r[4][j]), 1)
                    out[i] = {"wp_t0": float(r[1][j]), "v_t0": float(r[2][j]), "acts": r[3][j].copy(),
                              "pol": r[5][j, :nt].copy(), "q": r[6][j, :nt].copy(), "step": r[7][j, :nt].copy()}
                    if cfg[j, M.CFG_ASSIGN] == 1 and cfg[j, 4] < 0:
                        out[i]["assign"] = self.assignment(jobs[i][0], out[i]["acts"])
        return out


def verify(S):
    gis = S.full_set[:64]
    lbs = [S.lobby_pgd(gi) for gi in gis]
    S.prepare_context(np.concatenate([lb["rows"] for lb in lbs]))
    tens = [S.pgd_tensors(lb) for lb in lbs]
    out = {}
    # 1. flags off: identical to the personal kernel with the paper GD
    kc = [S.ctrl[lb["gi"]] ^ lb["first"] for lb in lbs]
    jobs = [(lb, S.cfg_row58(lb, k, 1, 0, 0), t[0], t[1]) for lb, k, t in zip(lbs, kc, tens)]
    a = S.run_pgd(jobs, 200, 11, use_pgd=0)
    b = S.run([(lb, S.cfg_row(lb, k, 1)) for lb, k in zip(lbs, kc)], 200, 11)
    out["flags_off_vs_personal_kernel"] = {
        "episodes": len(lbs), "identical_actions": all(np.array_equal(x["acts"], y["acts"]) for x, y in zip(a, b)),
        "identical_wp": all(x["wp_t0"] == y["wp_t0"] for x, y in zip(a, b)),
        "identical_root_visits": all(np.array_equal(x["pol"], y["pol"]) for x, y in zip(a, b))}
    # 2. kernel personalized-GD probabilities vs Python at real states (all 16 steps)
    key = lbs[0]["key"]
    items = [(lb, k) for lb in lbs for k in range(16) if lb["key"] == key]
    cfg = np.stack([S.cfg_row58(lb, M.DRAFT_TEAM[k], 1, 1, 0, decide=k) for lb, k in items])
    s = np.stack([lb["s"] for lb, _ in items])
    off = np.stack([lb["off"] for lb, _ in items])
    imit = np.stack([lb["imit"] for lb, _ in items])
    pool = np.stack([lb["pool"] for lb, _ in items])
    tmap = {lb["gi"]: t for lb, t in zip(lbs, tens)}
    pg = np.stack([tmap[lb["gi"]][0] for lb, _ in items])
    b1 = np.stack([tmap[lb["gi"]][1] for lb, _ in items])
    eng = S.engine(key, 0, len(items), 1)
    coefs = np.r_[S.coefs, np.float32(S.hp.alpha.item()), np.float32(S.hb.alpha.item())].astype(np.float32)
    kp = eng.pgd_eval(cfg, s, off, imit, pool, coefs, pg, b1)
    del eng
    diffs = []
    for (lb, k), pk in zip(items, kp):
        x = np.zeros(469, np.float32)
        taken = np.zeros(NUM_HEROES, bool)
        for j in range(k):
            h = int(S.from_sh[lb["acts"][j]])
            taken[h] = True
            if M.IS_PICK[j]:
                x[(0 if M.DRAFT_TEAM[j] == 0 else NUM_HEROES) + h] = 1
            else:
                x[2 * NUM_HEROES + h] = 1
        x[3 * NUM_HEROES + lb["map"]] = 1
        x[284 + lb["tier"]] = 1
        x[287] = k / 15.0
        x[288] = M.IS_PICK[k]
        day = int(S.L["day"][lb["gi"]]) - int(S.games["day0"])
        x[289:379] = S.games["meta_pick"][day, lb["tier"]]
        x[379:469] = S.games["meta_ban"][day, lb["tier"]]
        mask = ~taken
        if M.IS_PICK[k]:
            sl = lb["slot_of"][k]
            if M.assign_mode() == 1:
                team = range(5 * M.DRAFT_TEAM[k], 5 * M.DRAFT_TEAM[k] + 5)
                pm = np.logical_or.reduce([S.T["n"][S.T["pos"][int(lb["rows"][x])]] > 0 for x in team])
            else:
                pm = S.T["n"][S.T["pos"][int(lb["rows"][sl])]] > 0
            if (pm & mask).any():
                mask = mask & pm
        lp = P.logp_metagd(S.mg, x[None], mask[None])[0]
        if M.IS_PICK[k]:
            u = S.hp.alpha.item() * lp + S.ctx[int(lb["rows"][lb["slot_of"][k]])]["pick_bias"]
        else:
            row = 10 + 2 * (0 if k < 4 else 1) + M.DRAFT_TEAM[k]
            u = S.hb.alpha.item() * lp + tmap[lb["gi"]][0][row][S.to_sh]
        u = np.where(mask, u, -1e9)
        p = np.exp(u - u.max())
        p /= p.sum()
        diffs.append(np.abs(pk[S.to_sh] - p).max())
    out["kernel_pgd_vs_python"] = {"states": len(items), "max_abs_diff": float(max(diffs)),
                                   "mean_max_abs_diff": float(np.mean(diffs))}
    print(json.dumps(out, indent=1), flush=True)
    with open(os.path.join(C.RESULTS, "p3_pgd_verify.json"), "w") as f:
        json.dump(out, f, indent=1)


def stage_real(S, sims, n):
    gis = np.r_[S.full_set, S.extra_set][:n]
    lbs = [S.lobby_pgd(gi) for gi in gis]
    S.prepare_context(np.concatenate([lb["rows"] for lb in lbs]))
    tens = [S.pgd_tensors(lb) for lb in lbs]
    res = {"lobbies": [{"gi": lb["gi"], "first": lb["first"], "rows": lb["rows"], "slot_of": lb["slot_of"],
                        "acts_real": lb["acts"]} for lb in lbs]}
    for name, (pers, use_pgd, prior) in CFGS.items():
        jobs, meta_ = [], []
        for li, (lb, t) in enumerate(zip(lbs, tens)):
            for k in range(16):
                jobs.append((lb, S.cfg_row58(lb, M.DRAFT_TEAM[k], pers, use_pgd, prior, decide=k), t[0], t[1]))
                meta_.append((li, k))
        rr = S.run_pgd(jobs, sims, 5000, use_pgd)
        res[name] = [(m[0], m[1], r["pol"][0], r["q"][0]) for m, r in zip(meta_, rr)]
        print(f"real {name}", flush=True)
    return res


def stage_full(S, sims):
    lbs = [S.lobby_pgd(gi) for gi in S.full_set]
    S.prepare_context(np.concatenate([lb["rows"] for lb in lbs]))
    tens = [S.pgd_tensors(lb) for lb in lbs]
    kctrl = [S.ctrl[lb["gi"]] ^ lb["first"] for lb in lbs]
    res = {"lobbies": [{"gi": lb["gi"], "first": lb["first"], "rows": lb["rows"], "slot_of": lb["slot_of"],
                        "kctrl": kc, "acts_real": lb["acts"]} for lb, kc in zip(lbs, kctrl)]}
    for name, (pers, use_pgd, prior) in CFGS.items():
        for sp in (0, 1):
            jobs = [(lb, S.cfg_row58(lb, kc, pers, use_pgd, prior, selfplay=sp), t[0], t[1])
                    for lb, kc, t in zip(lbs, kctrl, tens)]
            res[f"{name}|{'self' if sp else 'ctrl'}"] = S.run_pgd(jobs, sims, 1000 + 7 * sp, use_pgd)
            print(f"full {name} {sp}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["verify", "real", "full", "analyze"])
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--nlobbies", type=int, default=2000)
    ap.add_argument("--search-mode", required=True, choices=["chance", "rollfwd"])
    ap.add_argument("--assign", default="slot", choices=["slot", "team"])
    a = ap.parse_args()
    M.set_search_mode(a.search_mode)
    M.set_assign_mode(a.assign)
    torch.set_num_threads(2)
    t0 = time.time()
    if a.stage == "analyze":
        import p3_pgd_analyze
        p3_pgd_analyze.main()
        return
    S = PGDSetup()
    if a.stage == "verify":
        verify(S)
        return
    res = stage_real(S, a.sims, a.nlobbies) if a.stage == "real" else stage_full(S, a.sims)
    res["assign"] = a.assign
    M.save_pickle(res, f"pgdmcts_{a.stage}_s{a.sims}.pkl.gz")
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
