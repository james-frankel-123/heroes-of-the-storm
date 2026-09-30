"""
P3 distillation, stage 4: the distilled prior as a standalone greedy drafter.

At each decision the controlled team picks argmax over valid heroes of
alpha log p_BC(state) + g(player features) (no search). The same code with
the prior term removed is the greedy BC drafter (non-personal baseline).
Same lobbies, controlled teams, opponents and conventions as the MCTS runs
(kernel team labels, real bans forced where still free, other bans and
opponent picks sampled from the pool-restricted GD policy or the imitation
model). Final drafts are valued with the drift-aware WP (mean-probability
ensemble, WPEval) and the personal combiner.

Outputs in the MCTS pickle formats so p3_mcts_analyze functions apply:
  full      a-gd|pers, a-imit|pers, b|pers (greedy distilled), and the same
            three with |pop (greedy BC)
  real      rankings at the real pick states of all 5,724 lobbies: pers =
            distilled prior, pop = BC prior (pol = prior probabilities)
  collapse  300 players x 30 contexts (same contexts as the MCTS runs)

Run (from training/):
  OMP_NUM_THREADS=4 nice -n 19 taskset -c 48-63 python3 personalization/p3_ds_greedy.py
Output: cache/dsgreedy_{full,real,collapse}.pkl.gz
"""
import os
import sys
import gzip
import time
import pickle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

import p3_hs_core as C
import p3_mcts_core as M
import p3_ds_common as DS
from p3_mcts_drafter import Setup


def main():
    t0 = time.time()
    torch.set_num_threads(4)
    S = Setup()
    import search as SR
    import p3_dr_core as D
    net = SR.policy_net("bc")
    model, ck = DS.load_model()
    alpha = float(ck["alpha"])
    wpE = D.WPEval(S.d["hero_names"])
    gd = D.GDPolicy(S.d["hero_names"])
    to_sh, from_sh = S.to_sh, S.from_sh
    iw0 = float(S.iw[0])
    b = S.coefs

    def lobby_all(gis, sub=None):
        return [S.lobby(gi) if sub is None else S.lobby(gi, sub=sub) for gi in gis]

    def biases(lbs):
        rows = np.unique(np.concatenate([lb["rows"] for lb in lbs]))
        F = DS.player_features(S, rows)
        with torch.no_grad():
            bb = model.bias(torch.tensor(F)).numpy()
        return {int(r): bb[i] for i, r in enumerate(rows)}   # my order

    def prior_scores(states, lbs, ks, slots, bias, personal):
        """Batched: returns (B, 90) logits in MY order over valid heroes."""
        B = len(states)
        X = np.zeros((B, 290), np.float32)
        Mk = np.zeros((B, 90), bool)
        for i, (st, lb, k, sl) in enumerate(zip(states, lbs, ks, slots)):
            x = X[i]
            for h in st["t0"]:
                x[to_sh[h]] = 1
            for h in st["t1"]:
                x[90 + to_sh[h]] = 1
            for h in st["bans"]:
                x[180 + to_sh[h]] = 1
            x[270 + lb["map"]] = 1
            x[284 + lb["tier"]] = 1
            x[287] = k / 15.0
            x[288] = 1.0
            x[289] = float(M.DRAFT_TEAM[k])
            row = int(lb["rows"][sl])
            pool = S.T["pool"][S.T["pos"][row]] & ~st["taken"]
            m = pool if pool.any() else ~st["taken"]
            Mk[i, to_sh] = m
        bl = DS.bc_logp(net, X, Mk)[:, to_sh]            # my order
        u = alpha * bl if personal else bl
        if personal:
            u = u + np.stack([bias[int(lb["rows"][sl])] for lb, sl in zip(lbs, slots)])
        return np.where(Mk[:, to_sh], u, -1e9)

    def gd_sample(states, lbs, k, rng, kind, is_pick):
        Bn = len(states)
        t0_ = np.zeros((Bn, 90), np.float32)
        t1_ = np.zeros((Bn, 90), np.float32)
        bans = np.zeros((Bn, 90), np.float32)
        avail = np.zeros((Bn, 90), bool)
        for i, (st, lb) in enumerate(zip(states, lbs)):
            t0_[i, list(st["t0"])] = 1
            t1_[i, list(st["t1"])] = 1
            bans[i, list(st["bans"])] = 1
            av = ~st["taken"]
            if is_pick:
                row = int(lb["rows"][lb["slot_of"][k]])
                pa = S.T["pool"][S.T["pos"][row]] & av
                av = pa if pa.any() else av
            avail[i] = av
        mo = np.stack([np.eye(14, dtype=np.float32)[lb["map"]] for lb in lbs])
        to = np.stack([np.eye(3, dtype=np.float32)[lb["tier"]] for lb in lbs])
        # GDPolicy one-hots use shared MAPS / SKILL_TIERS order, as the kernel's map/tier indices
        lp = gd.logprobs(t0_, t1_, bans, mo, to, np.full(Bn, k, np.float32), np.full(Bn, float(is_pick), np.float32),
                         avail)
        if kind == "imit" and is_pick:
            u = iw0 * lp + np.stack([S.ipers[int(lb["rows"][lb["slot_of"][k]])] for lb in lbs])  # my order
            u = np.where(avail, u, -1e9)
            u -= u.max(1, keepdims=True)
            p = np.exp(u)
        else:
            p = np.where(avail, np.exp(lp), 0)
        p /= p.sum(1, keepdims=True)
        r = rng.rand(Bn, 1)
        return np.minimum((p.cumsum(1) < r).sum(1), 89)

    def draft(lbs, controllers, opp_kind, bias, seed):
        """controllers: list of sets of kernel teams driven by the greedy
        prior; personal: distilled vs BC. Returns per lobby acts (shared)."""
        rng = np.random.RandomState(seed)
        states = [{"t0": [], "t1": [], "bans": [], "taken": np.zeros(90, bool), "acts": np.zeros(16, np.int32)}
                  for _ in lbs]
        for k in range(16):
            tm = M.DRAFT_TEAM[k]
            if not M.IS_PICK[k]:
                need = []
                for i, (st, lb) in enumerate(zip(states, lbs)):
                    h = int(from_sh[lb["forced"][k]])
                    if not st["taken"][h]:
                        st["bans"].append(h)
                        st["taken"][h] = True
                        st["acts"][k] = to_sh[h]
                    else:
                        need.append(i)
                if need:
                    hs = gd_sample([states[i] for i in need], [lbs[i] for i in need], k, rng, "gd", 0)
                    for i, h in zip(need, hs):
                        states[i]["bans"].append(int(h))
                        states[i]["taken"][h] = True
                        states[i]["acts"][k] = to_sh[h]
                continue
            greedy = [i for i, c in enumerate(controllers) if tm in c[0]]
            other = [i for i, c in enumerate(controllers) if tm not in c[0]]
            if greedy:
                sc = prior_scores([states[i] for i in greedy], [lbs[i] for i in greedy], [k] * len(greedy),
                                  [lbs[i]["slot_of"][k] for i in greedy], bias, controllers[greedy[0]][1])
                hs = sc.argmax(1)
                for i, h in zip(greedy, hs):
                    states[i]["t0" if tm == 0 else "t1"].append(int(h))
                    states[i]["taken"][h] = True
                    states[i]["acts"][k] = to_sh[h]
            if other:
                hs = gd_sample([states[i] for i in other], [lbs[i] for i in other], k, rng, opp_kind, 1)
                for i, h in zip(other, hs):
                    states[i]["t0" if tm == 0 else "t1"].append(int(h))
                    states[i]["taken"][h] = True
                    states[i]["acts"][k] = to_sh[h]
        out = []
        for st, lb in zip(states, lbs):
            wp = wpE.wp([(tuple(st["t0"]), tuple(st["t1"]))], lb["bidx"], lb["map_name"], lb["tier_name"])[0]
            out.append({"acts": st["acts"].copy(), "wp_t0": float(wp)})
        return out

    def add_names(lbs):
        for lb in lbs:
            lb["map_name"] = str(S.L["map"][lb["gi"]])
            lb["tier_name"] = str(S.L["tier"][lb["gi"]])
            lb["bidx"] = int(S.bidx_all[S.L["g"][lb["gi"]]])

    # ---------- full
    lbs = lobby_all(S.full_set)
    add_names(lbs)
    bias = biases(lbs)
    kctrl = [S.ctrl[lb["gi"]] ^ lb["first"] for lb in lbs]
    res = {"alpha": alpha}
    for pers, tag in ((True, "pers"), (False, "pop")):
        res[f"a-gd|{tag}"] = draft(lbs, [({kc}, pers) for kc in kctrl], "gd", bias, 1000)
        res[f"a-imit|{tag}"] = draft(lbs, [({kc}, pers) for kc in kctrl], "imit", bias, 1000)
        res[f"b|{tag}"] = draft(lbs, [({0, 1}, pers) for _ in lbs], "gd", bias, 1007)
        print(f"full {tag}: {time.time() - t0:.0f}s", flush=True)
    res["lobbies"] = [{"gi": lb["gi"], "first": lb["first"], "rows": lb["rows"], "slot_of": lb["slot_of"],
                       "kctrl": kc, "acts_real": lb["acts"]} for lb, kc in zip(lbs, kctrl)]
    with gzip.open(os.path.join(C.CACHE, "dsgreedy_full.pkl.gz"), "wb") as f:
        pickle.dump(res, f)
    # ---------- real-state rankings (all lobbies)
    gis = np.r_[S.full_set, S.extra_set]
    lbs = lobby_all(gis)
    bias = biases(lbs)
    dec, lob_meta = [], []
    for li, lb in enumerate(lbs):
        lob_meta.append({"gi": lb["gi"], "first": lb["first"], "rows": lb["rows"], "slot_of": lb["slot_of"],
                         "acts_real": lb["acts"]})
    for k in [k for k in range(16) if M.IS_PICK[k]]:
        states = []
        for lb in lbs:
            st = {"t0": [], "t1": [], "bans": [], "taken": np.zeros(90, bool)}
            for j in range(k):
                h = int(from_sh[lb["acts"][j]])
                st["taken"][h] = True
                if M.IS_PICK[j]:
                    st["t0" if M.DRAFT_TEAM[j] == 0 else "t1"].append(h)
                else:
                    st["bans"].append(h)
            states.append(st)
        for pers in (1, 0):
            sc = prior_scores(states, lbs, [k] * len(lbs), [lb["slot_of"][k] for lb in lbs], bias, bool(pers))
            p = np.exp(sc - sc.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            for li in range(len(lbs)):
                pol = np.zeros(90, np.float32)
                pol[to_sh] = p[li]
                dec.append((li, k, pers, pol, np.zeros(90, np.float32)))
        print(f"real step {k}: {time.time() - t0:.0f}s", flush=True)
    with gzip.open(os.path.join(C.CACHE, "dsgreedy_real.pkl.gz"), "wb") as f:
        pickle.dump({"decisions": dec, "lobbies": lob_meta}, f)
    # ---------- collapse (same contexts as the MCTS runs)
    rng = np.random.RandomState(9)
    ctx = []
    for pr in S.coll_players:
        for j in range(30):
            gi = int(rng.choice(S.full_set))
            st = S.L["steps"][gi]
            ks = [k for k in range(16) if st[k][1] == 1]
            k = int(rng.choice(ks))
            ctx.append((int(pr), gi, k))
    lbs = [S.lobby(gi, sub=(k, pr)) for pr, gi, k in ctx]
    bias = biases(lbs)
    cdec = []
    for pers in (1, 0):
        states = []
        for (pr, gi, k), lb in zip(ctx, lbs):
            st = {"t0": [], "t1": [], "bans": [], "taken": np.zeros(90, bool)}
            for j in range(k):
                h = int(from_sh[lb["acts"][j]])
                st["taken"][h] = True
                if M.IS_PICK[j]:
                    st["t0" if M.DRAFT_TEAM[j] == 0 else "t1"].append(h)
                else:
                    st["bans"].append(h)
            states.append(st)
        sc = prior_scores(states, lbs, [c[2] for c in ctx], [lb["slot_of"][c[2]] for c, lb in zip(ctx, lbs)], bias,
                          bool(pers))
        for (pr, gi, k), h in zip(ctx, sc.argmax(1)):
            cdec.append((pr, k, pers, int(to_sh[h])))
    with gzip.open(os.path.join(C.CACHE, "dsgreedy_collapse.pkl.gz"), "wb") as f:
        pickle.dump({"decisions": cdec}, f)
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
