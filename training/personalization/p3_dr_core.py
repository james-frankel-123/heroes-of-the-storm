"""
P3 personalized drafter: shared core.

Unified model (Max's design): the population WP carries the meta; identified
players add causal per-player x hero terms. Value of a finished draft from
team 0's side:

  V = sigmoid(b0 + b1 logit(WP_pop) + b2 (S_0 - S_1) + b3 (O_0 - O_1))

  WP_pop  paper 2's drift-aware d2c_cumprev ensemble (3 seeds, swap-
          symmetrized), features from cumulative stats through the previous
          build (the causal convention of p3_wp_scores)
  S_t     sum over team t's players of s(p, h) = experience offset(n_p, n_ph)
          + posterior mean of the phase-1 "+CF rank 2" kernel (pooled skill
          and similarity embedding), state through the previous day
  O_t     number of off-role players (fine role < 10% of >= 50 earlier games)
  b       the V1-fit combiner "skill + off-role count (fine)" (p3_ph_role)
The population drafter values drafts with WP_pop alone.

Lobbies: post-cutoff snapshot games with a standard 16-step draft. The
player making each pick is the one who played the hero picked at that step.
A player's available heroes ("pool") are the heroes he played on earlier
days (lag 1); the hero he actually played is not added.

Pieces here: lobby table, personal tables (s, off-role, pool, counts for
every candidate hero of every slot), GD policy (paper-1 generic-draft
models, 5-model average, masked to the acting player's pool), WP evaluator.
"""
import os
import sys
import gzip
import json

TRAINING = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TRAINING)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from p3_heroes import NUM_HEROES
import p3_hs_core as C

DRAFTS = os.path.join(C.CACHE, "drafts_post.json.gz")
LOBBY = os.path.join(C.CACHE, "dr_lobbies.npz")
PTAB = os.path.join(C.CACHE, "dr_personal_post.npz")
# GD pool for rollouts, opponents and imitation: the train-window models
# (p3_gd_trainonly.py, games before 2026-02-10) unless P3_GD=paper1 selects
# the paper-1 pool, which was trained on a random 98% of the snapshot and is
# in sample for V1/V2.
PAPER1_GD = os.path.join(TRAINING, "rerun2026", "models")
TRAINONLY_GD = os.path.join(C.CACHE, "gd_trainonly")


def gd_dir():
    d = PAPER1_GD if os.environ.get("P3_GD") == "paper1" else TRAINONLY_GD
    if not all(os.path.exists(os.path.join(d, f"generic_draft_{i}.pt")) for i in range(5)):
        raise SystemExit(f"GD pool missing in {d} (run p3_gd_trainonly.py, or set P3_GD=paper1)")
    return d


def bc_prior_path():
    if os.environ.get("P3_GD") == "paper1":
        return os.path.join(TRAINING, "overfit2026", "models", "bc_prior.pt")
    p = os.path.join(TRAINONLY_GD, "bc_prior.pt")
    if not os.path.exists(p):
        raise SystemExit(f"{p} missing (run p3_gd_trainonly.py bc, or set P3_GD=paper1)")
    return p
H = NUM_HEROES


# ------------------------------------------------------------------ lobbies

def build_lobbies(d):
    """Games with a 16-step draft whose 10 picks all join to player rows.
    steps: (hero, type 0 ban / 1 pick, team, row index of the picking
    player's slot or -1 for bans)."""
    names = [str(x) for x in d["hero_names"]]
    hidx = {n: i for i, n in enumerate(names)}
    dr = json.load(gzip.open(DRAFTS, "rt"))
    post = np.flatnonzero(~d["in_sample"])
    key = {(int(d["replay_id"][i]), int(d["hero"][i])): i for i in post}
    games, steps = [], []
    for k, g in dr.items():
        o = g["order"]
        if len(o) != 16:
            continue
        st = []
        ok = True
        for h, ty, pn, sl in o:
            if h not in hidx:
                ok = False
                break
            hi = hidx[h]
            if ty == 1:
                row = key.get((int(k), hi))
                if row is None:
                    ok = False
                    break
                st.append((hi, 1, int(d["team"][row]), row))
            else:
                st.append((hi, 0, 0 if sl == 1 else 1, -1))
        if not ok or sum(s[1] for s in st) != 10:
            continue
        games.append((int(k), g["map"], g["tier"]))
        steps.append(st)
    rid = np.array([g[0] for g in games], np.int64)
    steps = np.array(steps, np.int64)  # (G, 16, 4)
    first_row = steps[:, :, 3].max(1)
    np.savez(LOBBY, replay_id=rid, map=np.array([g[1] for g in games]),
             tier=np.array([g[2] for g in games]), steps=steps,
             day=d["day"][first_row], g=d["g"][first_row])
    print(f"lobbies: {len(rid):,} games with 16-step drafts and all picks joined")


def load_lobbies():
    z = np.load(LOBBY)
    return {k: z[k] for k in z.files}


# ------------------------------------------------------------------ personal tables

def personal_tables(d, rows, table, kernel_K, r_adj):
    """For each slot row (query), per candidate hero: s = mu + m (lag 1 day),
    counts before the day, off-role (fine), pool. Returns dict of (nq, 90)."""
    meta = C.hero_meta(d["hero_names"])
    fine = meta["fine"]
    order = np.argsort(d["day"][rows], kind="stable")
    rows = rows[order]
    qmask = np.zeros(len(d["pid"]), bool)
    qmask[rows] = True
    qidx, S, P, _, _, _, _ = C.online_state(d, np.ones(len(d["pid"]), bool), qmask, H, r_adj)
    assert np.array_equal(np.sort(qidx), np.sort(rows))
    # counts matrix per query (lag 1 day), same walk as online_state
    npl = int(d["n_players"])
    Nst = np.zeros((npl, H), np.float32)
    N = np.empty((len(qidx), H), np.float32)
    day = d["day"]
    ud, qs = np.unique(day[qidx], return_index=True)
    qe = np.r_[qs[1:], len(qidx)]
    ap = 0
    for t, a, b in zip(ud, qs, qe):
        e = np.searchsorted(day, t - 1, side="right")
        if e > ap:
            np.add.at(Nst, (d["pid"][ap:e], d["hero"][ap:e]), 1.0)
            ap = e
        N[a:b] = Nst[d["pid"][qidx[a:b]]]
    m_all, v_all = C.gp_all(S, P, kernel_K)
    del S, P
    n_p = N.sum(1)
    nb = len(C.NPH_EDGES)
    bp = np.searchsorted(C.NP_EDGES, n_p, side="right") - 1
    bh = np.searchsorted(C.NPH_EDGES, N, side="right") - 1
    mu = table[bp[:, None] * nb + bh]
    s = mu + m_all
    fine_cnt = np.zeros((len(qidx), len(meta["fine_names"])), np.float32)
    for f in range(len(meta["fine_names"])):
        fine_cnt[:, f] = N[:, fine == f].sum(1)
    share = fine_cnt[:, fine] / np.maximum(n_p[:, None], 1)
    off = (n_p[:, None] >= 50) & (share < 0.10)
    pool = N > 0  # heroes played on earlier days only; the game's own hero is not added
    # restore the caller's row order
    inv = np.empty(len(qidx), np.int64)
    pos = {r: i for i, r in enumerate(qidx)}
    idx = np.array([pos[r] for r in rows[np.argsort(order)]])
    return {"rows": qidx[idx], "s": s[idx].astype(np.float32), "sd": np.sqrt(v_all[idx]).astype(np.float32),
            "off": off[idx], "pool": pool[idx], "n": N[idx], "n_p": n_p[idx]}


def kernel_best():
    kz = np.load(os.path.join(C.CACHE, "hs_kernels.npz"))
    from p3_hs_fit import KERNELS
    bases = {k[6:]: kz[k] for k in kz.files if k.startswith("basis_")}
    return C.kernel_from([bases[b] for b in KERNELS["+CF rank 2"]], kz["theta_+CF rank 2"])


def combiner():
    with open(os.path.join(C.RESULTS, "p3_ph_role.json")) as f:
        c = json.load(f)["D_game"]["skill + off-role count (fine)"]["coef"]
    return np.array(c, float)  # b0, b_logitwp, b_skill, b_offrole


# ------------------------------------------------------------------ GD policy

class GDPolicy:
    def __init__(self, hero_names):
        from train_generic_draft import GenericDraftModel
        from shared import HEROES
        self.models = []
        for i in range(5):
            m = GenericDraftModel()
            m.load_state_dict(torch.load(os.path.join(gd_dir(), f"generic_draft_{i}.pt"),
                                         weights_only=True, map_location="cpu"))
            m.eval()
            self.models.append(m)
        # my index -> shared index
        sidx = {h: i for i, h in enumerate(HEROES)}
        self.to_shared = np.array([sidx[str(h)] for h in hero_names])

    def logprobs(self, t0, t1, bans, map_oh, tier_oh, step_num, step_type, avail):
        """Batched: t0/t1/bans (B, 90) multi-hot in MY index order; avail
        (B, 90) bool. Returns (B, 90) log-probs over MY index order."""
        B = t0.shape[0]
        X = np.zeros((B, 3 * NUM_HEROES + 19), np.float32)
        sh = self.to_shared
        X[:, sh] = t0
        X[:, NUM_HEROES + sh] = t1
        X[:, 2 * NUM_HEROES + sh] = bans
        X[:, 3 * NUM_HEROES:3 * NUM_HEROES + 14] = map_oh
        X[:, 284:287] = tier_oh
        X[:, 287] = step_num / 15.0
        X[:, 288] = step_type
        with torch.no_grad():
            xt = torch.from_numpy(X)
            p = sum(torch.softmax(m(xt), dim=1) for m in self.models) / len(self.models)
        p = p.numpy()[:, sh]
        p = np.where(avail, p, 0.0)
        tot = p.sum(1, keepdims=True)
        p = np.where(tot > 0, p / np.maximum(tot, 1e-12), avail / np.maximum(avail.sum(1, keepdims=True), 1))
        return np.log(np.maximum(p, 1e-12))


# ------------------------------------------------------------------ WP evaluator

class WPEval:
    def __init__(self, hero_names):
        from drift2026 import common
        common.setup()
        from sweep_enriched_wp import WinProbEnrichedModel, FEATURE_GROUPS
        from drift2026.train_drift_wp import enriched_cols
        self.common = common
        self.cols = np.array(enriched_cols())
        self.mask = [True] * len(FEATURE_GROUPS)
        self.names = [str(h) for h in hero_names]
        self.models = []
        for s in (42, 123, 777):
            ck = torch.load(os.path.join(common.MODELS_DIR, f"d2c_cumprev_s{s}.pt"),
                            map_location="cpu", weights_only=False)
            m = WinProbEnrichedModel(ck["input_dim"], ck["arch"], dropout=ck["dropout"])
            m.load_state_dict(ck["state_dict"])
            m.eval()
            self.models.append(m)
        self.builds = common.load_patch_index()["builds"]
        w = np.load(C.WP)
        self.present = np.unique(w["build_idx"])
        self._stats = {}
        self.cache = {}

    def stats_for_build(self, bidx):
        pos = int(np.searchsorted(self.present, bidx))
        key = self.builds[self.present[pos - 1]]
        if key not in self._stats:
            self._stats[key] = self.common.load_patch_stats("cumulative", key)
            if len(self._stats) > 4:
                self._stats.pop(next(iter(self._stats)))
        return key, self._stats[key]

    def wp(self, drafts, bidx, game_map, tier):
        """drafts: list of (tuple team0 hero idx, tuple team1 hero idx) for one
        game context. Returns WP_pop (team 0 wins), cached."""
        from sweep_enriched_wp import extract_features, _swap_features
        key, st = self.stats_for_build(bidx)
        out = np.empty(len(drafts))
        todo, tidx = [], []
        for i, (a, b) in enumerate(drafts):
            ck = (key, game_map, tier, tuple(sorted(a)), tuple(sorted(b)))
            v = self.cache.get(ck)
            if v is None:
                todo.append(ck)
                tidx.append(i)
            else:
                out[i] = v
        if todo:
            X, Xs = [], []
            for ck in todo:
                dd = {"team0_heroes": [self.names[h] for h in ck[3]],
                      "team1_heroes": [self.names[h] for h in ck[4]],
                      "game_map": game_map, "skill_tier": tier, "winner": 0}
                b_, e_ = extract_features(dd, st, self.mask)
                bs, es = _swap_features(b_, e_)
                X.append(np.concatenate([b_, e_[self.cols]]))
                Xs.append(np.concatenate([bs, es[self.cols]]))
            X = torch.tensor(np.array(X), dtype=torch.float32)
            Xs = torch.tensor(np.array(Xs), dtype=torch.float32)
            with torch.no_grad():
                p = np.mean([((m(X).view(-1) + 1 - m(Xs).view(-1)) / 2).numpy()
                             for m in self.models], axis=0)
            for j, (ck, i) in enumerate(zip(todo, tidx)):
                self.cache[ck] = float(p[j])
                out[i] = p[j]
            if len(self.cache) > 400000:
                self.cache.clear()
        return out


def one_hots(game_map, tier):
    from shared import MAPS, SKILL_TIERS
    mo = np.zeros(14, np.float32)
    mo[MAPS.index(game_map)] = 1
    to = np.zeros(3, np.float32)
    to[SKILL_TIERS.index(tier)] = 1
    return mo, to




def build_personal_tables():
    """Personal tables for every slot of every lobby (cache/dr_personal_post.npz)."""
    from p3_hs_fit import prepare
    d = C.load_slots()
    _, _, _, table, r_adj = prepare(d)
    L = load_lobbies()
    rows = np.unique(L["steps"][:, :, 3][L["steps"][:, :, 3] >= 0])
    T = personal_tables(d, rows, table, kernel_best(), r_adj)
    np.savez(PTAB, rows=T["rows"], s=T["s"], sd=T["sd"].astype(np.float16), off=T["off"],
             pool=T["pool"], n=T["n"].astype(np.float32), n_p=T["n_p"])
    print(f"personal tables: {len(rows):,} slots")


def load_personal():
    z = np.load(PTAB)
    T = {k: z[k] for k in z.files}
    T["pos"] = {int(r): i for i, r in enumerate(T["rows"])}
    return T


if __name__ == "__main__":
    if "tables" in sys.argv:
        build_personal_tables()
    else:
        build_lobbies(C.load_slots())
