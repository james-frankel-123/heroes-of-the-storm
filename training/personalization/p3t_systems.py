"""
P3 tournament: drafting systems (actors) on real lobbies.

Conventions (cuda_personal / p3_mcts_core): kernel team 0 is the real
first-pick team; Storm League order (bans 0,1,0,1; picks 0,1,1,0,0; bans 1,0;
picks 1,1,0,0,1); hero indices in shared.HEROES order. A draft is 16
actions `acts` (kernel order, -1 = not yet taken).

Lobby rules, the same for every system:
  - a pick by team t must be in the union of its five players' pools
    (heroes each played before the lobby day); if that leaves nothing, any
    free hero; bans are free;
  - the team picks; personal values use each team's best assignment of its
    five heroes to its five players (team mode), so the real pick order
    (who picked what) is never used.

Vintages (p3t_vintages.json): the population side of a drafter (MCTS prior
policy seeds, leaf WP with its statistics, GD opponent models) and the
player state it reads (p3t_data state_<v>) plus its combiner
V = sigmoid(b0 + b1 logit WP + b2 (S_ours - S_theirs) + b3 (O_ours - O_theirs)).

Systems (SYSTEMS): kind x vintage x player-aware flag
  mcts    decision-time search in the personal kernel (leaf = vintage WP,
          plus the personal terms when aware), prior = the vintage's
          paper-1 F policy (seed by draft), root argmax of visits among the
          heroes paper 1's structural constraint allows (constrained MCTS)
  policy  the F policy argmax under the structural constraint, no search
          (paper 1's constrained MCTS tournament agent)
  onestep p3_dr_drafter's one-step search (top 15 GD + top 15 personal
          candidates, 4 GD rollouts each), WP = the vintage WP; picks under
          the structural constraint; bans GD argmax
  gd      GD argmax (5-model average): behavioral cloning
  real    the real lobby's actions, replayed when still legal (else GD argmax)
"""
import json
import os
import sys

os.environ.setdefault("P1_TIERS", "site")   # paper-1 artifacts of the site-tier rebuild

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING = os.path.dirname(HERE)
sys.path.insert(0, TRAINING)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(TRAINING, "cuda_mcts"))
sys.path.insert(0, os.path.join(TRAINING, "overfit2026"))
import numpy as np
import torch

H = 90
DRAFT_TEAM = [0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1]
IS_PICK = [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1]
SIMS = int(os.environ.get("P3T_SIMS", "400"))
VFILE = os.path.join(HERE, "p3t_vintages.json")


def tpath(p):
    return p if os.path.isabs(p) else os.path.join(TRAINING, p)


# ------------------------------------------------------------------ systems

def system_table():
    """name -> kind, vint (population side), aware, state (player state read
    by the personal terms; default the vintage's own)."""
    S = {}
    for v in ("cur", "s4m", "s1y"):
        S[f"pMCTS-{v}"] = dict(kind="mcts", vint=v, aware=0)
        S[f"aMCTS-{v}"] = dict(kind="mcts", vint=v, aware=1)
        S[f"pPol-{v}"] = dict(kind="policy", vint=v, aware=0)
    # search over the behavioral (BC) prior: same leaf WP, statistics, GD and sims; only the prior differs
    # prior trained with player features (p3t_pprior): search and prior-only
    S["aMCTS-pp"] = dict(kind="mcts_pp", vint="cur", aware=1, prior="pprior")
    # player-state upkeep for the strong player-aware drafter: state frozen at the vintage cutoff
    S["aMCTS-pp@s4m"] = dict(kind="mcts_pp", vint="cur", aware=1, prior="pprior", state="s4m")
    S["aMCTS-pp@s1y"] = dict(kind="mcts_pp", vint="cur", aware=1, prior="pprior", state="s1y")
    S["aPol-pp"] = dict(kind="policy_pp", vint="cur", aware=1, prior="pprior")
    # prior trained by self-play with sampled real rosters (p3t_selfplay)
    S["aMCTS-sp"] = dict(kind="mcts_pp", vint="cur", aware=1, prior="selfplay_prior")
    S["aPol-sp"] = dict(kind="policy_pp", vint="cur", aware=1, prior="selfplay_prior")
    # value net with per-lobby player features (p3t_vnet; cuda_pvnet), behavioral prior
    S["aMCTS-vn"] = dict(kind="mcts_vn", vint="cur_bc_vn", aware=1)
    S["pMCTS-bc"] = dict(kind="mcts", vint="cur_bc", aware=0)
    # split-history robustness: drafter state from odd-day evidence only (judge: even days)
    S["aMCTS-bc@split"] = dict(kind="mcts", vint="cur_bc_split", aware=1, state="splitd")
    S["a1step@split"] = dict(kind="onestep", vint="cur_split", aware=1, state="splitd")
    S["aMCTS-bc"] = dict(kind="mcts", vint="cur_bc", aware=1)
    for v in ("s4m", "s1y"):
        # upkeep options of the player-blind search drafter under drift
        S[f"pMCTS-{v}-rs"] = dict(kind="mcts", vint=f"{v}_rs", aware=0)   # stale weights, current stats tables
        S[f"pMCTS-{v}-ft"] = dict(kind="mcts", vint=f"{v}_ft", aware=0)   # fine-tuned from the stale weights
        # player-state upkeep: current population side, player state frozen at the vintage cutoff
        S[f"aMCTS-cur@{v}"] = dict(kind="mcts", vint="cur", aware=1, state=v)
    for v in ("cur", "s1y"):
        S[f"a1step-{v}"] = dict(kind="onestep", vint=v, aware=1)
        S[f"p1step-{v}"] = dict(kind="onestep", vint=v, aware=0)
    # DraftRec (Lee et al. 2022) reimplemented on our data (p3t_draftrec): pick head, and the paper's p+v rule
    S["DraftRec"] = dict(kind="draftrec", vint="cur", aware=1, mode="p")
    S["DraftRec-pv"] = dict(kind="draftrec", vint="cur", aware=1, mode="pv")
    # rich learned prior (p3t_rprior) by training signal: prior only
    for sig in ("bcow", "onestep", "search"):
        S[f"aPol-rich-{sig}"] = dict(kind="policy_rich", vint="cur", aware=1, signal=sig)
    S["GD"] = dict(kind="gd", vint="cur", aware=0)
    S["Real"] = dict(kind="real", vint="cur", aware=0)
    return S


def vintage_state(v):
    """The player state a vintage reads: its own cutoff (s4m_rs -> s4m; cur_bc -> cur)."""
    return v.split("_")[0]


XFILE = os.path.join(HERE, "p3t_explore.json")


def explore_systems():
    """Exploration-design systems (kind mcts_x) from p3t_explore.json:
    name -> {base, prior, bias_scale, alpha, c_puct, opp, ban_bias, optimism, assign_bias, sims}."""
    if not os.path.exists(XFILE):
        return {}
    cf = json.load(open(XFILE))
    out = {}
    for n, c in cf.items():
        if n.startswith("_"):
            continue
        out[n] = dict(kind="mcts_x", vint=c.get("base", "cur"), aware=1, x=c)
        if c.get("state"):
            out[n]["state"] = c["state"]
    return out


SYSTEMS = system_table()
SYSTEMS.update(explore_systems())


# ------------------------------------------------------------------ vintages

class Vintage:
    """Population resources of one vintage (lazy)."""

    def __init__(self, name, spec):
        self.name, self.spec = name, spec
        self._pol_flat, self._pol_net, self._gd, self._gd_flat = {}, {}, None, None
        self._wp = None

    @property
    def n_seeds(self):
        return len(self.spec["policies"])

    def stats(self):
        if not hasattr(self, "_st"):
            from overfit2026 import feats
            from paper1_revision import core
            st = feats.stats_from_json(tpath(self.spec["stats"]), compositions=False)
            st.comp_data = core.comps_from_json(tpath(self.spec["comps"]))
            self._st = st
        return self._st

    def wp_model(self):
        if self._wp is None:
            from sweep_enriched_wp import WinProbEnrichedModel
            from paper1_revision import core
            meta = json.load(open(tpath(self.spec["wp_json"])))
            cols = core.group_cols(meta["spec"]["groups"])
            m = WinProbEnrichedModel(len(cols), meta["spec"]["arch"], dropout=0.3)
            wdir = os.path.dirname(tpath(self.spec["wp_json"]))
            name = os.path.basename(self.spec["wp_json"])[:-5]
            m.load_state_dict(torch.load(os.path.join(wdir, f"{name}_s{meta['selected_seed']}.pt"),
                                         map_location="cpu", weights_only=True))
            m.eval()
            self._wp = (m, np.asarray(cols))
        return self._wp

    def wcfg(self):
        if not hasattr(self, "_wcfg"):
            from extract_weights import extract_wp_weights, build_wp_net_offsets, extract_lookup_tables
            m, cols = self.wp_model()
            assert len(cols) == 283
            wf, wn = extract_wp_weights(m)
            wo = build_wp_net_offsets(m, wn, 283, use_enriched=True)
            self._wcfg = (wf, wo, extract_lookup_tables(self.stats()))
        return self._wcfg

    def policy_flat(self, i):
        if i not in self._pol_flat:
            import search as S
            self._pol_flat[i] = S.policy_flat("path:" + tpath(self.spec["policies"][i]))
        return self._pol_flat[i]

    def policy_net(self, i):
        if i not in self._pol_net:
            import search as S
            self._pol_net[i] = S.policy_net("path:" + tpath(self.spec["policies"][i]))
        return self._pol_net[i]

    def gds(self):
        if self._gd is None:
            from train_generic_draft import GenericDraftModel
            from extract_weights import extract_gd_weights
            self._gd, self._gd_flat = [], []
            for p in self.spec["gds"]:
                g = GenericDraftModel()
                g.load_state_dict(torch.load(tpath(p), weights_only=True, map_location="cpu"))
                g.eval()
                self._gd.append(g)
                self._gd_flat.append(extract_gd_weights(g))
        return self._gd

    def gd_flat(self, i):
        self.gds()
        return self._gd_flat[i % len(self._gd_flat)]

    def wp_drafts(self, rows):
        """Symmetrized P(team0 wins) for rows [(team0 names, team1 names, map, tier)]."""
        from paper1_revision import core
        m, cols = self.wp_model()
        sd = core.stats_dict(self.stats())
        Xf, Xs = core._chunk((rows, sd))
        with torch.no_grad():
            a = m(torch.tensor(Xf[:, cols])).view(-1).numpy()
            b = m(torch.tensor(Xs[:, cols])).view(-1).numpy()
        return 0.5 * (a + 1 - b)


_VINT = {}


def vintage(name):
    if name not in _VINT:
        spec = json.load(open(VFILE))[name]
        _VINT[name] = Vintage(name, spec)
    return _VINT[name]


# ------------------------------------------------------------------ state features

def features(acts, k, map_i, tier_i, our_team):
    """AlphaZeroDraftNet / GD input (290-d) at step k for drafts (B,16)."""
    B = len(acts)
    X = np.zeros((B, 290), np.float32)
    for j in range(k):
        h = acts[:, j]
        if IS_PICK[j]:
            X[np.arange(B), (H if DRAFT_TEAM[j] else 0) + h] = 1
        else:
            X[np.arange(B), 2 * H + h] = 1
    X[np.arange(B), 3 * H + map_i] = 1
    X[np.arange(B), 3 * H + 14 + tier_i] = 1
    X[:, 287] = k / 15.0
    X[:, 288] = float(IS_PICK[k])
    X[:, 289] = our_team
    return X


def legal_mask(acts, k, team_pool):
    """(B, H) legal actions at step k; team_pool (B, 2, H) bool."""
    B = len(acts)
    free = np.ones((B, H), bool)
    for j in range(k):
        free[np.arange(B), acts[:, j]] = False
    if not IS_PICK[k]:
        return free
    m = free & team_pool[:, DRAFT_TEAM[k]]
    empty = ~m.any(1)
    m[empty] = free[empty]
    return m


# ------------------------------------------------------------------ structural mask (paper 1's constraint)

from functools import lru_cache

_BLIZZ = ("Tank", "Bruiser", "Healer", "Ranged Assassin", "Melee Assassin", "Support")
_STACK = {"Tank", "Melee Assassin", "Support", "Healer"}
_CAP = tuple(2 if r in _STACK else 5 for r in _BLIZZ)


def _hero_role():
    from shared import HEROES, HERO_ROLE_FINE, FINE_TO_BLIZZ_ROLE
    return np.array([_BLIZZ.index(FINE_TO_BLIZZ_ROLE.get(HERO_ROLE_FINE.get(h, ""), "Ranged Assassin"))
                     for h in HEROES], np.int64)


HROLE = _hero_role()


def _valid_final(c):
    if any(x > cap for x, cap in zip(c, _CAP)):
        return False
    return c[2] >= 1 and c[0] + c[1] >= 1 and c[3] >= 1


@lru_cache(maxsize=None)
def _feasible(cur, avail, k):
    """rerun2026.constrained_search._feasible (exact, memoized)."""
    if any(c > cap for c, cap in zip(cur, _CAP)):
        return False
    if k == 0:
        return _valid_final(cur)
    for i in range(6):
        if avail[i] == 0 or cur[i] + 1 > _CAP[i]:
            continue
        c = list(cur)
        c[i] += 1
        a = list(avail)
        a[i] -= 1
        a = [min(x, k - 1) for x in a]
        if _feasible(tuple(c), tuple(a), k - 1):
            return True
    return False


def struct_mask(acts, k, legal):
    """Paper 1's structural constraint at pick step k: a candidate is kept iff
    the team can still complete >= 1 healer, >= 1 frontline, >= 1 ranged and
    <= 2 of each stacking role from its legal heroes (team pools, free). Falls
    back to `legal` where nothing is feasible. (B, H) bool."""
    if not IS_PICK[k]:
        return legal
    t = DRAFT_TEAM[k]
    out = np.zeros_like(legal)
    for i in range(len(acts)):
        cur = [0] * 6
        nown = 0
        for j in range(k):
            if IS_PICK[j] and DRAFT_TEAM[j] == t:
                cur[HROLE[acts[i, j]]] += 1
                nown += 1
        after = 5 - nown - 1
        av = np.bincount(HROLE[legal[i]], minlength=6)
        for h in np.flatnonzero(legal[i]):
            r = HROLE[h]
            c = list(cur)
            c[r] += 1
            a = list(av)
            a[r] -= 1
            a = [min(x, after) for x in a]
            out[i, h] = _feasible(tuple(c), tuple(a), after)
        if not out[i].any():
            out[i] = legal[i]
    return out


def gd_probs(vint, X289, mask):
    with torch.no_grad():
        xt = torch.from_numpy(X289)
        p = sum(torch.softmax(g(xt), 1) for g in vint.gds()) / len(vint.gds())
    p = p.numpy() * mask
    tot = p.sum(1, keepdims=True)
    return np.where(tot > 0, p / np.maximum(tot, 1e-12), mask / mask.sum(1, keepdims=True))


# ------------------------------------------------------------------ actors

def act_gd(vint, ctx):
    k = ctx["k"]
    X = features(ctx["acts"], k, ctx["map"], ctx["tier"], DRAFT_TEAM[k])
    m = legal_mask(ctx["acts"], k, ctx["team_pool"])
    return gd_probs(vint, X[:, :289], m).argmax(1)


def act_policy(vint, ctx):
    k = ctx["k"]
    X = features(ctx["acts"], k, ctx["map"], ctx["tier"], DRAFT_TEAM[k])
    m = struct_mask(ctx["acts"], k, legal_mask(ctx["acts"], k, ctx["team_pool"]))
    out = np.empty(len(X), np.int64)
    for i in np.unique(ctx["seed_i"] % vint.n_seeds):
        sel = np.flatnonzero(ctx["seed_i"] % vint.n_seeds == i)
        net = vint.policy_net(int(i))
        with torch.no_grad():
            logits, _ = net(torch.from_numpy(X[sel]), torch.from_numpy(m[sel].astype(np.float32)))
        lg = logits.numpy()
        out[sel] = np.where(m[sel], lg, -1e30).argmax(1)
    return out


def act_real(vint, ctx):
    k = ctx["k"]
    real = ctx["real_acts"][:, k]
    m = legal_mask(ctx["acts"], k, np.ones_like(ctx["team_pool"]))
    ok = m[np.arange(len(real)), real]
    out = real.copy()
    if (~ok).any():
        sub = {kk: (v[~ok] if isinstance(v, np.ndarray) and v.shape[:1] == ok.shape else v) for kk, v in ctx.items()}
        out[~ok] = act_gd(vint, sub)
    ctx.setdefault("fallback", np.zeros(len(real), bool))[:] = ~ok
    return out


def kernel_module():
    import p3_mcts_core as M
    return M.personal_kernel()


def prior_path(name):
    import p3t_data as TD
    return os.path.join(TD.OUT, f"{name}.pt")


def act_policy_pp(vint, ctx, prior="pprior"):
    """argmax of the player-feature prior alpha log p_F + team bias (no search)."""
    import p3t_pprior as PP
    k = ctx["k"]
    if not IS_PICK[k]:
        return act_policy(vint, ctx)
    lp, m = PP.f_logp(ctx["acts"], k, ctx["map"], ctx["tier"], ctx["team_pool"], ctx["seed_i"])
    m = struct_mask(ctx["acts"], k, m)
    bias, alpha = PP.bias_for(ctx["s"], ctx["off"], ctx["n"], float(vint.spec["combiner"][2]), prior_path(prior))
    u = alpha * lp + bias[:, 5 * DRAFT_TEAM[k]]
    return np.where(m, u, -1e30).argmax(1)


def _assign_gain(s5, own_heroes, b2):
    """b2 * (best assignment value of the team with hero h added minus without),
    for every hero h (90,); s5 (5, 90) the team's players, own_heroes the team's picks so far."""
    import itertools
    m = len(own_heroes)
    best0 = 0.0
    if m:
        best0 = max(sum(s5[p[i], own_heroes[i]] for i in range(m)) for p in itertools.permutations(range(5), m))
    perms = np.array(list(itertools.permutations(range(5), m + 1)), np.int64)        # (P, m+1)
    base = np.zeros(len(perms))
    for i, h in enumerate(own_heroes):
        base += s5[perms[:, i], h]
    tot = base[:, None] + s5[perms[:, m], :]                                          # (P, 90)
    return b2 * (tot.max(0) - best0)


def opp_model_path():
    import p3_hs_core as C
    return os.path.join(C.RESULTS, "p3t", "opp_model.json")


def opp_bias(n5, w):
    """Opponent pick propensity bias (90,) for a team's 5 players: log mean_i
    exp(w1 log1p n_ih + w2 [n_ih == 0]); the team mixture (who picks is unknown)."""
    u = w["w1"] * np.log1p(n5) + w["w2"] * (n5 == 0)
    m = u.max(0)
    return m + np.log(np.exp(u - m).mean(0))


def act_mcts_x(vint, ctx, x, device=0):
    """Decide-only search with the exploration design x (see explore_systems)."""
    import p3_mcts_core as M
    k = ctx["k"]
    t = DRAFT_TEAM[k]
    B = len(ctx["acts"])
    use_ex = (bool(x.get("opp")) or bool(x.get("ban_bias")) or bool(x.get("ban_dr"))
              or (bool(x.get("rich")) and not IS_PICK[k]) or "stranger" in ctx)
    kmod = M.load_module("explore_kernel" if use_ex else "pvnet_kernel",
                         os.path.join(HERE, "cuda_explore" if use_ex else "cuda_pvnet"))
    cfg = np.full((B, M.CFG_LEN), -1, np.int32)
    cfg[:, 0], cfg[:, 1], cfg[:, 2] = ctx["map"], ctx["tier"], t
    cfg[:, 3], cfg[:, 4], cfg[:, 5], cfg[:, 6], cfg[:, 7] = 0, k, 1, 0, 1
    cfg[:, 8:24] = np.where(np.arange(16)[None] < k, ctx["acts"], -1)
    cfg[:, 24:40] = ctx["slot_of"]
    cfg[:, 40:56] = -1
    if "stranger" in ctx:
        # party mode (p3t_party.py): our steps taken by stranger teammates are chance nodes
        cfg[:, 40:56] = np.where(ctx["stranger"], -2, -1)
    cfg[:, 56:M.CFG_LEN] = 0
    cfg[:, M.CFG_ASSIGN] = 1
    s = ctx["s"].astype(np.float32).copy()
    off = ctx["off"].astype(np.float32)
    b = list(vint.spec["combiner"])
    # (e) uncertainty-aware value: optimism on our cells, pessimism on theirs
    ku = float(x.get("optimism", 0.0))
    if ku:
        sd = ctx["sd"].astype(np.float32)
        s[:, 5 * t:5 * t + 5] += ku * sd[:, 5 * t:5 * t + 5]
        s[:, 5 * (1 - t):5 * (1 - t) + 5] -= ku * sd[:, 5 * (1 - t):5 * (1 - t) + 5]
    bias = np.zeros((B, 10, H), np.float32)
    alpha = x.get("alpha", 1.0)
    if isinstance(alpha, dict):
        alpha = float(alpha.get(str(k), alpha.get("default", 1.0)))
    if x.get("prior"):
        import p3t_pprior as PP
        pb, a_fit = PP.bias_for(ctx["s"], ctx["off"], ctx["n"], float(b[2]), prior_path(x["prior"]))
        bias += float(x.get("bias_scale", 1.0)) * pb
        if x.get("alpha") is None:
            alpha = a_fit
    la = float(x.get("assign_bias", 0.0))
    if la and IS_PICK[k]:
        for i in range(B):
            own = [int(ctx["acts"][i, j]) for j in range(k) if IS_PICK[j] and DRAFT_TEAM[j] == t]
            g = _assign_gain(ctx["s"][i, 5 * t:5 * t + 5].astype(np.float64), own, float(b[2]))
            bias[i, 5 * t:5 * t + 5] += la * g[None, :].astype(np.float32)
    rich = x.get("rich")
    use_rich_ban = False
    if rich:
        import p3t_rprior as RP
        u, lpF, msk = RP.logits(rich, ctx["acts"], k, ctx["map"], ctx["tier"], ctx["s"].astype(np.float32),
                                ctx["sd"].astype(np.float32), ctx["n"].astype(np.float32), ctx["seed_i"], float(b[2]))
        rb = np.where(msk, u - lpF, 0.0).astype(np.float32)
        alpha = 1.0                                    # root prior = softmax(rich logits) exactly
        if IS_PICK[k]:
            bias[:, 5 * t:5 * t + 5] += rb[:, None, :]
        else:
            use_rich_ban = True
            r = 5 * (1 - t)
            bias[:, r] += rb
            cfg[:, 24 + k] = r
    if x.get("prior") or la or (rich and IS_PICK[k]):
        cfg[:, 56] = 1
    if use_rich_ban:
        cfg[:, 56] = 2
    imit = np.zeros((B, 10, H), np.float32)
    coefs = np.array(b + [1.0, alpha], np.float32)
    dr_deny = None
    if x.get("opp") and ctx.get("imit_override") is not None:
        # party mode: per-player DraftRec rows for the opponents and the stranger teammates (p3t_party.py)
        imit, w0 = ctx["imit_override"]
        imit = np.ascontiguousarray(imit, np.float32)
        cfg[:, 6] = 3 if x.get("opp_chance_only") else 2
        coefs[4] = w0
    elif x.get("opp") == "dr":
        # DraftRec opponent (p3t_oppdr.py): static per-decision rows read at the root;
        # opponent picks on their kernel slot rows, opponent bans on rows of our team
        import p3t_oppdr as OD
        w = json.load(open(OD.out_file()))               # fails loudly if the fit is missing
        imit, ban_rows, dr_deny = OD.rows(ctx, t, k, w)
        for s_, r_ in ban_rows.items():
            cfg[:, 24 + s_] = r_
        cfg[:, 6] = 3 if x.get("opp_chance_only") else 2
        coefs[4] = w["w0"]
    elif x.get("opp"):
        w = json.load(open(opp_model_path()))           # p3t_oppmodel.py; fails loudly if missing
        for i in range(B):
            ob = opp_bias(ctx["n"][i, 5 * (1 - t):5 * (1 - t) + 5].astype(np.float32), w)
            imit[i, 5 * (1 - t):5 * (1 - t) + 5] = float(x["opp"]) * ob[None, :]
        cfg[:, 6] = 3 if x.get("opp_chance_only") else 2
        coefs[4] = w["w0"]
    lbd = float(x.get("ban_dr", 0.0))
    if lbd and dr_deny is None:
        import p3t_oppdr as OD
        _, _, dr_deny = OD.rows(ctx, t, k, {"lp": 0.0, "lb": 0.0})
    if lbd and dr_deny is not None:
        # denial bans aimed by DraftRec: log-probability of the opponent's next pick (read at the root)
        r = 5 * (1 - t)
        bias[:, r] += lbd * dr_deny
        for j in range(16):
            if not IS_PICK[j] and DRAFT_TEAM[j] == t:
                cfg[:, 24 + j] = r
        cfg[:, 56] = 2
    lb = float(x.get("ban_bias", 0.0))
    if lb:
        # our bans: denial bias in the (otherwise unused) first row of the opponent team:
        # their best player's personal term on each hero (b2 * max_i s_ih)
        r = 5 * (1 - t)
        bias[:, r] += lb * float(b[2]) * ctx["s"][:, 5 * (1 - t):5 * (1 - t) + 5].max(1)   # adds to a rich ban bias
        for j in range(16):
            if not IS_PICK[j] and DRAFT_TEAM[j] == t:
                cfg[:, 24 + j] = r
        cfg[:, 56] = 2
    feat = np.zeros((B, 10, H), np.float32)
    vnet = np.zeros(0, np.float32)
    if x.get("vnet"):
        # value net leaf with player features (cuda_pvnet / cuda_explore cfg[57] = 1)
        vs_ = vintage(x["vnet"]).spec
        vnet = np.asarray(vs_["vnet"], np.float32)
        feat = np.log1p(ctx["n"].astype(np.float32))
        coefs[1] = vs_["vnet_b1"]
        cfg[:, 57] = 1
    if x.get("blind"):
        cfg[:, 7] = 0                                    # population leaf only (the pMCTS-cur search through this kernel)
    sims = int(x.get("sims", SIMS))
    cp = float(x.get("c_puct", 2.0))
    out = np.empty(B, np.int64)
    wf, wo, lut = vint.wcfg()
    seeds = ctx["seed_i"] % vint.n_seeds
    Eng = getattr(kmod, "PersonalEngine", None) or getattr(kmod, "PriorEngine")
    for i in np.unique(seeds):
        sel = np.flatnonzero(seeds == i)
        pf, po = vint.policy_flat(int(i))
        gf, go = vint.gd_flat(int(i))
        for b0 in range(0, len(sel), 256):
            ii = sel[b0:b0 + 256]
            eng = Eng(pf, gf, wf, po, go, wo, lut, max_concurrent=len(ii), device_id=device)
            r_ = eng.run(np.ascontiguousarray(cfg[ii]), np.ascontiguousarray(s[ii]), np.ascontiguousarray(off[ii]),
                         np.ascontiguousarray(imit[ii]), np.ascontiguousarray(ctx["pool_bits"][ii]), coefs,
                         np.ascontiguousarray(bias[ii]), np.ascontiguousarray(feat[ii]), vnet, sims, cp,
                         int(ctx["kseed"] + 7919 * k + b0 + 104729 * int(i)), 0.0, 0.3, 0.0, 1, search_mode=1)
            del eng
            pol = r_[5][:, 0, :]
            q = np.nan_to_num(r_[6][:, 0, :], nan=-1.0)
            legal = legal_mask(ctx["acts"][ii], k, ctx["team_pool"][ii])
            sm = struct_mask(ctx["acts"][ii], k, legal)
            out[ii] = np.where(sm, pol + 1e-6 * q, -1e30).argmax(1)
            assert legal[np.arange(len(ii)), out[ii]].all(), "illegal action"
    return out


def act_policy_rich(vint, ctx, signal):
    """argmax of the rich prior (structural mask at picks)."""
    import p3t_rprior as RP
    k = ctx["k"]
    b2 = float(vint.spec["combiner"][2])
    u, _, msk = RP.logits(signal, ctx["acts"], k, ctx["map"], ctx["tier"], ctx["s"].astype(np.float32),
                          ctx["sd"].astype(np.float32), ctx["n"].astype(np.float32), ctx["seed_i"], b2)
    m = struct_mask(ctx["acts"], k, legal_mask(ctx["acts"], k, ctx["team_pool"]))
    return np.where(m & msk, u, -1e30).argmax(1)


def act_mcts(vint, ctx, aware, device=0, kmod=None, pp=False, prior="pprior", vn=False):
    """Decide-only search at step k for each draft (grouped by prior seed)."""
    import p3_mcts_core as M
    kmod = kmod or kernel_module()
    k = ctx["k"]
    t = DRAFT_TEAM[k]
    B = len(ctx["acts"])
    cfg = np.full((B, M.CFG_LEN), -1, np.int32)
    cfg[:, 0], cfg[:, 1], cfg[:, 2] = ctx["map"], ctx["tier"], t
    cfg[:, 3], cfg[:, 4], cfg[:, 5], cfg[:, 6], cfg[:, 7] = 0, k, 1, 0, aware
    cfg[:, 8:24] = np.where(np.arange(16)[None] < k, ctx["acts"], -1)
    cfg[:, 24:40] = ctx["slot_of"]
    cfg[:, 40:56] = -1
    cfg[:, 56:M.CFG_LEN] = 0
    cfg[:, M.CFG_ASSIGN] = 1
    s, off = ctx["s"].astype(np.float32), ctx["off"].astype(np.float32)
    imit = np.zeros_like(s)
    pool = ctx["pool_bits"]
    coefs = np.array(list(vint.spec["combiner"]) + [0.0], np.float32)
    if pp:
        import p3t_pprior as PP
        import p3_mcts_core as M2
        kmod = M2.load_module("prior_kernel", os.path.join(HERE, "cuda_prior"))
        bias, alpha = PP.bias_for(s, off, ctx["n"], float(vint.spec["combiner"][2]), prior_path(prior))
        coefs = np.array(list(vint.spec["combiner"]) + [0.0, alpha], np.float32)
        cfg[:, 56] = 1
    if vn:
        import p3_mcts_core as M2
        kmod = M2.load_module("pvnet_kernel", os.path.join(HERE, "cuda_pvnet"))
        cfg[:, 57] = 1
        feat = np.log1p(ctx["n"].astype(np.float32))
        vnet = np.asarray(vint.spec["vnet"], np.float32)
        cb = list(vint.spec["combiner"])
        coefs = np.array([0.0, vint.spec["vnet_b1"], cb[2], cb[3], 0.0, 1.0], np.float32)
        bias = np.zeros_like(s)
    out = np.empty(B, np.int64)
    wf, wo, lut = vint.wcfg()
    seeds = ctx["seed_i"] % vint.n_seeds
    for i in np.unique(seeds):
        sel = np.flatnonzero(seeds == i)
        pf, po = vint.policy_flat(int(i))
        gf, go = vint.gd_flat(int(i))
        for b0 in range(0, len(sel), 256):
            ii = sel[b0:b0 + 256]
            eng = (getattr(kmod, "PersonalEngine", None) or kmod.PriorEngine)(pf, gf, wf, po, go, wo, lut, max_concurrent=len(ii), device_id=device)
            extra = (np.ascontiguousarray(bias[ii]),) if pp else ()
            if vn:
                extra = (np.ascontiguousarray(bias[ii]), np.ascontiguousarray(feat[ii]), vnet)
            r = eng.run(np.ascontiguousarray(cfg[ii]), np.ascontiguousarray(s[ii]), np.ascontiguousarray(off[ii]),
                        np.ascontiguousarray(imit[ii]), np.ascontiguousarray(pool[ii]), coefs, *extra, SIMS, 2.0,
                        int(ctx["kseed"] + 7919 * k + b0 + 104729 * int(i)), 0.0, 0.3, 0.0, 1,
                        search_mode=1)
            del eng
            pol = r[5][:, 0, :]
            q = np.nan_to_num(r[6][:, 0, :], nan=-1.0)
            legal = legal_mask(ctx["acts"][ii], k, ctx["team_pool"][ii])
            sm = struct_mask(ctx["acts"][ii], k, legal)
            # root choice under paper 1's constraint: most visits among allowed
            # heroes; ties and unvisited allowed heroes broken by Q
            score = np.where(sm, pol + 1e-6 * q, -1e30)
            out[ii] = score.argmax(1)
            bad = ~legal[np.arange(len(ii)), out[ii]]
            assert not bad.any(), "illegal action"
            ctx.setdefault("root_changed_by_mask", np.zeros(B, bool))[ii] = out[ii] != np.where(legal, pol, -1.0).argmax(1)
    return out


class OneStep:
    """p3_dr_drafter's one-step search with the vintage WP and GD (CPU pool)."""

    def __init__(self, vint_name, procs):
        self.v = vint_name
        self.procs = procs
        self.pool = None

    def start(self):
        import multiprocessing as mp
        if self.pool is None:
            self.pool = mp.get_context("fork").Pool(self.procs, initializer=_os_init, initargs=(self.v,))

    def act(self, ctx, aware):
        self.start()
        k = ctx["k"]
        if not IS_PICK[k]:
            return act_gd(vintage(self.v), ctx)
        tasks = [(ctx["dr_lob"][i], ctx["acts"][i], k, int(ctx["kseed"] + 31 * k + i), aware)
                 for i in range(len(ctx["acts"]))]
        return np.array(self.pool.map(_os_decide, tasks, chunksize=2), np.int64)

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None


_OS = {}


class _GDPol:
    """GDPolicy interface of p3_dr_core over the vintage's GD models (shared order)."""

    def __init__(self, vint):
        self.models = vint.gds()
        self.to_shared = np.arange(H)

    def logprobs(self, t0, t1, bans, map_oh, tier_oh, step_num, step_type, avail):
        B = t0.shape[0]
        X = np.zeros((B, 289), np.float32)
        X[:, :H], X[:, H:2 * H], X[:, 2 * H:3 * H] = t0, t1, bans
        X[:, 3 * H:3 * H + 14] = map_oh
        X[:, 284:287] = tier_oh
        X[:, 287] = step_num / 15.0
        X[:, 288] = step_type
        with torch.no_grad():
            p = sum(torch.softmax(m(torch.from_numpy(X)), 1) for m in self.models) / len(self.models)
        p = np.where(avail, p.numpy(), 0.0)
        tot = p.sum(1, keepdims=True)
        p = np.where(tot > 0, p / np.maximum(tot, 1e-12), avail / np.maximum(avail.sum(1, keepdims=True), 1))
        return np.log(np.maximum(p, 1e-12))


class _WPv:
    """WPEval interface of p3_dr_core over the vintage WP (shared order), cached."""

    def __init__(self, vint):
        from shared import HEROES
        self.v = vint
        self.names = HEROES
        self.cache = {}

    def wp(self, drafts, bidx, game_map, tier):
        out = np.empty(len(drafts))
        todo, idx = [], []
        for i, (a, b) in enumerate(drafts):
            ck = (game_map, tier, tuple(sorted(a)), tuple(sorted(b)))
            v = self.cache.get(ck)
            if v is None:
                todo.append(ck)
                idx.append(i)
            else:
                out[i] = v
        if todo:
            rows = [([self.names[h] for h in ck[2]], [self.names[h] for h in ck[3]], game_map, tier) for ck in todo]
            p = self.v.wp_drafts(rows)
            for ck, i, pv in zip(todo, idx, p):
                self.cache[ck] = float(pv)
                out[i] = pv
            if len(self.cache) > 300000:
                self.cache.clear()
        return out


def _os_init(vname):
    torch.set_num_threads(1)
    import p3_dr_drafter as DR
    v = vintage(vname)
    DR._W.update({"gd": _GDPol(v), "wp": _WPv(v), "iw0": 0.0})
    DR.ASSIGN = "team"
    _OS["DR"] = DR


def _os_decide(task):
    lob, acts, k, seed, aware = task
    DR = _OS["DR"]
    DR._W["b"] = np.asarray(lob["b"], float)
    st = {"t0": {}, "t1": {}, "bans": set(), "taken": np.zeros(H, bool)}
    for j in range(k):
        st = DR._apply_ban(st, int(acts[j])) if not IS_PICK[j] else DR._apply(st, j, int(acts[j]), lob)
    rng = np.random.RandomState(seed)
    dd = DR._decide(st, k, lob, rng, "gd")
    legal = legal_mask(np.asarray(acts)[None], k, np.stack([lob["team_pool"][0], lob["team_pool"][1]])[None])
    sm = struct_mask(np.asarray(acts)[None], k, legal)[0]
    ok = sm[dd["cand"]]
    val = np.asarray(dd["V"] if aware else dd["wp"], float)
    if ok.any():
        val = np.where(ok, val, -1e9)
    return int(dd["cand"][int(np.argmax(val))])


def dr_lobby(kl, s, off, pool, b):
    """p3_dr_drafter lobby payload in KERNEL team labels (team 0 = first pick).
    Rows are kernel slots 0-9 (team t: 5t .. 5t+4; each pick step gets the
    next slot of its team, a placeholder: team mode never uses it)."""
    from shared import MAPS, SKILL_TIERS
    steps, cnt = [], [0, 0]
    for k in range(16):
        t = DRAFT_TEAM[k]
        if IS_PICK[k]:
            steps.append((-1, 1, t, 5 * t + cnt[t]))
            cnt[t] += 1
        else:
            steps.append((-1, 0, t, -1))
    mo = np.zeros(14, np.float32)
    mo[kl["map"]] = 1
    to = np.zeros(3, np.float32)
    to[kl["tier"]] = 1
    team_rows = {t: list(range(5 * t, 5 * t + 5)) for t in (0, 1)}
    return {"steps": steps, "map": MAPS[kl["map"]], "tier": SKILL_TIERS[kl["tier"]], "bidx": 0,
            "mo": mo, "to": to, "s": {r: s[r] for r in range(10)},
            "off": {r: off[r].astype(float) for r in range(10)},
            "pool": {r: pool[r] for r in range(10)}, "team_rows": team_rows,
            "team_pool": {t: np.any(pool[5 * t:5 * t + 5], axis=0) for t in (0, 1)}, "b": list(b)}
