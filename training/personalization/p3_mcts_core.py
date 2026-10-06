"""
P3 personalized MCTS: host side for cuda_personal/personal_kernel.

Kernel conventions
  - Draft order is fixed (team 0 picks first):
      step  0-3  bans   team 0,1,0,1
      step  4-8  picks  team 0,1,1,0,0
      step  9-10 bans   team 1,0
      step 11-15 picks  team 1,1,0,0,1
    This is Storm League order. Real lobbies where the real team 1 picked
    first are relabeled: kernel team 0 = the real first-picking team.
  - Pick-step -> player (slot) rule: the player at a pick step is the player
    who picked at that step in the real draft (draft_order). Kernel slots
    0-4 are kernel team 0's players in the order of their picks, 5-9 kernel
    team 1's.
  - Assign mode (cfg[CFG_ASSIGN] = 1, hosts' --assign team): the slot rule
    above uses who ended up playing the hero (after in-draft trades), mild
    oracle information. In assign mode the TEAM picks: candidates at a pick
    step are the union of the acting team's 5 slot pools, and each team's
    personal terms S_t, O_t come from the assignment of its 5 heroes to its
    5 slots maximizing b2 S_t + b3 O_t (both teams assign for themselves;
    first maximum over permutations in lexicographic order, identity = the
    pick order first). The kernel values complete drafts only (rollouts run
    to the end). step_slot still selects the per-slot PUCT prior bias,
    imitation bias and personalized-GD pick term at a pick step (priors and
    opponent model, not valuation).
  - Hero indices are the shared.HEROES order used by all kernels.
  - Leaf value: population WP of the complete draft = the 3 d2c_cumprev
    seeds combined as a mean-logit net (the overfit2026 K-member path) per
    orientation, then swap-symmetrized; plus the personal logit terms.

Personal tensors per lobby (10 slots x 90 heroes): s (experience offset +
pooled skill), off-role indicator, imitation bias; pool bitsets.
"""
import os
import sys

TRAINING = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TRAINING)
sys.path.insert(0, os.path.join(TRAINING, "cuda_mcts"))
sys.path.insert(0, os.path.join(TRAINING, "overfit2026"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
import numpy as np
from p3_heroes import NUM_HEROES
from p3_assign import best_assignment, team_picks, assignment_terms  # noqa: F401 (re-exported)
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
KDIR = os.path.join(HERE, "cuda_personal")
CFG_LEN = 59      # shared by personal / prior / pgd kernels; [56], [57] kernel-specific flags
CFG_ASSIGN = 58   # 0 slot mode (pick-step -> player), 1 assign mode (team pick + best assignment)
DRAFT_TEAM = [0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1]
IS_PICK = [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1]


SEARCH_MODES = {"legacy": 0, "chance": 1, "rollfwd": 2}
_SEARCH_MODE = None


def set_search_mode(name):
    """Every host sets the search mode once from its required --search-mode."""
    global _SEARCH_MODE
    if name not in ("chance", "rollfwd"):
        raise SystemExit(f"search mode {name!r}: use chance (headline) or rollfwd")
    env = os.environ.get("P3_SEARCH_MODE")
    if env not in (None, name):
        raise SystemExit(f"--search-mode {name} but P3_SEARCH_MODE={env}")
    os.environ["P3_SEARCH_MODE"] = name  # readers (result_path, the distilled prior path) follow it
    _SEARCH_MODE = name


def search_mode():
    """Kernel search_mode for engine.run; refuses to run before set_search_mode."""
    if _SEARCH_MODE is None:
        raise SystemExit("search mode not set: pass --search-mode chance|rollfwd")
    return SEARCH_MODES[_SEARCH_MODE]


def search_mode_name():
    return _SEARCH_MODE


ASSIGN_MODES = {"slot": 0, "team": 1}
_ASSIGN_MODE = "slot"


def set_assign_mode(name):
    """Hosts set this from --assign {slot,team} (default slot)."""
    global _ASSIGN_MODE
    if name not in ASSIGN_MODES:
        raise SystemExit(f"assign mode {name!r}: use slot or team")
    env = os.environ.get("P3_ASSIGN")
    if env not in (None, name):
        raise SystemExit(f"--assign {name} but P3_ASSIGN={env}")
    os.environ["P3_ASSIGN"] = name
    _ASSIGN_MODE = name


def assign_mode():
    """cfg[CFG_ASSIGN] value for the current assign mode."""
    return ASSIGN_MODES[_ASSIGN_MODE]


def assign_mode_name():
    return _ASSIGN_MODE


def out_path(fname):
    """Search outputs of the fixed kernel live apart from the legacy-tree
    caches: cache/mcts_v2/<search mode>/<fname>. Written atomically."""
    d = os.path.join(HERE, "cache", "mcts_v2", search_mode_name() or "unset")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, fname)


def save_pickle(obj, fname):
    """Assign-mode (team) outputs get an _assign-team suffix."""
    import gzip
    import pickle
    if _ASSIGN_MODE != "slot":
        base = fname[:-len(".pkl.gz")] if fname.endswith(".pkl.gz") else fname
        fname = f"{base}_assign-{_ASSIGN_MODE}.pkl.gz"
    p = out_path(fname)
    with gzip.open(p + ".tmp", "wb") as f:
        pickle.dump(obj, f)
    os.replace(p + ".tmp", p)
    return p


class BatchCache:
    """Pause-robust search runs: results of finished batches are kept in
    cache/mcts_v2/<mode>/partial_<tag>[_assign-team].pkl.gz (written at most
    every 5 minutes and on SIGTERM), so a resumed run redoes only the
    batches in flight. Results do not depend on batch order (each batch has
    its own seed and GD model), so a resumed run equals an uninterrupted one."""

    def __init__(self, tag):
        import gzip
        import pickle
        import signal
        import time
        sfx = "" if _ASSIGN_MODE == "slot" else f"_assign-{_ASSIGN_MODE}"
        self.path = out_path(f"partial_{tag}{sfx}.pkl.gz")
        self.d = {}
        if os.path.exists(self.path):
            with gzip.open(self.path, "rb") as f:
                self.d = pickle.load(f)
            print(f"resume: {len(self.d)} finished batches from {os.path.basename(self.path)}", flush=True)
        self.t = time.time()
        self._time = time.time
        prev = signal.getsignal(signal.SIGTERM)

        def on_term(signum, frame):
            self.flush()
            if callable(prev):
                prev(signum, frame)
            raise SystemExit(143)
        signal.signal(signal.SIGTERM, on_term)

    def get(self, k):
        return self.d.get(k)

    def put(self, k, v):
        self.d[k] = v
        if self._time() - self.t > 300:
            self.flush()

    def flush(self):
        import gzip
        import pickle
        with gzip.open(self.path + ".tmp", "wb") as f:
            pickle.dump(self.d, f)
        os.replace(self.path + ".tmp", self.path)
        self.t = self._time()

    def done(self):
        if os.path.exists(self.path):
            os.remove(self.path)


SEARCH_OUTPUTS = ("mcts_", "dsmcts_", "ds_targets_", "pgdmcts_")


def result_path(name):
    """Where analysis scripts read a search output: with P3_SEARCH_MODE set
    (chance or rollfwd; P3_ASSIGN slot or team, default team), the fixed-
    kernel file cache/mcts_v2/<mode>/<name>[_assign-team]; otherwise the
    legacy cache/<name>."""
    import p3_hs_core as C
    mode = os.environ.get("P3_SEARCH_MODE")
    if not mode or not name.startswith(SEARCH_OUTPUTS):
        return os.path.join(C.CACHE, name)
    assign = os.environ.get("P3_ASSIGN", "team")
    if "_collapse_" in name:
        assign = "slot"  # collapse probes ask what one given player gets: always slot mode
    if assign != "slot":
        name = f"{name[:-len('.pkl.gz')]}_assign-{assign}.pkl.gz"
    return os.path.join(C.CACHE, "mcts_v2", mode, name)


def team_mixture(x):
    """Assign mode: who on a team makes a given pick is not known before the
    draft, so a per-slot behavioral term (imitation bias, distilled prior
    bias) is replaced, for every slot of a team, by the team mixture
    log(mean_i exp(x_i)) over its five players. x: (10, H)."""
    out = np.empty_like(x)
    for t in (0, 1):
        blk = x[5 * t:5 * t + 5].astype(np.float64)
        m = blk.max(0)
        out[5 * t:5 * t + 5] = (m + np.log(np.exp(blk - m).mean(0)))[None, :]
    return out


def clear_partials(tag):
    """Remove a finished run's batch caches (a later run must not reuse them)."""
    d = os.path.dirname(out_path("x"))
    for f in os.listdir(d):
        if f.startswith(f"partial_{tag}_"):
            os.remove(os.path.join(d, f))


def load_module(name, d):
    """Load a P3 kernel build. Refuses builds without the v2 search or whose
    composition fallback is not in the WP's tier order (mid, high, low):
    rebuild with personalization/build_p3_kernels.sh."""
    so = [f for f in os.listdir(d) if f.startswith(name) and f.endswith(".so")]
    if len(so) != 1:
        raise SystemExit(f"{d}: expected one {name}*.so, found {so}")
    spec = importlib.util.spec_from_file_location(name, os.path.join(d, so[0]))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    order = getattr(mod, "COMP_FALLBACK_ORDER", None)
    if order != "mid_high_low":
        raise SystemExit(f"{name}: COMP_FALLBACK_ORDER is {order}; rebuild with "
                         "HOTS_COMP_FALLBACK_MID_HIGH_LOW=1 (personalization/build_p3_kernels.sh)")
    if getattr(mod, "SEARCH_CHANCE", None) != 1:
        raise SystemExit(f"{name}: build predates the v2 (chance) search")
    return mod


def personal_kernel():
    return load_module("personal_kernel", KDIR)


def ref_kernel():
    return load_module("pop_ref_kernel", os.path.join(KDIR, "ref"))


class Weights:
    """Policy prior (bc), GD models, WP ensemble + LUT per stats key."""

    def __init__(self):
        from drift2026 import common
        common.setup()
        import search as S
        from sweep_enriched_wp import WinProbEnrichedModel
        self.S = S
        self.common = common
        import p3_dr_core as D
        from train_generic_draft import GenericDraftModel
        from extract_weights import extract_gd_weights
        # prior and in-tree GD pool: train-window models unless P3_GD=paper1
        self.policy = S.policy_flat("path:" + D.bc_prior_path())
        self.gds = []
        for i in range(5):
            g = GenericDraftModel()
            g.load_state_dict(torch.load(os.path.join(D.gd_dir(), f"generic_draft_{i}.pt"),
                                         weights_only=True, map_location="cpu"))
            self.gds.append(extract_gd_weights(g.eval()))
        self.models = []
        for s in (42, 123, 777):
            ck = torch.load(os.path.join(common.MODELS_DIR, f"d2c_cumprev_s{s}.pt"), map_location="cpu",
                            weights_only=False)
            m = WinProbEnrichedModel(ck["input_dim"], ck["arch"], dropout=ck["dropout"])
            m.load_state_dict(ck["state_dict"])
            m.eval()
            self.models.append(m)
        self.builds = common.load_patch_index()["builds"]
        w = np.load(os.path.join(HERE, "cache", "wp_drift.npz"))
        self.present = np.unique(w["build_idx"])
        self._wcfg = {}

    def stats_key(self, bidx):
        pos = int(np.searchsorted(self.present, bidx))
        return self.builds[self.present[pos - 1]]

    def wp_cfg(self, key):
        if key not in self._wcfg:
            st = self.common.load_patch_stats("cumulative", key)
            self._wcfg[key] = (self.S.wp_config(self.models, st, lcb_lambda=0.0), st)
            if len(self._wcfg) > 3:
                self._wcfg.pop(next(iter(self._wcfg)))
        return self._wcfg[key]


def make_engine(kmod, W, key, gd_i, max_conc, device):
    (wf, wo, lut), _ = W.wp_cfg(key)
    pf, po = W.policy
    gf, go = W.gds[gd_i % len(W.gds)]
    cls = getattr(kmod, "PersonalEngine", None) or getattr(kmod, "OfitEngine")
    return cls(pf, gf, wf, po, go, wo, lut, max_concurrent=max_conc, device_id=device)


def empty_cfg(n):
    c = np.full((n, CFG_LEN), -1, np.int32)
    c[:, 3:8] = 0
    c[:, 4] = -1
    c[:, 56:CFG_LEN] = 0
    return c


def zeros_personal(n):
    return (np.zeros((n, 10, NUM_HEROES), np.float32), np.zeros((n, 10, NUM_HEROES), np.float32),
            np.zeros((n, 10, NUM_HEROES), np.float32), np.zeros((n, 10, 3), np.uint32))


def pool_bits(mask90):
    b = np.zeros(3, np.uint32)
    for h in np.flatnonzero(mask90):
        b[h // 32] |= np.uint32(1 << (h % 32))
    return b


def reference_value(W, key, map_name, tier, t0_names, t1_names, our_team, S, O, coefs,
                    assign=False, s=None, off=None, acts=None):
    """float64 mirror of the kernel leaf: mean-logit ensemble per
    orientation, symmetrized, our side; then personal terms. assign=True:
    S, O are ignored and recomputed by the best assignment (assignment_terms)
    from s, off (10, NUM_HEROES) and the 16 kernel-order actions acts."""
    if assign:
        S, O, _ = assignment_terms(s, off, acts, float(coefs[2]), float(coefs[3]))
    from sweep_enriched_wp import extract_features, _swap_features, FEATURE_GROUPS
    from drift2026.train_drift_wp import enriched_cols
    _, st = W.wp_cfg(key)
    cols = np.array(enriched_cols())
    dd = {"team0_heroes": t0_names, "team1_heroes": t1_names, "game_map": map_name,
          "skill_tier": tier, "winner": 0}
    b_, e_ = extract_features(dd, st, [True] * len(FEATURE_GROUPS))
    bs, es = _swap_features(b_, e_)
    X = torch.tensor(np.array([np.concatenate([b_, e_[cols]]), np.concatenate([bs, es[cols]])]),
                     dtype=torch.float64)
    if not hasattr(W, "models_d"):
        import copy
        W.models_d = [copy.deepcopy(m).double().eval() for m in W.models]
    with torch.no_grad():
        zs = []
        for m in W.models_d:
            p = m(X).view(-1).clamp(1e-12, 1 - 1e-12)
            zs.append(torch.log(p / (1 - p)))
        z = torch.stack(zs).mean(0).numpy()
    pn, ps = 1 / (1 + np.exp(-z[0])), 1 / (1 + np.exp(-z[1]))
    wp0 = (pn + 1 - ps) / 2
    p_our = wp0 if our_team == 0 else 1 - wp0
    b0, b1, b2, b3 = coefs[:4]
    adj = b2 * (S[our_team] - S[1 - our_team]) + b3 * (O[our_team] - O[1 - our_team])
    b0s = b0 if our_team == 0 else -b0
    if adj == 0 and b0s == 0 and b1 == 1:
        return p_our, p_our
    pc = min(max(p_our, 1e-6), 1 - 1e-6)
    v = 1 / (1 + np.exp(-(b0s + b1 * np.log(pc / (1 - pc)) + adj)))
    return p_our, v
