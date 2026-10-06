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
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
KDIR = os.path.join(HERE, "cuda_personal")
CFG_LEN = 56
DRAFT_TEAM = [0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1]
IS_PICK = [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1]


SEARCH_MODES = {"legacy": 0, "chance": 1, "rollfwd": 2}
_SEARCH_MODE = None


def set_search_mode(name):
    """Every host sets the search mode once from its required --search-mode."""
    global _SEARCH_MODE
    if name not in ("chance", "rollfwd"):
        raise SystemExit(f"search mode {name!r}: use chance (headline) or rollfwd")
    _SEARCH_MODE = name


def search_mode():
    """Kernel search_mode for engine.run; refuses to run before set_search_mode."""
    if _SEARCH_MODE is None:
        raise SystemExit("search mode not set: pass --search-mode chance|rollfwd")
    return SEARCH_MODES[_SEARCH_MODE]


def search_mode_name():
    return _SEARCH_MODE


def out_path(fname):
    """Search outputs of the fixed kernel live apart from the legacy-tree
    caches: cache/mcts_v2/<search mode>/<fname>. Written atomically."""
    d = os.path.join(HERE, "cache", "mcts_v2", search_mode_name() or "unset")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, fname)


def save_pickle(obj, fname):
    import gzip
    import pickle
    p = out_path(fname)
    with gzip.open(p + ".tmp", "wb") as f:
        pickle.dump(obj, f)
    os.replace(p + ".tmp", p)
    return p


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
        self.policy = S.policy_flat("bc")
        self.gds = S.gd_flats()
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
    return c


def zeros_personal(n):
    return (np.zeros((n, 10, NUM_HEROES), np.float32), np.zeros((n, 10, NUM_HEROES), np.float32),
            np.zeros((n, 10, NUM_HEROES), np.float32), np.zeros((n, 10, 3), np.uint32))


def pool_bits(mask90):
    b = np.zeros(3, np.uint32)
    for h in np.flatnonzero(mask90):
        b[h // 32] |= np.uint32(1 << (h % 32))
    return b


def reference_value(W, key, map_name, tier, t0_names, t1_names, our_team, S, O, coefs):
    """float64 mirror of the kernel leaf: mean-logit ensemble per
    orientation, symmetrized, our side; then personal terms."""
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
