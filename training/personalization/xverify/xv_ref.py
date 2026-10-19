"""
Independent CPU replica of the MCTS kernels' v2 (chance-node) search, for
CPU/GPU equivalence tests. Written from the kernel sources, not from the
kernel authors' Python helpers:

  training/cuda_mcts/search_v2.cuh                      (paper 1, mode 1)
  training/personalization/cuda_*/p3_search_v2.inc       (P3 copies)
  training/personalization/cuda_*/<name>_kernel.cu       (P3 hooks)

What is mirrored exactly
  - curand XORWOW (curand_init(seed, 0, 0) and curand_uniform), consumed in the
    kernel's order: one draw per chance-node selection, per rollout step and per
    executed action; none for GD caching, expansion or PUCT. Dirichlet root
    noise is not mirrored (every tournament search runs with dir_eps = 0).
  - Draft order, state encoding (290-d, step / 15 in float32), legal masks,
    pool masks (slot and assign mode), forced bans.
  - Decision nodes: lazy children, untried action Q = 0, N = 0, PUCT
    q + c * p * sqrt(N) / (1 + n), ties to the lowest action.
  - Chance nodes: GD (optionally player-aware) distribution cached at first
    visit, outcome sampled with v2_sample; progressive widening
    limit = max(1, ceil(k N^alpha - 1e-3)), at the limit v2_sample_existing.
  - Leaf: unexpanded own decision (expanded with the policy) or terminal;
    rollout to the end with GD for both teams (P3: forced bans, pool masks,
    explore: player-aware opponent on opponent steps); value = symmetrized WP
    from our side, then the P3 personal value (slot, assign, pvnet).
  - Backup: every node on the path gets N += 1, W += v (our perspective).
  - Root outputs: visit fractions and child Q; executed action rules.

What is not bit-exact (and how the tests handle it)
  The networks run in PyTorch float32 on the CPU, the kernel runs its own
  float32 code with --use_fast_math. Probabilities agree to about 1e-6 and
  leaf values to about 1e-5. A sampled action can therefore differ only when
  r falls within that distance of a cumulative boundary, and a PUCT choice
  only when two scores are that close. The replica records both margins for
  every simulation so the tests can tell a float near-tie from a logic error.

Networks: a vintage's F policy (AlphaZeroDraftNet), one GenericDraftModel
and the enriched WP with the Python featurization (sweep_enriched_wp), all
from p3t_vintages.json paths.
"""
import itertools
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# XV_TRAINING: the repo's training/ dir when this file runs from a copy (remote workers)
TRAINING = os.environ.get("XV_TRAINING") or os.path.dirname(os.path.dirname(HERE))
P3 = os.path.join(TRAINING, "personalization")
for p in (TRAINING, P3, os.path.join(TRAINING, "cuda_mcts"), os.path.join(TRAINING, "overfit2026")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("P1_TIERS", "site")
import numpy as np

H = 90
TEAM = [0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1]
PICK = [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1]
M32 = 0xFFFFFFFF
F32 = np.float32
CHILD_NONE, CHILD_LAZY = -2, -1
KIND_DECISION, KIND_CHANCE, KIND_TERMINAL = 0, 1, 2
CFG_LEN, CFG_ASSIGN = 59, 58
# kernel revision: XV_KREV=2ad31c1 selects the prior floor of 2ad31c1 (default: the earlier rule)
PRIOR_FLOOR = os.environ.get("XV_KREV", "") == "2ad31c1"


# ------------------------------------------------------------------ curand XORWOW

_C = F32(2.3283064e-10)
_C2 = F32(_C / F32(2.0))


_JUMP = {}


def _xorwow_matrix():
    """160 x 160 GF(2) matrix of one XORWOW step on v[0..4] (bit 32 j + b = bit b of v[j])."""
    T = np.zeros((160, 160), np.uint8)
    I32 = np.eye(32, dtype=np.uint8)
    sh = lambda k: np.eye(32, k=-k, dtype=np.uint8) if k > 0 else np.eye(32, k=-k, dtype=np.uint8)
    # x << k: out bit b = in bit b - k ; x >> k: out bit b = in bit b + k
    L = lambda k: np.eye(32, k=-k, dtype=np.uint8)
    Rr = lambda k: np.eye(32, k=k, dtype=np.uint8)
    for j in range(4):                     # v[j] <- v[j + 1]
        T[32 * j:32 * j + 32, 32 * (j + 1):32 * (j + 2)] = I32
    t = (I32 + Rr(2)) % 2                  # t = v0 ^ (v0 >> 2)
    tt = (t + L(1) @ t) % 2                # t ^ (t << 1)
    v4 = (I32 + L(4)) % 2                  # v4 ^ (v4 << 4)
    T[128:160, 0:32] = tt
    T[128:160, 128:160] = v4
    del sh
    return T


def _matpow2(M, k):
    for _ in range(k):
        M = (M.astype(np.int64) @ M.astype(np.int64) % 2).astype(np.uint8)
    return M


def _jump(kind, j):
    """M^(2^j) for kind 'seq' (M = T^(2^67), one subsequence) or 'off' (M = T)."""
    key = (kind, j)
    if key not in _JUMP:
        if j == 0:
            T = _xorwow_matrix()
            _JUMP[key] = _matpow2(T, 67) if kind == "seq" else T
        else:
            _JUMP[key] = _matpow2(_jump(kind, j - 1), 1)
    return _JUMP[key]


class XorWow:
    """curandStateXORWOW_t with curand_init(seed, subsequence, offset, &s)."""

    def __init__(self, seed, fma=True, subsequence=0, offset=0):
        seed = int(seed) & 0xFFFFFFFFFFFFFFFF
        s0 = (seed & M32) ^ 0xAAD26B49
        s1 = ((seed >> 32) & M32) ^ 0xF7DCEFDD
        t0 = (1099087573 * s0) & M32
        t1 = (2591861531 * s1) & M32
        self.d = (6615241 + t1 + t0) & M32
        self.v = [(123456789 + t0) & M32, 362436069 ^ t0, (521288629 + t1) & M32, 88675123 ^ t1,
                  (5783321 + t0) & M32]
        self.fma = fma
        self.n = 0
        for kind, x in (("seq", int(subsequence)), ("off", int(offset))):
            if not x:
                continue
            bits = np.array([(self.v[i // 32] >> (i % 32)) & 1 for i in range(160)], np.int64)
            j = 0
            while x:
                if x & 1:
                    bits = _jump(kind, j).astype(np.int64) @ bits % 2
                x >>= 1
                j += 1
            self.v = [int(sum(int(bits[32 * w + b]) << b for b in range(32))) for w in range(5)]
            if kind == "off":
                self.d = (self.d + 362437 * (int(offset) & M32)) & M32

    def next_u32(self):
        v = self.v
        t = v[0] ^ (v[0] >> 2)
        v[0], v[1], v[2], v[3] = v[1], v[2], v[3], v[4]
        v[4] = (v[4] ^ ((v[4] << 4) & M32)) ^ (t ^ ((t << 1) & M32))
        self.d = (self.d + 362437) & M32
        self.n += 1
        return (v[4] + self.d) & M32

    def uniform(self):
        """curand_uniform: x * 2^-32 + 2^-33 in float32, in (0, 1]. nvcc
        contracts it to one fused multiply-add (fma=True); fma=False is the
        unfused form."""
        x = F32(self.next_u32())          # uint -> float, round to nearest
        if self.fma:
            return F32(float(x) * float(_C) + float(_C2))
        return F32(F32(x * _C) + _C2)


# ------------------------------------------------------------------ samplers (float32, kernel order)

def v2_sample(p, r):
    """first index whose float32 running sum exceeds r; zero entries skipped;
    last positive entry on overrun. Returns (index, margin) where margin is
    the distance from r to the nearest boundary of the chosen interval."""
    cum = F32(0.0)
    last = -1
    lo = F32(0.0)
    for i in range(H):
        x = p[i]
        if x <= 0:
            continue
        last = i
        lo = cum
        cum = F32(cum + x)
        if cum > r:
            return i, float(min(abs(float(r) - float(lo)), abs(float(cum) - float(r))))
    return last, 0.0


def v2_sample_existing(p, child, r):
    tot = F32(0.0)
    for i in range(H):
        if child[i] >= 0:
            tot = F32(tot + p[i])
    x = F32(r * tot)
    cum = F32(0.0)
    last = -1
    lo = F32(0.0)
    for i in range(H):
        if child[i] < 0:
            continue
        last = i
        lo = cum
        cum = F32(cum + p[i])
        if cum > x:
            return i, float(min(abs(float(x) - float(lo)), abs(float(cum) - float(x))))
    return last, 0.0


def legacy_cum_sample(p, r):
    """P3 main-loop sampler: `action = 0; cum += p[i]; if (cum > r) ...`
    (falls back to hero 0 on overrun, taken or not)."""
    cum = F32(0.0)
    for i in range(H):
        cum = F32(cum + p[i])
        if cum > r:
            return i
    return 0


def pw_limit(n, k, alpha):
    if k <= 0:
        return 1 << 30
    return max(1, math.ceil(k * float(n) ** alpha - 1e-3))


# ------------------------------------------------------------------ draft state

class Draft:
    __slots__ = ("t0", "t1", "bans", "taken", "step", "map", "tier", "our", "acts")

    def __init__(self, mp, tier, our):
        self.t0 = self.t1 = self.bans = self.taken = 0
        self.step = 0
        self.map, self.tier, self.our = mp, tier, our
        self.acts = [-1] * 16

    def copy(self):
        d = Draft.__new__(Draft)
        d.t0, d.t1, d.bans, d.taken, d.step = self.t0, self.t1, self.bans, self.taken, self.step
        d.map, d.tier, d.our = self.map, self.tier, self.our
        d.acts = list(self.acts)
        return d

    def apply(self, h, team, is_pick):
        b = 1 << h
        self.taken |= b
        if self.step < 16:
            self.acts[self.step] = h
        if not is_pick:
            self.bans |= b
        elif team == 0:
            self.t0 |= b
        else:
            self.t1 |= b
        self.step += 1

    def is_taken(self, h):
        return (self.taken >> h) & 1

    def terminal(self):
        return self.step >= 16

    def team(self):
        return TEAM[self.step]

    def is_pick(self):
        return PICK[self.step]

    def key(self):
        return (self.t0, self.t1, self.bans, self.step, self.our, self.map, self.tier)

    def free(self):
        return np.array([not ((self.taken >> h) & 1) for h in range(H)])

    def features(self):
        x = np.zeros(290, np.float32)
        for h in range(H):
            if (self.t0 >> h) & 1:
                x[h] = 1
            if (self.t1 >> h) & 1:
                x[H + h] = 1
            if (self.bans >> h) & 1:
                x[2 * H + h] = 1
        x[3 * H + self.map] = 1
        x[3 * H + 14 + self.tier] = 1
        if self.step < 16:
            x[287] = F32(self.step) / F32(15.0)
            x[288] = 1.0 if PICK[self.step] else 0.0
        else:
            x[287] = x[288] = 1.0
        x[289] = float(self.our)
        return x

    def team_heroes(self, t):
        bits = self.t0 if t == 0 else self.t1
        return [h for h in range(H) if (bits >> h) & 1][:5]


def kind_of(s):
    if s.terminal():
        return KIND_TERMINAL
    return KIND_DECISION if s.team() == s.our else KIND_CHANCE


# ------------------------------------------------------------------ networks (PyTorch float32)

class Nets:
    """policy prior, one GD model and the enriched WP of a p3t vintage."""

    def __init__(self, vint="cur", seed_i=0, gd_i=None):
        import torch
        import p3t_systems as SY
        torch.set_num_threads(1)
        self.torch = torch
        v = SY.vintage(vint)
        self.v = v
        self.pol = v.policy_net(seed_i % v.n_seeds)
        self.gd = v.gds()[(seed_i if gd_i is None else gd_i) % len(v.gds())]
        m, cols = v.wp_model()
        self.wp_m, self.wp_cols = m, cols
        from paper1_revision import core
        self.sd = core.stats_dict(v.stats())
        self._pc, self._gc, self._wc = {}, {}, {}

    def policy_probs(self, s, mask):
        k = (s.key(), mask.tobytes())
        r = self._pc.get(k)
        if r is None:
            t = self.torch
            with t.no_grad():
                lg, _ = self.pol(t.from_numpy(s.features()[None]), t.from_numpy(mask.astype(np.float32)[None]))
                r = t.softmax(lg, 1)[0].numpy().astype(np.float32)
            r = np.where(mask, r, F32(0.0)).astype(np.float32)
            self._pc[k] = r
        return r

    def gd_probs(self, s, mask):
        k = (s.t0, s.t1, s.bans, s.step, s.map, s.tier, mask.tobytes())
        r = self._gc.get(k)
        if r is None:
            t = self.torch
            with t.no_grad():
                lg = self.gd(t.from_numpy(s.features()[None, :289]), t.from_numpy(mask.astype(np.float32)[None]))
                r = t.softmax(lg, 1)[0].numpy().astype(np.float32)
            r = np.where(mask, r, F32(0.0)).astype(np.float32)
            if len(self._gc) > 200000:
                self._gc.clear()
            self._gc[k] = r
        return r

    def wp_t0(self, t0, t1, mp, tier):
        """symmetrized P(team 0 wins) of a complete draft (Python featurization)."""
        k = (tuple(sorted(t0)), tuple(sorted(t1)), mp, tier)
        r = self._wc.get(k)
        if r is None:
            from paper1_revision import core
            from shared import HEROES, MAPS, SKILL_TIERS
            rows = [([HEROES[h] for h in t0], [HEROES[h] for h in t1], MAPS[mp], SKILL_TIERS[tier])]
            Xf, Xs = core._chunk((rows, self.sd))
            t = self.torch
            with t.no_grad():
                a = float(self.wp_m(t.from_numpy(Xf[:, self.wp_cols])).view(-1)[0])
                b = float(self.wp_m(t.from_numpy(Xs[:, self.wp_cols])).view(-1)[0])
            r = (a + (1.0 - b)) / 2.0
            self._wc[k] = r
        return r


# ------------------------------------------------------------------ P3 episode inputs

class P3Inputs:
    """One episode's kernel inputs. flavor: personal | prior | pvnet | explore
    (which hooks the kernel has). pool: (10, 90) bool."""

    def __init__(self, flavor, cfg, s, off, imit, pool, coefs, bias=None, feat=None, vnet=None):
        self.flavor = flavor
        self.cfg = np.asarray(cfg, np.int64)
        self.s = np.asarray(s, np.float32)
        self.off = np.asarray(off, np.float32)
        self.imit = np.asarray(imit, np.float32)
        self.pool = np.asarray(pool, bool)
        c = np.zeros(6, np.float32)
        c[:len(coefs)] = coefs
        self.b0, self.b1, self.b2, self.b3, self.w0, self.alpha = (float(x) for x in c)
        self.c = c
        self.bias = np.zeros((10, H), np.float32) if bias is None else np.asarray(bias, np.float32)
        self.feat = np.zeros((10, H), np.float32) if feat is None else np.asarray(feat, np.float32)
        self.vnet = np.zeros(225, np.float32) if vnet is None or len(vnet) == 0 else np.asarray(vnet, np.float32)


def p3_mask(s, e):
    """ep_valid_mask: free heroes; at pick steps with use_pool the pool rule
    (assign mode: union of the acting team's 5 pools; slot mode: the acting
    slot's pool; if nothing is left, any free hero)."""
    m = s.free()
    st = s.step
    if st < 16 and e.cfg[5] and PICK[st]:
        if e.cfg[CFG_ASSIGN] == 1:
            u = e.pool[5 * TEAM[st]:5 * TEAM[st] + 5].any(0)
        else:
            slot = int(e.cfg[24 + st])
            if slot < 0:
                return m
            u = e.pool[slot]
        if (m & u).any():
            m = m & u
    return m


def p3_forced(s, e):
    st = s.step
    if st >= 16 or PICK[st]:
        return -1
    h = int(e.cfg[40 + st])
    if h < 0 or s.is_taken(h):
        return -1
    return h


def prior_bias(pri, mask, step, e):
    """apply_prior_bias (prior/pvnet: picks; explore also bans with cfg[56] = 2)."""
    if e.flavor not in ("prior", "pvnet", "explore") or not e.cfg[56] or step >= 16:
        return pri
    if not (PICK[step] or (e.flavor == "explore" and e.cfg[56] == 2)):
        return pri
    slot = int(e.cfg[24 + step])
    if slot < 0:
        return pri
    if PRIOR_FLOOR:
        # 2ad31c1: every valid hero, prior floored at 1e-30 before the log
        ok = mask.copy()
        u = np.where(ok, e.alpha * np.log(np.maximum(pri.astype(np.float64), 1e-30)) + e.bias[slot], -np.inf)
    else:
        ok = mask & (pri > 0)
        u = np.where(ok, e.alpha * np.log(np.where(ok, pri, 1.0)) + e.bias[slot], -np.inf)
    mx = u[ok].max()
    w = np.where(ok, np.exp(u - mx), 0.0)
    return (w / w.sum()).astype(np.float32)


def opp_personalize(p, mask, step, our, e):
    """explore cfg[6] = 2: softmax(w0 log p + imit[row]) at opponent steps."""
    if e.flavor != "explore" or e.cfg[6] != 2 or step < 0 or step >= 16 or TEAM[step] == our:
        return p
    row = int(e.cfg[24 + step])
    if row < 0 or row >= 10:
        return p
    b = e.imit[row]
    if e.w0 == 1.0 and not b.any():
        return p
    ok = (p > 0) & mask
    u = np.where(ok, e.w0 * np.log(np.where(ok, p, 1.0)) + b, -np.inf)
    mx = u[ok].max()
    w = np.where(ok, np.exp(u - mx), 0.0)
    return (w / w.sum()).astype(np.float32)


def best_assignment(tm, acts, step, es, eo, b2, b3):
    """assign mode: first permutation (lexicographic, identity first) that
    maximizes b2 S + b3 O; slot 5 tm + i plays pick perm[i]. Returns S, O, perm."""
    h = [acts[st] for st in range(min(step, 16)) if PICK[st] and TEAM[st] == tm][:5]
    n = len(h)
    base = 5 * tm
    best, bS, bO, bp = -1e30, 0.0, 0.0, (0, 1, 2, 3, 4)
    for p in itertools.permutations(range(5)):
        sS = sum(float(es[base + i, h[p[i]]]) for i in range(5) if p[i] < n)
        sO = sum(float(eo[base + i, h[p[i]]]) for i in range(5) if p[i] < n)
        obj = b2 * sS + b3 * sO
        if obj > best:
            best, bS, bO, bp = obj, sS, sO, p
    return bS, bO, bp


def pvnet_h(F, w):
    W1 = w[:192].reshape(16, 12).astype(np.float64)
    c1 = w[192:208].astype(np.float64)
    w2 = w[208:224].astype(np.float64)
    c2 = float(w[224])
    return float(w2 @ np.maximum(W1 @ F + c1, 0.0) + c2)


def pvnet_features(tm, s, e):
    S, O, bp = best_assignment(tm, s.acts, s.step, e.s, e.off, e.b2, e.b3)
    h = [s.acts[st] for st in range(min(s.step, 16)) if PICK[st] and TEAM[st] == tm][:5]
    n = len(h)
    base = 5 * tm
    sv = sorted([float(e.s[base + i, h[bp[i]]]) if bp[i] < n else 0.0 for i in range(5)], reverse=True)
    fv = sorted([float(e.feat[base + i, h[bp[i]]]) if bp[i] < n else 0.0 for i in range(5)], reverse=True)
    return np.array([S, O] + sv + fv, np.float64)


def personal_value(p_our, s, our, e):
    """P3 personal_value (all flavors), float64."""
    if not e.cfg[7]:
        return p_our
    if e.flavor in ("pvnet", "explore") and e.cfg[57] == 1:
        h0 = pvnet_h(pvnet_features(0, s, e), e.vnet)
        h1 = pvnet_h(pvnet_features(1, s, e), e.vnet)
        dh = h0 - h1 if our == 0 else h1 - h0
        b0s = e.b0 if our == 0 else -e.b0
        pc = min(max(p_our, 1e-6), 1 - 1e-6)
        return 1.0 / (1.0 + math.exp(-(b0s + e.b1 * math.log(pc / (1 - pc)) + dh)))
    S, O = [0.0, 0.0], [0.0, 0.0]
    if e.cfg[CFG_ASSIGN] == 1:
        for t in (0, 1):
            S[t], O[t], _ = best_assignment(t, s.acts, s.step, e.s, e.off, e.b2, e.b3)
    else:
        for st in range(min(s.step, 16)):
            if not PICK[st]:
                continue
            slot = int(e.cfg[24 + st])
            if slot < 0:
                continue
            tm = 0 if slot < 5 else 1
            S[tm] += float(e.s[slot, s.acts[st]])
            O[tm] += float(e.off[slot, s.acts[st]])
    adj = e.b2 * (S[our] - S[1 - our]) + e.b3 * (O[our] - O[1 - our])
    b0s = e.b0 if our == 0 else -e.b0
    if adj == 0.0 and b0s == 0.0 and e.b1 == 1.0:
        return p_our
    pc = min(max(p_our, 1e-6), 1 - 1e-6)
    return 1.0 / (1.0 + math.exp(-(b0s + e.b1 * math.log(pc / (1 - pc)) + adj)))


# ------------------------------------------------------------------ the v2 search

class Node:
    __slots__ = ("parent", "action", "visits", "wsum", "p", "child", "n_kids", "kind")

    def __init__(self, parent, action, kind):
        self.parent, self.action, self.kind = parent, action, kind
        self.visits, self.wsum, self.n_kids = 0, 0.0, 0
        self.p = None
        self.child = None


class Search:
    """One kernel episode block. e: P3Inputs or None (paper-1 kernel: no hooks)."""

    def __init__(self, nets, e=None, c_puct=2.0, pw_k=1.0, pw_alpha=0.5, seed=0, fma=True, subsequence=0):
        self.nets, self.e = nets, e
        self.c, self.k, self.alpha = c_puct, pw_k, pw_alpha
        self.rng = XorWow(seed, fma=fma, subsequence=subsequence)

    # hooks ---------------------------------------------------------
    def mask(self, s):
        return p3_mask(s, self.e) if self.e is not None else s.free()

    def forced(self, s):
        return p3_forced(s, self.e) if self.e is not None else -1

    def apply_forced(self, s):
        while not s.terminal():
            fa = self.forced(s)
            if fa < 0:
                break
            s.apply(fa, s.team(), 0)

    def priors(self, s, mask):
        p = self.nets.policy_probs(s, mask)
        if self.e is not None:
            p = prior_bias(p, mask, s.step, self.e)
        return p

    def gd(self, s, mask):
        p = self.nets.gd_probs(s, mask)
        if self.e is not None:
            p = opp_personalize(p, mask, s.step, s.our, self.e)
        return p

    def value(self, s):
        t0, t1 = s.team_heroes(0), s.team_heroes(1)
        w0 = self.nets.wp_t0(t0, t1, s.map, s.tier)
        p_our = w0 if s.our == 0 else 1.0 - w0
        return personal_value(p_our, s, s.our, self.e) if self.e is not None else p_our

    # search --------------------------------------------------------
    def search(self, root, sims, snap=None):
        """root: Draft at our decision. Returns dict with per-sim margins and,
        at each k in snap, root (visits, Q) after k sims."""
        nodes = [Node(-1, -1, KIND_DECISION)]
        rm = self.mask(root)
        r0 = nodes[0]
        r0.p = self.priors(root, rm)
        r0.child = [CHILD_LAZY if rm[a] else CHILD_NONE for a in range(H)]
        self.root_prior = r0.p.copy()
        margins, snaps, leafv = [], {}, []
        snap = set(snap or [])
        for sim in range(sims):
            s = root.copy()
            path = [0]
            i = 0
            mpuct, msamp = 1e9, 1e9
            leaf_expand = False
            while True:
                if s.terminal():
                    break
                n = nodes[i]
                if s.team() == s.our:
                    if n.p is None:
                        leaf_expand = True
                        break
                    sq = math.sqrt(n.visits)
                    best, second, ba = -1e30, -1e30, -1
                    # exact ties that both implementations reproduce (no visits at the
                    # node; unvisited children with prior exactly 0) are not near-ties
                    det = lambda nn_, p_: nn_ == 0 and p_ == 0.0
                    bdet = sdet = False
                    for a in range(H):
                        c = n.child[a]
                        if c == CHILD_NONE:
                            continue
                        nn, q = (0, 0.0) if c < 0 else (nodes[c].visits, (nodes[c].wsum / nodes[c].visits
                                                                       if nodes[c].visits else 0.0))
                        sc = q + self.c * float(n.p[a]) * sq / (1.0 + nn)
                        if sc > best:
                            second, sdet = best, bdet
                            best, ba, bdet = sc, a, det(nn, float(n.p[a]))
                        elif sc > second or (sc == second and not det(nn, float(n.p[a]))):
                            second, sdet = sc, det(nn, float(n.p[a]))
                    if second > -1e29 and n.visits > 0 and not (bdet and sdet):
                        mpuct = min(mpuct, best - second)
                    a = ba
                else:
                    if n.p is None:
                        m = self.mask(s)
                        n.p = self.gd(s, m)
                        n.child = [CHILD_LAZY if n.p[x] > 0 else CHILD_NONE for x in range(H)]
                    r = self.rng.uniform()
                    lim = pw_limit(n.visits, self.k, self.alpha)
                    if n.n_kids < lim:
                        a, mg = v2_sample(n.p, r)
                    else:
                        a, mg = v2_sample_existing(n.p, n.child, r)
                    msamp = min(msamp, mg)
                c = n.child[a]
                if c < 0:
                    c = len(nodes)
                    nodes.append(Node(i, a, KIND_DECISION))
                    n.child[a] = c
                    n.n_kids += 1
                s.apply(a, s.team(), s.is_pick())
                self.apply_forced(s)
                nodes[c].kind = kind_of(s)
                i = c
                path.append(i)
            if leaf_expand:
                m = self.mask(s)
                nd = nodes[i]
                nd.p = self.priors(s, m)
                nd.child = [CHILD_LAZY if m[a] else CHILD_NONE for a in range(H)]
            # rollout
            while not s.terminal():
                fa = self.forced(s)
                if fa >= 0:
                    s.apply(fa, s.team(), 0)
                    continue
                m = self.mask(s)
                p = self.gd(s, m)
                r = self.rng.uniform()
                a, mg = v2_sample(p, r)
                msamp = min(msamp, mg)
                s.apply(a, s.team(), s.is_pick())
            v = self.value(s)
            leafv.append(v)
            for j in path:
                nodes[j].visits += 1
                nodes[j].wsum += v
            margins.append((mpuct, msamp))
            if (sim + 1) in snap:
                snaps[sim + 1] = self.root_stats(nodes)
        self.nodes = nodes
        return {"margins": margins, "snaps": snaps, "root": self.root_stats(nodes), "leafv": leafv}

    @staticmethod
    def root_stats(nodes):
        r = nodes[0]
        vis = np.zeros(H, np.int64)
        q = np.zeros(H)
        for a in range(H):
            c = r.child[a]
            if c >= 0:
                vis[a] = nodes[c].visits
                q[a] = nodes[c].wsum / nodes[c].visits if nodes[c].visits else 0.0
        return vis, q


# ------------------------------------------------------------------ whole episodes

def decide_p3(nets, e, sims, c_puct, seed, pw_k=1.0, pw_alpha=0.5, snap=None, fma=True, subsequence=0):
    """P3 kernel decide-only episode (cfg[4] = decide step): replay the prefix,
    search once for the acting team. Returns the Search result plus the root
    prior and the executed action (argmax rule of the P3 main loop, root_temp 0)."""
    cfg = e.cfg
    s = Draft(int(cfg[0]), int(cfg[1]), int(cfg[2]))
    ds = int(cfg[4])
    while s.step < ds:
        s.apply(int(cfg[8 + s.step]), s.team(), s.is_pick())
    # main loop at the decide step: a forced ban there is applied and the loop
    # stops at the next step (s_stop) without searching
    fa = p3_forced(s, e)
    if fa >= 0:
        return None
    s.our = s.team()
    S = Search(nets, e, c_puct, pw_k, pw_alpha, seed, fma=fma, subsequence=subsequence)
    out = S.search(s, sims, snap)
    vis, q = out["root"]
    pol = vis / max(vis.sum(), 1)
    chosen, best = 0, -1.0
    for i in range(H):
        if pol[i] > best:
            best, chosen = pol[i], i
    out.update(chosen=chosen, root_prior=S.root_prior, rng_draws=S.rng.n, state=s)
    return out


def episode_paper1(nets, mp, tier, our, sims, c_puct, seed, root_temp=0.0, pw_k=1.0, pw_alpha=0.5, fma=True,
                   subsequence=0):
    """paper-1 kernel (search_v2.cuh, mode 1) full episode with dir_eps = 0.
    Returns per-turn visit fractions, executed actions and the terminal WP."""
    s = Draft(mp, tier, our)
    S = Search(nets, None, c_puct, pw_k, pw_alpha, seed, fma=fma, subsequence=subsequence)
    pols, acts, margins = [], [], []
    while not s.terminal():
        team = s.team()
        if team == our:
            out = S.search(s, sims)
            vis, _ = out["root"]
            margins.append(min(min(m) for m in out["margins"]) if out["margins"] else 1e9)
            pol = (vis / vis.sum()).astype(np.float32) if vis.sum() > 0 else np.zeros(H, np.float32)
            pols.append(pol)
            r = S.rng.uniform()
            chosen = -1
            if root_temp < 0.05:
                best = -1.0
                for i in range(H):
                    if pol[i] > best:
                        best, chosen = pol[i], i
            elif abs(root_temp - 1.0) < 1e-6:
                chosen, _ = v2_sample(pol, r)
            else:
                w = np.where(pol > 0, pol.astype(np.float64) ** (1.0 / root_temp), 0.0).astype(np.float32)
                w = (w / w.sum()).astype(np.float32)
                chosen, _ = v2_sample(w, r)
            if chosen < 0 or s.is_taken(chosen):
                chosen, _ = v2_sample(S.root_prior, r)
            s.apply(chosen, team, s.is_pick())
        else:
            m = s.free()
            p = nets.gd_probs(s, m)
            r = S.rng.uniform()
            a, mg = v2_sample(p, r)
            margins.append(mg)
            s.apply(a, team, s.is_pick())
        acts.append(s.acts[s.step - 1])
    w0 = nets.wp_t0(s.team_heroes(0), s.team_heroes(1), mp, tier)
    return {"policies": pols, "acts": acts, "wp": w0 if our == 0 else 1 - w0, "margins": margins}

