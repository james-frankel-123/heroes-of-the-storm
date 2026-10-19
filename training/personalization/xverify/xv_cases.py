"""
Randomized and adversarial decide-only test cases for the P3 kernels, shared
by the GPU runner (xv_gpu.py) and the CPU comparison (xv_compare.py).

A case is one kernel episode: cfg (59 ints, kernel layout), per-lobby tensors
s, off, imit, prior bias, feat (10 x 90), pools (10 x 90 bool). Coefficients
and the value-net weights are per launch, so they are per flavor here.

Categories (cycled over the cases of a flavor):
  ban_early   decide at a ban step 0-3 (bans only so far)
  mid         decide at a pick step 4-8
  ban_late    decide at ban step 9 or 10
  late        decide at a pick step 11-15 (15 = the last action)
  pool_tiny   1-2 heroes per player pool
  pool_full   every hero in every pool
  pool_gone   the acting team's pools are all taken: falls back to any free hero
  pool_one    exactly one free hero in the acting team's pools
  same_tables all ten players share one table
  extreme     s ~ N(0, 3^2), off ~ Bernoulli(0.5), imit/bias ~ N(0, 4^2)
  mirror      team 1's tables are team 0's (rows 5-9 = rows 0-4)
  forced      forced bans at some later ban steps (cfg[40:56])
  slot_mode   cfg[58] = 0 (pick-step -> player slots)
"""
import numpy as np

H = 90
TEAM = [0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1]
PICK = [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1]
CATS = ["ban_early", "mid", "ban_late", "late", "pool_tiny", "pool_full", "pool_gone", "pool_one",
        "same_tables", "extreme", "mirror", "forced", "slot_mode"]
FLAVORS = ("personal", "prior", "pvnet", "explore")
COEFS = {   # b0, b1, b2, b3, imit_w0, prior alpha (per launch)
    "personal": [0.03, 1.2015, 4.5406, -0.0147, 0.0, 0.0],
    "prior": [0.0, 1.2015, 4.5406, -0.0147, 0.0, 0.3],
    "pvnet": [0.05, 1.1, 4.5406, -0.0147, 0.0, 1.0],
    "explore": [0.0, 1.2015, 4.5406, -0.0147, 0.8, 0.156],
}


def slot_of_default():
    out, cnt = [], [0, 0]
    for k in range(16):
        if PICK[k]:
            t = TEAM[k]
            out.append(5 * t + cnt[t])
            cnt[t] += 1
        else:
            out.append(-1)
    return np.array(out, np.int64)


def legal_mask(acts, k, pools):
    """lobby rule at step k (team pool union for picks, else any free)."""
    free = np.ones(H, bool)
    for j in range(k):
        free[acts[j]] = False
    if not PICK[k]:
        return free
    u = pools[5 * TEAM[k]:5 * TEAM[k] + 5].any(0)
    m = free & u
    return m if m.any() else free


def make_case(rng, flavor, cat, idx):
    k = {"ban_early": int(rng.integers(0, 4)), "mid": int(rng.integers(4, 9)),
         "ban_late": int(rng.integers(9, 11)), "late": int(rng.integers(11, 16))}.get(cat, int(rng.integers(0, 16)))
    if cat in ("pool_gone", "pool_one") and not PICK[k]:
        k = int(rng.choice([4, 5, 6, 7, 8, 11, 12, 13, 14, 15]))
    t = TEAM[k]
    pools = rng.random((10, H)) < rng.uniform(0.08, 0.4)
    if cat == "pool_tiny":
        pools = np.zeros((10, H), bool)
        for r in range(10):
            pools[r, rng.choice(H, int(rng.integers(1, 3)), replace=False)] = True
    if cat == "pool_full":
        pools[:] = True
    acts = np.full(16, -1, np.int64)
    if cat in ("pool_gone", "pool_one"):
        # the acting team's pool union: heroes taken in the prefix (all, or all but one)
        u = rng.choice(H, int(rng.integers(k, k + 2)) if cat == "pool_one" else k, replace=False)
        pools[5 * t:5 * t + 5] = False
        for j, h in enumerate(u):
            pools[5 * t + j % 5, h] = True
        order = list(u[:k]) if len(u) >= k else list(u)
        rest = [h for h in rng.permutation(H) if h not in set(u)]
        while len(order) < k:
            order.append(rest.pop())
        acts[:k] = np.array(order[:k])
    else:
        for j in range(k):
            m = legal_mask(acts, j, pools)
            acts[j] = int(rng.choice(np.flatnonzero(m)))
    ext = cat == "extreme"
    s = rng.normal(0, 3.0 if ext else 0.25, (10, H)).astype(np.float32)
    off = (rng.random((10, H)) < (0.5 if ext else 0.1)).astype(np.float32)
    imit = rng.normal(0, 4.0 if ext else 0.7, (10, H)).astype(np.float32)
    bias = rng.normal(0, 4.0 if ext else 0.7, (10, H)).astype(np.float32)
    feat = np.log1p(rng.poisson(5, (10, H))).astype(np.float32)
    if cat == "same_tables":
        for x in (s, off, imit, bias, feat):
            x[1:] = x[0]
        pools[1:] = pools[0]
    if cat == "mirror":
        for x in (s, off, imit, bias, feat):
            x[5:] = x[:5]
        pools[5:] = pools[:5]
    cfg = np.full(59, -1, np.int64)
    cfg[0], cfg[1], cfg[2] = int(rng.integers(0, 14)), int(rng.integers(0, 3)), t
    cfg[3], cfg[4], cfg[5], cfg[6], cfg[7] = 0, k, 1, 0, 1
    cfg[8:24] = acts
    cfg[24:40] = slot_of_default()
    cfg[40:56] = -1
    cfg[56:59] = 0
    cfg[58] = 0 if cat == "slot_mode" else 1
    if cat == "forced":
        later = [j for j in range(k + 1, 16) if not PICK[j]]
        for j in later:
            if rng.random() < 0.7:
                cfg[40 + j] = int(rng.integers(0, H))      # may be taken already: then not forced
    if idx % 7 == 3:
        cfg[7] = 0                                         # personal valuation off
    if idx % 11 == 5:
        cfg[5] = 0                                         # no pool rule
    if flavor in ("prior", "pvnet", "explore"):
        cfg[56] = int(rng.random() < 0.8)
    if flavor in ("pvnet", "explore") and cfg[58] == 1 and cfg[7] == 1:
        cfg[57] = int(rng.random() < 0.6)
    if flavor == "explore":
        if rng.random() < 0.6:
            cfg[56] = 2
            r = 5 * (1 - t)
            for j in range(16):
                if not PICK[j] and TEAM[j] == t and rng.random() < 0.8:
                    cfg[24 + j] = r
        if rng.random() < 0.6:
            cfg[6] = 2
            if rng.random() < 0.5:   # opponent ban rows too
                for j in range(16):
                    if not PICK[j] and TEAM[j] != t:
                        cfg[24 + j] = int(5 * (1 - t) + rng.integers(0, 5))
        elif rng.random() < 0.3:
            cfg[6] = 1               # imitation: main loop only (no effect in decide-only)
    return dict(cfg=cfg.astype(np.int32), s=s, off=off, imit=imit, bias=bias, feat=feat, pools=pools, cat=cat)


def make_cases(flavor, n, seed):
    rng = np.random.default_rng(seed)
    out = [make_case(rng, flavor, CATS[i % len(CATS)], i) for i in range(n)]
    return stack(out)


def stack(cases):
    z = {k: np.stack([c[k] for c in cases]) for k in ("cfg", "s", "off", "imit", "bias", "feat", "pools")}
    z["cat"] = np.array([c["cat"] for c in cases])
    return z


def pool_bits(pools):
    """(n, 10, 90) bool -> (n, 10, 3) uint32."""
    n = len(pools)
    b = np.zeros((n, 10, 3), np.uint64)
    for h in range(H):
        b[:, :, h // 32] |= (pools[:, :, h].astype(np.uint64) << np.uint64(h % 32))
    return b.astype(np.uint32)


def vnet_weights(seed):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.3, 225).astype(np.float32)


def complete_drafts(rng, n, weird_frac=0.4):
    """random complete drafts (kernel order). weird_frac: one team stacked with
    one role class (healers / tanks / supports) so its composition is rare or
    unknown and the composition fallback runs."""
    import p3t_systems as SY
    roles = SY.HROLE
    out = np.zeros((n, 16), np.int64)
    for i in range(n):
        if rng.random() < weird_frac:
            r = int(rng.choice([0, 2, 5, 4]))          # tank, healer, support, melee assassin
            cls = list(rng.permutation(np.flatnonzero(roles == r)))
            others = [h for h in rng.permutation(H) if h not in set(cls)]
            tw = int(rng.integers(0, 2))
            picks = {0: [], 1: []}
            for j in range(16):
                if PICK[j]:
                    pool = cls if TEAM[j] == tw and cls else others
                    picks[TEAM[j]].append(pool.pop(0))
            it = {0: iter(picks[0]), 1: iter(picks[1])}
            rest = iter(h for h in others)
            used = set(picks[0]) | set(picks[1])
            for j in range(16):
                if PICK[j]:
                    out[i, j] = next(it[TEAM[j]])
                else:
                    h = next(rest)
                    while h in used:
                        h = next(rest)
                    used.add(h)
                    out[i, j] = h
        else:
            out[i] = rng.permutation(H)[:16]
    return out
