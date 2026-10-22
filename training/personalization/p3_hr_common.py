"""
Shared data for P3_HERO_RECS_AND_REPLAYS (hero recommendations backtest, B1
scoreboard studies). Everything runs on the full_1017 panel, the same rows
as P3_TAILS: the A1 slot table plus the full-history merged Heroes Profile
rows, the `a_shrink` scoreboard card, and the fitted `full_ms` (main) and
`full_imp2` params.

`build` runs p3_rcf.load_aug once and stores a compact copy of every
history row (cache/hr/aug.npz, gitignored: it holds player ids):
day, pid, hero, r (raw residual: y - WP on slot rows, y - hero quarterly WR
on HP rows), r_adj (minus the experience offset), v, y, slot index (-1 for
HP rows), the 15 scoreboard z values (float16, NaN where missing), plus the
slot-table fields the tests need (window, game index, team, WP).

Run from training/:  python personalization/p3_hr_common.py build
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("P3_CACHE", os.path.join(HERE, "cache_full"))
os.environ.setdefault("P3_RESULTS", os.path.join(HERE, "results", "rcf_full"))
os.environ.setdefault("P3_RCF_ROWS", os.path.join(HERE, "cache_full", "merged_rows_full_1017.npz"))
os.environ.setdefault("P3_RCF_KERNELS", os.path.join(HERE, "cache_full", "hs_kernels.npz"))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

HR_CACHE = os.path.join(HERE, "cache", "hr")
HR_RES = os.path.join(HERE, "results", "hr")
AUG = os.path.join(HR_CACHE, "aug.npz")
PARAMS = os.path.join(HERE, "results", "rcf_full", "params")
H = 90


def build():
    import p3_rcf as R
    import p3_c_tests as T
    t0 = time.time()
    d, aug = R.load_aug("merged")
    Z, W = R.aug_metrics(aug, "a_shrink", d)
    w, med = T.windows(d)
    pulled = np.load(R.ROWS)["pulled_pid"]
    os.makedirs(HR_CACHE, exist_ok=True)
    out = dict(
        day=aug["day"].astype(np.int32), pid=aug["pid"].astype(np.int32), hero=aug["hero"].astype(np.int8),
        r=aug["r"].astype(np.float32), r_adj=aug["r_adj"].astype(np.float32), v=aug["v"].astype(np.float32),
        y=aug["y"].astype(np.int8), slot=aug["slot"].astype(np.int32), Z=Z, table=aug["table"],
        s_win=w, s_g=d["g"].astype(np.int32), s_team=d["team"], s_wp=d["wp_pop"].astype(np.float32),
        s_day=d["day"], s_pid=d["pid"].astype(np.int32), s_hero=d["hero"].astype(np.int8),
        s_y=d["y"].astype(np.int8), s_replay_id=d["replay_id"], s_in_sample=d["in_sample"],
        player_keys=d["player_keys"], hero_names=d["hero_names"], pulled_pid=pulled,
        n_players=np.int64(d["n_players"]), split_day=np.float64(med),
        metrics=np.array([str(m) for m in np.load(os.path.join(R.RCF_CACHE, "sb_a_shrink.npz"))["metrics"]]))
    np.savez(AUG, **out)
    print(f"[hr build] {len(out['day']):,} rows, {len(out['s_win']):,} slots, pulled {len(pulled):,}; "
          f"{time.time() - t0:.0f}s -> {AUG}", flush=True)


_CACHE = {}


def load():
    if "aug" not in _CACHE:
        z = np.load(AUG)
        _CACHE["aug"] = {k: z[k] for k in z.files}
    return _CACHE["aug"]


def params(tag):
    import p3_rcf as R
    if tag == "today":
        return R.Params.from_kernel_file(os.environ["P3_RCF_KERNELS"])
    p = os.path.join(PARAMS, f"{tag}.npz")
    if not os.path.exists(p):
        p = os.path.join(HR_RES, "ablate", f"{tag}.npz")    # fits made for this study (p3_hr_ablate.py)
    return R.Params.load(p)


def state_at(a, t, players, metrics=True, row_mask=None):
    """As-of state for the given players from every history row with day < t:
    S, P (win evidence with r_adj), N (games), X, Cm (metric sums and counts,
    unit weights as the main model). Returns dict of (n, H[, M]) arrays."""
    sel = a["day"] < t
    if row_mask is not None:
        sel &= row_mask
    pos = np.full(int(a["n_players"]), -1, np.int64)
    pos[players] = np.arange(len(players))
    rows = np.flatnonzero(sel & (pos[a["pid"]] >= 0))
    n = len(players)
    cell = pos[a["pid"][rows]] * H + a["hero"][rows].astype(np.int64)
    v = a["v"][rows].astype(np.float64)
    out = {"S": np.bincount(cell, a["r_adj"][rows] / v, n * H).reshape(n, H),
           "P": np.bincount(cell, 1.0 / v, n * H).reshape(n, H),
           "N": np.bincount(cell, None, n * H).reshape(n, H).astype(np.float64),
           "R": np.bincount(cell, a["r"][rows], n * H).reshape(n, H)}
    if metrics:
        M = a["Z"].shape[1]
        X = np.zeros((n * H, M), np.float32)
        Cc = np.zeros((n * H, M), np.float32)
        for k in range(M):
            x = a["Z"][rows, k].astype(np.float32)
            g = ~np.isnan(x)
            X[:, k] = np.bincount(cell[g], x[g], n * H)
            Cc[:, k] = np.bincount(cell[g], None, n * H)
        out["X"] = X.reshape(n, H, M)
        out["Cm"] = Cc.reshape(n, H, M)
    return out


def post_all(par, st, use_metrics=True, use_z=True, chunk=20000, wipe_win=None):
    """All-hero posterior mean and variance (residual units) for every player
    of a state, in chunks. wipe_win: optional (n, H) bool mask of cells whose
    win evidence is hidden (S = P = 0)."""
    import p3_rcf as R
    n = st["S"].shape[0]
    m_out = np.empty((n, H), np.float32)
    v_out = np.empty((n, H), np.float32)
    for a_ in range(0, n, chunk):
        b_ = min(n, a_ + chunk)
        S, P = st["S"][a_:b_].copy(), st["P"][a_:b_].copy()
        if wipe_win is not None:
            S[wipe_win[a_:b_]] = 0.0
            P[wipe_win[a_:b_]] = 0.0
        kw = {}
        if use_z and par.W is not None:
            kw["N"] = st["N"][a_:b_]
        if use_metrics and par.G is not None:
            kw["X"], kw["Cm"] = st["X"][a_:b_], st["Cm"][a_:b_]
        mm, vv = R.posterior(par, S, P, np.zeros(b_ - a_, np.int64), want_all=True, **kw)
        m_out[a_:b_], v_out[a_:b_] = mm, vv
    return m_out, v_out


def hero_roles(hero_names):
    import p3_hs_core as C
    meta = C.hero_meta(hero_names)
    return meta


def save_json(name, obj):
    os.makedirs(HR_RES, exist_ok=True)
    p = os.path.join(HR_RES, name)
    with open(p, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    print(f"  wrote {p}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build()
