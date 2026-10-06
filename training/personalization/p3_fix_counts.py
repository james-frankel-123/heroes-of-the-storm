"""
Audit fix P3-03 (and P3-12 role counts): one count contract everywhere.

Contract: games on EARLIER DAYS only (lag 1 day). It is the contract the
drafter tables and the P3_EXTENSIONS evaluations already serve with, and it
is available in production (no same-day upload-order ambiguity).

  lag1_counts(d)      n_p, n_ph per row: the player's games on days < day,
                      overall and on this hero
  lag1_key_counts     the same for any grouping key (e.g. player x fine role)
  prepare_l1(d)       drop-in for p3_hs_fit.prepare: the experience table is
                      REFIT on lag-1 counts (E window) and r_adj uses it
  run_fixed(module)   runs another p3_* script's main() with prepare (and the
                      role prior counts) replaced, results redirected to
                      results/fix/ and caches it writes redirected to cache/fix_*

The old phase-1/phase-2 convention ordered same-day games by replay_id
(upload order), and the drafter combiner was fit on those counts but served
lag-1 counts (AUDIT P3-03).
"""
import os
import sys
import importlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from p3_heroes import NUM_HEROES, HKEY
import p3_hs_core as C

# With P3_RESULTS set (the October 2026 reruns), every result of the run,
# fixed or not, goes to that one directory.
FIX_RESULTS = C.RESULTS if os.environ.get("P3_RESULTS") else os.path.join(C.RESULTS, "fix")


def lag1_key_counts(d, key):
    """Number of earlier-day rows with the same key, per row."""
    day = d["day"].astype(np.int64)
    o = np.lexsort((day, key))
    k, dd = key[o], day[o]
    comp_first = np.r_[True, (k[1:] != k[:-1])]
    grp_start = np.maximum.accumulate(np.where(comp_first, np.arange(len(o)), 0))
    # first row of the same (key, day) block
    blk_first = np.r_[True, (k[1:] != k[:-1]) | (dd[1:] != dd[:-1])]
    blk_start = np.maximum.accumulate(np.where(blk_first, np.arange(len(o)), 0))
    c = blk_start - grp_start
    out = np.empty(len(o), np.int64)
    out[o] = c
    return out


def lag1_counts(d):
    return lag1_key_counts(d, d["pid"]), lag1_key_counts(d, d["pid"] * HKEY + d["hero"])


def prepare_l1(d):
    days = d["day"]
    e_mask = d["in_sample"] & (days >= C.day_of(C.E_START))
    n_p, n_ph = lag1_counts(d)
    table = C.fit_experience(d["r"], n_p, n_ph, e_mask)
    r_adj = d["r"] - table[C.exp_bins(n_p, n_ph)]
    return e_mask, n_p, n_ph, table, r_adj


def role_prior_counts(d, key, order_second=None):
    """Replacement for p3_ph_role.prior_counts: lag-1-day counts."""
    return lag1_key_counts(d, key)


def _recency_l1_kernel():
    from numba import njit, prange

    @njit(parallel=True, cache=False)
    def k(starts, ends, day, hero, qslot, out_e20, out_e100, out_last):
        a20 = 1 - np.exp(-np.log(2) / 20.0)
        a100 = 1 - np.exp(-np.log(2) / 100.0)
        for p in prange(starts.shape[0]):
            e20 = np.zeros(NUM_HEROES)
            e100 = np.zeros(NUM_HEROES)
            w20 = 0.0
            w100 = 0.0
            last = np.full(NUM_HEROES, -1)
            r = starts[p]
            while r < ends[p]:
                b = r
                while b < ends[p] and day[b] == day[r]:
                    b += 1
                for i in range(r, b):  # queries see earlier days only
                    q = qslot[i]
                    if q >= 0:
                        for h in range(NUM_HEROES):
                            out_e20[q, h] = e20[h] / w20 if w20 > 0 else 0.0
                            out_e100[q, h] = e100[h] / w100 if w100 > 0 else 0.0
                            out_last[q, h] = day[i] - last[h] if last[h] >= 0 else -1
                for i in range(r, b):
                    h0 = hero[i]
                    for h in range(NUM_HEROES):
                        e20[h] *= 1 - a20
                        e100[h] *= 1 - a100
                    e20[h0] += a20
                    e100[h0] += a100
                    w20 = (1 - a20) * w20 + a20
                    w100 = (1 - a100) * w100 + a100
                    last[h0] = day[i]
                r = b
    return k


_RK = None


def recency_features_l1(d, qrows):
    """p3_dr_imitation.recency_features on the lag-1 contract (P3-24): the
    EWMA shares and days-since use only games on earlier days."""
    global _RK
    if _RK is None:
        _RK = _recency_l1_kernel()
    gt = np.load(os.path.join(C.CACHE, "gametime_2024q2.npz"))
    o = np.argsort(gt["replay_ids"])
    zs = gt["replay_ids"][o]
    j = np.minimum(np.searchsorted(zs, d["replay_id"]), len(zs) - 1)
    ts = np.where(zs[j] == d["replay_id"], gt["ts"][o][j], d["day"] * 86400 + 43200)
    srt = np.lexsort((d["replay_id"], ts, d["day"], d["pid"]))
    pid = d["pid"][srt]
    brk = np.flatnonzero(np.r_[True, pid[1:] != pid[:-1]])
    ends = np.r_[brk[1:], len(pid)]
    qpos = np.full(len(d["pid"]), -1, np.int64)
    qpos[qrows] = np.arange(len(qrows))
    qslot = qpos[srt]
    nq = len(qrows)
    e20 = np.zeros((nq, NUM_HEROES), np.float32)
    e100 = np.zeros((nq, NUM_HEROES), np.float32)
    last = np.zeros((nq, NUM_HEROES), np.float32)
    _RK(brk.astype(np.int64), ends.astype(np.int64), d["day"][srt].astype(np.int64),
        d["hero"][srt].astype(np.int64), qslot, e20, e100, last)
    return e20, e100, last


def run_fixed(modname, argv=None, extra_patch=None):
    os.makedirs(FIX_RESULTS, exist_ok=True)
    C.RESULTS = FIX_RESULTS
    import p3_hs_fit
    p3_hs_fit.prepare = prepare_l1
    mod = importlib.import_module(modname)
    if hasattr(mod, "prepare"):
        mod.prepare = prepare_l1
    if hasattr(mod, "prior_counts"):
        mod.prior_counts = role_prior_counts
    if hasattr(mod, "PRED"):
        mod.PRED = os.path.join(C.CACHE, "fix_" + os.path.basename(mod.PRED))
    if extra_patch:
        extra_patch(mod)
    if argv is not None:
        sys.argv = [modname] + list(argv)
    mod.main()


if __name__ == "__main__":
    # quick check of the count builder against a direct loop on a sample
    d = C.load_slots()
    n_p, n_ph = lag1_counts(d)
    rng = np.random.RandomState(0)
    for i in rng.choice(len(n_p), 2000, replace=False):
        m = (d["pid"] == d["pid"][i]) & (d["day"] < d["day"][i])
        assert n_p[i] == m.sum()
        assert n_ph[i] == (m & (d["hero"] == d["hero"][i])).sum()
    print("lag-1 counts verified on 2,000 random rows")
    _, n_p_rid, n_ph_rid = (None,) + C.experience_counts_upload_order(d)
    same = (d["day"] >= 0)
    print(f"rows whose overall count differs from the old replay_id-order count: {(n_p != n_p_rid).mean():.3f}")
