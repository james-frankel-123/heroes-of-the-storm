"""
P3 arm A2: the merged as-of history store (p3_full_history) wired into P3's
experience counts and GP skill state.

A2 = the slot table of an arm (A0 window cache or A1 cache_full) PLUS every
Heroes Profile row of the same players that is not already a slot row
(store rows with an HP source, matched on (player, replay_id) against the
slot table). Those rows have no draft, so no WP score:
  counts      n_p, n_ph (the experience offset and every count term) count
              them like slot rows
  skill state r = y - wp_proxy, v = wp_proxy (1 - wp_proxy), r_adj = r minus
              the experience offset at that row's own counts; wp_proxy is the
              hero's win rate in its calendar quarter from in-sample slot
              rows (before the WP training cutoff, shrunk to the hero's
              in-sample rate with 200 pseudo-games; quarters after the cutoff
              use the last in-sample quarter), or 0.5 (proxy "half")

As-of contract (same as the slot rows): a row enters the counts and the
state of a scored game on day t only if its day <= t - 1, so its game end is
strictly before the scored game's start, and the scored game's own row is
never an extra row (it is a slot row, excluded by replay_id). Rule "seen"
uses the row's estimated upload day (max of end and upload, p3_full_history
seen_ts) instead: only matches already uploaded when the scored game started
(A2-seen). Clock guard: slot days of games from 2026-07-09 on come from
exports made before the 2026-10-16 timezone fix and may run up to a day late
(stored end = UTC end + US Eastern offset), so an extra row from that date
on is placed on the day of end + Eastern offset. That only delays rows,
never moves one earlier.

Library (used by p3_c_tests.py and tests/test_merged.py):
  extra_rows(store, player_keys, slot_pid, slot_rid, hero_names)
  wp_proxy(d, ex, mode)
  merged_predict(d, ex, K, query_mask, rule, ...) -> per-query posterior,
      counts and the as-of record, through p3_x_common.online_predict

CLI (from training/, with P3_CACHE set to the arm cache):
  python3 personalization/p3_c_merged.py rows --store oct05
    writes $P3_CACHE/merged_rows_<store>.npz: the extra rows keyed to the
    pids of $P3_CACHE/x_slots_ext.npz (holds no battletags; gitignored cache)
  python3 personalization/p3_c_merged.py drtables --store oct05 --out <dir>
    the drafter's personal tables with the merged state in a linked cache
    <dir>; then P3_CACHE=<dir> p3_dr_drafter.py / p3_dr_analyze.py
"""
import argparse
import datetime
import hashlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import p3_hs_core as C
from p3_heroes import HKEY

DAY = 86400
CLOCK_GUARD_TS = int(datetime.datetime(2026, 7, 9, tzinfo=datetime.timezone.utc).timestamp())
PROXY_PRIOR = 200.0


# ------------------------------------------------------------------ rows

def slot_fingerprint(d):
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(d["player_keys"]).tobytes())
    h.update(np.int64(len(d["pid"])).tobytes())
    return h.hexdigest()[:16]


def guard_day(ts):
    """Day of a timestamp, delayed by the US Eastern offset from
    CLOCK_GUARD_TS on (see the module doc)."""
    import p3_full_history as FH
    ts = np.asarray(ts, np.int64)
    out = ts // DAY
    late = ts >= CLOCK_GUARD_TS
    if late.any():
        out[late] = (ts[late] + FH.eastern_offset(ts[late])) // DAY
    return out.astype(np.int32)


def extra_rows(st, player_keys, slot_pid, slot_rid, hero_names):
    """Store rows with an HP source of players in player_keys that are not
    slot rows (same pid and replay_id). Returns pid (slot table ids), hero
    (slot hero order), end_day / seen_day (clock-guarded), y, replay_id,
    src, end_ts, sorted by (end_day, pid)."""
    import p3_full_history as FH
    pk = np.asarray(player_keys, np.int64)
    o = np.argsort(pk, kind="stable")
    spk = pk[o]
    key_row = st.keys[st.kidx]
    j = np.minimum(np.searchsorted(spk, key_row), len(spk) - 1)
    sel = ((st.a["src"] & FH.SRC_HP) > 0) & (st.a["hero"] >= 0) & (spk[j] == key_row)
    idx = np.flatnonzero(sel)
    pid = o[j[idx]].astype(np.int64)
    rid = st.a["replay_id"][idx].astype(np.int64)
    assert rid.max(initial=0) < (1 << 32) and int(np.max(slot_pid, initial=0)) < (1 << 31)
    comp = (pid << 32) | rid
    slot_comp = np.unique((np.asarray(slot_pid, np.int64) << 32) | np.asarray(slot_rid, np.int64))
    k = np.minimum(np.searchsorted(slot_comp, comp), max(len(slot_comp) - 1, 0))
    is_slot = (slot_comp[k] == comp) if len(slot_comp) else np.zeros(len(comp), bool)
    idx, pid, rid = idx[~is_slot], pid[~is_slot], rid[~is_slot]
    hm = st.hero_map(hero_names)
    assert (hm >= 0).all(), "slot hero list misses a store hero"
    end_ts = st.a["end_ts"][idx]
    seen_ts = np.maximum(st.a["seen_ts"][idx], end_ts)
    ex = {"pid": pid, "hero": hm[st.a["hero"][idx].astype(np.int64)],
          "end_day": guard_day(end_ts), "seen_day": guard_day(seen_ts),
          "y": st.a["win"][idx].astype(np.float32), "replay_id": rid,
          "src": st.a["src"][idx], "end_ts": end_ts}
    s = np.lexsort((ex["pid"], ex["end_day"]))
    ex = {k: v[s] for k, v in ex.items()}
    ex["n_slot_rows_excluded"] = np.int64(is_slot.sum())
    return ex


def quarter(day):
    m = np.asarray(day, np.int64).astype("datetime64[D]").astype("datetime64[M]").astype(np.int64)
    return m // 3                                  # quarters since 1970


def wp_proxy(d, ex, mode="hero_era"):
    """Expected win probability of an extra row (no draft, so no WP)."""
    if mode == "half" or len(ex["pid"]) == 0:
        return np.full(len(ex["pid"]), 0.5)
    assert mode == "hero_era", mode
    H = len(d["hero_names"])
    m = d["in_sample"]
    hero, y = d["hero"][m], d["y"][m].astype(np.float64)
    q = quarter(d["day"][m])
    sh = np.bincount(hero, weights=y, minlength=H)
    ch = np.bincount(hero, minlength=H)
    base = (sh + 0.5 * PROXY_PRIOR) / (ch + PROXY_PRIOR)
    q0, q1 = int(q.min()), int(q.max())
    nq = q1 - q0 + 1
    cell = hero * nq + (q - q0)
    s = np.bincount(cell, weights=y, minlength=H * nq).reshape(H, nq)
    c = np.bincount(cell, minlength=H * nq).reshape(H, nq)
    tab = (s + PROXY_PRIOR * base[:, None]) / (c + PROXY_PRIOR)
    qe = quarter(ex["end_day"])
    qi = np.clip(qe - q0, 0, nq - 1)
    out = tab[ex["hero"], qi]
    out = np.where(qe < q0, base[ex["hero"]], out)
    return np.clip(out, 0.2, 0.8)


# ------------------------------------------------------------------ counts

def lag1_key_sums(day, key, w=None):
    """Per row: sum of w over rows with the same key on EARLIER days
    (w = 1: p3_fix_counts.lag1_key_counts)."""
    day = np.asarray(day, np.int64)
    o = np.lexsort((day, key))
    k, dd = key[o], day[o]
    wv = np.ones(len(o)) if w is None else np.asarray(w, np.float64)[o]
    cs = np.r_[0.0, np.cumsum(wv)]
    ar = np.arange(len(o))
    grp_start = np.maximum.accumulate(np.where(np.r_[True, k[1:] != k[:-1]], ar, 0))
    blk_start = np.maximum.accumulate(np.where(np.r_[True, (k[1:] != k[:-1]) | (dd[1:] != dd[:-1])], ar, 0))
    out = np.empty(len(o))
    out[o] = cs[blk_start] - cs[grp_start]
    return out


# ------------------------------------------------------------------ predict

def merged_predict(d, ex, K, query_mask, rule="game", proxy="hero_era", extra_K=None, lag=1,
                   table=None, ex_weight=1.0, exp_counts="merged", aug_only=False,
                   half_life=None):
    """Causal GP posterior at each query's hero with the slot rows of d plus
    the extra rows ex (None: slot rows only, i.e. the arm without HP rows).
    exp_counts: "merged" = the experience offset reads counts with the extras;
    "slots" = it reads slot-row counts only (the extras then feed only the
    skill state).
    Returns a dict over query rows (in d's row order): qidx, mean (GP), var,
    mu (experience offset), pred = mu + mean, nh, np (lag-1 counts with the
    extras), nh_slot, np_slot (slot rows only), rec (sum of raw residuals on
    the hero to date: y - WP on slot rows, y - proxy on extras), rec_slot,
    kdiag (prior variance at the query hero), extra means; plus the table."""
    import p3_x_common as X
    n = len(d["pid"])
    m = 0 if ex is None else len(ex["pid"])
    e_mask = d["in_sample"] & (d["day"] >= C.day_of(C.E_START))
    if m:
        pr = wp_proxy(d, ex, proxy)
        avail = ex["end_day"] if rule == "game" else np.maximum(ex["end_day"], ex["seen_day"])
        pid_all = np.r_[d["pid"], ex["pid"]].astype(np.int64)
        hero_all = np.r_[d["hero"], ex["hero"]].astype(np.int64)
        day_q = np.r_[d["day"], avail].astype(np.int64)
        day_g = np.r_[d["day"], ex["end_day"]].astype(np.int64)
        r_all = np.r_[d["r"], ex["y"] - pr]
    else:
        pr = np.zeros(0)
        pid_all, hero_all = d["pid"].astype(np.int64), d["hero"].astype(np.int64)
        day_q = day_g = d["day"].astype(np.int64)
        r_all = d["r"]
    kph = pid_all * HKEY + hero_all
    n_p = lag1_key_sums(day_q, pid_all)
    n_ph = lag1_key_sums(day_q, kph)
    rec = lag1_key_sums(day_q, kph, r_all)
    assert exp_counts in ("merged", "slots")
    w_cnt = None if exp_counts == "merged" else np.r_[np.ones(n), np.zeros(m)]
    if m:
        # extras' own experience at play time (end day order)
        n_p_g = lag1_key_sums(day_g, pid_all, w_cnt)[n:]
        n_ph_g = lag1_key_sums(day_g, kph, w_cnt)[n:]
    if exp_counts == "merged":
        np_s, nph_s = n_p[:n].astype(np.int64), n_ph[:n].astype(np.int64)
    else:
        np_s = lag1_key_sums(d["day"], d["pid"].astype(np.int64)).astype(np.int64)
        nph_s = lag1_key_sums(d["day"], kph[:n]).astype(np.int64)
    if table is None:
        table = C.fit_experience(d["r"], np_s, nph_s, e_mask)
    r_adj = np.empty(n + m)
    r_adj[:n] = d["r"] - table[C.exp_bins(np_s, nph_s)]
    v_all = np.empty(n + m)
    v_all[:n] = d["v"]
    if m:
        r_adj[n:] = (ex["y"] - pr) - table[C.exp_bins(n_p_g.astype(np.int64), n_ph_g.astype(np.int64))]
        # ex_weight < 1 down-weights extra rows (variance / w); 1 = same as a slot row
        v_all[n:] = pr * (1 - pr) / ex_weight
    o = np.argsort(day_q, kind="stable")
    pos = np.empty(n + m, np.int64)
    pos[o] = np.arange(n + m)
    y_all = np.r_[d["y"], ex["y"]] if m else d["y"]
    aug = {"day": day_q[o], "pid": pid_all[o], "hero": hero_all[o], "v": v_all[o],
           "y": y_all[o], "r": r_all[o], "n_players": d["n_players"]}
    qrows = np.flatnonzero(query_mask)
    qm = np.zeros(n + m, bool)
    qm[pos[qrows]] = True
    if aug_only:
        # the augmented, day-sorted rows for another state walk (drafter tables)
        return {"aug": aug, "r_adj": r_adj[o], "qmask": qm, "order": o, "table": table}
    qi, mean, var, Nh, Np, exm = X.online_predict(aug, np.ones(n + m, bool), qm, K, r_adj[o], lag=lag,
                                                  extra=extra_K, half_life=half_life)
    srow = o[qi]
    assert (srow < n).all()
    # the state counts must equal the lag-1 counts when lag is 1
    if lag == 1:
        assert np.array_equal(Nh.astype(np.int64), n_ph[srow].astype(np.int64))
        assert np.array_equal(Np.astype(np.int64), n_p[srow].astype(np.int64))
    # slot-only counts and record (what the arm without extras sees)
    if m:
        kph_s = kph[:n]
        nh_slot = lag1_key_sums(d["day"], kph_s)[srow]
        np_slot = lag1_key_sums(d["day"], d["pid"].astype(np.int64))[srow]
        rec_slot = lag1_key_sums(d["day"], kph_s, d["r"])[srow]
    else:
        nh_slot, np_slot, rec_slot = n_ph[srow], n_p[srow], rec[srow]
    if exp_counts == "merged":
        mu = table[C.exp_bins(Np.astype(np.int64), Nh.astype(np.int64))]
    else:
        mu = table[C.exp_bins(np_s[srow], nph_s[srow])]
    hq = d["hero"][srow]
    out = {"qidx": srow, "mean": mean, "var": var, "mu": mu, "pred": mu + mean,
           "nh": Nh, "np": Np, "nh_slot": nh_slot.astype(np.float32), "np_slot": np_slot.astype(np.float32),
           "rec": rec[srow].astype(np.float32), "rec_slot": rec_slot.astype(np.float32),
           "kdiag": K[hq, hq], "table": table, "n_extra": m,
           "extra_means": exm}
    # results come back in d's row order
    s = np.argsort(srow, kind="stable")
    for k, v in list(out.items()):
        if isinstance(v, np.ndarray) and v.shape[:1] == (len(srow),):
            out[k] = v[s]
    out["extra_means"] = [e[s] for e in exm]
    return out


def ex_static_state(d, ex, e_last_day, n_players, H, r_adj_slot, r_adj_ex, v_ex):
    """Static E-window state (slot E rows plus extras with day in E), for
    refitting the kernel log-variances on merged history."""
    e_mask = d["in_sample"] & (d["day"] >= C.day_of(C.E_START))
    ein = (ex["end_day"] >= C.day_of(C.E_START)) & (ex["end_day"] <= e_last_day)
    idx = np.r_[d["pid"][e_mask] * H + d["hero"][e_mask], ex["pid"][ein] * H + ex["hero"][ein]]
    w_s = np.r_[r_adj_slot[e_mask] / d["v"][e_mask], r_adj_ex[ein] / v_ex[ein]]
    w_p = np.r_[1 / d["v"][e_mask], 1 / v_ex[ein]]
    size = int(n_players) * H
    S = np.bincount(idx, weights=w_s, minlength=size).reshape(int(n_players), H)
    P = np.bincount(idx, weights=w_p, minlength=size).reshape(int(n_players), H)
    return S, P


# ------------------------------------------------------------------ drafter

def build_dr_tables(store, out, exp_counts="slots", rule="game", proxy="hero_era"):
    """The personalized drafter's per-slot tables (p3_dr_core
    dr_personal_post.npz) with the merged skill state, in a new cache
    directory `out` that links every other file of $P3_CACHE. The pool,
    off-role flags and counts stay slot-row based (exp_counts "slots"), so
    only the skill state differs from the A0 drafter."""
    import p3_dr_core as D
    import p3_full_history as FH
    t0 = time.time()
    d = C.load_slots()
    H = len(d["hero_names"])
    st = FH.Store.load(store if store.endswith(".npz") else os.path.join(C.CACHE, "full_history", store, "history.npz"))
    ex = extra_rows(st, d["player_keys"], d["pid"], d["replay_id"], d["hero_names"])
    del st
    L = D.load_lobbies()
    rows = np.unique(L["steps"][:, :, 3][L["steps"][:, :, 3] >= 0])
    q = np.zeros(len(d["pid"]), bool)
    q[rows] = True
    A = merged_predict(d, ex, None, q, rule=rule, proxy=proxy, exp_counts=exp_counts, aug_only=True)
    aug, o, table = A["aug"], A["order"], A["table"]
    qi, S, P, _, _, _, _ = C.online_state(aug, np.ones(len(o), bool), A["qmask"], H, A["r_adj"])
    srow = o[qi]
    assert (srow < len(d["pid"])).all() and np.array_equal(np.sort(srow), rows)
    # slot-row count matrix per query, lag 1 day (as p3_dr_core.personal_tables)
    Nst = np.zeros((int(d["n_players"]), H), np.float32)
    N = np.empty((len(srow), H), np.float32)
    day = d["day"]
    qd = day[srow]
    assert (np.diff(qd) >= 0).all()
    ud, qs = np.unique(qd, return_index=True)
    qe = np.r_[qs[1:], len(srow)]
    ap = 0
    for t, a_, b_ in zip(ud, qs, qe):
        e = np.searchsorted(day, t - 1, side="right")
        if e > ap:
            np.add.at(Nst, (d["pid"][ap:e], d["hero"][ap:e]), 1.0)
            ap = e
        N[a_:b_] = Nst[d["pid"][srow[a_:b_]]]
    m_all, v_all = C.gp_all(S, P, D.kernel_best())
    del S, P
    meta = C.hero_meta(d["hero_names"])
    fine = meta["fine"]
    n_p = N.sum(1)
    nb = len(C.NPH_EDGES)
    bp = np.searchsorted(C.NP_EDGES, n_p, side="right") - 1
    bh = np.searchsorted(C.NPH_EDGES, N, side="right") - 1
    s = table[bp[:, None] * nb + bh] + m_all
    fine_cnt = np.zeros((len(srow), len(meta["fine_names"])), np.float32)
    for f in range(len(meta["fine_names"])):
        fine_cnt[:, f] = N[:, fine == f].sum(1)
    share = fine_cnt[:, fine] / np.maximum(n_p[:, None], 1)
    off = (n_p[:, None] >= 50) & (share < 0.10)
    srt = np.argsort(srow)
    out = os.path.abspath(out)
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(C.CACHE):
        if f.startswith("dr_personal_post") or f.startswith("dr_runs") or f.startswith("merged_rows") \
                or f == "full_history":
            continue
        dst = os.path.join(out, f)
        if not os.path.lexists(dst):
            os.symlink(os.path.join(C.CACHE, f), dst)
    np.savez(os.path.join(out, "dr_personal_post.npz"), rows=srow[srt], s=s[srt].astype(np.float32),
             sd=np.sqrt(v_all[srt]).astype(np.float16), off=off[srt], pool=(N > 0)[srt],
             n=N[srt].astype(np.float32), n_p=n_p[srt])
    print(f"drafter tables with merged state: {len(srow):,} slots, {len(ex['pid']):,} extra rows "
          f"-> {out} ({time.time() - t0:.0f}s)", flush=True)


# ------------------------------------------------------------------ CLI

def build_rows(store, slots_path=None):
    """store: a tag under $P3_CACHE/full_history or a history.npz path."""
    import p3_full_history as FH
    import p3_x_common as X
    t0 = time.time()
    slots_path = slots_path or X.EXT
    z = np.load(slots_path)
    d = {k: z[k] for k in ("pid", "replay_id", "player_keys", "hero_names")}
    if store.endswith(".npz"):
        st = FH.Store.load(store)
        store_tag = os.path.basename(os.path.dirname(os.path.abspath(store)))
    else:
        st = FH.Store.load(os.path.join(C.CACHE, "full_history", store, "history.npz"))
        store_tag = store
    ex = extra_rows(st, d["player_keys"], d["pid"], d["replay_id"], d["hero_names"])
    pulled = np.isin(d["player_keys"], st.a["pulled_keys"])
    out = os.path.join(C.CACHE, f"merged_rows_{store_tag}.npz")
    np.savez(out + ".tmp.npz", **{k: v for k, v in ex.items()}, pulled_pid=np.flatnonzero(pulled),
             fingerprint=np.array(slot_fingerprint({"player_keys": d["player_keys"], "pid": d["pid"]})),
             slots_path=np.array(os.path.abspath(slots_path)))
    os.replace(out + ".tmp.npz", out)
    print(f"wrote {out}: {len(ex['pid']):,} extra rows for {len(np.unique(ex['pid'])):,} players "
          f"({int(ex['n_slot_rows_excluded']):,} store rows dropped as slot rows); "
          f"pulled players in the slot table {int(pulled.sum()):,}; {time.time() - t0:.0f}s", flush=True)
    return out


def load_rows(path, d):
    z = np.load(path)
    fp = str(z["fingerprint"])
    assert fp == slot_fingerprint(d), f"{path} was built for another slot table"
    ex = {k: z[k] for k in z.files if k not in ("fingerprint", "slots_path", "pulled_pid", "n_slot_rows_excluded")}
    return ex, z["pulled_pid"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["rows", "drtables"])
    ap.add_argument("--store", required=True,
                    help="merged store: tag ($P3_CACHE/full_history/<tag>/history.npz) or a history.npz path")
    ap.add_argument("--slots", help="slot table (default $P3_CACHE/x_slots_ext.npz)")
    ap.add_argument("--out", help="drtables: new cache directory")
    ap.add_argument("--exp-counts", default="slots", choices=["slots", "merged"])
    ap.add_argument("--rule", default="game", choices=["game", "seen"])
    a = ap.parse_args()
    if a.what == "rows":
        build_rows(a.store, a.slots)
    else:
        build_dr_tables(a.store, a.out, a.exp_counts, a.rule)


if __name__ == "__main__":
    main()
