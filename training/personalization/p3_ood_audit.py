"""
P3 out-of-support audit (paper 2): does the personalized drafter exploit sparse cells of its
player-value function s(p, h), the way paper 1's MCTS exploited unsupported hero combinations?
Measurement only; nothing here changes a drafter, judge or result file.

Build def1027, headline tournament (results/p3t_def1027_headline). Windows:
  drafter state  s(p, h) as of the day before each lobby (lobbies 2026-05-23 to 06-30)
  judge window   2026-07-01 to 09-11: realized edges y - WP_pop (slot rows, post-snapshot,
                 not the sealed personalization build 2.55.17.98025); after every drafter state
  later window   2026-09-28 to 10-26, patch 2.57 Storm League games from the weekly build wk261101
                 (its slot table excludes the 5,000 sealed 2.57 replays; they and the sealed build are
                 filtered again here through p3_sealed): after every judge state. The def1027 table
                 has almost nothing between 09-12 and 09-27 that is not the sealed build.

Stages (from training/, main box, nice 19, cores 44-47, single-threaded BLAS):
  prep      rebuild the player states of the headline lobbies and of every tournament-window
            player locally (p3_rcf_states.answer on new query files; the judge states are the
            existing answer files cache_def_def1027/judge_a_{orig,gd,v2}), past and realized
            records; writes <work>/cells.npz
  verify    reproduce the headline score files (judge-free record, J's personal sum, the
            drafter's own V) from the rebuilt states
  exposure  part 1: games-on-hero, record, sd and s of drafted cells, by arm
  curse     part 2: predicted s vs realized edge, all cells and selected cells, by n
  judges    part 3: each judge's personal term vs realized edge, by n
  record    part 4: the judge-free paired record by the n of the drafted cells
  report    print the headline numbers (results/P3_OOD_AUDIT.md is written by hand from the JSON)
Inputs not in this checkout: the headline pass's lobby table, drafts and scores
(cache_def_def1027_headline/p3t/{lobbies.npz,tourney,scores} on the workers), copied to --hl.
Output: results/p3_ood_audit.json (aggregates only, no player keys).
"""
import argparse
import json
import os
import sys
import time

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING = os.path.dirname(HERE)
os.environ.setdefault("P1_TIERS", "site")
_D = "def1027"
os.environ.setdefault("P3_CACHE", os.path.join(HERE, f"cache_def_{_D}"))
os.environ.setdefault("P3_RESULTS", os.path.join(HERE, "results", f"def_{_D}"))
os.environ.setdefault("P3_RCF_KERNELS", os.path.join(os.environ["P3_CACHE"], "hs_kernels.npz"))
if "P3_RCF_ROWS" not in os.environ:
    _rows = sorted(f for f in os.listdir(os.environ["P3_CACHE"]) if f.startswith(f"merged_rows_{_D}_"))
    os.environ["P3_RCF_ROWS"] = os.path.join(os.environ["P3_CACHE"], _rows[0])
sys.path.insert(0, HERE)
sys.path.insert(0, TRAINING)
import numpy as np

import p3_hs_core as C

H = 90
RES = os.path.join(HERE, "results")
OUT_JSON = os.path.join(RES, "p3_ood_audit.json")
OUT_MD = os.path.join(RES, "P3_OOD_AUDIT.md")
HL_RES = os.path.join(RES, "p3t_def1027_headline")
JUDGE_A = {"J": "judge_a_orig", "gd": "judge_a_gd", "v2": "judge_a_v2"}
JUDGE_JSON = {"J": HL_RES, "gd": HL_RES + "_judgegd", "v2": HL_RES + "_judgev2"}
SEALED_BUILD = "2.55.17.98025"
J0, J1 = C.day_of("2026-07-01"), C.day_of("2026-09-11")
L0, L1 = C.day_of("2026-09-28"), C.day_of("2026-10-26")
LATER_TABLE = os.path.join(HERE, "cache_def_wk261101", "x_slots_ext.npz")
SEL = "x-combo1-dr-chance"
P1 = "pMCTS-cur"
ARMS = [SEL, P1, "aMCTS-pp", "aMCTS-cur", "a1step-cur", "p1step-cur", "DraftRec", "DraftRec-pv", "GD", "Real"]
BINS = [(0, 0, "0"), (1, 1, "1"), (2, 4, "2-4"), (5, 19, "5-19"), (20, 10 ** 9, "20+")]
SKILL_PICK = [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1]
SKILL_TEAM = [0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1]
HOT_R, HOT_W, HOT_N = 0.2, 0.8, 5
BOOT = 1000


def _perms():
    import itertools
    return np.array(list(itertools.permutations(range(5))), np.int64)


PERMS = _perms()


def best_assign(s5, o5, heroes, c2, c3):
    """p3t_judge.best_assign: the c2*S + c3*O-best of the 120 hero-to-player assignments.
    Returns S, O, perm (perm[i] = index into heroes of player i)."""
    hs = np.asarray(heroes)
    sv = s5[:, hs]
    ov = o5[:, hs]
    i = np.arange(5)[None, :]
    sp = sv[i, PERMS].sum(1)
    op = ov[i, PERMS].sum(1)
    b = int(np.argmax(c2 * sp + c3 * op))
    return sp[b], op[b], PERMS[b]


def bin_of(n):
    n = np.asarray(n)
    out = np.full(n.shape, -1, np.int64)
    for k, (lo, hi, _) in enumerate(BINS):
        out[(n >= lo) & (n <= hi)] = k
    return out


def off_role_shared(n_sh):
    """p3t_data.off_role on merged counts in shared.HEROES order."""
    from shared import HEROES
    fine = C.hero_meta(HEROES)["fine"]
    n_p = n_sh.sum(1)
    cnt = np.zeros((len(n_sh), fine.max() + 1), np.float32)
    for f in range(fine.max() + 1):
        cnt[:, f] = n_sh[:, fine == f].sum(1)
    share = cnt[:, fine] / np.maximum(n_p[:, None], 1)
    return (n_p[:, None] >= 50) & (share < 0.10)


# ------------------------------------------------------------------ prep

def _query(keys, as_of):
    n = len(keys)
    return {"player_key": keys.astype(np.int64), "as_of_day": as_of.astype(np.int32),
            "ev_lo": np.full(n, -1, np.int32), "ev_parity": np.zeros(n, np.int8),
            "exclude_lo": np.full(n, -1, np.int32), "exclude_hi": np.full(n, -1, np.int32),
            "half": np.zeros(n, np.int8), "hero": np.full(n, -1, np.int16)}


def _cell_sums(pid_rows, hero_rows, val_rows, upid, n_pl):
    """(len(upid), H) sums of val over rows, rows already filtered; hero in shared order."""
    pos = np.full(n_pl, -1, np.int64)
    pos[upid] = np.arange(len(upid))
    p = pos[pid_rows]
    k = p >= 0
    cell = p[k] * H + hero_rows[k]
    return np.bincount(cell, weights=val_rows[k], minlength=len(upid) * H).reshape(-1, H)


def prep(a):
    import p3_rcf as R
    import p3_rcf_states as S
    import p3_sealed
    t0 = time.time()
    work = a.work
    os.makedirs(os.path.join(work, "q"), exist_ok=True)
    L = np.load(os.path.join(a.hl, "lobbies.npz"))
    gi = np.load(os.path.join(a.hl, "tourney", "lobbies.npz"))["gi"]
    n = len(gi)
    steps = L["steps"][gi]
    first = L["first_team"][gi].astype(np.int64)
    srow = np.zeros((n, 10), np.int64)
    real_h = np.zeros((n, 10), np.int64)
    for i in range(n):
        cnt = [0, 0]
        j = 0
        for k in range(16):
            h, ty, tm, r = steps[i, k]
            if ty == 1:
                kt = tm ^ first[i]
                assert kt == SKILL_TEAM[k] and SKILL_PICK[k] == 1
                sl = 5 * kt + cnt[kt]
                cnt[kt] += 1
                srow[i, sl] = L["slot_rows"][gi[i], j]
                assert srow[i, sl] == r
                real_h[i, sl] = h
                j += 1
    lday = L["day"][gi].astype(np.int64)
    d, aug = R.load_aug("merged")
    print(f"loaded slot table and merged history ({time.time() - t0:.0f}s)", flush=True)
    from shared import HEROES
    sh = np.array([HEROES.index(str(x)) for x in d["hero_names"]], np.int64)   # d hero -> shared
    assert (sh[d["hero"][srow]] == real_h).all(), "real picks do not join to their slot rows"
    keys_all = np.asarray(d["player_keys"]).astype(np.int64)
    spid = d["pid"][srow].astype(np.int64)
    skey = keys_all[spid]
    # --- query files: (1) the drafter state of each headline slot (as p3t_data states, vintage cur)
    qc_k = skey.ravel()
    qc_d = np.repeat(lday - 1, 10)
    uk, inv_c = np.unique(qc_k * 100000 + qc_d, return_inverse=True)
    qcur = _query(uk // 100000, uk % 100000)
    ref = np.load(os.path.join(C.CACHE, "judge_q", "p3t_data_states__0.npz"))
    refk = ref["player_key"].astype(np.int64) * 100000 + ref["as_of_day"].astype(np.int64)
    assert np.isin(uk, refk).all(), "headline drafter queries are not in the definitive cur query file"
    for f in ("ev_lo", "ev_parity", "exclude_lo", "exclude_hi", "half", "hero"):
        assert np.unique(ref[f]).size == 1 and ref[f][0] == qcur[f][0], f
    # (2) every tournament-window player (the judge-state players) as of 2026-06-30, drafter fit
    jq = np.load(os.path.join(C.CACHE, "judge_q", "p3t_data_states__3.npz"))
    bkeys = jq["player_key"].astype(np.int64)
    assert (jq["as_of_day"] == J1).all() and len(np.unique(bkeys)) == len(bkeys)
    qasof = _query(bkeys, np.full(len(bkeys), J0 - 1))
    np.savez(os.path.join(work, "q", "cur_headline.npz"), **qcur)
    np.savez(os.path.join(work, "q", "asof_0630.npz"), **qasof)
    adir = os.path.join(work, "a")
    if not (os.path.exists(os.path.join(adir, "cur_headline.npz")) and os.path.exists(os.path.join(adir, "asof_0630.npz"))):
        ctx = {"d": d, "aug": aug, "cache": {}, "mcache": {}}
        S.answer(os.path.join(work, "q"), adir, ctx=ctx)
        del ctx
    ac = np.load(os.path.join(adir, "cur_headline.npz"))
    ab = np.load(os.path.join(adir, "asof_0630.npz"))
    assert ac["known"].all() and ab["known"].all()
    print(f"answers ready ({time.time() - t0:.0f}s)", flush=True)
    # --- past records (merged history, every row with day <= as-of day): sums of r and y, counts
    ko = np.argsort(keys_all)
    pid_of = lambda k: ko[np.searchsorted(keys_all[ko], k)]
    upid_h = np.unique(spid)
    hpos = {int(p): i for i, p in enumerate(upid_h)}
    sel = np.flatnonzero(np.isin(aug["pid"], upid_h))
    a_pid, a_day, a_hero = aug["pid"][sel].astype(np.int64), aug["day"][sel].astype(np.int64), sh[aug["hero"][sel]]
    a_r, a_y = aug["r"][sel], aug["y"][sel].astype(np.float64)
    o = np.lexsort((a_day, a_pid))
    a_pid, a_day, a_hero, a_r, a_y = a_pid[o], a_day[o], a_hero[o], a_r[o], a_y[o]
    past = np.zeros((3, n, 10, H))
    for i in range(n):
        for sl in range(10):
            lo = np.searchsorted(a_pid, spid[i, sl])
            hi = np.searchsorted(a_pid, spid[i, sl], side="right")
            e = lo + np.searchsorted(a_day[lo:hi], lday[i] - 1, side="right")
            hh = a_hero[lo:e]
            for c, w in enumerate((a_r[lo:e], a_y[lo:e], np.ones(e - lo))):
                past[c, i, sl] = np.bincount(hh, weights=w, minlength=H)
    del sel, a_pid, a_day, a_hero, a_r, a_y
    bpid = pid_of(bkeys)
    assert (keys_all[bpid] == bkeys).all()
    selb = np.flatnonzero(np.isin(aug["pid"], bpid) & (aug["day"] <= J0 - 1))
    pb = [_cell_sums(aug["pid"][selb], sh[aug["hero"][selb]], w, bpid, int(d["n_players"]))
          for w in (aug["r"][selb], aug["y"][selb].astype(np.float64), np.ones(len(selb)))]
    del selb
    print(f"past records ({time.time() - t0:.0f}s)", flush=True)
    # --- realized records from slot rows (as p3t_data states: post, not the sealed build; and not a sealed id)
    vs = [str(x) for x in d["versions"]]
    sb = vs.index(SEALED_BUILD) if SEALED_BUILD in vs else -999
    sealed = np.isin(d["replay_id"].astype(np.int64), p3_sealed.sealed_ids())
    base = d["post"].astype(bool) & (d["version"] != sb) & ~sealed
    rec = {}
    w = np.flatnonzero(base & (d["day"] >= J0) & (d["day"] <= J1))
    win = {"J": (d["pid"][w].astype(np.int64), sh[d["hero"][w]], d["r"][w], d["replay_id"][w], d["day"][w])}
    # later window: 2.57 games 2026-09-28 to 10-26 from the weekly build wk261101 (its slot table excludes the
    # sealed list; the sealed ids and the sealed build are filtered again here), players joined by player key
    zl = np.load(LATER_TABLE)
    ldy = zl["day"]
    lvs = [str(x) for x in zl["versions"]]
    lsb = lvs.index(SEALED_BUILD) if SEALED_BUILD in lvs else -999
    lrid = zl["replay_id"].astype(np.int64)
    m = (ldy >= L0) & (ldy <= L1) & zl["post"].astype(bool) & (zl["version"] != lsb)
    m &= ~np.isin(lrid, p3_sealed.sealed_ids())
    w = np.flatnonzero(m)
    lkeys = np.asarray(zl["player_keys"]).astype(np.int64)[zl["pid"][w].astype(np.int64)]
    j = np.minimum(np.searchsorted(keys_all[ko], lkeys), len(keys_all) - 1)
    known = keys_all[ko][j] == lkeys
    lsh = np.array([HEROES.index(str(x)) for x in zl["hero_names"]], np.int64)
    w, j = w[known], j[known]
    win["later"] = (ko[j].astype(np.int64), lsh[zl["hero"][w]], (zl["y"][w] - zl["wp"][w]).astype(np.float64),
                    lrid[w], ldy[w])
    del zl, ldy, lrid, m
    for nm, (pr, hr, rr, rid_, dd_) in win.items():
        rec[nm + "_b"] = [_cell_sums(pr, hr, v, bpid, int(d["n_players"])) for v in (rr, np.ones(len(pr)))]
        rec[nm + "_h"] = [_cell_sums(pr, hr, v, upid_h, int(d["n_players"])) for v in (rr, np.ones(len(pr)))]
        rec[nm + "_games"] = int(len(np.unique(rid_)))
        rec[nm + "_days"] = [int(dd_.min()), int(dd_.max())] if len(pr) else None
        print(f"window {nm}: {len(pr):,} slot rows, {rec[nm + '_games']:,} games, days {rec[nm + '_days']}", flush=True)
    rec["sealed_ids_in_slot_table"] = int(sealed.sum())
    hidx = np.array([[hpos[int(p)] for p in row] for row in spid])
    # --- judge states (existing answer files, as of 2026-09-11, judge-window evidence only)
    jpos = {int(k): i for i, k in enumerate(bkeys)}
    jrow_h = np.array([[jpos[int(k)] for k in row] for row in skey])
    judge = {}
    for nm, sub in JUDGE_A.items():
        z = np.load(os.path.join(C.CACHE, sub, "p3t_data_states__3.npz"))
        assert np.array_equal(z["player_key"], jq["player_key"]) and z["known"].all()
        judge[nm] = (z["s"].astype(np.float32), z["sd"].astype(np.float32), z["n"].astype(np.float32))
    js_n = judge["J"][2]
    for nm in judge:
        assert np.array_equal(judge[nm][2], js_n)
    # --- save
    cur_s = ac["s"][inv_c].astype(np.float16).astype(np.float32).reshape(n, 10, H)   # stored float16 as the run
    cur_sd = ac["sd"][inv_c].astype(np.float16).astype(np.float32).reshape(n, 10, H)
    cur_n = ac["n"][inv_c].astype(np.uint16).astype(np.float32).reshape(n, 10, H)
    assert np.array_equal(cur_n, past[2].astype(np.float32)), "answer n differs from the past merged counts"
    out = dict(
        gi=gi, lday=lday, map=L["map"][gi], tier=L["tier"][gi], first=first, real_acts=steps[:, :, 0],
        real_h=real_h, slot_player=hidx.astype(np.int32),
        cur_s=cur_s, cur_sd=cur_sd, cur_n=cur_n,
        cur_off=off_role_shared(cur_n.reshape(-1, H)).reshape(n, 10, H),
        past_rsum=past[0].astype(np.float32), past_ysum=past[1].astype(np.float32),
        recJ_sum=rec["J_h"][0][hidx].astype(np.float32), recJ_n=rec["J_h"][1][hidx].astype(np.float32),
        recL_sum=rec["later_h"][0][hidx].astype(np.float32), recL_n=rec["later_h"][1][hidx].astype(np.float32),
        b_player=np.arange(len(bkeys), dtype=np.int32),
        b_s=ab["s"].astype(np.float32), b_sd=ab["sd"].astype(np.float32), b_n=ab["n"].astype(np.float32),
        b_past_rsum=pb[0].astype(np.float32), b_past_ysum=pb[1].astype(np.float32),
        b_recJ_sum=rec["J_b"][0].astype(np.float32), b_recJ_n=rec["J_b"][1].astype(np.float32),
        b_recL_sum=rec["later_b"][0].astype(np.float32), b_recL_n=rec["later_b"][1].astype(np.float32),
        jn=js_n, j_off=off_role_shared(js_n),
        meta=np.array(json.dumps({k: rec[k] for k in rec if not k.endswith(("_b", "_h"))})))
    assert np.array_equal(out["b_n"], pb[2].astype(np.float32)), "as-of answer n differs from the past merged counts"
    for nm, (s_, sd_, _) in judge.items():
        out[f"j_{nm}_s"], out[f"j_{nm}_sd"] = s_, sd_
        out[f"hj_{nm}_s"] = s_[jrow_h]
    out["hj_n"] = js_n[jrow_h]
    out["hj_off"] = out["j_off"][jrow_h]
    out["h_brow"] = jrow_h.astype(np.int32)        # headline slot -> row of the tournament-window player set
    np.savez(os.path.join(work, "cells.npz"), **out)
    print(f"cells.npz written ({time.time() - t0:.0f}s)", flush=True)


# ------------------------------------------------------------------ shared loaders

class Ctx:
    def __init__(self, a):
        self.a = a
        z = np.load(os.path.join(a.work, "cells.npz"))
        self.z = {k: z[k] for k in z.files}
        self.n = len(self.z["gi"])
        self.judges = {k: json.load(open(os.path.join(v, "judge.json"))) for k, v in JUDGE_JSON.items()}
        vint = json.load(open(os.path.join(HL_RES, "p3t_vintages.json")))
        self.comb = vint["cur"]["combiner"]
        self._files = {}

    def acts(self, a_, b_):
        k = (a_, b_)
        if k not in self._files:
            p = os.path.join(self.a.hl, "tourney", f"{a_}__{b_}.npz")
            self._files[k] = np.load(p)["acts"] if os.path.exists(p) else None
        return self._files[k]

    def scores(self, a_, b_):
        p = os.path.join(self.a.hl, "scores", f"{a_}__{b_}.npz")
        return np.load(p) if os.path.exists(p) else None


def team_heroes(acts, i, t):
    return [int(acts[i, k]) for k in range(16) if SKILL_PICK[k] and SKILL_TEAM[k] == t]


def assign_cells(X, acts, i, t, how):
    """Hero (shared index) of each player of kernel team t, slot order, under an assignment rule."""
    z = X.z
    hs = team_heroes(acts, i, t)
    sl = slice(5 * t, 5 * t + 5)
    if how == "own":           # the drafter's value model: combiner b2 * S + b3 * O on the cur state
        _, _, perm = best_assign(z["cur_s"][i, sl], z["cur_off"][i, sl].astype(float), hs, X.comb[2], X.comb[3])
    elif how == "of":          # outcome-free: most earlier games (the judge-free record's rule)
        _, _, perm = best_assign(np.log1p(z["cur_n"][i, sl]), np.zeros((5, H)), hs, 1.0, 0.0)
    elif how == "J":
        jd = X.judges["J"]
        _, _, perm = best_assign(z["hj_J_s"][i, sl], z["hj_off"][i, sl].astype(float), hs,
                                 jd["c2_skill"], jd["c3_offrole"])
    else:
        raise ValueError(how)
    return np.array([hs[p] for p in perm])


def wls_cluster(x, y, w, g):
    """Weighted LS y = a + b x, SEs clustered by g. Returns a, b, se_a, se_b."""
    X = np.column_stack([np.ones_like(x), x])
    XtWX = X.T @ (X * w[:, None])
    beta = np.linalg.solve(XtWX, X.T @ (w * y))
    u = y - X @ beta
    sc = X * (w * u)[:, None]
    _, gi = np.unique(g, return_inverse=True)
    S = np.zeros((gi.max() + 1, 2))
    np.add.at(S, gi, sc)
    Ainv = np.linalg.inv(XtWX)
    V = Ainv @ (S.T @ S) @ Ainv
    G = gi.max() + 1
    V *= G / max(G - 1, 1)
    return float(beta[0]), float(beta[1]), float(np.sqrt(V[0, 0])), float(np.sqrt(V[1, 1]))


def wmean_cluster(v, w, g):
    m = float(np.sum(w * v) / np.sum(w))
    _, gi = np.unique(g, return_inverse=True)
    S = np.bincount(gi, weights=w * (v - m))
    G = len(S)
    return m, float(np.sqrt(np.sum(S ** 2) * G / max(G - 1, 1)) / np.sum(w))


def calib_block(pred, real_sum, real_n, g, nbin, scale=1.0, min_cells=30):
    """Game-weighted calibration of realized edge (cell mean of y - WP_pop) on a prediction, overall
    and by n bin: slope (realized on predicted), mean predicted, mean realized, gap = realized - predicted
    (all in WP points), SEs clustered by player."""
    ok = real_n > 0
    out = {}
    for name, m in [("all", ok)] + [(BINS[k][2], ok & (nbin == k)) for k in range(len(BINS))]:
        if m.sum() < min_cells:
            out[name] = {"cells": int(m.sum())}
            continue
        p = scale * pred[m].astype(np.float64)
        e = real_sum[m] / real_n[m]
        w = real_n[m].astype(np.float64)
        r = {"cells": int(m.sum()), "games": int(w.sum()), "players": int(len(np.unique(g[m])))}
        mp, sp = wmean_cluster(p, w, g[m])
        me, se_ = wmean_cluster(e, w, g[m])
        gap, sg = wmean_cluster(e - p, w, g[m])
        r.update(pred_pp=round(100 * mp, 3), pred_se=round(100 * sp, 3), real_pp=round(100 * me, 3),
                 real_se=round(100 * se_, 3), gap_pp=round(100 * gap, 3), gap_se=round(100 * sg, 3))
        if np.std(p) > 1e-9:
            _, b, _, sb = wls_cluster(p, e, w, g[m])
            r.update(slope=round(b, 3), slope_se=round(sb, 3))
        out[name] = r
    return out


def save_part(key, val):
    j = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {}
    j[key] = val
    j["_about"] = ("P3 out-of-support audit (p3_ood_audit.py), build def1027, headline tournament. "
                   "WP points = 100 x fraction of win probability on the y - WP_pop scale. "
                   f"n bins = merged games on the hero before the as-of day: {[b[2] for b in BINS]}. "
                   "No player keys.")
    json.dump(j, open(OUT_JSON, "w"), indent=1)


# ------------------------------------------------------------------ verify

def verify(a):
    """Rebuild the score files' judge-free record (rec_slot), J's personal sum (dS_judge) and the
    drafter's own V from the local states; they must match the headline files."""
    X = Ctx(a)
    z = X.z
    jd = X.judges["J"]
    out = {}
    for a_, b_ in [(SEL, P1), (P1, SEL), (SEL, "Real"), ("Real", SEL), ("aMCTS-pp", P1)]:
        acts = X.acts(a_, b_)
        sc = X.scores(a_, b_)
        n = X.n
        rec = np.full((n, 10), np.nan)
        dS = np.zeros(n)
        dSo = np.zeros(n)
        dOo = np.zeros(n)
        for i in range(n):
            for t in (0, 1):
                sl = slice(5 * t, 5 * t + 5)
                hs = team_heroes(acts, i, t)
                sg = 1 if t == 0 else -1
                S, _, _ = best_assign(z["hj_J_s"][i, sl], z["hj_off"][i, sl].astype(float), hs, jd["c2_skill"], jd["c3_offrole"])
                dS[i] += sg * S
                h_of = assign_cells(X, acts, i, t, "of")
                for p in range(5):
                    c = z["recJ_n"][i, 5 * t + p, h_of[p]]
                    if c > 0:
                        rec[i, 5 * t + p] = z["recJ_sum"][i, 5 * t + p, h_of[p]] / c
                So, Oo, _ = best_assign(z["cur_s"][i, sl], z["cur_off"][i, sl].astype(float), hs, X.comb[2], X.comb[3])
                dSo[i] += sg * So
                dOo[i] += sg * Oo
        wp = sc["own_wp_a"]
        lo = np.log(np.clip(wp, 1e-6, 1 - 1e-6) / np.clip(1 - wp, 1e-6, 1 - 1e-6))
        b = X.comb
        V = 1 / (1 + np.exp(-(b[0] + b[1] * lo + b[2] * dSo + b[3] * dOo)))
        own_ok = SEL == a_ or a_ == "aMCTS-pp"
        same = np.isfinite(rec) == np.isfinite(sc["rec_slot"])
        out[f"{a_}__{b_}"] = {
            "rec_slot_max_abs_diff": float(np.nanmax(np.abs(rec - sc["rec_slot"]))), "rec_slot_cover_agree": float(same.mean()),
            "dS_judge_max_abs_diff": float(np.abs(dS - sc["dS_judge"]).max()),
            "own_V_a_max_abs_diff": float(np.abs(V - sc["own_V_a"]).max()) if own_ok else None}
    print(json.dumps(out, indent=1))
    save_part("verify", {"what": "local rebuild vs the headline score files (max abs differences)", "pairs": out})


# ------------------------------------------------------------------ part 1: exposure

def arm_cells(X, arm, how, opps=None):
    """Cells (lobby, slot, hero) the arm drafted in every seating against opponents in opps."""
    z = X.z
    opps = [o for o in (opps or ARMS) if o != arm]
    L_, S_, Hh, O_ = [], [], [], []
    for o in opps:
        for (x, y, t) in ((arm, o, 0), (o, arm, 1)):
            acts = X.acts(x, y)
            if acts is None:
                continue
            for i in range(X.n):
                h = assign_cells(X, acts, i, t, how)
                L_ += [i] * 5
                S_ += list(range(5 * t, 5 * t + 5))
                Hh += list(h)
                O_ += [o] * 5
    return np.array(L_), np.array(S_), np.array(Hh), np.array(O_)


def cell_stats(X, Li, Si, Hi):
    z = X.z
    n = z["cur_n"][Li, Si, Hi]
    rs = z["past_rsum"][Li, Si, Hi]
    ys = z["past_ysum"][Li, Si, Hi]
    with np.errstate(invalid="ignore", divide="ignore"):
        rec = np.where(n > 0, rs / np.maximum(n, 1), np.nan)
        ws = np.where(n > 0, ys / np.maximum(n, 1), np.nan)
    hot = (n >= 1) & (n <= HOT_N) & ((rec >= HOT_R) | (ws >= HOT_W))
    return {"n": n, "rec": rec, "ws": ws, "hot": hot, "s": z["cur_s"][Li, Si, Hi], "sd": z["cur_sd"][Li, Si, Hi],
            "bin": bin_of(n)}


def summarize(st, Li):
    n = st["n"]
    r = {"cells": int(len(n)), "n_median": float(np.median(n)), "n_mean": round(float(n.mean()), 1),
         "share_by_n": {BINS[k][2]: round(100 * float((st["bin"] == k).mean()), 2) for k in range(len(BINS))},
         "hot_thin_pct": round(100 * float(st["hot"].mean()), 2),
         "thin_le5_pct": round(100 * float(((n >= 1) & (n <= HOT_N)).mean()), 2),
         "s_mean_pp": round(100 * float(st["s"].mean()), 3), "sd_mean_pp": round(100 * float(st["sd"].mean()), 3),
         "past_record_mean_pp": round(100 * float(np.nanmean(st["rec"])), 3),
         "past_winshare_mean": round(float(np.nanmean(st["ws"])), 4)}
    by = {}
    for k in range(len(BINS)):
        m = st["bin"] == k
        if m.sum():
            by[BINS[k][2]] = {"s_mean_pp": round(100 * float(st["s"][m].mean()), 3),
                              "sd_mean_pp": round(100 * float(st["sd"][m].mean()), 3),
                              "past_record_mean_pp": (round(100 * float(np.nanmean(st["rec"][m])), 3)
                                                      if np.isfinite(st["rec"][m]).any() else None)}
    r["by_n"] = by
    return r


def boot_share_diff(Xa, La, Xb, Lb, key, n, seed=0):
    """Lobby-bootstrap SE of share(a) - share(b) for a boolean cell flag."""
    rng = np.random.default_rng(seed)
    ca = np.bincount(La, weights=Xa.astype(float), minlength=n)
    na = np.bincount(La, minlength=n)
    cb = np.bincount(Lb, weights=Xb.astype(float), minlength=n)
    nb = np.bincount(Lb, minlength=n)
    bs = []
    for _ in range(BOOT):
        ix = rng.integers(0, n, n)
        bs.append(ca[ix].sum() / na[ix].sum() - cb[ix].sum() / nb[ix].sum())
    pt = Xa.mean() - Xb.mean()
    return round(100 * float(pt), 2), round(100 * float(np.std(bs)), 2)


def exposure(a):
    X = Ctx(a)
    z = X.z
    n = X.n
    out = {"lobbies": n, "hot_thin_rule": f"1 <= n <= {HOT_N} and (mean past y - WP_pop >= {HOT_R} or win share >= {HOT_W})",
           "assignments": {"own": "drafter's value model: best of 120 by b2 S + b3 O on the cur state (b2 = %.3f)" % X.comb[2],
                           "of": "outcome-free: most earlier games (the judge-free record's rule and outcome-free J)",
                           "actual": "the human who actually played the hero"},
           "opponents": "each arm's drafts in both seats against the other arms of " + ", ".join(ARMS)}
    Li = np.repeat(np.arange(n), 10)
    Si = np.tile(np.arange(10), n)
    Hi = z["real_h"].ravel()
    real = cell_stats(X, Li, Si, Hi)
    out["humans_actual"] = summarize(real, Li)
    # pool context: every hero a player could be drafted on (team pool) vs his own played heroes
    out["pool"] = {"own_pool_median_heroes": float(np.median((z["cur_n"] > 0).sum(2))),
                   "own_pool_share_n1_4": round(100 * float(((z["cur_n"] >= 1) & (z["cur_n"] <= 4)).sum() /
                                                       (z["cur_n"] > 0).sum()), 2)}
    arms = {}
    keep = {}
    for arm in ARMS:
        r = {}
        for how in ("own", "of"):
            L_, S_, H_, O_ = arm_cells(X, arm, how)
            st = cell_stats(X, L_, S_, H_)
            r[how] = summarize(st, L_)
            keep[(arm, how)] = (st, L_, S_, H_, O_)
        # the same-player head-to-head with the paper-1 MCTS only
        if arm != P1:
            L_, S_, H_, _ = arm_cells(X, arm, "own", opps=[P1])
            r["own_vs_pMCTS_only"] = summarize(cell_stats(X, L_, S_, H_), L_)
        arms[arm] = r
        print(arm, r["own"]["share_by_n"], "hot", r["own"]["hot_thin_pct"], flush=True)
    out["arms"] = arms
    # agreement of the selected design's own assignment with the outcome-free one
    st_o, L_o, S_o, H_o, _ = keep[(SEL, "own")]
    st_f, L_f, S_f, H_f, _ = keep[(SEL, "of")]
    assert np.array_equal(L_o, L_f) and np.array_equal(S_o, S_f)
    out["sel_own_vs_of_same_hero_pct"] = round(100 * float((H_o == H_f).mean()), 2)
    # contrasts with lobby-bootstrap SEs
    con = {}
    for other, label in ((P1, "vs paper-1 MCTS"), ("Real", "vs Real arm"), ("aMCTS-pp", "vs aMCTS-pp"), ("GD", "vs GD")):
        a_ = keep[(SEL, "own")]
        b_ = keep[(other, "own")]
        con[label] = {}
        for flag, f in (("hot_thin", lambda s: s["hot"]), ("n_le4", lambda s: (s["n"] >= 1) & (s["n"] <= 4)),
                        ("n_0", lambda s: s["n"] == 0), ("n_ge20", lambda s: s["n"] >= 20)):
            con[label][flag] = boot_share_diff(f(a_[0]), a_[1], f(b_[0]), b_[1], flag, n)
    a_ = keep[(SEL, "own")]
    con["vs humans' actual picks"] = {}
    for flag, f in (("hot_thin", lambda s: s["hot"]), ("n_le4", lambda s: (s["n"] >= 1) & (s["n"] <= 4)),
                    ("n_0", lambda s: s["n"] == 0), ("n_ge20", lambda s: s["n"] >= 20)):
        con["vs humans' actual picks"][flag] = boot_share_diff(f(a_[0]), a_[1], f(real), Li, flag, n)
    out["contrasts_pp_se"] = con
    save_part("exposure", out)
    print(json.dumps(con, indent=1))


# ------------------------------------------------------------------ part 2: winner's curse

def curse(a):
    X = Ctx(a)
    z = X.z
    out = {"window": "realized: 2026-07-01 to 09-11 (judge window; slot rows, post-snapshot, not the sealed build, "
                     "not a sealed id). Predictions made before it, so outcome-independent of every fitted piece.",
           "realized": "game-weighted cell mean of y - WP_pop; slope of realized on predicted; gap = realized - predicted",
           "bins": "n = merged games on the hero before the as-of day"}
    # (A) every tournament-window player, drafter state as of 2026-06-30
    s, nn = z["b_s"], z["b_n"]
    P = np.repeat(z["b_player"][:, None], H, 1)
    rs, rn = z["b_recJ_sum"], z["b_recJ_n"]
    nb = bin_of(nn)
    pool = nn > 0
    A = {}
    A["all_cells_played"] = calib_block(s.ravel(), rs.ravel(), rn.ravel(), P.ravel(), nb.ravel())
    A["all_cells_played_n0_excluded_from_pool"] = calib_block(s[pool], rs[pool], rn[pool], P[pool], nb[pool])
    # selected cells: top-s pool hero per player; top-s among thin (1-4 games) pool heroes; hot-thin cells
    sm = np.where(pool, s, -np.inf)
    top = np.argmax(sm, 1)
    r_ = np.arange(len(s))
    A["top_s_hero"] = calib_block(s[r_, top], rs[r_, top], rn[r_, top], z["b_player"], nb[r_, top])
    A["top_s_hero_played_share_pct"] = round(100 * float((rn[r_, top] > 0).mean()), 2)
    thin = (nn >= 1) & (nn <= 4)
    st_ = np.where(thin, s, -np.inf)
    has = thin.any(1)
    tt = np.argmax(st_, 1)
    A["top_s_among_thin_1_4"] = calib_block(s[r_, tt][has], rs[r_, tt][has], rn[r_, tt][has], z["b_player"][has],
                                            nb[r_, tt][has])
    A["top_s_among_thin_played_share_pct"] = round(100 * float((rn[r_, tt][has] > 0).mean()), 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = z["b_past_rsum"] / np.maximum(nn, 1)
        pws = z["b_past_ysum"] / np.maximum(nn, 1)
    hot = (nn >= 1) & (nn <= HOT_N) & ((prec >= HOT_R) | (pws >= HOT_W))
    A["hot_thin_cells"] = calib_block(s[hot], rs[hot], rn[hot], P[hot], nb[hot])
    A["hot_thin_played_share_pct"] = round(100 * float((rn[hot] > 0).mean()), 2)
    A["players"] = int(len(s))
    out["tournament_window_players_asof_0630"] = A
    # (B) the headline lobbies: drafter state at the lobby, cells drafted by the selected design
    #     (own assignment) and by paper 1's MCTS, and the humans' actual cells
    B = {}
    for arm in (SEL, P1, "aMCTS-pp"):
        L_, S_, H_, _ = arm_cells(X, arm, "own")
        key = np.unique(L_ * 1000 + S_ * 100 + H_)       # each drafted (lobby, player, hero) cell once
        Lk, Sk, Hk = key // 1000, (key // 100) % 10, key % 100
        g = z["slot_player"][Lk, Sk]
        B[arm] = calib_block(z["cur_s"][Lk, Sk, Hk], z["recJ_sum"][Lk, Sk, Hk], z["recJ_n"][Lk, Sk, Hk], g,
                             bin_of(z["cur_n"][Lk, Sk, Hk]), min_cells=15)
        B[arm]["played_share_pct"] = round(100 * float((z["recJ_n"][Lk, Sk, Hk] > 0).mean()), 2)
    Li = np.repeat(np.arange(X.n), 10)
    Si = np.tile(np.arange(10), X.n)
    Hi = z["real_h"].ravel()
    B["humans_actual"] = calib_block(z["cur_s"][Li, Si, Hi], z["recJ_sum"][Li, Si, Hi], z["recJ_n"][Li, Si, Hi],
                                     z["slot_player"].ravel(), bin_of(z["cur_n"][Li, Si, Hi]), min_cells=15)
    B["humans_actual"]["played_share_pct"] = round(100 * float((z["recJ_n"][Li, Si, Hi] > 0).mean()), 2)
    out["headline_lobbies_drafted_cells"] = B
    out["selection_caveat"] = ("Realized edges exist only for cells the player chose to play in the window. Players "
                               "keep playing heroes that go well and drop ones that go badly, so the played subset "
                               "of thin cells is selected on the player's own (unobserved) form; the share of "
                               "selected cells with any window games is reported next to each block.")
    save_part("curse", out)
    print(json.dumps({k: (v.get("all"), v.get("1"), v.get("2-4")) if isinstance(v, dict) else v
                      for k, v in A.items()}, indent=1, default=str))


# ------------------------------------------------------------------ part 3: judges

def judges(a):
    X = Ctx(a)
    z = X.z
    P = np.repeat(z["b_player"][:, None], H, 1)
    out = {"definition": "predicted personal term of one player-hero cell in WP points at p = 0.5: "
                         "(c2 / 4) s_judge (J, gd, v2) and (cal_k c2 / 4) s_judge (J_cal); CONS has no personal "
                         "term (predicts 0). Judge state: evidence 2026-07-01 to 09-11 only, as of 09-11.",
           "windows": {"in_window": "2026-07-01 to 09-11, the window the judges are fit and scored on (in-sample: "
                                    "the judge's own evidence; the c2 fit uses the same games)",
                       "later": "2026-09-28 to 10-26, patch 2.57 (weekly build wk261101 slot table, sealed replays excluded): out of sample for every judge, but a new patch 2.5 to 6.5 weeks after the judge state"},
           "bins": {"judge_n": "games on the hero in the judge window (the judge's evidence)",
                    "lifetime_n": "merged games on the hero before 2026-07-01 (the drafter's n)"}}
    meta = json.loads(str(z["meta"]))
    out["later_window"] = {"games": meta.get("later_games"), "days": meta.get("later_days")}
    nJ = z["b_recJ_n"]
    nlife = z["b_n"]
    res = {}
    for nm, label in (("J", "J"), ("J", "J_cal"), ("gd", "chosen_flip_gd"), ("v2", "chosen_v2flip"), (None, "CONS")):
        if nm is None:
            pred = np.zeros_like(z["b_s"])
            sc = 1.0
        else:
            jd = X.judges[nm]
            pred = z[f"j_{nm}_s"]
            sc = jd["c2_skill"] / 4 * (jd["cal_k"] if label == "J_cal" else 1.0)
        r = {"scale_c2_over_4": round(sc, 4)}
        for wn, rs_, rn_ in (("in_window", z["b_recJ_sum"], z["b_recJ_n"]), ("later", z["b_recL_sum"], z["b_recL_n"])):
            r[wn] = {"by_judge_n": calib_block(pred.ravel(), rs_.ravel(), rn_.ravel(), P.ravel(), bin_of(nJ).ravel(), sc),
                     "by_lifetime_n": calib_block(pred.ravel(), rs_.ravel(), rn_.ravel(), P.ravel(), bin_of(nlife).ravel(), sc)}
        res[label] = r
        print(label, json.dumps({w: {k: r[w]["by_judge_n"].get(k) for k in ("all", "1", "2-4")} for w in ("in_window", "later")}),
              flush=True)
    out["judges"] = res
    # the cells the judge credits in the headline drafts: J's best assignment for the selected design vs paper 1's MCTS
    hd = {}
    for arm in (SEL, P1):
        L_, S_, H_, _ = arm_cells(X, arm, "J")
        jn = z["recJ_n"][L_, S_, H_]          # the judge's evidence: games on the hero in the judge window
        pt = X.judges["J"]["c2_skill"] / 4 * z["hj_J_s"][L_, S_, H_]
        hd[arm] = {"share_by_judge_n": {BINS[k][2]: round(100 * float((bin_of(jn) == k).mean()), 2) for k in range(len(BINS))},
                   "mean_personal_term_pp_by_judge_n": {BINS[k][2]: round(100 * float(pt[bin_of(jn) == k].mean()), 3)
                                                        for k in range(len(BINS)) if (bin_of(jn) == k).any()},
                   "mean_personal_term_pp": round(100 * float(pt.mean()), 3)}
    out["headline_cells_J_assignment"] = hd
    save_part("judges", out)


# ------------------------------------------------------------------ part 4: paired record by exposure

def record(a):
    X = Ctx(a)
    z = X.z
    n = X.n
    out = {"definition": "judge-free paired record (p3t_analyze.paired_record): a player's judge-window mean "
                         "y - WP_pop on the hero arm A put him on minus on the hero arm B put him on (heroes to "
                         "players by most earlier games), players covered under both arms; per lobby the mean over "
                         "covered players, then the mean over lobbies. Contribution of a bin = its share of that "
                         "double mean (contributions add to the total). Bins: the n of A's cell (and of B's cell).",
           "pairs": {}}
    rng = np.random.default_rng(0)
    bidx = [rng.integers(0, n, n) for _ in range(BOOT)]
    for opp in (P1, "Real", "GD", "DraftRec", "aMCTS-pp"):
        sa, sb = X.scores(SEL, opp), X.scores(opp, SEL)
        if sa is None or sb is None:
            continue
        acts_ab, acts_ba = X.acts(SEL, opp), X.acts(opp, SEL)
        A = np.concatenate([sa["rec_slot"][:, :5], sb["rec_slot"][:, 5:]], 1)
        Bv = np.concatenate([sb["rec_slot"][:, :5], sa["rec_slot"][:, 5:]], 1)
        hA = np.zeros((n, 10), np.int64)
        hB = np.zeros((n, 10), np.int64)
        hA_own = np.zeros((n, 10), np.int64)
        for i in range(n):
            hA[i, :5] = assign_cells(X, acts_ab, i, 0, "of")
            hA[i, 5:] = assign_cells(X, acts_ba, i, 1, "of")
            hB[i, :5] = assign_cells(X, acts_ba, i, 0, "of")
            hB[i, 5:] = assign_cells(X, acts_ab, i, 1, "of")
            hA_own[i, :5] = assign_cells(X, acts_ab, i, 0, "own")
            hA_own[i, 5:] = assign_cells(X, acts_ba, i, 1, "own")
        Li = np.repeat(np.arange(n)[:, None], 10, 1)
        Si = np.repeat(np.arange(10)[None], n, 0)
        nA = z["cur_n"][Li, Si, hA]
        nB = z["cur_n"][Li, Si, hB]
        both = np.isfinite(A) & np.isfinite(Bv)
        d = np.where(both, A - Bv, np.nan)
        cnt = both.sum(1)
        lob = cnt > 0
        nl = lob.sum()
        w = np.where(both, 1.0 / np.maximum(cnt, 1)[:, None], 0.0) / nl     # weights of the double mean
        total = float(np.nansum(w * np.nan_to_num(d)))
        with np.errstate(invalid="ignore"):
            check = float(np.nanmean(np.nanmean(d[lob], 1)))
        assert abs(total - check) < 1e-12

        def stat(ix, m):
            ww = np.where(both[ix], 1.0 / np.maximum(cnt[ix], 1)[:, None], 0.0) / max((cnt[ix] > 0).sum(), 1)
            return float(np.sum(ww * np.nan_to_num(d[ix]) * m[ix]))

        def mean_in(ix, m):
            mm = both[ix] & m[ix]
            return float(np.nanmean(d[ix][mm])) if mm.any() else np.nan

        r = {"total_pp": round(100 * total, 3),
             "total_se": round(100 * float(np.std([stat(ix, np.ones((n, 10), bool)) for ix in bidx])), 3),
             "covered_players": int(both.sum())}
        for lab, nmat in (("by_n_of_selected_design_cell", nA), ("by_n_of_opponent_cell", nB)):
            bb = bin_of(nmat)
            rr = {}
            for k in range(len(BINS)):
                m = bb == k
                if not (m & both).any():
                    continue
                rr[BINS[k][2]] = {"covered_share_pct": round(100 * float((m & both).sum() / both.sum()), 2),
                                  "contribution_pp": round(100 * stat(np.arange(n), m), 3),
                                  "contribution_se": round(100 * float(np.std([stat(ix, m) for ix in bidx])), 3),
                                  "mean_diff_pp": round(100 * mean_in(np.arange(n), m), 3),
                                  "mean_diff_se": round(100 * float(np.nanstd([mean_in(ix, m) for ix in bidx])), 3)}
            r[lab] = rr
        # same hero under both arms vs different heroes; and A deeper / shallower than B
        for lab, m in (("same_hero", hA == hB), ("A_cell_deeper", (hA != hB) & (nA > nB)),
                       ("A_cell_shallower", (hA != hB) & (nA < nB)), ("different_hero_equal_n", (hA != hB) & (nA == nB))):
            r.setdefault("by_relation", {})[lab] = {
                "covered_share_pct": round(100 * float((m & both).sum() / both.sum()), 2),
                "contribution_pp": round(100 * stat(np.arange(n), m), 3),
                "contribution_se": round(100 * float(np.std([stat(ix, m) for ix in bidx])), 3)}
        # thin cells the selected design drafts (n 1-4) and hot-thin cells (past record)
        with np.errstate(invalid="ignore", divide="ignore"):
            prec = z["past_rsum"][Li, Si, hA] / np.maximum(nA, 1)
            pws = z["past_ysum"][Li, Si, hA] / np.maximum(nA, 1)
        hot = (nA >= 1) & (nA <= HOT_N) & ((prec >= HOT_R) | (pws >= HOT_W))
        r["hot_thin_selected_cell"] = {"covered_share_pct": round(100 * float((hot & both).sum() / both.sum()), 2),
                                       "contribution_pp": round(100 * stat(np.arange(n), hot), 3),
                                       "contribution_se": round(100 * float(np.std([stat(ix, hot) for ix in bidx])), 3)}
        r["record_rule_vs_own_assignment_same_hero_pct"] = round(100 * float((hA == hA_own).mean()), 2)
        r["coverage_pct"] = {"A": round(100 * float(np.isfinite(A).mean()), 2), "B": round(100 * float(np.isfinite(Bv).mean()), 2),
                             "both": round(100 * float(both.mean()), 2)}
        out["pairs"][f"{SEL} vs {opp}"] = r
        print(opp, r["total_pp"], r["total_se"], json.dumps(r["by_n_of_selected_design_cell"]), flush=True)
    # the record under the drafter's own assignment (how it would read if players sat where the drafter's value model put them)
    own = {}
    for opp in (P1, "Real"):
        acts_ab, acts_ba = X.acts(SEL, opp), X.acts(opp, SEL)
        recs = {}
        for who, (fa, ta, fb, tb) in (("A", (acts_ab, 0, acts_ba, 1)), ("B", (acts_ba, 0, acts_ab, 1))):
            M = np.full((n, 10), np.nan)
            for i in range(n):
                for acts, t in ((fa, ta), (fb, tb)):
                    how = "own" if who == "A" else "of"
                    hs = assign_cells(X, acts, i, t, how)
                    for p in range(5):
                        sl = 5 * t + p
                        c = z["recJ_n"][i, sl, hs[p]]
                        if c > 0:
                            M[i, sl] = z["recJ_sum"][i, sl, hs[p]] / c
            recs[who] = M
        both = np.isfinite(recs["A"]) & np.isfinite(recs["B"])
        d = np.where(both, recs["A"] - recs["B"], np.nan)
        lob = both.any(1)
        with np.errstate(invalid="ignore"):
            pt = float(np.nanmean(np.nanmean(d[lob], 1)))
            bs = [float(np.nanmean(np.nanmean(d[ix][lob[ix]], 1))) for ix in bidx]
        own[f"{SEL} (own assignment) vs {opp} (record rule)"] = {"diff_pp": round(100 * pt, 3), "se": round(100 * float(np.std(bs)), 3),
                                                                 "coverage_A_pct": round(100 * float(np.isfinite(recs['A']).mean()), 2)}
    out["own_assignment_record"] = own
    save_part("record", out)


# ------------------------------------------------------------------ report

def report(a):
    """Print the headline numbers of each part (results/P3_OOD_AUDIT.md is written by hand from them)."""
    j = json.load(open(OUT_JSON))
    e = j["exposure"]
    print("part 1: share by n (own assignment), hot-thin %")
    print("  humans", e["humans_actual"]["share_by_n"], e["humans_actual"]["hot_thin_pct"])
    for arm, r in e["arms"].items():
        print(" ", arm, r["own"]["share_by_n"], r["own"]["hot_thin_pct"])
    c = j["curse"]["tournament_window_players_asof_0630"]
    print("part 2:", {k: (v["all"].get("slope"), v["all"].get("gap_pp")) for k, v in c.items() if isinstance(v, dict)})
    print("part 3:", {k: v["later"]["by_judge_n"]["all"].get("slope") for k, v in j["judges"]["judges"].items()})
    print("part 4:", {k: (v["total_pp"], v["total_se"]) for k, v in j["record"]["pairs"].items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "verify", "exposure", "curse", "judges", "record", "report", "all"])
    ap.add_argument("--hl", default=os.environ.get("P3OOD_HL"), help="copy of cache_def_def1027_headline/p3t (lobbies.npz, tourney/, scores/)")
    ap.add_argument("--work", default=os.environ.get("P3OOD_WORK"), help="scratch dir for queries, answers and cells.npz")
    a = ap.parse_args()
    assert a.hl and a.work, "--hl and --work (or P3OOD_HL, P3OOD_WORK) are required"
    stages = ["verify", "exposure", "curse", "judges", "record"] if a.stage == "all" else [a.stage]
    for s in stages:
        t0 = time.time()
        globals()[s](a)
        print(f"stage {s} done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
