"""
Offline player cards and name index for the site (docs/personalization-site-design.md).

The website never calls Heroes Profile. Everything it shows about a player is
computed here, on the main box, from our research store, then published:

  build     read the merged history store (corpus plus the Heroes Profile
            history pull: cache/full_history/<tag>/history.npz), pick every
            account with a game in the last WINDOW days of the store, compute
            its per-hero estimates with the exported model
            (src/lib/personal/model-artifact.json; the main model where the
            account has corpus scoreboard rows, else imp2), read its latest
            visible name, discriminator and MMR from the local research DB,
            drop accounts that are private in player_privacy, and write
            cache/site/accounts.npz plus a size report.
  load      replace the local table personal_accounts with the build (one
            transaction). sync/publish-site.ts then pushes it to Neon with
            the other site tables (it is in SITE_TABLES).
  simulate  name-matching accuracy on held-out lobbies (see match_sim).

Data rules. The store holds no game from 2026-09-28 on (the P3 snapshot cap:
no 2.57 game, so the sealed replays and the scilearn C, F and pool tranches
never enter). The cards are therefore as of the store cap. Battletags are
read from the local store at build time and go only to the local table and
Neon, never to the repo.

Card format (bytea, little endian), version 1:
  u8 version, u8 model (1 main, 2 imp2), u32 games,
  3 x f32 factor mean a_hat of the draft model (s for unplayed heroes is
      offset(games, 0) + U_h . a_hat; see src/lib/personal/card.ts),
  u8 k, then k played heroes, each:
      u8 hero, u16 games, u16 last played (days since 2014-01-01),
      i16 display mean (0.01 pp), u8 display sd (0.1 pp), i16 draft s (0.01 pp)
  u8 j, then j never-played suggestions: u8 hero, i16 mean (0.01 pp), u8 sd (0.1 pp)

Usage (from training/, source ../.env; numpy and psycopg2, no torch or numba):
  nice -n 19 taskset -c 48-51 python3 personalization/p3_site_publish.py build
  python3 personalization/p3_site_publish.py load
  nice -n 19 taskset -c 48-51 python3 personalization/p3_site_publish.py simulate
"""
import argparse
import datetime
import json
import os
import struct
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
ARTIFACT = os.path.join(ROOT, "src", "lib", "personal", "model-artifact.json")
PARAMS = os.path.join(HERE, "results", "rcf_full", "params", "full_ms.npz")
SB_IDX = os.path.join(HERE, "cache", "rcf", "site_sb_by_blizz")
OUT = os.path.join(HERE, "cache", "site")
REGION_SHIFT = 40
DAY = 86400
EPOCH14 = (datetime.date(2014, 1, 1) - datetime.date(1970, 1, 1)).days
CARD_VERSION = 1
H = 90


def log(msg):
    print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)


# ------------------------------------------------------------------ model pieces

class Model:
    def __init__(self):
        A = json.load(open(ARTIFACT))
        self.A = A
        self.names = A["heroNames"]
        self.ms_U = np.array(A["main"]["U"])
        self.ms_s2 = np.array(A["main"]["s2"])
        self.im_U = np.array(A["imp2"]["U"])
        self.im_s2 = np.array(A["imp2"]["s2"])
        self.W = np.array(A["imp2"]["W"])
        self.zmean = np.array(A["imp2"]["zmean"])
        self.zsd = np.array(A["imp2"]["zsd"])
        self.exp = np.array(A["expTable"])
        self.npe = np.array(A["npEdges"])
        self.nphe = np.array(A["nphEdges"])
        self.ptab = np.array(A["proxy"]["table"])
        self.pbase = np.array(A["proxy"]["base"])
        self.q0 = A["proxy"]["q0"]
        self.vol = np.array(A["main"]["volParams"])
        self.vol_edges = np.array(A["main"]["volEdges"])
        self.sc_disp = A["main"]["scaleNoLayer"]
        self.sc_imp = A["imp2"]["scale"]
        p = np.load(PARAMS)
        self.G, self.c, self.rho2, self.eps2 = (p[k].astype(np.float64) for k in ("G", "c", "rho2", "eps2"))
        assert np.allclose(p["U"], self.ms_U), "full_ms params do not match the artifact"

    def offset(self, n_p, n_ph):
        a = np.searchsorted(self.npe, n_p, side="right") - 1
        b = np.searchsorted(self.nphe, n_ph, side="right") - 1
        return self.exp[a * len(self.nphe) + b]

    def proxy(self, hero, day):
        m = day.astype("datetime64[D]").astype("datetime64[M]").astype(np.int64)
        q = m // 3
        nq = self.ptab.shape[1]
        out = self.ptab[hero, np.clip(q - self.q0, 0, nq - 1)]
        out = np.where(q < self.q0, self.pbase[hero], out)
        return np.clip(out, 0.2, 0.8)

    def zfeat(self, N):
        L = np.log1p(N)
        tot = L.sum(1, keepdims=True)
        s = np.where(tot > 0, L / np.maximum(tot, 1e-9), 0.0)
        F = np.concatenate([s, np.log1p(N.sum(1, keepdims=True))], 1)
        Z = ((F - self.zmean) / self.zsd).astype(np.float32).astype(np.float64)
        Z[tot[:, 0] <= 0] = 0.0
        return Z

    def metric_evidence(self, X, C):
        """(n, H, M) metric sums -> Lam_m (n, K, K), eta_m (n, K) (p3_site_export.evidence, batched)."""
        wm = C / (self.rho2[None, None, :] * C + self.eps2[None, None, :])
        xb = np.where(C > 0, X / np.maximum(C, 1e-9), 0.0) - self.c.T[None]
        G = np.transpose(self.G, (1, 0, 2))                       # (H, M, K)
        Lam = np.einsum("nhm,hmk,hml->nkl", wm, G, G, optimize=True)
        eta = np.einsum("nhm,hmk->nk", wm * xb, G, optimize=True)
        return Lam, eta


def factor_post(U, s2, S, P, Lam_extra=None, eta_extra=None):
    """Batched posterior pieces (p3_rcf.posterior): Lam, eta, Linv, ahat, w, wy."""
    n = S.shape[0]
    K = U.shape[1]
    den = 1.0 + s2[None] * P
    w = P / den
    wy = S / den
    Lam = np.einsum("nh,hk,hl->nkl", w, U, U, optimize=True) + np.eye(K)[None]
    eta = wy @ U
    if Lam_extra is not None:
        Lam = Lam + Lam_extra
    if eta_extra is not None:
        eta = eta + eta_extra
    Linv = np.linalg.inv(Lam)
    ahat = np.einsum("nkl,nl->nk", Linv, eta)
    return Lam, eta, Linv, ahat, w, wy


def all_heroes(U, s2, Linv, ahat, w, wy):
    a = 1.0 - s2[None] * w
    mean = a * (ahat @ U.T) + s2[None] * wy
    q = np.einsum("hk,nkl,hl->nh", U, Linv, U, optimize=True)
    var = a * a * q + s2[None] * a
    return mean, var


def vol_post(M, m0, v0, Sh, Ph, nh, npl):
    """p3_rcf_tails.post, card "vol", for many (player, hero) pairs."""
    b = np.searchsorted(M.vol_edges, nh, side="right") - 1
    b = b * 2 + ((nh / np.maximum(npl, 1)) >= 0.2).astype(int)
    xg = np.linspace(-10, 10, 401)
    out_m = np.empty(len(m0))
    out_v = np.empty(len(m0))
    for i0 in range(0, len(m0), 50000):
        sl = slice(i0, i0 + 50000)
        mu = M.vol[b[sl]][:, None]
        s = np.exp(M.vol[10 + b[sl]])[:, None]
        s0 = np.sqrt(np.maximum(v0[sl], 1e-10))[:, None]
        P = Ph[sl][:, None]
        yb = np.where(P > 0, Sh[sl][:, None] / np.maximum(P, 1e-12), 0.0)
        th = m0[sl][:, None] + s0 * xg[None]
        lw = -0.5 * ((xg[None] - mu) / s) ** 2 - np.log(s) + np.where(P > 0, -0.5 * P * (yb - th) ** 2, 0.0)
        lw -= lw.max(1, keepdims=True)
        wgt = np.exp(lw)
        wgt /= wgt.sum(1, keepdims=True)
        out_m[sl] = (wgt * th).sum(1)
        out_v[sl] = (wgt * th * th).sum(1) - out_m[sl] ** 2
    return out_m, out_v


def lag1(day, key):
    """Rows (any order) -> number of rows with the same key on EARLIER days."""
    o = np.lexsort((day, key))
    k, d = key[o], day[o]
    ar = np.arange(len(o))
    grp = np.maximum.accumulate(np.where(np.r_[True, k[1:] != k[:-1]], ar, 0))
    blk = np.maximum.accumulate(np.where(np.r_[True, (k[1:] != k[:-1]) | (d[1:] != d[:-1])], ar, 0))
    out = np.empty(len(o), np.int64)
    out[o] = blk - grp
    return out


# ------------------------------------------------------------------ store

def load_store(tag):
    z = np.load(os.path.join(HERE, "cache", "full_history", tag, "history.npz"))
    meta = json.load(open(os.path.join(HERE, "cache", "full_history", tag, "meta.json")))
    return z, meta


def active_rows(z, cap_ts, window_days, M):
    """Rows of every account with a game in [cap - window, cap), known heroes only."""
    ptr = z["ptr"]
    keys = z["keys"]
    end = z["end_ts"]
    nk = len(keys)
    kidx = np.repeat(np.arange(nk), np.diff(ptr))
    recent = np.bincount(kidx, weights=(end >= cap_ts - window_days * DAY) & (end < cap_ts), minlength=nk)
    act = np.flatnonzero(recent > 0)
    pos = np.full(nk, -1, np.int64)
    pos[act] = np.arange(len(act))
    rows = np.flatnonzero(pos[kidx] >= 0)
    hmap = np.array([M.names.index(str(h)) if str(h) in M.names else -1 for h in z["hero_names"]])
    hero = np.where(z["hero"][rows] >= 0, hmap[np.maximum(z["hero"][rows], 0)], -1)
    keep = (hero >= 0) & (end[rows] < cap_ts)
    rows = rows[keep]
    return {"keys": keys[act], "recent": recent[act].astype(np.int64),
            "acct": pos[kidx[rows]], "hero": hero[keep], "win": z["win"][rows].astype(np.float64),
            "day": (end[rows] // DAY).astype(np.int64), "rid": z["replay_id"][rows].astype(np.int64),
            "src": z["src"][rows], "mmr": z["post_mmr"][rows]}


# ------------------------------------------------------------------ build

def compute_cards(M, R, chunk=10000):
    n = len(R["keys"])
    acct, hero, day = R["acct"], R["hero"], R["day"]
    log(f"accounts {n:,}, rows {len(acct):,}")
    n_p = lag1(day, acct)
    n_ph = lag1(day, acct * 128 + hero)
    pr = M.proxy(hero, day)
    r = R["win"] - pr
    v = pr * (1 - pr)
    r_adj = r - M.offset(n_p, n_ph)
    cell = acct * H + hero
    S = np.bincount(cell, weights=r_adj / v, minlength=n * H).reshape(n, H)
    P = np.bincount(cell, weights=1 / v, minlength=n * H).reshape(n, H)
    N = np.bincount(cell, minlength=n * H).reshape(n, H).astype(np.float64)
    W_ = np.bincount(cell, weights=R["win"], minlength=n * H).reshape(n, H)
    last = np.full(n * H, -1, np.int64)
    np.maximum.at(last, cell, day)
    last = last.reshape(n, H)
    log("state built; joining corpus scoreboard rows")
    # metric sums: corpus rows joined to the a_shrink card on (replay_id, blizz_id)
    sb_bl = np.load(os.path.join(SB_IDX, "blizz_id.npy"), mmap_mode="r")
    sb_rid = np.load(os.path.join(SB_IDX, "replay_id.npy"), mmap_mode="r")
    sb_Z = np.load(os.path.join(SB_IDX, "Z.npy"), mmap_mode="r")
    sb_W = np.load(os.path.join(SB_IDX, "W.npy"), mmap_mode="r")
    comp = np.asarray(sb_rid) * (1 << 34) + np.asarray(sb_bl)       # rid < 2^27, blizz < 2^34: exact
    so = np.argsort(comp)
    cs = comp[so]
    blz = R["keys"][acct] & ((1 << REGION_SHIFT) - 1)
    assert int(blz.max()) < (1 << 34)
    q = R["rid"] * (1 << 34) + blz
    j = np.minimum(np.searchsorted(cs, q), len(cs) - 1)
    hit = ((R["src"] & 1) > 0) & (cs[j] == q)
    hrow = np.flatnonzero(hit)
    sbrow = so[j[hit]]
    log(f"scoreboard rows matched {len(hrow):,}")
    Mm = sb_Z.shape[1]
    has_m = np.bincount(acct[hrow], minlength=n) > 0
    # per account outputs
    model = np.where(has_m, 1, 2).astype(np.int8)
    ahat_draft = np.zeros((n, 3))
    disp_mean = np.zeros((n, H))
    disp_sd = np.zeros((n, H))
    draft_s = np.zeros((n, H))
    sugg_mean = np.zeros((n, H))
    sugg_sd = np.zeros((n, H))
    ntot = N.sum(1)
    order = np.argsort(acct[hrow], kind="stable")
    hrow, sbrow = hrow[order], sbrow[order]
    hacct = acct[hrow]
    for i0 in range(0, n, chunk):
        i1 = min(n, i0 + chunk)
        sl = slice(i0, i1)
        m = i1 - i0
        off = M.offset(np.repeat(ntot[sl, None], H, 1).astype(np.int64), N[sl].astype(np.int64))
        # imp2 (all accounts): cards without metrics and the never-played ranking
        _, _, Li, ai, w, wy = factor_post(M.im_U, M.im_s2, S[sl], P[sl], eta_extra=M.zfeat(N[sl]) @ M.W.T)
        mi, vi = all_heroes(M.im_U, M.im_s2, Li, ai, w, wy)
        sc = M.sc_imp
        sugg_mean[sl] = mi
        sugg_sd[sl] = np.sqrt(sc["c"] * vi + sc["k0"] ** 2)
        disp_mean[sl] = off + mi
        disp_sd[sl] = sugg_sd[sl]
        draft_s[sl] = off + mi
        ahat_draft[sl] = ai
        # main model where the account has scoreboard rows
        a0, a1 = np.searchsorted(hacct, [i0, i1])
        X = np.zeros((m, H, Mm))
        C = np.zeros((m, H, Mm))
        if a1 > a0:
            la = hacct[a0:a1] - i0
            hh = hero[hrow[a0:a1]]
            sr = np.sort(sbrow[a0:a1])
            inv = np.argsort(np.argsort(sbrow[a0:a1]))
            Zr = np.asarray(sb_Z[sr]).astype(np.float64)[inv]
            Wr = np.asarray(sb_W[sr]).astype(np.float64)[inv]
            for k in range(Mm):
                x = Zr[:, k]
                g = ~np.isnan(x)
                X[:, :, k] = np.bincount(la[g] * H + hh[g], weights=(Wr * x)[g], minlength=m * H).reshape(m, H)
                C[:, :, k] = np.bincount(la[g] * H + hh[g], weights=Wr[g], minlength=m * H).reshape(m, H)
        sel = np.flatnonzero(has_m[sl])
        if len(sel):
            Lm, em = M.metric_evidence(X[sel], C[sel])
            Ss, Ps = S[sl][sel], P[sl][sel]
            Lam, eta, Lg, ag, w, wy = factor_post(M.ms_U, M.ms_s2, Ss, Ps, Lm, em)
            mg, vg = all_heroes(M.ms_U, M.ms_s2, Lg, ag, w, wy)
            ok = off[sel]
            draft_s[i0 + sel] = ok + mg
            ahat_draft[i0 + sel] = ag
            # volume card for played heroes: prior from everything except the hero's own record
            pi, ph = np.nonzero(N[sl][sel] > 0)
            U = M.ms_U
            Lloo = Lam[pi] - w[pi, ph][:, None, None] * U[ph][:, :, None] * U[ph][:, None, :]
            eloo = eta[pi] - wy[pi, ph][:, None] * U[ph]
            al = np.linalg.solve(Lloo, eloo[:, :, None])[:, :, 0]
            m0 = (U[ph] * al).sum(1)
            v0 = np.einsum("nk,nkl,nl->n", U[ph], np.linalg.inv(Lloo), U[ph]) + M.ms_s2[ph]
            nh = N[sl][sel][pi, ph]
            vm, _ = vol_post(M, m0, v0, Ss[pi, ph], Ps[pi, ph], nh, ntot[sl][sel][pi])
            dm = np.array(ok)
            dm[pi, ph] = ok[pi, ph] + vm
            sd = np.sqrt(M.sc_disp["c"] * vg + M.sc_disp["k0"] ** 2)
            disp_mean[i0 + sel] = np.where(N[sl][sel] > 0, dm, disp_mean[i0 + sel])
            disp_sd[i0 + sel] = np.where(N[sl][sel] > 0, sd, disp_sd[i0 + sel])
        if (i0 // chunk) % 4 == 0:
            log(f"  cards {i1:,}/{n:,}")
    return {"model": model, "N": N, "wins": W_, "last": last, "ahat": ahat_draft, "disp_mean": disp_mean,
            "disp_sd": disp_sd, "draft_s": draft_s, "sugg_mean": sugg_mean, "sugg_sd": sugg_sd,
            "games": ntot.astype(np.int64)}


def encode_card(Cd, i, n_sugg=5):
    N = Cd["N"][i]
    played = np.flatnonzero(N > 0)
    played = played[np.argsort(-N[played], kind="stable")][:255]
    b = bytearray(struct.pack("<BBI3f", CARD_VERSION, int(Cd["model"][i]), int(Cd["games"][i]),
                              *[float(x) for x in Cd["ahat"][i]]))
    b += struct.pack("<B", len(played))
    for h in played:
        lastd = int(Cd["last"][i, h]) - EPOCH14
        b += struct.pack("<BHHhBh", int(h), int(min(N[h], 65535)), max(0, min(lastd, 65535)),
                         int(np.clip(round(Cd["disp_mean"][i, h] * 1e4), -32767, 32767)),
                         int(np.clip(round(Cd["disp_sd"][i, h] * 1e3), 0, 255)),
                         int(np.clip(round(Cd["draft_s"][i, h] * 1e4), -32767, 32767)))
    never = np.flatnonzero(N == 0)
    top = never[np.argsort(-Cd["sugg_mean"][i, never], kind="stable")][:n_sugg]
    b += struct.pack("<B", len(top))
    for h in top:
        b += struct.pack("<BhB", int(h), int(np.clip(round(Cd["sugg_mean"][i, h] * 1e4), -32767, 32767)),
                         int(np.clip(round(Cd["sugg_sd"][i, h] * 1e3), 0, 255)))
    return bytes(b)


def read_names(keys, cap_ts, window_days):
    """Latest battletag, region and player MMR per account from the local
    research DB, over corpus rows of games in the window (before the cap)."""
    import psycopg2
    con = psycopg2.connect(os.environ["DATABASE_URL_RESEARCH"])
    con.set_session(readonly=True, autocommit=True)
    cur = con.cursor()
    lo = datetime.datetime.fromtimestamp(cap_ts - window_days * DAY, datetime.timezone.utc).replace(tzinfo=None)
    hi = datetime.datetime.fromtimestamp(cap_ts, datetime.timezone.utc).replace(tzinfo=None)
    cur.execute("SELECT min(replay_id), max(replay_id) FROM replay_draft_data WHERE game_date >= %s AND game_date < %s",
                (lo, hi))
    rlo, rhi = cur.fetchone()
    t0 = time.time()
    cur.execute("""
        SELECT DISTINCT ON (r, rp.blizz_id) COALESCE(rp.region, d.region) AS r, rp.blizz_id, rp.battletag
        FROM replay_players rp JOIN replay_draft_data d USING (replay_id)
        WHERE rp.replay_id BETWEEN %s AND %s AND rp.blizz_id > 0 AND rp.battletag LIKE '%%#%%'
          AND d.game_date < %s
        ORDER BY r, rp.blizz_id, rp.replay_id DESC""", (rlo, rhi, hi))
    names = {}
    for reg, bl, tag in cur:
        if reg is None:
            continue
        names[(int(reg) << REGION_SHIFT) | int(bl)] = tag
    log(f"names from the window: {len(names):,} accounts ({time.time() - t0:.0f}s)")
    cur.execute("SELECT battletag, region FROM player_privacy WHERE state = 'private'")
    private = {(t, int(r)) for t, r in cur.fetchall()}
    con.close()
    out = []
    for k in keys:
        tag = names.get(int(k))
        reg = int(k) >> REGION_SHIFT
        out.append(None if tag is None or (tag, reg) in private else tag)
    return out, len(private)


def build(a):
    t0 = time.time()
    M = Model()
    z, meta = load_store(a.store)
    cap = int(meta["cap_ts"])
    R = active_rows(z, cap, a.window, M)
    Cd = compute_cards(M, R)
    tags, n_private = read_names(R["keys"], cap, a.window)
    keys = R["keys"]
    n = len(keys)
    # latest MMR per account (HP post-game rating of its last row)
    lastrow = np.full(n, -1, np.int64)
    np.maximum.at(lastrow, R["acct"], np.arange(len(R["acct"])))
    mmr = np.where(lastrow >= 0, R["mmr"][np.maximum(lastrow, 0)], np.nan)
    keep = np.array([t is not None for t in tags])
    log(f"accounts with a visible name: {keep.sum():,} of {n:,} (private-listed tags: {n_private:,})")
    rows = []
    sizes = []
    for i in np.flatnonzero(keep):
        name, disc = tags[i].rsplit("#", 1)
        if not disc.isdigit() or not name:
            continue
        card = encode_card(Cd, i)
        N = Cd["N"][i]
        top = np.argsort(-N, kind="stable")[:3]
        top = [int(h) for h in top if N[h] > 0]
        lastday = int(Cd["last"][i].max())
        rows.append((int(keys[i] >> REGION_SHIFT), int(keys[i] & ((1 << REGION_SHIFT) - 1)), name[:24], int(disc),
                     None if not np.isfinite(mmr[i]) else int(round(float(mmr[i]))),
                     (datetime.date(1970, 1, 1) + datetime.timedelta(days=lastday)).isoformat(),
                     int(Cd["games"][i]), int(R["recent"][i]), bytes(top), card))
        sizes.append(len(card))
    os.makedirs(OUT, exist_ok=True)
    cols = ["region", "blizz_id", "name", "discriminator", "mmr", "last_played", "games", "recent_games",
            "top_heroes", "card"]
    np.savez(os.path.join(OUT, "accounts.npz"), cols=np.array(cols),
             **{c: np.array([r[j] for r in rows], dtype=object) for j, c in enumerate(cols)})
    sizes = np.array(sizes)
    names_len = np.array([len(r[2].encode()) for r in rows])
    # Postgres estimate: 24 B tuple header + 4 B item pointer, fixed columns
    # 2+4+4+2+4+4+4 = 24 B, varlena headers 3 x 1-4 B, plus two btree indexes
    # (pk ~ 24 B per row, lower(name) ~ 16 B + name per row)
    heap = 28 + 24 + (names_len + 1) + (3 + 1) + (sizes + 4)
    idx = 24 + 16 + names_len
    rep = {"store": a.store, "cap": datetime.date.fromtimestamp(cap).isoformat(), "window_days": a.window,
           "accounts": len(rows), "main_model": int(sum(r[9][1] == 1 for r in rows)),
           "card_bytes_mean": float(sizes.mean()), "card_bytes_p95": float(np.quantile(sizes, 0.95)),
           "neon_estimate_mb": float((heap.sum() * 1.12 + idx.sum() * 1.3) / 1e6),
           "by_recent_games": {str(t): int((np.array([r[7] for r in rows]) >= t).sum()) for t in (1, 3, 5, 10, 20)},
           "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "seconds": round(time.time() - t0)}
    for t in (1, 3, 5, 10):
        m = np.array([r[7] for r in rows]) >= t
        rep[f"neon_estimate_mb_min{t}"] = float((heap[m].sum() * 1.12 + idx[m].sum() * 1.3) / 1e6)
    with open(os.path.join(OUT, "report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    log(json.dumps(rep))


# ------------------------------------------------------------------ load (local table)

DDL = """
CREATE TABLE IF NOT EXISTS personal_accounts (
  region smallint NOT NULL,
  blizz_id integer NOT NULL,
  name varchar(24) NOT NULL,
  discriminator integer NOT NULL,
  mmr smallint,
  last_played date NOT NULL,
  games integer NOT NULL,
  recent_games integer NOT NULL,
  top_heroes bytea NOT NULL,
  card bytea NOT NULL,
  CONSTRAINT personal_accounts_region_blizz_id_pk PRIMARY KEY (region, blizz_id)
);
CREATE INDEX IF NOT EXISTS personal_accounts_name_idx ON personal_accounts (lower(name));
"""


def load(a):
    import io
    import psycopg2
    zf = np.load(os.path.join(OUT, "accounts.npz"), allow_pickle=True)
    z = {k: zf[k] for k in zf.files}                  # NpzFile re-reads an array on every access
    keep = np.array(z["recent_games"], dtype=np.int64) >= a.min_recent
    n = int(keep.sum())
    url = os.environ["DATABASE_URL"]
    assert "neon.tech" not in url, "load writes the LOCAL store; publish-site.ts copies it to Neon"
    con = psycopg2.connect(url)
    cur = con.cursor()
    cur.execute(DDL)
    buf = io.StringIO()
    for i in np.flatnonzero(keep):
        mmr = z["mmr"][i]
        name = str(z["name"][i]).replace("\\", "\\\\").replace("\t", " ").replace("\n", " ")
        buf.write("\t".join([str(z["region"][i]), str(z["blizz_id"][i]), name, str(z["discriminator"][i]),
                             "\\N" if mmr is None else str(int(mmr)), str(z["last_played"][i]), str(z["games"][i]),
                             str(z["recent_games"][i]), "\\\\x" + bytes(z["top_heroes"][i]).hex(),
                             "\\\\x" + bytes(z["card"][i]).hex()]) + "\n")
    buf.seek(0)
    cur.execute("DELETE FROM personal_accounts")
    cur.copy_expert("COPY personal_accounts FROM STDIN", buf)
    con.commit()
    con.autocommit = True
    cur.execute("VACUUM personal_accounts")       # dead rows of the previous build become reusable
    cur.execute("SELECT pg_total_relation_size('personal_accounts')")
    log(f"personal_accounts: {n:,} rows (recent games >= {a.min_recent}); local size {cur.fetchone()[0] / 1e6:.1f} MB")
    con.close()


# ------------------------------------------------------------------ name-matching simulation

def match_score(cand, ctx, p):
    """Higher is better. cand: dict arrays for the candidates of one name."""
    s = np.where(cand["region"] == ctx["region"], p["region"], 0.0)
    known = np.isfinite(cand["mmr"]) & np.isfinite(ctx["mmr"])
    s = s - np.where(known, np.abs(cand["mmr"] - ctx["mmr"]) / p["mmr_scale"], p["mmr_unknown"])
    s = s + p["recency"] * np.exp(-np.maximum(ctx["day"] - cand["last"], 0) / p["recency_days"])
    return s


PARAMS_MATCH = {"region": 10.0, "mmr_scale": 300.0, "mmr_unknown": 1.0, "recency": 1.0, "recency_days": 30.0,
                "margin": 1.5, "min_score": 9.5}


def simulate(a):
    """Hide the #numbers of the other nine players in held-out lobbies and
    match each by visible name against an index built only from games
    before the test period, with the party's region and MMR as context.

    Index as of CUT (accounts with a game in the WINDOW days before CUT;
    last played, games and MMR as of CUT). Test lobbies: corpus games from
    CUT to the store cap. The party is one random player of the lobby; its
    region and MMR as of CUT are the context (the lobby MMR context uses the
    mean MMR of the party only, as the page knows nothing else)."""
    M = Model()
    z, meta = load_store(a.store)
    cap = int(meta["cap_ts"])
    cut = cap - a.test_days * DAY
    keys_all = z["keys"]
    ptr = z["ptr"]
    kidx = np.repeat(np.arange(len(keys_all)), np.diff(ptr))
    end = z["end_ts"]
    before = end < cut
    # as-of index stats per account
    inwin = before & (end >= cut - a.window * DAY)
    nk = len(keys_all)
    idx_acct = np.bincount(kidx, weights=inwin, minlength=nk) >= a.min_recent
    lastday = np.full(nk, -1, np.int64)
    np.maximum.at(lastday, kidx[before], end[before] // DAY)
    lastrow = np.full(nk, -1, np.int64)
    br = np.flatnonzero(before)
    np.maximum.at(lastrow, kidx[br], br)
    mmr = np.where(lastrow >= 0, z["post_mmr"][np.maximum(lastrow, 0)], np.nan).astype(np.float64)
    # names: the production build's name map (the latest name in the window)
    accf = np.load(os.path.join(OUT, "accounts.npz"), allow_pickle=True)
    acc = {k: accf[k] for k in ("region", "blizz_id", "name")}
    akey = (acc["region"].astype(np.int64) << REGION_SHIFT) | acc["blizz_id"].astype(np.int64)
    kpos = np.searchsorted(keys_all, akey)
    ok = (kpos < nk) & (keys_all[np.minimum(kpos, nk - 1)] == akey)
    name_of = np.full(nk, None, dtype=object)
    name_of[kpos[ok]] = [str(x).lower() for x in acc["name"][ok]]
    # index: name -> account indices
    from collections import defaultdict
    by_name = defaultdict(list)
    for k in np.flatnonzero(idx_acct & (name_of != None)):  # noqa: E711
        by_name[name_of[k]].append(k)
    # test lobbies: corpus rows of games in [cut, cap)
    test = (end >= cut) & (end < cap) & ((z["src"] & 1) > 0)
    tr = np.flatnonzero(test)
    rid = z["replay_id"][tr]
    o = np.argsort(rid, kind="stable")
    tr, rid = tr[o], rid[o]
    starts = np.flatnonzero(np.r_[True, rid[1:] != rid[:-1]])
    stops = np.r_[starts[1:], len(rid)]
    rng = np.random.default_rng(a.seed)
    games = np.flatnonzero((stops - starts) >= 6)
    games = rng.choice(games, min(a.games, len(games)), replace=False)
    region = keys_all >> REGION_SHIFT
    rec = []
    p = dict(PARAMS_MATCH, **json.loads(a.params or "{}"))
    for gi, g in enumerate(games):
        members = kidx[tr[starts[g]:stops[g]]]
        gday = int(end[tr[starts[g]]] // DAY)
        party = members[rng.integers(len(members))]
        ctx = {"region": region[party], "mmr": mmr[party], "day": gday}
        for k in members:
            if k == party:
                continue
            nm = name_of[k]
            if nm is None:
                rec.append((0, 0, 0, 0, 0, 0, gi))     # no visible name known
                continue
            cands = by_name.get(nm, [])
            in_idx = k in cands
            if not cands:
                rec.append((1, 0, 0, 0, 0, 0, gi))
                continue
            cand = {"region": region[cands], "mmr": mmr[cands], "last": lastday[cands]}
            sc = match_score(cand, ctx, p)
            order = np.argsort(-sc, kind="stable")
            ranked = [cands[i] for i in order]
            top1 = ranked[0] == k
            top3 = k in ranked[:3]
            clear = (sc[order[0]] >= p["min_score"]) and (
                len(cands) == 1 or (sc[order[0]] - sc[order[1]] >= p["margin"]))
            rec.append((1, int(in_idx), int(top1), int(top3), int(clear), len(cands), gi))
    R = np.array(rec)
    half = R[:, 6] >= len(games) // 2          # report on the second half; tune on the first
    out = {"cut": datetime.date.fromtimestamp(cut).isoformat(), "test_days": a.test_days,
           "lobbies": int(len(games)), "params": p, "min_recent": a.min_recent}
    for nm_, part in (("tune", R[~half]), ("report", R[half])):
        named, inidx = part[:, 0] == 1, part[:, 1] == 1
        hasc = part[:, 5] > 0
        clear = hasc & (part[:, 4] == 1)
        shown = hasc & ~clear
        out[nm_] = {
            "players": int(len(part)),
            "share_in_index": float(inidx.mean()),
            "indexed_top1": float(part[inidx, 2].mean()), "indexed_top3": float(part[inidx, 3].mean()),
            "indexed_unique_name": float((part[inidx, 5] == 1).mean()),
            "auto_selected": float(clear.mean()),
            "auto_correct": float((clear & (part[:, 2] == 1)).mean()),
            "auto_wrong": float((clear & (part[:, 2] == 0)).mean()),
            "auto_precision": float(part[clear, 2].mean()),
            "choices_shown": float(shown.mean()),
            "choices_contain_true_top3": float(part[shown, 3].mean()) if shown.any() else None,
            "no_candidate": float((named & ~hasc).mean()),
        }
    print(json.dumps(out, indent=1))
    with open(os.path.join(HERE, "results", "site_match_sim.json"), "w") as f:
        json.dump(out, f, indent=1)


# ------------------------------------------------------------------ checks and fixtures

def check(a):
    """compute_cards against the research reference (p3_site_export.reference, which calls
    p3_rcf.posterior and p3_rcf_tails.post) on random active accounts. Needs numba
    (/usr/bin/python3). Writes results/site_publish_check.json (errors only, no ids)."""
    from p3_site_export import load_inputs, reference
    M = Model()
    T, ms, imp, consts, _, _ = load_inputs()
    z, meta = load_store(a.store)
    R = active_rows(z, int(meta["cap_ts"]), a.window, M)
    cnt = np.bincount(R["acct"], minlength=len(R["keys"]))
    rng = np.random.default_rng(a.seed)
    pick = rng.choice(np.flatnonzero((cnt >= 5) & (cnt <= 400)), a.n_check, replace=False)
    sel = np.isin(R["acct"], pick)
    remap = np.full(len(R["keys"]), -1, np.int64)
    remap[pick] = np.arange(len(pick))
    Rs = {k: (v[sel] if k not in ("keys", "recent") else v[pick]) for k, v in R.items()}
    Rs["acct"] = remap[R["acct"][sel]]
    Cd = compute_cards(M, Rs, chunk=len(pick))
    # the same corpus scoreboard rows, as per-hero metric sums, for the reference
    sb_bl = np.load(os.path.join(SB_IDX, "blizz_id.npy"), mmap_mode="r")
    sb_rid = np.load(os.path.join(SB_IDX, "replay_id.npy"), mmap_mode="r")
    sb_Z = np.load(os.path.join(SB_IDX, "Z.npy"), mmap_mode="r")
    sb_W = np.load(os.path.join(SB_IDX, "W.npy"), mmap_mode="r")
    err = {"draft_s": 0.0, "disp_mean_played": 0.0, "imp2": 0.0, "gauss_var_sd": 0.0}
    for i in range(len(pick)):
        m = Rs["acct"] == i
        rows = {"hero": Rs["hero"][m], "y": Rs["win"][m].astype(np.int64), "day": Rs["day"][m]}
        o = np.argsort(rows["day"], kind="stable")
        rows = {k: v[o] for k, v in rows.items()}
        blz = int(Rs["keys"][i] & ((1 << REGION_SHIFT) - 1))
        lo, hi = np.searchsorted(sb_bl, blz, "left"), np.searchsorted(sb_bl, blz, "right")
        rid_m = Rs["rid"][m][((Rs["src"][m] & 1) > 0)]
        hero_m = Rs["hero"][m][((Rs["src"][m] & 1) > 0)]
        X = np.zeros((H, sb_Z.shape[1]))
        C = np.zeros((H, sb_Z.shape[1]))
        hero_of = dict(zip(rid_m.tolist(), hero_m.tolist()))
        for j in range(lo, hi):
            h = hero_of.get(int(sb_rid[j]))
            if h is None:
                continue
            x = np.asarray(sb_Z[j], np.float64)
            w = float(sb_W[j])
            g = ~np.isnan(x)
            X[h, g] += w * x[g]
            C[h, g] += w
        withm = C.sum() > 0
        ref = reference(T, ms, imp, consts, rows, X if withm else None, C if withm else None,
                        int(rows["day"].max()) + 1)
        off = np.array(ref["offset"])
        played = np.array(ref["N"]) > 0
        err["imp2"] = max(err["imp2"], float(np.abs(Cd["sugg_mean"][i] - np.array(ref["imp2"]["mean"])).max()))
        if withm:
            e = ref["main"]
            err["draft_s"] = max(err["draft_s"], float(np.abs(Cd["draft_s"][i] - (off + np.array(e["gauss_mean"]))).max()))
            err["disp_mean_played"] = max(err["disp_mean_played"], float(
                np.abs(Cd["disp_mean"][i][played] - (off + np.array(e["vol_mean"]))[played]).max()))
            sd_ref = np.sqrt(M.sc_disp["c"] * np.array(e["gauss_var"]) + M.sc_disp["k0"] ** 2)
            err["gauss_var_sd"] = max(err["gauss_var_sd"], float(np.abs(Cd["disp_sd"][i][played] - sd_ref[played]).max()))
    out = {"accounts": int(len(pick)), "max_abs_error": err,
           "note": "win-probability units; reference = p3_site_export.reference (research functions)"}
    print(json.dumps(out, indent=1))
    with open(os.path.join(HERE, "results", "site_publish_check.json"), "w") as f:
        json.dump(out, f, indent=1)
    assert max(err.values()) < 1e-6, err


def fixture(a):
    """Synthetic cards (no real accounts) for src/lib/personal/__tests__: the encoded bytes,
    what they decode to, the draft term for every hero, and match scores."""
    from types import SimpleNamespace  # noqa: F401
    M = Model()
    rng = np.random.default_rng(a.seed)
    cases = []
    for c in range(4):
        n = 1
        N = np.zeros((n, H))
        k = [0, 3, 25, 40][c]
        heroes = rng.choice(H, k, replace=False)
        N[0, heroes] = rng.integers(1, 300, k)
        ahat = rng.normal(0, 0.02, (n, 3))
        model = np.array([1 if c % 2 == 0 else 2], np.int8)
        U = M.ms_U if model[0] == 1 else M.im_U
        ntot = N.sum(1)
        off = M.offset(np.repeat(ntot[:, None], H, 1).astype(np.int64), N.astype(np.int64))
        draft = off + ahat @ U.T
        draft[0, heroes] += rng.normal(0, 0.02, k)        # played heroes: own evidence moves them
        Cd = {"model": model, "N": N, "games": ntot.astype(np.int64), "ahat": ahat,
              "last": np.where(N > 0, 20500 + rng.integers(0, 300, (n, H)), -1),
              "disp_mean": draft + rng.normal(0, 0.005, (n, H)), "disp_sd": rng.uniform(0.02, 0.09, (n, H)),
              "draft_s": draft, "sugg_mean": rng.normal(0, 0.02, (n, H)), "sugg_sd": rng.uniform(0.02, 0.06, (n, H))}
        card = encode_card(Cd, 0)
        # the values as stored (quantized)
        q = lambda x, s: np.round(np.asarray(x) * s) / s
        draft_exp = off[0] + np.asarray(np.float32(ahat[0]), np.float64) @ U.T
        draft_exp[heroes] = q(draft[0, heroes], 1e4)
        cases.append({"card_hex": card.hex(), "model": "main" if model[0] == 1 else "imp2", "games": int(ntot[0]),
                      "played": {M.names[h]: {"games": int(N[0, h]), "draft": float(q(draft[0, h], 1e4)),
                                              "mean": float(q(Cd["disp_mean"][0, h], 1e4))} for h in heroes},
                      "draft_s": draft_exp.tolist()})
    # match scores
    ctx = {"region": 2, "mmr": 2600.0, "day": 20700}
    cand = {"region": np.array([2, 2, 1, 2]), "mmr": np.array([2650.0, np.nan, 2600.0, 3400.0]),
            "last": np.array([20690, 20500, 20700, 20699])}
    sc = match_score(cand, ctx, PARAMS_MATCH)
    iso = lambda d: (datetime.date(1970, 1, 1) + datetime.timedelta(days=int(d))).isoformat()
    match = {"ctx": {"region": 2, "mmr": 2600, "asOf": iso(20700)},
             "candidates": [{"region": int(cand["region"][i]), "mmr": None if np.isnan(cand["mmr"][i]) else
                             float(cand["mmr"][i]), "lastPlayed": iso(cand["last"][i])} for i in range(4)],
             "scores": sc.tolist(), "params": PARAMS_MATCH}
    p = os.path.join(ROOT, "src", "lib", "personal", "__tests__", "fixtures", "site_cards.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump({"note": "Synthetic cards (no real accounts), from p3_site_publish.py fixture",
                   "cards": cases, "match": match}, f)
    print(f"wrote {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "load", "simulate", "check", "fixture"])
    ap.add_argument("--n-check", type=int, default=40)
    ap.add_argument("--store", default="full_1017")
    ap.add_argument("--window", type=int, default=180, help="days before the store cap that make an account active")
    ap.add_argument("--min-recent", type=int, default=3, help="load and simulate: minimum games in the window")
    ap.add_argument("--test-days", type=int, default=14)
    ap.add_argument("--games", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--params", help="simulate: JSON overrides of PARAMS_MATCH")
    a = ap.parse_args()
    {"build": build, "load": load, "simulate": simulate, "check": check, "fixture": fixture}[a.stage](a)


if __name__ == "__main__":
    main()
