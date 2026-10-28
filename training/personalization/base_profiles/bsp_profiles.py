"""
BSP: as-of hero x map x tier scoreboard profiles (5090, CPU, numpy only).

Per player-game row (sb.npz joined to games.npz):
  rate        value per minute of game length; time_spent_dead as a share of
              the game; highest_kill_streak raw. Missing "can be zero"
              metrics (healing, shields, cc, ...) count as 0 when the row has
              a hero_damage value; rows without hero_damage are dropped.
              Winsorized at the 99.9th percentile of each metric. "cc" is
              stun + root + silence seconds (time_cc_enemy_heroes is always
              0 in the store).
Weekly cell sums over (week, hero, map, tier, outcome), weeks from Monday.

Profile of (hero, map, tier) as of week w, window L weeks (weeks w-L..w-1:
every game used is from a strictly earlier week than the scored game):
  for each outcome o separately,
    m_h   = mean over the hero's games in the window (all maps, tiers);
            if the window has none, over every earlier week
    m_hm  = (S_hm + K m_h) / (n_hm + K)
    m_hmt = (S_hmt + K m_hm) / (n_hmt + K)
  profile = (m_hmt(win) + m_hmt(loss)) / 2
so the profile does not depend on how often the hero wins. Heroes with no
earlier game at all get the all-hero mean (flag kept).

As-of hero win rate (control, from every game in games.npz, same windows):
  wr_hmt shrunk to hero x map, then hero, then 0.5 (K_WR games).
"""
import os

import numpy as np

BSP = os.environ.get("BSP", os.path.expanduser("~/hots/base_profiles"))
DATA = os.path.join(BSP, "data")
MON0 = 4        # 1970-01-05 (day 4) was a Monday
K = 40.0
K_WR = 200.0

# profile dimensions (name, source metric)
DIMS = [("dmg", "hero_damage"), ("siege", "siege_damage"), ("struct", "structure_damage"),
        ("heal", "healing"), ("self_heal", "self_healing"), ("shield", "protection_allies"),
        ("taken", "damage_taken"), ("xp", "experience_contribution"), ("cc", "stunning_enemies+rooting_enemies+silencing_enemies"),
        ("stun", "stunning_enemies"), ("root", "rooting_enemies"), ("silence", "silencing_enemies"),
        ("merc", "merc_camp_captures"), ("deaths", "deaths"), ("dead_share", "time_spent_dead"),
        ("escapes", "escapes"), ("globes", "regen_globes"), ("tf_dmg", "teamfight_hero_damage"),
        ("tf_taken", "teamfight_damage_taken"), ("tf_heal", "teamfight_healing"),
        ("takedowns", "takedowns"), ("outnum_deaths", "outnumbered_deaths")]
DIM_NAMES = [d for d, _ in DIMS]


def week_of(day):
    return (np.asarray(day) - MON0) // 7


def load_games():
    g = np.load(os.path.join(DATA, "games.npz"), allow_pickle=True)   # our own export; version is an object array
    out = {k: g[k] for k in g.files}
    out["version"] = out["version"].astype("U16")
    out["day"] = out["end_ts"] // 86400
    o = np.argsort(out["replay_id"])
    for k in list(out):
        if k not in ("heroes", "maps") and len(out[k]) == len(o):
            out[k] = out[k][o]
    return out


def build_cells(G):
    """-> dict(S (nW,H,M,T,2,D) sums, N (nW,H,M,T,2) counts, w0, nW, info)."""
    sb = np.load(os.path.join(DATA, "sb.npz"), mmap_mode="r")
    rid = np.asarray(sb["replay_id"])
    j = np.searchsorted(G["replay_id"], rid)
    j = np.minimum(j, len(G["replay_id"]) - 1)
    hit = G["replay_id"][j] == rid
    hero = np.asarray(sb["hero"]).astype(np.int64)
    win = np.asarray(sb["winner"]).astype(np.int64)
    glen = G["game_length"][j] / 60.0
    hd = np.asarray(sb["hero_damage"])
    ok = (hit & (hero >= 0) & (win >= 0) & (G["map"][j] >= 0) & (G["tier"][j] >= 0) & (glen > 3)
          & ~np.isnan(hd))
    idx = np.where(ok)[0]
    j = j[idx]
    hero, win, glen = hero[idx], win[idx], glen[idx]
    wk = week_of(G["day"][j])
    w0 = int(wk.min())
    nW = int(wk.max()) - w0 + 1
    H, M, T = len(G["heroes"]), len(G["maps"]), 3
    cell = ((((wk - w0) * H + hero) * M + G["map"][j]) * T + G["tier"][j]) * 2 + win
    ncell = nW * H * M * T * 2
    N = np.bincount(cell, minlength=ncell).astype(np.float64)
    S = np.zeros((ncell, len(DIMS)), np.float64)
    info = {"rows_used": int(len(idx)), "rows_total": int(len(rid)), "first_week_day": int(w0 * 7 + MON0),
            "metrics": {}}
    for d, (name, m) in enumerate(DIMS):
        v = sum(np.asarray(sb[part])[idx].astype(np.float64) for part in m.split("+"))
        nan = np.isnan(v)
        v[nan] = 0.0
        if m == "time_spent_dead":
            r = v / (glen * 60.0)
        elif m == "highest_kill_streak":
            r = v
        else:
            r = v / glen
        cap = np.percentile(r, 99.9)
        r = np.minimum(r, cap)
        S[:, d] = np.bincount(cell, weights=r, minlength=ncell)
        info["metrics"][name] = {"nan_share": float(nan.mean()), "nonzero": float((r > 0).mean()),
                                 "cap_p999": float(cap), "mean": float(r.mean())}
    return {"S": S.reshape(nW, H, M, T, 2, len(DIMS)), "N": N.reshape(nW, H, M, T, 2), "w0": w0,
            "nW": nW, "info": info}


def cumulate(C):
    """prefix sums over weeks: P[i] = sum of weeks < i (i = 0..nW)."""
    S, N = C["S"], C["N"]
    PS = np.zeros((S.shape[0] + 1,) + S.shape[1:], np.float64)
    PN = np.zeros((N.shape[0] + 1,) + N.shape[1:], np.float64)
    np.cumsum(S, 0, out=PS[1:])
    np.cumsum(N, 0, out=PN[1:])
    return PS, PN


def profile_table(PS, PN, w0, week, L):
    """-> (prof (H,M,T,D), n_hmt (H,M,T) games in window, has_hero (H,))."""
    i1 = int(np.clip(week - w0, 0, PS.shape[0] - 1))
    i0 = int(np.clip(week - L - w0, 0, i1)) if L else 0
    S = PS[i1] - PS[i0]                         # (H,M,T,2,D)
    N = PN[i1] - PN[i0]                         # (H,M,T,2)
    Sall, Nall = PS[i1], PN[i1]
    S_h = S.sum((1, 2))                         # (H,2,D)
    N_h = N.sum((1, 2))
    Sa_h, Na_h = Sall.sum((1, 2)), Nall.sum((1, 2))
    m_h = np.where(N_h[..., None] > 0, S_h / np.maximum(N_h, 1)[..., None],
                   Sa_h / np.maximum(Na_h, 1)[..., None])
    has = (Na_h > 0).all(1)
    glob = (Sa_h.sum(0) / np.maximum(Na_h.sum(0), 1)[:, None])        # (2,D)
    m_h = np.where(has[:, None, None], m_h, glob[None])
    S_hm, N_hm = S.sum(2), N.sum(2)             # (H,M,2,D)
    m_hm = (S_hm + K * m_h[:, None]) / (N_hm + K)[..., None]
    m_hmt = (S + K * m_hm[:, :, None]) / (N + K)[..., None]
    prof = 0.5 * (m_hmt[..., 0, :] + m_hmt[..., 1, :])
    return prof, N.sum(-1), has


def wr_cells(G):
    """weekly (hero, map, tier) games and wins from every valid game."""
    ok = (G["map"] >= 0) & (G["tier"] >= 0) & (G["t0"] >= 0).all(1) & (G["t1"] >= 0).all(1) & (G["winner"] >= 0)
    idx = np.where(ok)[0]
    wk = week_of(G["day"][idx])
    w0 = int(wk.min())
    nW = int(wk.max()) - w0 + 1
    H, M, T = len(G["heroes"]), len(G["maps"]), 3
    Ng = np.zeros(nW * H * M * T)
    Nw = np.zeros_like(Ng)
    for team, tcol in ((0, "t0"), (1, "t1")):
        won = (G["winner"][idx] == team).astype(np.float64)
        for s in range(5):
            c = (((wk - w0) * H + G[tcol][idx, s]) * M + G["map"][idx]) * T + G["tier"][idx]
            Ng += np.bincount(c, minlength=len(Ng))
            Nw += np.bincount(c, weights=won, minlength=len(Ng))
    sh = (nW, H, M, T)
    PG = np.zeros((nW + 1,) + sh[1:])
    PW = np.zeros_like(PG)
    np.cumsum(Ng.reshape(sh), 0, out=PG[1:])
    np.cumsum(Nw.reshape(sh), 0, out=PW[1:])
    return PG, PW, w0


def wr_table(PG, PW, w0, week, L):
    i1 = int(np.clip(week - w0, 0, PG.shape[0] - 1))
    i0 = int(np.clip(week - L - w0, 0, i1)) if L else 0
    g, w = PG[i1] - PG[i0], PW[i1] - PW[i0]
    wr_h = (w.sum((1, 2)) + K_WR * 0.5) / (g.sum((1, 2)) + K_WR)
    wr_hm = (w.sum(2) + K_WR * wr_h[:, None]) / (g.sum(2) + K_WR)
    return (w + K_WR * wr_hm[:, :, None]) / (g + K_WR)


def load_games_maps():
    return np.load(os.path.join(DATA, "games.npz"))["maps"]


def load_games_heroes():
    return np.load(os.path.join(DATA, "games.npz"))["heroes"]
