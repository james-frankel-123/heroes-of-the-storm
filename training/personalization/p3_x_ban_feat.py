"""
P3 extensions, ban redo: causal per-player x hero features from games that
ended before the current game (play-time order: game_date, replay_id), over
the snapshot slot table.

MAWP is Max's formula exactly as in src/lib/mawp.ts / sync/compute-derived.ts:
  over the player's Storm League games on the hero before the draft, newest
  first (rank 1 = most recent):
    w_g = 1 for rank <= 30, else exp(-ln2/30 (rank - 30))
    w_t = 1 if days <= 180, else exp(-ln2/90 (days - 180)); eff = y w_t + 0.5 (1 - w_t)
    if games < 30: add (30 - games) phantom 0.5 observations at weight 1
    MAWP = (sum w_g eff + phantoms * 0.5) / (sum w_g + phantoms); 0.5 if no games
  days = (draft time - that game's end) / 86400, draft time = this game's start
  (game_date - game_length). Only the 400 most recent games per hero are kept:
  rank 400 has w_g = 2e-4, so the truncation changes MAWP by < 1e-3.
EWMA proxies (per hero, half-lives in games on that hero): residual r = y - WP
at 10 and 30 games, win rate (y - 0.5) at 10 games.
For every slot: the player's main (most games so far), its share, games, the
90-day-decayed share, MAWP and EWMA residual of the main, and MAWP / EWMA of
the hero actually played. For query slots: all 90 heroes.

Library (used by p3_x_ban_nat.py and p3_x_ban_model.py).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from numba import njit
from p3_heroes import NUM_HEROES
import p3_hs_core as C

GAMETIME = os.path.join(C.CACHE, "gametime_2024q2.npz")
KEEP = 400
LG = np.log(2) / 30.0
LT = np.log(2) / 90.0


@njit(cache=True)
def _mawp(buf_t, buf_y, cnt, head, now):
    """buf_*: ring of the hero's last KEEP games (end time s, outcome);
    cnt = games so far on the hero; head = next write position."""
    if cnt == 0:
        return 0.5
    m = cnt if cnt < buf_t.shape[0] else buf_t.shape[0]
    num = 0.0
    den = 0.0
    for rank in range(1, m + 1):
        j = (head - rank) % buf_t.shape[0]
        wg = 1.0 if rank <= 30 else np.exp(-LG * (rank - 30))
        days = (now - buf_t[j]) / 86400.0
        wt = 1.0 if days <= 180 else np.exp(-LT * (days - 180))
        eff = buf_y[j] * wt + 0.5 * (1 - wt)
        num += wg * eff
        den += wg
    if cnt < 30:
        num += (30 - cnt) * 0.5
        den += 30 - cnt
    return num / den


@njit(cache=True)
def _walk(starts, ends, hero, y, r, t_end, t_start, qslot,
          o_main, o_tot, o_topc, o_dshare, o_mawp_main, o_ewr_main, o_mawp_own, o_ewr_own, o_nown,
          q_mawp, q_ewr10, q_ewr30, q_eww10, q_cnt, q_days):
    a10 = 1 - np.exp(-np.log(2) / 10.0)
    a30 = 1 - np.exp(-np.log(2) / 30.0)
    lam_d = np.log(2) / 90.0
    bt = np.zeros((NUM_HEROES, 400))
    by = np.zeros((NUM_HEROES, 400))
    for p in range(starts.shape[0]):
        cnt = np.zeros(NUM_HEROES)
        dec = np.zeros(NUM_HEROES)
        ewr10 = np.zeros(NUM_HEROES)
        ewr30 = np.zeros(NUM_HEROES)
        eww10 = np.zeros(NUM_HEROES)
        w10 = np.zeros(NUM_HEROES)
        w30 = np.zeros(NUM_HEROES)
        last = np.full(NUM_HEROES, -1.0)
        head = np.zeros(NUM_HEROES, np.int64)  # ring entries beyond cnt are never read
        tlast = -1.0
        ap = starts[p]
        for i in range(starts[p], ends[p]):
            now = t_start[i]
            # decay the calendar-decayed counts to now
            if tlast >= 0:
                f = np.exp(-lam_d * max(now - tlast, 0.0) / 86400.0)
                for h in range(NUM_HEROES):
                    dec[h] *= f
            tlast = max(tlast, now)
            # games enter the history only once they have ended before this
            # game started (end-time order: the first unfinished one blocks)
            while ap < i and t_end[ap] <= now:
                h0 = hero[ap]
                cnt[h0] += 1
                dec[h0] += np.exp(-lam_d * max(now - t_end[ap], 0.0) / 86400.0)
                ewr10[h0] = (1 - a10) * ewr10[h0] + a10 * r[ap]
                w10[h0] = (1 - a10) * w10[h0] + a10
                ewr30[h0] = (1 - a30) * ewr30[h0] + a30 * r[ap]
                w30[h0] = (1 - a30) * w30[h0] + a30
                eww10[h0] = (1 - a10) * eww10[h0] + a10 * (y[ap] - 0.5)
                last[h0] = t_end[ap]
                bt[h0, head[h0] % 400] = t_end[ap]
                by[h0, head[h0] % 400] = y[ap]
                head[h0] += 1
                ap += 1
            tot = 0.0
            mx = -1.0
            mh = -1
            dsum = 0.0
            for h in range(NUM_HEROES):
                tot += cnt[h]
                dsum += dec[h]
                if cnt[h] > mx:
                    mx = cnt[h]
                    mh = h
            o_main[i] = mh
            o_tot[i] = tot
            o_topc[i] = mx
            o_dshare[i] = dec[mh] / dsum if dsum > 0 else 0.0
            o_mawp_main[i] = _mawp(bt[mh], by[mh], int(cnt[mh]), head[mh], now)
            o_ewr_main[i] = ewr10[mh] / w10[mh] if w10[mh] > 0 else 0.0
            h0 = hero[i]
            o_mawp_own[i] = _mawp(bt[h0], by[h0], int(cnt[h0]), head[h0], now)
            o_ewr_own[i] = ewr10[h0] / w10[h0] if w10[h0] > 0 else 0.0
            o_nown[i] = cnt[h0]
            q = qslot[i]
            if q >= 0:
                for h in range(NUM_HEROES):
                    q_mawp[q, h] = _mawp(bt[h], by[h], int(cnt[h]), head[h], now)
                    q_ewr10[q, h] = ewr10[h] / w10[h] if w10[h] > 0 else 0.0
                    q_ewr30[q, h] = ewr30[h] / w30[h] if w30[h] > 0 else 0.0
                    q_eww10[q, h] = eww10[h] / w10[h] if w10[h] > 0 else 0.0
                    q_cnt[q, h] = cnt[h]
                    q_days[q, h] = (now - last[h]) / 86400.0 if last[h] >= 0 else -1.0


def play_times(d):
    z = np.load(GAMETIME)
    o = np.argsort(z["replay_ids"])
    j = np.searchsorted(z["replay_ids"][o], d["replay_id"])
    t_end = z["ts"][o][j].astype(np.float64)
    gl = z["game_length"][o][j].astype(np.float64)
    t_start = t_end - np.where(gl > 0, gl, 1200.0)
    return t_end, t_start


def compute(d, qrows=None):
    """Returns dict of per-slot arrays (row-aligned with d) and, if qrows is
    given, (len(qrows), 90) arrays for those rows."""
    t_end, t_start = play_times(d)
    n = len(d["pid"])
    srt = np.lexsort((d["replay_id"], t_end, d["pid"]))
    pid = d["pid"][srt]
    brk = np.flatnonzero(np.r_[True, pid[1:] != pid[:-1]])
    ends = np.r_[brk[1:], n]
    qpos = np.full(n, -1, np.int64)
    nq = 0
    if qrows is not None:
        qpos[qrows] = np.arange(len(qrows))
        nq = len(qrows)
    qs = qpos[srt]
    outs = [np.zeros(n, np.int64)] + [np.zeros(n) for _ in range(8)]
    qa = [np.zeros((max(nq, 1), NUM_HEROES), np.float32) for _ in range(6)]
    _walk(brk.astype(np.int64), ends.astype(np.int64), d["hero"][srt].astype(np.int64),
          d["y"][srt].astype(np.float64), (d["y"] - d["wp"])[srt].astype(np.float64),
          t_end[srt], t_start[srt], qs, *outs, *qa)
    names = ["main", "tot", "top_cnt", "dshare_main", "mawp_main", "ewr_main", "mawp_own", "ewr_own", "n_own"]
    res = {}
    for nm, a in zip(names, outs):
        b = np.empty_like(a)
        b[srt] = a
        res[nm] = b
    if qrows is not None:
        for nm, a in zip(["q_mawp", "q_ewr10", "q_ewr30", "q_eww10", "q_cnt", "q_days"], qa):
            res[nm] = a
    return res


def real_tier(replay_ids):
    """Real league tier per game: stored league_tier - 1 (1 Bronze .. 5 Diamond),
    NULL = Master (6). -2 if unknown."""
    z = np.load(os.path.join(C.CACHE, "x_tiers.npz"))
    o = np.argsort(z["replay_ids"])
    s = z["replay_ids"][o]
    j = np.searchsorted(s, replay_ids)
    ok = (j < len(s)) & (s[np.minimum(j, len(s) - 1)] == replay_ids)
    lt = z["league_tier"][o][np.minimum(j, len(s) - 1)].astype(int)
    t = np.where(lt < 0, 6, lt - 1)
    return np.where(ok, t, -2)


TIER_NAMES = {1: "Bronze", 2: "Silver", 3: "Gold", 4: "Platinum", 5: "Diamond", 6: "Master"}
