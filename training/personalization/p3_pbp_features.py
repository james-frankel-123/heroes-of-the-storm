"""
P3 play-by-play study (results/P3_PLAY_BY_PLAY.md): per player-game features
from the play-by-play timelines (training/replays/extract_pbp.py output,
/home/max/hots-replays/pbp/<build>/<replay_id>.json.gz).

Every replay id is checked against training/replays/cohorts/sealed_ids.txt
(asserted zero overlap); nothing under hots-replays/sealed is read.

Families (raw values; normalization in p3_pbp_eval.py):
  mech   APM (commands, retargets, key presses, camera), ability casts and
         right-clicks per minute, ability share, idle share while alive (seconds
         in command gaps of 5 s or more), burstiness (CV of 10 s action counts),
         ability-mix entropy and deviation from the hero's mix, talent pick
         latency, camera away from the hero
  death  isolated deaths, unpaid deaths (no enemy hero died within 10 s),
         traded share, deaths in the 45 s before a map objective, late deaths,
         first death of the game, forward position of deaths, outnumbered
         exposure and survival when outnumbered
  fight  presence at team fights, takedown share in fights, died in fights,
         distance to the team centroid, front/back position relative to the
         team (all time and in fights), base time, standing still
  macro  XP globes, minion last hits, enemy minion deaths nearby (soak proxy,
         before and after 10:00), merc-unit kill share, merc presence,
         structure kill presence, objective presence, structure last hits
Rows also carry the end-of-game score screen (15 model metrics and extras) so
the replay games can be compared with the DB scoreboard.

Run (from training/): nice -n 19 taskset -c 56-59 python3 personalization/p3_pbp_features.py [--procs 4]
Output: cache/pbp/rows.npz (gitignored: replay ids and toon ids).
"""
import argparse
import gzip
import json
import math
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PBP = os.environ.get("PBP_DIR", "/home/max/hots-replays/pbp")
SEALED = os.path.join(HERE, "..", "replays", "cohorts", "sealed_ids.txt")
OUT = os.environ.get("PBP_ROWS", os.path.join(HERE, "cache", "pbp", "rows.npz"))

SB = [("HeroDamage", "hero_damage"), ("SiegeDamage", "siege_damage"), ("StructureDamage", "structure_damage"),
      ("Healing", "healing"), ("SelfHealing", "self_healing"), ("ProtectionGivenToAllies", "protection_allies"),
      ("DamageTaken", "damage_taken"), ("Deaths", "deaths"), ("SoloKill", "kills"), ("Assists", "assists"),
      ("ExperienceContribution", "experience_contribution"), ("MercCampCaptures", "merc_camp_captures"),
      ("OutnumberedDeaths", "outnumbered_deaths"), ("EscapesPerformed", "escapes"), ("RegenGlobes", "regen_globes")]
EXTRA = [("TimeCCdEnemyHeroes", "cc_time")]

FAMILIES = {
    "mech": ["apm_cmd", "apm_retarget", "apm_key", "apm_camera", "abil_pm", "rclick_pm", "abil_share",
             "idle_alive", "burst_cv", "abil_entropy", "abil_dev", "talent_latency", "cam_away"],
    "death": ["d_isolated", "d_unpaid", "d_traded_share", "d_preobj", "d_late", "d_first_blood",
              "d_axis_mean", "outnum_exposure", "outnum_survive"],
    "fight": ["f_presence", "f_takedown_share", "f_died_share", "p_centroid_dist", "p_front", "p_front_fight",
              "p_base_share", "p_still_share", "p_move_pm"],
    "macro": ["x_globes", "x_lasthits", "x_soak_early", "x_soak_late", "x_merc_kill_share", "x_merc_presence",
              "x_struct_presence", "x_obj_presence", "x_struct_kills"],
}
ALLF = [f for fam in FAMILIES.values() for f in fam]
FIGHT_GAP = 12.0
FIGHT_R = 20.0
NEAR_R = 12.0
SOAK_R = 15.0


def sealed_ids():
    s = set(int(x) for x in open(SEALED) if x.strip())
    assert len(s) >= 5000
    return s


class Track:
    """Hero position by linear interpolation of the 15 s samples, NaN while dead or
    more than 20 s from any sample."""

    def __init__(self, pos, dead):
        P = np.asarray(pos, np.float64) if pos else np.zeros((0, 3))
        self.t, self.x, self.y = (P[:, 0], P[:, 1], P[:, 2]) if len(P) else (np.zeros(0),) * 3
        self.dead = dead   # list of (t0, t1)

    def at(self, ts):
        ts = np.atleast_1d(np.asarray(ts, np.float64))
        if len(self.t) < 2:
            return np.full((len(ts), 2), np.nan)
        x = np.interp(ts, self.t, self.x)
        y = np.interp(ts, self.t, self.y)
        j = np.clip(np.searchsorted(self.t, ts), 1, len(self.t) - 1)
        gap = np.minimum(np.abs(ts - self.t[j - 1]), np.abs(self.t[j] - ts))
        bad = (gap > 20) | (ts < self.t[0] - 20) | (ts > self.t[-1] + 20)
        for a, b in self.dead:
            bad |= (ts >= a) & (ts < b)
        out = np.column_stack([x, y])
        out[bad] = np.nan
        return out

    def alive(self, ts):
        ts = np.atleast_1d(np.asarray(ts, np.float64))
        ok = np.ones(len(ts), bool)
        for a, b in self.dead:
            ok &= ~((ts >= a) & (ts < b))
        return ok


def game_rows(path):
    d = json.load(gzip.open(path))
    g0 = d.get("gates_open_s")
    pb = d.get("pbp")
    if g0 is None or pb is None or len(d.get("players") or []) != 10:
        return None
    Ls = d["length_s"] - g0
    mins = max(Ls / 60.0, 1.0)
    players = d["players"]
    team = {p["pid"]: p["team"] for p in players}
    pids = [p["pid"] for p in players]
    deaths = sorted(d.get("deaths", []), key=lambda e: e["t"])
    revives = pb.get("revives", [])
    dead = {pid: [] for pid in pids}
    for de in deaths:
        v = de["victim"]
        if v not in dead:
            continue
        nxt = [r[0] for r in revives if r[1] == v and r[0] > de["t"]]
        dead[v].append((de["t"], min(nxt) if nxt else de["t"] + 60.0))
    tr = {p["pid"]: Track(p["positions"], dead[p["pid"]]) for p in players}
    # core axis
    core = {}
    for e in d.get("notable_units", []):
        if e["ev"] == "born" and e["type"] == "KingsCore" and e.get("team") in (0, 1):
            core.setdefault(e["team"], np.array([e["x"], e["y"]], float))
    have_axis = len(core) == 2

    def axis(tm, xy):
        """0 at own core, 1 at the enemy core."""
        if not have_axis:
            return np.full(len(xy), np.nan)
        c0, c1 = core[tm], core[1 - tm]
        ax = c1 - c0
        return (np.asarray(xy, float) - c0) @ ax / max(ax @ ax, 1e-9)

    # objective times: starts of map-event clusters after the gates
    ot = sorted(e["t"] for e in d.get("map_events", []) if e["t"] > g0 + 30)
    obj_starts = [t for i, t in enumerate(ot) if i == 0 or t - ot[i - 1] > 60]
    # fights: clusters of hero deaths no more than 12 s apart, 2+ deaths
    fights, cur = [], []
    for de in deaths:
        if de.get("x") is None:
            continue
        if cur and de["t"] - cur[-1]["t"] > FIGHT_GAP:
            if len(cur) >= 2:
                fights.append(cur)
            cur = []
        cur.append(de)
    if len(cur) >= 2:
        fights.append(cur)
    grid = np.arange(g0 + 15.0, d["length_s"], 15.0)
    P = {pid: tr[pid].at(grid) for pid in pids}
    first_death = deaths[0]["victim"] if deaths else None
    minions = np.asarray(pb.get("minions") or np.zeros((0, 5)), float).reshape(-1, 5)
    globes = np.asarray(pb.get("globes") or np.zeros((0, 4)), float).reshape(-1, 4)
    mercu = np.asarray(pb.get("merc_units") or np.zeros((0, 4)), float).reshape(-1, 4)
    structs = [s for s in d.get("structures", []) if s.get("kind") == "unit" and s.get("x") is not None
               and s.get("team") in (0, 1) and "Wall" not in s["type"] and "Gate" not in s["type"]]
    notable = [e for e in d.get("notable_units", []) if e["t"] > g0 + 30 and e.get("x") is not None
               and e["type"] != "KingsCore" and not e["type"].startswith("Merc")]
    apm = d.get("actions_per_min") or {}
    tlev = {int(k): v for k, v in (d.get("team_levels") or {}).items()}
    rows = []
    for p in players:
        pid, tm = p["pid"], p["team"]
        T = tr[pid]
        pp = pb["players"].get(str(pid), {})
        f = {}
        # ---- mech
        a = apm.get(str(pid), {})
        f["apm_cmd"] = a.get("cmd", np.nan)
        f["apm_retarget"] = a.get("cmd_retarget", np.nan)
        f["apm_key"] = a.get("key", np.nan)
        f["apm_camera"] = a.get("camera", np.nan)
        ab = pp.get("abil", {})
        n_abil = sum(v for k, v in ab.items() if k != "-")
        n_rc = ab.get("-", 0)
        f["abil_pm"] = n_abil / mins
        f["rclick_pm"] = n_rc / mins
        f["abil_share"] = n_abil / (n_abil + n_rc) if (n_abil + n_rc) else np.nan
        pr = np.array([v for k, v in ab.items() if k != "-"], float)
        f["abil_entropy"] = float(-(pr / pr.sum() * np.log(pr / pr.sum())).sum()) if n_abil >= 10 else np.nan
        sec = np.frombuffer(pp.get("sec", "").encode("latin-1"), np.uint8).astype(np.int32)
        if len(sec) >= 60:
            ts = g0 + np.arange(len(sec)) + 0.5
            al = T.alive(ts)
            z = (sec == 0).astype(np.int8)
            # runs of zeros of length >= 5
            idle = np.zeros(len(sec), bool)
            i = 0
            while i < len(z):
                if z[i]:
                    j = i
                    while j < len(z) and z[j]:
                        j += 1
                    if j - i >= 5:
                        idle[i:j] = True
                    i = j
                else:
                    i += 1
            f["idle_alive"] = float(idle[al].mean()) if al.sum() >= 60 else np.nan
            nb = len(sec) // 10
            c10 = sec[:nb * 10].reshape(nb, 10).sum(1)
            al10 = al[:nb * 10].reshape(nb, 10).all(1)
            v = c10[al10]
            f["burst_cv"] = float(v.std() / v.mean()) if len(v) >= 6 and v.mean() > 0 else np.nan
        else:
            f["idle_alive"] = f["burst_cv"] = np.nan
        f["abil_dev"] = np.nan          # needs the hero's mix: second pass
        lat = []
        lv = {int(L): t for L, t in tlev.get(tm, [])}
        for tal in p.get("talents", []):
            lvl = {1: 1, 2: 4, 3: 7, 4: 10, 5: 13, 6: 16, 7: 20}.get(tal["tier"])
            if lvl and lvl > 1 and lvl in lv:
                dt = tal["t"] - lv[lvl]
                if -1 <= dt <= 300:
                    lat.append(min(max(dt, 0.0), 120.0))
        f["talent_latency"] = float(np.mean(lat)) if len(lat) >= 2 else np.nan
        cam = np.asarray(pp.get("cam", []), float).reshape(-1, 2) / 2.0
        if len(cam) >= 12:
            tb = g0 + 5.0 * np.arange(len(cam)) + 2.5
            hp = T.at(tb)
            ok = np.isfinite(hp[:, 0]) & (cam[:, 0] >= 0)
            dist = np.hypot(cam[ok, 0] - hp[ok, 0], cam[ok, 1] - hp[ok, 1])
            f["cam_away"] = float((dist > 25).mean()) if ok.sum() >= 12 else np.nan
        else:
            f["cam_away"] = np.nan
        # ---- death
        my = [de for de in deaths if de["victim"] == pid]
        iso = unpaid = traded = preobj = late = 0
        axs = []
        for de in my:
            t = de["t"]
            if de.get("x") is not None:
                xy = np.array([[de["x"], de["y"]]])
                allies = [q for q in pids if q != pid and team[q] == tm]
                near = 0
                for q in allies:
                    qa = tr[q].at([t - 2.0])[0]
                    if np.isfinite(qa[0]) and math.hypot(qa[0] - xy[0, 0], qa[1] - xy[0, 1]) <= NEAR_R:
                        near += 1
                iso += near == 0
                axs.append(float(axis(tm, xy)[0]))
            enemy_died = any(abs(e2["t"] - t) <= 10 and team.get(e2["victim"]) == 1 - tm for e2 in deaths)
            unpaid += not enemy_died
            traded += enemy_died
            preobj += any(0 <= o - t <= 45 for o in obj_starts)
            late += (t - g0) > 16 * 60
        f["d_isolated"] = iso / mins
        f["d_unpaid"] = unpaid / mins
        f["d_traded_share"] = traded / len(my) if my else np.nan
        f["d_preobj"] = preobj / mins if obj_starts else np.nan
        f["d_late"] = late / max((Ls - 16 * 60) / 60.0, 1.0) if Ls > 18 * 60 else np.nan
        f["d_first_blood"] = float(first_death == pid) if deaths else np.nan
        f["d_axis_mean"] = float(np.mean(axs)) if axs else np.nan
        # outnumbered exposure / survival at 15 s samples
        me = P[pid]
        alive_s = np.isfinite(me[:, 0])
        n_en = np.zeros(len(grid))
        n_al = np.zeros(len(grid))
        for q in pids:
            if q == pid:
                continue
            dq = np.hypot(P[q][:, 0] - me[:, 0], P[q][:, 1] - me[:, 1])
            if team[q] == tm:
                n_al += np.nan_to_num(dq, nan=1e9) <= NEAR_R
            else:
                n_en += np.nan_to_num(dq, nan=1e9) <= NEAR_R
        expo = alive_s & (n_en >= 2) & (n_en >= n_al + 2)
        f["outnum_exposure"] = float(expo.sum() / max(alive_s.sum() * 15 / 60.0, 1.0))
        if expo.sum() >= 2:
            dts = np.array([de["t"] for de in my])
            surv = [not ((dts >= t) & (dts <= t + 12)).any() for t in grid[expo]]
            f["outnum_survive"] = float(np.mean(surv))
        else:
            f["outnum_survive"] = np.nan
        # ---- fight / position
        pres = tk = died = nf = 0
        front_f = []
        for fg in fights:
            t0, t1 = fg[0]["t"], fg[-1]["t"]
            cx = np.mean([e["x"] for e in fg])
            cy = np.mean([e["y"] for e in fg])
            tm_ = (t0 + t1) / 2
            hp = T.at([t0 - 3.0])[0]
            if not np.isfinite(hp[0]):
                hp = T.at([tm_])[0]
            if not T.alive([t0 - 3.0])[0]:
                continue
            nf += 1
            here = np.isfinite(hp[0]) and math.hypot(hp[0] - cx, hp[1] - cy) <= FIGHT_R
            pres += here
            en = [e for e in fg if team.get(e["victim"]) == 1 - tm]
            if en:
                tk += any(pid in e["killers"] for e in en) / 1.0
            died += any(e["victim"] == pid for e in fg)
            if here and have_axis:
                al = [tr[q].at([t0 - 3.0])[0] for q in pids if team[q] == tm]
                al = np.array([a_ for a_ in al if np.isfinite(a_[0])])
                if len(al) >= 3:
                    front_f.append(float(axis(tm, hp[None])[0] - axis(tm, al).mean()))
        n_en_f = sum(1 for fg in fights if any(team.get(e["victim"]) == 1 - tm for e in fg))
        f["f_presence"] = pres / nf if nf >= 2 else np.nan
        f["f_takedown_share"] = tk / n_en_f if n_en_f >= 2 else np.nan
        f["f_died_share"] = died / max(pres, 1) if pres >= 2 else np.nan
        f["p_front_fight"] = float(np.mean(front_f)) if front_f else np.nan
        team_xy = np.stack([P[q] for q in pids if team[q] == tm and q != pid])   # (4, n, 2)
        cnt = np.isfinite(team_xy[:, :, 0]).sum(0)
        cxy = np.nanmean(np.where(np.isfinite(team_xy), team_xy, np.nan), 0) if True else None
        okc = alive_s & (cnt >= 2)
        f["p_centroid_dist"] = float(np.hypot(me[okc, 0] - cxy[okc, 0], me[okc, 1] - cxy[okc, 1]).mean()) \
            if okc.sum() >= 8 else np.nan
        if have_axis and okc.sum() >= 8:
            f["p_front"] = float((axis(tm, me[okc]) - axis(tm, cxy[okc])).mean())
            ax_me = axis(tm, me[alive_s])
            f["p_base_share"] = float((ax_me < 0.08).mean())
        else:
            f["p_front"] = f["p_base_share"] = np.nan
        Pr = np.asarray(p["positions"], float) if p["positions"] else np.zeros((0, 3))
        if len(Pr) >= 6:
            dtp = np.diff(Pr[:, 0])
            ok = (dtp > 0) & (dtp <= 16.5) & T.alive(Pr[1:, 0]) & T.alive(Pr[:-1, 0])
            dist = np.hypot(np.diff(Pr[:, 1]), np.diff(Pr[:, 2]))
            f["p_still_share"] = float((dist[ok] < 2.0).mean()) if ok.sum() >= 6 else np.nan
            f["p_move_pm"] = float(dist[ok].sum() / max(dtp[ok].sum(), 1.0) * 60.0) if ok.sum() >= 6 else np.nan
        else:
            f["p_still_share"] = f["p_move_pm"] = np.nan
        # ---- macro
        if len(globes):
            og = globes[globes[:, 3] == 11 + tm]      # collected by the player's team (killer = team AI id)
            hp = T.at(og[:, 0])
            f["x_globes"] = float((np.nan_to_num(np.hypot(hp[:, 0] - og[:, 1], hp[:, 1] - og[:, 2]),
                                                 nan=1e9) <= 8).sum()) / mins
        else:
            f["x_globes"] = np.nan
        f["x_lasthits"] = float((minions[:, 3] == pid).sum()) / mins if len(minions) else np.nan
        if len(minions):
            em = minions[minions[:, 4] == 1 - tm]
            hp = T.at(em[:, 0])
            near = np.hypot(hp[:, 0] - em[:, 1], hp[:, 1] - em[:, 2]) <= SOAK_R
            early = (em[:, 0] - g0) < 600
            f["x_soak_early"] = float(near[early].sum()) / max(min(Ls, 600) / 60.0, 1.0)
            f["x_soak_late"] = float(near[~early].sum()) / max((Ls - 600) / 60.0, 1.0) if Ls > 900 else np.nan
        else:
            f["x_soak_early"] = f["x_soak_late"] = np.nan
        if len(mercu):
            own = np.isin(mercu[:, 3], [q for q in pids if team[q] == tm])
            f["x_merc_kill_share"] = float((mercu[own, 3] == pid).mean()) if own.sum() >= 4 else np.nan
        else:
            f["x_merc_kill_share"] = np.nan
        own_m = [m for m in d.get("mercs", []) if m.get("team") == tm and m.get("x") is not None]
        if own_m:
            hp = T.at([m["t"] - 3 for m in own_m])
            f["x_merc_presence"] = float(np.mean(np.hypot(hp[:, 0] - np.array([m["x"] for m in own_m]),
                                                          hp[:, 1] - np.array([m["y"] for m in own_m])) <= 15))
        else:
            f["x_merc_presence"] = np.nan
        es = [s for s in structs if s["team"] == 1 - tm]
        if len(es) >= 2:
            hp = T.at([s["t"] for s in es])
            f["x_struct_presence"] = float(np.mean(np.nan_to_num(np.hypot(hp[:, 0] - np.array([s["x"] for s in es]),
                                                                          hp[:, 1] - np.array([s["y"] for s in es])),
                                                                 nan=1e9) <= 20))
            f["x_struct_kills"] = float(sum(s.get("killer") == pid for s in es)) / mins
        else:
            f["x_struct_presence"] = f["x_struct_kills"] = np.nan
        if len(notable) >= 2:
            hp = T.at([e["t"] for e in notable])
            f["x_obj_presence"] = float(np.mean(np.nan_to_num(np.hypot(hp[:, 0] - np.array([e["x"] for e in notable]),
                                                                       hp[:, 1] - np.array([e["y"] for e in notable])),
                                                              nan=1e9) <= 25))
        else:
            f["x_obj_presence"] = np.nan
        sc = p["end"].get("score", {})
        row = {"replay_id": int(d["replay_id"] or os.path.basename(path).split(".")[0]),
               "unix": d["time_utc_filetime"] / 1e7 - 11644473600.0, "region": p.get("region", -1),
               "toon_id": p["toon_id"], "hero": p.get("hero") or "", "team": tm, "win": int(bool(p["win"])),
               "map": d.get("map_id") or "", "minutes": mins, "abil": json.dumps(ab)}
        for k, nm in SB + EXTRA:
            v = sc.get(k)
            row["sb_" + nm] = float(v) if v is not None else np.nan
        row.update(f)
        rows.append(row)
    return rows


def abil_dev(z, fit):
    """1 - cosine(player's ability-cast mix, the hero's mean mix over fit rows)."""
    mixes = [json.loads(s) for s in z["abil"]]
    hero = z["hero"]
    mean = {}
    for i in np.flatnonzero(fit):
        m = mixes[i]
        tot = sum(v for k, v in m.items() if k != "-")
        if tot < 10:
            continue
        acc = mean.setdefault(hero[i], {})
        for k, v in m.items():
            if k != "-":
                acc[k] = acc.get(k, 0.0) + v / tot
    out = np.full(len(hero), np.nan)
    for i, m in enumerate(mixes):
        mu = mean.get(hero[i])
        tot = sum(v for k, v in m.items() if k != "-")
        if not mu or tot < 10:
            continue
        keys = set(mu) | {k for k in m if k != "-"}
        a = np.array([m.get(k, 0) / tot for k in keys])
        b = np.array([mu.get(k, 0.0) for k in keys])
        out[i] = 1 - float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--fit-before", default="2026-10-16", help="hero ability mixes from games before this day")
    a = ap.parse_args()
    sealed = sealed_ids()
    paths = []
    for b in sorted(os.listdir(PBP)):
        dd = os.path.join(PBP, b)
        if os.path.isdir(dd):
            paths += [os.path.join(dd, fn) for fn in os.listdir(dd) if fn.endswith(".json.gz")]
    ids = [int(os.path.basename(p).split(".")[0]) for p in paths]
    overlap = len(set(ids) & sealed)
    assert overlap == 0, f"sealed overlap {overlap}"
    print(f"pbp timelines {len(paths):,}; sealed overlap {overlap}", flush=True)
    with Pool(a.procs) as pool:
        res = pool.map(game_rows, paths, chunksize=20)
    rows = [r for g in res if g for r in g]
    out = {}
    for c in rows[0].keys():
        v = [r[c] for r in rows]
        if c in ("hero", "map", "abil"):
            out[c] = np.array(v)
        elif c in ("replay_id", "toon_id"):
            out[c] = np.array(v, np.int64)
        elif c in ("region", "team", "win"):
            out[c] = np.array(v, np.int32)
        else:
            out[c] = np.array(v, np.float64)
    import datetime
    cut = datetime.datetime.fromisoformat(a.fit_before).replace(tzinfo=datetime.timezone.utc).timestamp()
    out["abil_dev"] = abil_dev(out, out["unix"] < cut)
    out["sealed_overlap"] = np.int64(overlap)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, **out)
    print(f"wrote {OUT}: {len(rows):,} player rows from {len(set(out['replay_id'])):,} games", flush=True)
    for c in ALLF:
        print(f"  {c:20s} coverage {np.isfinite(out[c]).mean():.3f}  median {np.nanmedian(out[c]):.3f}", flush=True)


if __name__ == "__main__":
    main()
