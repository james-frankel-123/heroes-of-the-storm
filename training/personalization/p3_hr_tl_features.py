"""
P3 hero recs and replays (results/P3_HERO_RECS_AND_REPLAYS.md, part B2):
per-player-game features from parsed replay timelines.

Reads /home/max/hots-replays/timelines/<build>/<replay_id>.json.gz (never the
sealed tree; every id is checked against training/replays/cohorts/sealed_ids.txt).
Writes cache/hr/tl_rows.npz: one row per player-game with identity fields
(replay_id, unix time, region, toon_id = blizz_id, hero, team, win, map, length,
cohort) and raw feature columns. Normalization and analysis: p3_hr_tl_analyze.py.

Time origin for "before 10:00" is gates open (t - gates_open_s).

Run (from training/): nice -n 19 taskset -c 60-61 python3 personalization/p3_hr_tl_features.py
"""
import gzip
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TL = "/home/max/hots-replays/timelines"
MANIFEST = "/home/max/hots-replays/manifest.jsonl"
SEALED = os.path.join(HERE, "..", "replays", "cohorts", "sealed_ids.txt")
OUT = os.path.join(HERE, "cache", "hr", "tl_rows.npz")

# end-of-game score screen: the 15 scoreboard metrics of the model (HP names in comments)
SB = [("HeroDamage", "hero_damage"), ("SiegeDamage", "siege_damage"), ("StructureDamage", "structure_damage"),
      ("Healing", "healing"), ("SelfHealing", "self_healing"), ("ProtectionGivenToAllies", "protection_allies"),
      ("DamageTaken", "damage_taken"), ("Deaths", "deaths"), ("SoloKill", "kills"), ("Assists", "assists"),
      ("ExperienceContribution", "experience_contribution"), ("MercCampCaptures", "merc_camp_captures"),
      ("OutnumberedDeaths", "outnumbered_deaths"), ("EscapesPerformed", "escapes"), ("RegenGlobes", "regen_globes")]
# score-screen stats that the Heroes Profile scoreboard does not carry (or carries as 0)
EXTRA = [("TimeCCdEnemyHeroes", "cc_time"), ("TimeStunningEnemyHeroes", "stun_time"),
         ("TimeRootingEnemyHeroes", "root_time"), ("TimeSilencingEnemyHeroes", "silence_time"),
         ("TeamfightHeroDamage", "tf_hero_damage"), ("TeamfightDamageTaken", "tf_damage_taken"),
         ("TeamfightHealingDone", "tf_healing"), ("MinionDamage", "minion_damage"), ("CreepDamage", "creep_damage")]
# timeline-only features (computed below)
TLF = ["deaths_pre10", "deaths_pre_obj", "death_enemy_half", "death_isolated", "death_overext",
       "early_kp", "obj_kp", "enemy_half_share", "move_per_min", "merc_part",
       "apm_cmd", "apm_key", "apm_camera", "time_dead_share", "first_death_min",
       "xp_lead10", "level_lead10", "first_merc_lead"]
TEAM_FEATS = {"xp_lead10", "level_lead10", "first_merc_lead"}
ALLY_R = 10.0      # map units: "an ally nearby" at a death
NEAR_DT = 8.0      # seconds: position sample matched to an event
MERC_R = 12.0
MERC_DT = 20.0
OBJ_DT = 30.0


def sealed_ids():
    return set(int(x) for x in open(SEALED) if x.strip())


def pos_at(pos, t, dt=NEAR_DT):
    """Nearest position sample within dt seconds of t, or None."""
    if pos is None or len(pos) == 0:
        return None
    ts = pos[:, 0]
    j = np.searchsorted(ts, t)
    best = None
    for k in (j - 1, j):
        if 0 <= k < len(ts) and abs(ts[k] - t) <= dt and (best is None or abs(ts[k] - t) < abs(ts[best] - t)):
            best = k
    return None if best is None else pos[best, 1:]


def game_rows(path):
    d = json.load(gzip.open(path))
    g0 = d.get("gates_open_s")
    if g0 is None or not d.get("players") or len(d["players"]) != 10:
        return None
    L = d["length_s"] - g0
    mins = max(L / 60.0, 1.0)
    players = d["players"]
    team = {p["pid"]: p["team"] for p in players}
    pos = {p["pid"]: (np.asarray(p["positions"], np.float64) if p["positions"] else None) for p in players}
    # core positions per team (KingsCore born) for the enemy-half axis; fallback: team position centroids
    core = {}
    for e in d.get("notable_units", []):
        if e["ev"] == "born" and e["type"] == "KingsCore" and e.get("team") in (0, 1):
            core.setdefault(e["team"], (e["x"], e["y"]))
    if len(core) < 2:
        for tm in (0, 1):
            arr = [pos[p][:, 1:] for p in pos if team[p] == tm and pos[p] is not None and len(pos[p])]
            if arr:
                core[tm] = tuple(np.concatenate(arr).mean(0))
    have_axis = len(core) == 2

    def enemy_side(tm, xy):
        c0, c1 = np.asarray(core[tm], float), np.asarray(core[1 - tm], float)
        ax = c1 - c0
        return float(np.dot(np.asarray(xy, float) - c0, ax) / max(np.dot(ax, ax), 1e-9)) > 0.5

    obj_t = sorted(e["t"] for e in d.get("map_events", []) if e["t"] > g0)
    first_obj = obj_t[0] if obj_t else None
    deaths = d.get("deaths", [])
    # team takedowns (enemy deaths) by window
    td_early = {0: [], 1: []}
    td_obj = {0: [], 1: []}
    for de in deaths:
        v = de["victim"]
        if v not in team:
            continue
        kt = 1 - team[v]
        ks = set(k for k in de["killers"] if k in team and team[k] == kt)
        if de["t"] - g0 < 600:
            td_early[kt].append(ks)
        if obj_t and min(abs(de["t"] - o) for o in obj_t) <= OBJ_DT:
            td_obj[kt].append(ks)
    # team XP / level lead at 10:00
    xp = {0: None, 1: None}
    lvl = {0: None, 1: None}
    for e in d.get("xp_periodic", []):
        if e["t"] - g0 <= 600 + 30 and e["team"] in (0, 1):
            tot = sum((e.get(k) or 0.0) for k in ("minion", "creep", "structure", "hero", "trickle"))
            xp[e["team"]] = tot
            lvl[e["team"]] = e.get("level")
    first_merc = {0: None, 1: None}
    for m in d.get("mercs", []):
        if m.get("team") in (0, 1) and first_merc[m["team"]] is None:
            first_merc[m["team"]] = m["t"] - g0
    apm = d.get("actions_per_min") or {}
    rows = []
    for p in players:
        pid, tm = p["pid"], p["team"]
        sc = p["end"].get("score", {})
        f = {}
        my_d = [de for de in deaths if de["victim"] == pid]
        f["deaths_pre10"] = float(sum(de["t"] - g0 < 600 for de in my_d)) if L >= 600 else np.nan
        f["deaths_pre_obj"] = float(sum(de["t"] < first_obj for de in my_d)) if first_obj else np.nan
        f["first_death_min"] = (min(de["t"] for de in my_d) - g0) / 60.0 if my_d else L / 60.0
        eh = iso = ov = 0
        for de in my_d:
            xy = (de["x"], de["y"])
            if xy[0] is None:
                continue
            e_side = have_axis and enemy_side(tm, xy)
            allies = 0
            for q in players:
                if q["pid"] == pid or q["team"] != tm or pos[q["pid"]] is None:
                    continue
                a = pos_at(pos[q["pid"]], de["t"])
                if a is not None and np.hypot(a[0] - xy[0], a[1] - xy[1]) <= ALLY_R:
                    allies += 1
            eh += e_side
            iso += allies == 0
            ov += e_side and allies == 0
        f["death_enemy_half"], f["death_isolated"], f["death_overext"] = float(eh), float(iso), float(ov)
        f["early_kp"] = (np.mean([pid in ks for ks in td_early[tm]]) if td_early[tm] else np.nan)
        f["obj_kp"] = (np.mean([pid in ks for ks in td_obj[tm]]) if td_obj[tm] else np.nan)
        P = pos[pid]
        if P is not None and len(P) >= 4 and have_axis:
            f["enemy_half_share"] = float(np.mean([enemy_side(tm, xy) for xy in P[:, 1:]]))
            dtp = np.diff(P[:, 0])
            ok = (dtp > 0) & (dtp <= 16.5)
            dist = np.hypot(np.diff(P[:, 1]), np.diff(P[:, 2]))
            f["move_per_min"] = float(dist[ok].sum() / max(dtp[ok].sum(), 1.0) * 60.0) if ok.sum() >= 3 else np.nan
        else:
            f["enemy_half_share"] = f["move_per_min"] = np.nan
        own_m = [m for m in d.get("mercs", []) if m.get("team") == tm and m.get("x") is not None]
        if own_m and P is not None:
            hits = 0
            for m in own_m:
                sel = (P[:, 0] <= m["t"] + 2) & (P[:, 0] >= m["t"] - MERC_DT)
                if sel.any() and (np.hypot(P[sel, 1] - m["x"], P[sel, 2] - m["y"]) <= MERC_R).any():
                    hits += 1
            f["merc_part"] = hits / len(own_m)
        else:
            f["merc_part"] = np.nan
        a = apm.get(str(pid), {})
        f["apm_cmd"] = a.get("cmd", np.nan) if a else np.nan
        f["apm_key"] = a.get("key", np.nan) if a else np.nan
        f["apm_camera"] = a.get("camera", np.nan) if a else np.nan
        td = p["end"].get("time_dead")
        f["time_dead_share"] = (td / L) if (td is not None and L > 0) else np.nan
        f["xp_lead10"] = (xp[tm] - xp[1 - tm]) if (xp[0] is not None and xp[1] is not None and L >= 600) else np.nan
        f["level_lead10"] = (lvl[tm] - lvl[1 - tm]) if (lvl[0] is not None and lvl[1] is not None and L >= 600) else np.nan
        fm, fe = first_merc[tm], first_merc[1 - tm]
        f["first_merc_lead"] = ((fe if fe is not None else L) - (fm if fm is not None else L)) / 60.0
        row = {"replay_id": d["replay_id"] if d.get("replay_id") else int(os.path.basename(path).split(".")[0]),
               "unix": d["time_utc_filetime"] / 1e7 - 11644473600.0, "region": p.get("region", -1),
               "toon_id": p["toon_id"], "hero": p.get("hero") or "", "team": tm, "win": int(bool(p["win"])),
               "map": d.get("map_id") or "", "minutes": mins, "build": d["build"]}
        for k, nm in SB + EXTRA:
            v = sc.get(k)
            row[nm] = float(v) if v is not None else np.nan
        row.update(f)
        rows.append(row)
    return rows


def main():
    sealed = sealed_ids()
    coh = {}
    for line in open(MANIFEST):
        r = json.loads(line)
        if r.get("status") in ("ok", "linked"):
            coh.setdefault(int(r["replay_id"]), set()).add(r["cohort"])
    paths = []
    for b in sorted(os.listdir(TL)):
        dd = os.path.join(TL, b)
        if not os.path.isdir(dd):
            continue
        for fn in os.listdir(dd):
            if fn.endswith(".json.gz"):
                paths.append(os.path.join(dd, fn))
    ids = [int(os.path.basename(p).split(".")[0]) for p in paths]
    overlap = len(set(ids) & sealed)
    assert overlap == 0, f"sealed overlap {overlap}"
    print(f"timelines {len(paths):,}; sealed overlap {overlap}", flush=True)
    with Pool(2) as pool:
        res = pool.map(game_rows, paths, chunksize=50)
    rows = [r for g in res if g for r in g]
    cols = list(rows[0].keys())
    out = {}
    for c in cols:
        v = [r[c] for r in rows]
        if c in ("hero", "map"):
            out[c] = np.array(v)
        elif c in ("replay_id", "toon_id"):
            out[c] = np.array(v, np.int64)
        elif c in ("region", "team", "win", "build"):
            out[c] = np.array(v, np.int32)
        else:
            out[c] = np.array(v, np.float64)
    cmap = {"stratified_b": 1, "surprise_v1": 2, "p2_panel_v1": 3}
    out["cohort"] = np.array([max((cmap.get(c, 0) for c in coh.get(int(i), {"?"})), default=0)
                              for i in out["replay_id"]], np.int8)
    out["sealed_overlap"] = np.int64(overlap)
    out["n_timelines"] = np.int64(len(paths))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, **out)
    print(f"wrote {OUT}: {len(rows):,} player rows from {len(set(out['replay_id'])):,} games", flush=True)
    for c in [k for k, _ in []] + [nm for _, nm in SB + EXTRA] + TLF:
        print(f"  {c:22s} coverage {np.isfinite(out[c]).mean():.3f}  median {np.nanmedian(out[c]):.3f}", flush=True)


if __name__ == "__main__":
    main()
