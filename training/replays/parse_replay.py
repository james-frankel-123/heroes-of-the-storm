"""
Parse one .StormReplay into a compact per-game timeline (dict, written as .json.gz).

Uses Blizzard's heroprotocol. The PyPI release imports the removed `imp` module and only
ships protocols up to build 91756; GitHub master stops at 96477, which is byte-identical
to 91756 apart from comments. So builds without their own protocol file use the newest
available one, and the timeline records which protocol decoded it.

Time: `t` is seconds since replay start (gameloop / 16). `gates_open_s` is when the gates
opened; the in-game clock is roughly t - gates_open_s.

Players are indexed by `pid`, the tracker PlayerID (1-10). Team is 0/1 (details order,
same as Heroes Profile's team index; checked in validate_timelines.py). No player names
are stored; `toon_id` is the Battle.net account id (= blizz_id in replay_players).

The sealed holdout must never reach this code: parse() refuses any path under a `sealed`
directory and any replayID in training/replays/cohorts/sealed_ids.txt.
"""
import gzip
import importlib
import json
import os
import re
import sys
import types
from collections import Counter, defaultdict

sys.modules.setdefault("imp", types.ModuleType("imp"))   # heroprotocol imports it, unused
import mpyq                                               # noqa: E402
import heroprotocol.versions as hpv                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SEALED_IDS = os.path.join(HERE, "cohorts", "sealed_ids.txt")
REF = os.path.join(HERE, "ref", "heroes_talents_ref.json")
LOOPS_PER_S = 16.0
FIX = 4096.0          # fixed-point scale of SStatGameEvent m_fixedData

_sealed = None
_ref = None
_protocols = {}


class SealedError(RuntimeError):
    pass


def sealed_ids():
    global _sealed
    if _sealed is None:
        _sealed = set()
        if os.path.exists(SEALED_IDS):
            with open(SEALED_IDS) as f:
                _sealed = {int(x) for x in f if x.strip()}
    return _sealed


def ref():
    global _ref
    if _ref is None:
        with open(REF) as f:
            _ref = json.load(f)
    return _ref


def protocol_for(build):
    if build in _protocols:
        return _protocols[build]
    try:
        mod, used = importlib.import_module(f"heroprotocol.versions.protocol{build}"), build
    except ImportError:
        avail = sorted(int(m[8:-3]) for m in os.listdir(os.path.dirname(hpv.__file__))
                       if re.fullmatch(r"protocol\d+\.py", m))
        older = [b for b in avail if b <= build]
        used = max(older) if older and build else avail[-1]
        mod = importlib.import_module(f"heroprotocol.versions.protocol{used}")
    _protocols[build] = (mod, used)
    return mod, used


def s(x):
    return x.decode("utf-8", "replace") if isinstance(x, bytes) else x


def t_of(loop):
    return round(loop / LOOPS_PER_S, 2)


def stat_fields(e):
    """Flatten an SStatGameEvent into {key: value}; repeated keys become lists."""
    out = {}

    def put(k, v):
        k = s(k)
        if k in out:
            out[k] = out[k] if isinstance(out[k], list) else [out[k]]
            out[k].append(v)
        else:
            out[k] = v
    for kv in e.get("m_stringData") or []:
        put(kv["m_key"], s(kv["m_value"]))
    for kv in e.get("m_intData") or []:
        put(kv["m_key"], kv["m_value"])
    for kv in e.get("m_fixedData") or []:
        put(kv["m_key"], round(kv["m_value"] / FIX, 3))
    return out


def norm(x):
    return re.sub(r"[^a-z0-9]", "", (x or "").lower())


def hero_name(unit_id):
    """'HeroAnubarak' -> "Anub'arak" via the reference, else strip 'Hero'."""
    h = ref()["heroes"].get(unit_id)
    if h:
        return h
    return unit_id[4:] if unit_id and unit_id.startswith("Hero") else unit_id


def talent_name(tree_id):
    t = ref()["talents"].get(tree_id)
    return t["name"] if t else None


# Unit types worth a born/died record. Everything else only lands in unit_type_counts.
NOTABLE = re.compile(
    r"^(Town(?!Wall|Gate)|KingsCore|VanndarStormpike|DrekThar|Merc|JungleGrave|Boss|Vehicle|"
    r"Underworld|Slime|Soul|Raven|Zerg|Immortal|Punisher|Shrine|Dragon|Terror|Garden|Plant|"
    r"Payload|Nuke|Beacon|Temple|Tribute|Skull|Altar|Lane|Monster|Golem|Avatar|Hive|Holdout|Seed$)",
)
NOT_NOTABLE = re.compile(r"(Minion|Dummy|Missile|Weapon|Sconce|Target|Pathing|Footprint|Marker|Wall)")
KNOWN_STATS = {
    "GameStart", "PlayerInit", "PlayerSpawned", "LevelUp", "TalentChosen", "PlayerDeath",
    "TownStructureInit", "TownStructureDeath", "JungleCampInit", "JungleCampCapture",
    "PeriodicXPBreakdown", "GatesOpen", "RegenGlobePickedUp", "EndOfGameXPBreakdown",
    "EndOfGameTimeSpentDead", "EndOfGameTalentChoices", "EndOfGameRegenMasterStacks",
    "EndOfGameUpVotesCollected", "LootSprayUsed", "LootWheelUsed", "LootVoiceLineUsed",
    "EndOfGameAwards",
}
ACTION_EVENTS = {
    "NNet.Game.SCmdEvent": "cmd",
    "NNet.Game.SCmdUpdateTargetPointEvent": "cmd_retarget",
    "NNet.Game.SCmdUpdateTargetUnitEvent": "cmd_retarget",
    "NNet.Game.SCameraUpdateEvent": "camera",
    "NNet.Game.STriggerKeyPressedEvent": "key",
    "NNet.Game.SSelectionDeltaEvent": "selection",
    "NNet.Game.SUnitClickEvent": "unit_click",
    "NNet.Game.STriggerPingEvent": "ping",
    "NNet.Game.STriggerChatMessageEvent": "chat",
    "NNet.Game.SHeroTalentTreeSelectedEvent": "talent_pick",
}


def parse(path, replay_id=None, with_actions=True):
    if replay_id is None:
        m = re.search(r"(\d+)\.StormReplay$", path)
        replay_id = int(m.group(1)) if m else None
    if "/sealed/" in os.path.abspath(path) or (replay_id is not None and replay_id in sealed_ids()):
        raise SealedError(f"refusing to parse sealed replay {replay_id}")

    arc = mpyq.MPQArchive(path)
    header = protocol_for(0)[0].decode_replay_header(arc.header["user_data_header"]["content"])
    build = header["m_version"]["m_baseBuild"]
    proto, used = protocol_for(build)
    v = header["m_version"]
    details = proto.decode_replay_details(arc.read_file("replay.details"))
    init = proto.decode_replay_initdata(arc.read_file("replay.initData"))
    lobby = init["m_syncLobbyState"]
    gdesc = lobby["m_gameDescription"]

    # ---- players: details order -> tracker pid via toon handle
    dplayers = details["m_playerList"]
    toon_to_didx = {}
    for i, p in enumerate(dplayers):
        tn = p["m_toon"]
        toon_to_didx[f"{tn['m_region']}-{s(tn['m_programId'])}-{tn['m_realm']}-{tn['m_id']}"] = i
    user_to_didx = {}
    for slot in lobby["m_lobbyState"]["m_slots"]:
        if slot["m_userId"] is not None and slot["m_workingSetSlotId"] is not None:
            user_to_didx[slot["m_userId"]] = slot["m_workingSetSlotId"]

    tracker = proto.decode_replay_tracker_events(arc.read_file("replay.tracker.events"))

    players = {}          # pid -> dict
    didx_to_pid = {}
    gates_loop = None
    levels = defaultdict(list)          # pid -> [(level, t)]
    talents = defaultdict(list)         # pid -> [{tier, id, name, t}]
    deaths, structures, mercs, map_events, xp_periodic = [], [], [], [], []
    town_info, camp_info = {}, {}
    globes = Counter()
    eog = defaultdict(dict)
    unit_types = {}       # (index, recycle) -> (type, control pid, born loop)
    unit_counts = defaultdict(Counter)   # team/"neutral" -> type -> born count
    notable = []
    hero_units = {}       # unit index -> pid
    positions = defaultdict(list)       # pid -> [[t, x, y], ...]
    score = None
    bans, picks = [], []
    end_loop = header["m_elapsedGameLoops"]

    def team_of_ctrl(pid):
        if pid in players:
            return players[pid]["team"]
        if pid == 11:
            return 0
        if pid == 12:
            return 1
        return None

    for e in tracker:
        ev = e["_event"].rsplit(".", 1)[-1]
        loop = e["_gameloop"]
        if ev == "SStatGameEvent":
            name = s(e["m_eventName"])
            f = stat_fields(e)
            if name == "PlayerInit":
                didx = toon_to_didx.get(f.get("ToonHandle"))
                pid = f["PlayerID"]
                if didx is not None:
                    dp = dplayers[didx]
                    players[pid] = {"pid": pid, "team": dp["m_teamId"], "toon_id": dp["m_toon"]["m_id"],
                                    "region": dp["m_toon"]["m_region"], "win": dp["m_result"] == 1,
                                    "controller": f.get("Controller")}
                    didx_to_pid[didx] = pid
            elif name == "PlayerSpawned":
                pid = f["PlayerID"]
                if pid in players and "hero_id" not in players[pid]:
                    players[pid]["hero_id"] = f.get("Hero")
                    players[pid]["hero"] = hero_name(f.get("Hero"))
            elif name == "GatesOpen":
                gates_loop = loop
            elif name == "LevelUp":
                levels[f["PlayerID"]].append([f["Level"], t_of(loop)])
            elif name == "TalentChosen":
                pid = f["PlayerID"]
                tid = f.get("PurchaseName")
                talents[pid].append({"tier": len(talents[pid]) + 1, "id": tid,
                                     "name": talent_name(tid), "t": t_of(loop)})
            elif name == "PlayerDeath":
                k = f.get("KillingPlayer")
                killers = k if isinstance(k, list) else ([k] if k is not None else [])
                deaths.append({"t": t_of(loop), "victim": f["PlayerID"], "killers": killers,
                               "x": f.get("PositionX"), "y": f.get("PositionY")})
            elif name == "TownStructureInit":
                town_info[f["TownID"]] = {"team": f.get("Team", 0) - 1, "lane": f.get("Lane"),
                                          "x": f.get("PositionX"), "y": f.get("PositionY")}
            elif name == "TownStructureDeath":
                ti = town_info.get(f.get("TownID"), {})
                structures.append({"t": t_of(loop), "kind": "town", "type": f.get("UnitType"),
                                   "town": f.get("TownID"), "team": ti.get("team"), "lane": ti.get("lane"),
                                   "x": ti.get("x"), "y": ti.get("y")})
            elif name == "JungleCampInit":
                camp_info[f["CampID"]] = {"x": f.get("PositionX"), "y": f.get("PositionY")}
            elif name == "JungleCampCapture":
                ci = camp_info.get(f.get("CampID"), {})
                tm = f.get("TeamID")
                mercs.append({"t": t_of(loop), "camp": f.get("CampID"), "type": f.get("CampType"),
                              "team": int(tm) - 1 if tm is not None else None,
                              "x": ci.get("x"), "y": ci.get("y")})
            elif name == "PeriodicXPBreakdown":
                xp_periodic.append({"t": t_of(loop), "team": f.get("Team", 0) - 1,
                                    "level": f.get("TeamLevel"), "minion": f.get("MinionXP"),
                                    "creep": f.get("CreepXP"), "structure": f.get("StructureXP"),
                                    "hero": f.get("HeroXP"), "trickle": f.get("TrickleXP")})
            elif name == "RegenGlobePickedUp":
                globes[f["PlayerID"]] += 1
            elif name == "EndOfGameXPBreakdown":
                eog[f["PlayerID"]]["xp"] = {k: f.get(k) for k in
                                            ("MinionXP", "CreepXP", "StructureXP", "HeroXP", "TrickleXP")}
            elif name == "EndOfGameTimeSpentDead":
                eog[f["PlayerID"]]["time_dead"] = f.get("Time")
            elif name == "EndOfGameTalentChoices":
                pid = f["PlayerID"]
                eog[pid]["talent_choices"] = [f.get(f"Tier {i} Choice") for i in range(1, 8)
                                              if f.get(f"Tier {i} Choice")]
                eog[pid]["map_id"] = f.get("Map")
            elif name in ("GameStart", "LootSprayUsed", "LootWheelUsed", "LootVoiceLineUsed",
                          "EndOfGameUpVotesCollected", "EndOfGameRegenMasterStacks"):
                pass
            else:
                map_events.append({"t": t_of(loop), "name": name, **f})
        elif ev == "SUnitBornEvent":
            typ = s(e["m_unitTypeName"])
            ctrl = e["m_controlPlayerId"]
            key = (e["m_unitTagIndex"], e["m_unitTagRecycle"])
            unit_types[key] = (typ, ctrl, loop)
            tm = team_of_ctrl(ctrl)
            unit_counts["neutral" if tm is None else str(tm)][typ] += 1
            if typ.startswith("Hero") and 1 <= ctrl <= 10 and ctrl in players:
                hero_units[e["m_unitTagIndex"]] = ctrl
            elif NOTABLE.match(typ) and not NOT_NOTABLE.search(typ) and not (loop == 0 and typ.startswith("Town")):
                notable.append({"t": t_of(loop), "ev": "born", "type": typ, "team": tm,
                                "x": e["m_x"], "y": e["m_y"]})
        elif ev == "SUnitDiedEvent":
            key = (e["m_unitTagIndex"], e["m_unitTagRecycle"])
            info = unit_types.get(key)
            if info is None:
                continue
            typ, ctrl, _ = info
            if typ.startswith("Hero"):
                continue
            if (NOTABLE.match(typ) and not NOT_NOTABLE.search(typ)) or typ.startswith("TownWall") \
                    or typ.startswith("TownGate"):
                rec = {"t": t_of(loop), "ev": "died", "type": typ, "team": team_of_ctrl(ctrl),
                       "killer": e["m_killerPlayerId"], "x": e["m_x"], "y": e["m_y"]}
                if typ.startswith("Town") or typ in ("KingsCore", "VanndarStormpike", "DrekThar"):
                    structures.append({**rec, "kind": "unit"})
                else:
                    notable.append(rec)
        elif ev == "SUnitPositionsEvent":
            idx = e["m_firstUnitIndex"]
            items = e["m_items"]
            for i in range(0, len(items), 3):
                idx += items[i]
                pid = hero_units.get(idx)
                if pid is not None:
                    positions[pid].append([t_of(loop), items[i + 1], items[i + 2]])
        elif ev == "SHeroBannedEvent":
            if s(e["m_hero"]) != "NONE":
                bans.append({"hero_id": s(e["m_hero"]), "team": e["m_controllingTeam"] - 1})
        elif ev == "SHeroPickedEvent":
            picks.append({"hero_id": s(e["m_hero"]), "pid": e["m_controllingPlayer"]})
        elif ev == "SScoreResultEvent":
            score = e["m_instanceList"]

    if score:
        for inst in score:
            nm = s(inst["m_name"])
            for slot, vals in enumerate(inst["m_values"]):
                pid = slot + 1
                if pid in players and vals:
                    eog[pid].setdefault("score", {})[nm] = vals[0]["m_value"]

    actions = None
    if with_actions:
        counts = defaultdict(Counter)
        for e in proto.decode_replay_game_events(arc.read_file("replay.game.events")):
            kind = ACTION_EVENTS.get(e["_event"])
            if kind is None:
                continue
            didx = user_to_didx.get(e["_userid"]["m_userId"])
            pid = didx_to_pid.get(didx)
            if pid is not None and (gates_loop is None or e["_gameloop"] >= gates_loop):
                counts[pid][kind] += 1
        minutes = max(1e-6, (end_loop - (gates_loop or 0)) / LOOPS_PER_S / 60)
        actions = {str(pid): {k: round(n / minutes, 2) for k, n in sorted(c.items())}
                   for pid, c in sorted(counts.items())}

    for pid, p in players.items():
        p["talents"] = talents.get(pid, [])
        p["levels"] = levels.get(pid, [])
        p["regen_globes"] = globes.get(pid, 0)
        p["end"] = eog.get(pid, {})
        p["positions"] = positions.get(pid, [])

    team_levels = {}
    for tm in (0, 1):
        pids = [pid for pid, p in players.items() if p["team"] == tm]
        if pids:
            team_levels[str(tm)] = max((levels.get(pid, []) for pid in pids), key=len)

    map_id = next((e.get("map_id") for e in eog.values() if e.get("map_id")), None)
    return {
        "schema": 1,
        "replay_id": replay_id,
        "version": f"{v['m_major']}.{v['m_minor']}.{v['m_revision']}.{v['m_build']}",
        "build": build,
        "protocol": used,
        "map_id": map_id,
        "map_title_local": s(details["m_title"]),
        "amm_id": gdesc["m_gameOptions"].get("m_ammId"),
        "time_utc_filetime": details["m_timeUTC"],
        "loops": end_loop,
        "length_s": t_of(end_loop),
        "gates_open_s": t_of(gates_loop) if gates_loop is not None else None,
        "winner_team": next((p["team"] for p in players.values() if p["win"]), None),
        "players": [players[k] for k in sorted(players)],
        "bans": bans,
        "picks": picks,
        "team_levels": team_levels,
        "xp_periodic": xp_periodic,
        "deaths": deaths,
        "structures": structures,
        "mercs": mercs,
        "map_events": map_events,
        "notable_units": notable,
        "unit_type_counts": {k: dict(v) for k, v in unit_counts.items()},
        "actions_per_min": actions,
    }


def write(timeline, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    with gzip.open(tmp, "wt", compresslevel=6) as f:
        json.dump(timeline, f, separators=(",", ":"))
    os.replace(tmp, out_path)


if __name__ == "__main__":
    tl = parse(sys.argv[1])
    print(json.dumps({k: (v if not isinstance(v, (list, dict)) else f"<{len(v)}>") for k, v in tl.items()}))
