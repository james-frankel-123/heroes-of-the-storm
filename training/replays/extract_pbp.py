"""
Play-by-play extension of the timeline parse (P3_PLAY_BY_PLAY.md).

For each replay: the normal timeline (parse_replay.parse, with the same action rates)
plus a `pbp` block with the event detail the timeline drops:

  per player (key = pid as str):
    abil       {"<abilLink>:<cmdIndex>": count} of SCmdEvents with an ability after gates
               ("-" = no ability: right-click move/attack)
    sec        per-second command counts after gates (SCmd + retarget + key presses), as a
               latin-1 string of bytes capped at 255 (index 0 = gates open)
    cam        camera centre every 5 s after gates, map units x2 (int), -1 = no update yet
    tgt        last right-click target point in each 5 s bin, map units x2 (int), -1 = none
    abil_bins  ability casts per 10 s bin after gates
  minions      [t, x, y, killer pid (0 = none, 11/12 = team AI), minion team] per lane minion death
  globes       [t, x, y, collector pid] per minion XP globe collected
  merc_units   [t, x, y, killer pid] per neutral merc-camp unit death
  revives      [t, pid, x, y] per hero revive (respawn)

Times are seconds since replay start, as in the timeline. Sealed replays are refused by
parse_replay.parse() and again here.

  python extract_pbp.py --root /home/max/hots-replays --list LIST --workers 12 [--out pbp]
"""
import argparse
import gzip
import json
import os
import sys
import time
import traceback
import types
from collections import Counter, defaultdict
from multiprocessing import Pool

sys.modules.setdefault("imp", types.ModuleType("imp"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpyq  # noqa: E402
import parse_replay as P  # noqa: E402

BIN = 5.0
ABIN = 10.0


def extras(path, tl):
    arc = mpyq.MPQArchive(path)
    header = P.protocol_for(0)[0].decode_replay_header(arc.header["user_data_header"]["content"])
    proto, _ = P.protocol_for(header["m_version"]["m_baseBuild"])
    details = proto.decode_replay_details(arc.read_file("replay.details"))
    init = proto.decode_replay_initdata(arc.read_file("replay.initData"))
    lobby = init["m_syncLobbyState"]
    toon_pid = {p["toon_id"]: p["pid"] for p in tl["players"]}
    didx_pid = {i: toon_pid.get(p["m_toon"]["m_id"]) for i, p in enumerate(details["m_playerList"])}
    user_pid = {}
    for slot in lobby["m_lobbyState"]["m_slots"]:
        if slot["m_userId"] is not None and slot["m_workingSetSlotId"] is not None:
            pid = didx_pid.get(slot["m_workingSetSlotId"])
            if pid is not None:
                user_pid[slot["m_userId"]] = pid
    g0 = tl["gates_open_s"] or 0.0
    L = max(tl["length_s"] - g0, 1.0)
    nsec = int(L) + 1
    nb = int(L / BIN) + 1
    na = int(L / ABIN) + 1
    team = {p["pid"]: p["team"] for p in tl["players"]}

    # ---- tracker: minions, globes, merc units, revives
    born = {}
    hero_tag = {}
    minions, globes, mercu, revives = [], [], [], []
    for e in proto.decode_replay_tracker_events(arc.read_file("replay.tracker.events")):
        ev = e["_event"]
        if ev.endswith("SUnitBornEvent"):
            typ = P.s(e["m_unitTypeName"])
            key = (e["m_unitTagIndex"], e["m_unitTagRecycle"])
            born[key] = (typ, e["m_controlPlayerId"])
            if typ.startswith("Hero") and e["m_controlPlayerId"] in team:
                hero_tag[e["m_unitTagIndex"]] = e["m_controlPlayerId"]
        elif ev.endswith("SUnitDiedEvent"):
            info = born.get((e["m_unitTagIndex"], e["m_unitTagRecycle"]))
            if info is None:
                continue
            typ, ctrl = info
            t = P.t_of(e["_gameloop"])
            k = e["m_killerPlayerId"] or 0
            if typ == "ExperienceGlobeMinion":
                globes.append([t, e["m_x"], e["m_y"], k])
            elif typ.endswith("Minion") and ctrl in (11, 12):
                minions.append([t, e["m_x"], e["m_y"], k, ctrl - 11])
            elif typ.startswith("Merc") and ctrl not in (11, 12) and "Laner" not in typ \
                    and "Dummy" not in typ:
                mercu.append([t, e["m_x"], e["m_y"], k])
        elif ev.endswith("SUnitRevivedEvent"):
            pid = hero_tag.get(e["m_unitTagIndex"])
            if pid is not None:
                revives.append([P.t_of(e["_gameloop"]), pid, e["m_x"], e["m_y"]])

    # ---- game events: action rates (as parse_replay), ability mix, per-second activity, camera, targets
    counts = defaultdict(Counter)
    abil = defaultdict(Counter)
    sec = {pid: bytearray(nsec) for pid in team}
    cam = {pid: [-1] * (2 * nb) for pid in team}
    tgt = {pid: [-1] * (2 * nb) for pid in team}
    abins = {pid: [0] * na for pid in team}
    gl = g0 * P.LOOPS_PER_S
    for e in proto.decode_replay_game_events(arc.read_file("replay.game.events")):
        evn = e["_event"]
        kind = P.ACTION_EVENTS.get(evn)
        if kind is None:
            continue
        pid = user_pid.get(e["_userid"]["m_userId"])
        if pid is None or e["_gameloop"] < gl:
            continue
        counts[pid][kind] += 1
        tt = (e["_gameloop"] - gl) / P.LOOPS_PER_S
        si = min(int(tt), nsec - 1)
        bi = min(int(tt / BIN), nb - 1)
        if kind in ("cmd", "cmd_retarget", "key"):
            if sec[pid][si] < 255:
                sec[pid][si] += 1
        if evn == "NNet.Game.SCmdEvent":
            a = e.get("m_abil")
            if a is None:
                abil[pid]["-"] += 1
                d = e.get("m_data") or {}
                tp = d.get("TargetPoint")
                if tp:
                    tgt[pid][2 * bi] = int(round(tp["x"] / 2048.0))
                    tgt[pid][2 * bi + 1] = int(round(tp["y"] / 2048.0))
            else:
                abil[pid][f"{a['m_abilLink']}:{a['m_abilCmdIndex']}"] += 1
                abins[pid][min(int(tt / ABIN), na - 1)] += 1
        elif evn == "NNet.Game.SCameraUpdateEvent":
            tg = e.get("m_target")
            if tg:
                cam[pid][2 * bi] = int(round(tg["x"] / 128.0))
                cam[pid][2 * bi + 1] = int(round(tg["y"] / 128.0))
    minutes = max(1e-6, L / 60.0)
    tl["actions_per_min"] = {str(pid): {k: round(n / minutes, 2) for k, n in sorted(c.items())}
                             for pid, c in sorted(counts.items())}
    # carry the camera forward through bins without an update (camera stays where it was)
    for pid in team:
        c = cam[pid]
        for b in range(1, nb):
            if c[2 * b] < 0 and c[2 * b - 2] >= 0:
                c[2 * b], c[2 * b + 1] = c[2 * b - 2], c[2 * b - 1]
    tl["pbp"] = {
        "schema": 1, "bin_s": BIN, "abin_s": ABIN,
        "players": {str(pid): {"abil": dict(abil[pid]), "sec": sec[pid].decode("latin-1"),
                               "cam": cam[pid], "tgt": tgt[pid], "abil_bins": abins[pid]} for pid in team},
        "minions": minions, "globes": globes, "merc_units": mercu, "revives": revives,
    }
    return tl


def job(args):
    root, rel, rid, outdir = args
    build = rel.split("/")[-2]
    out = os.path.join(root, outdir, build, f"{rid}.json.gz")
    if os.path.exists(out):
        return {"replay_id": rid, "status": "exists"}
    t0 = time.time()
    try:
        src = os.path.join(root, rel)
        tl = P.parse(src, rid, with_actions=False)
        tl = extras(src, tl)
        P.write(tl, out)
        return {"replay_id": rid, "status": "ok", "secs": round(time.time() - t0, 2)}
    except P.SealedError as e:
        return {"replay_id": rid, "status": "refused_sealed", "error": str(e)}
    except Exception as e:  # noqa: BLE001
        return {"replay_id": rid, "status": "error", "error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc()[-800:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/max/hots-replays")
    ap.add_argument("--list", required=True)
    ap.add_argument("--out", default="pbp")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    sealed = P.sealed_ids()
    assert len(sealed) >= 5000, "sealed id list missing"
    items = []
    for line in open(a.list):
        rel = line.strip()
        if rel:
            rid = int(os.path.basename(rel).split(".")[0])
            if rid not in sealed and not rel.startswith("sealed/") and "/sealed/" not in rel:
                items.append((rel, rid))
    if a.limit:
        items = items[:a.limit]
    os.makedirs(os.path.join(a.root, a.out), exist_ok=True)
    index = open(os.path.join(a.root, a.out, "index.jsonl"), "a")
    t0 = time.time()
    cnt = Counter()
    with Pool(a.workers, maxtasksperchild=100) as pool:
        for i, res in enumerate(pool.imap_unordered(job, [(a.root, rel, rid, a.out) for rel, rid in items],
                                                    chunksize=4)):
            cnt[res["status"]] += 1
            if res["status"] != "exists":
                index.write(json.dumps(res) + "\n")
            if (i + 1) % 500 == 0:
                index.flush()
                print(f"{i + 1}/{len(items)} {dict(cnt)} {time.time() - t0:.0f}s", flush=True)
    index.close()
    print(f"done {len(items)} in {time.time() - t0:.0f}s: {dict(cnt)}", flush=True)


if __name__ == "__main__":
    main()
