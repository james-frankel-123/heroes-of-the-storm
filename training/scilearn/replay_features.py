"""Replay feature store and the M tranche (FRAMING 4.1 row M, 9.1, 11.12). Harness side only.

  python -m training.scilearn.replay_features build      one row per (game, team) from timelines
  python -m training.scilearn.replay_features status     counts by cohort and M side, no outcomes

M membership is decided by the frozen rules, never by outcomes:
  - 2.57 games only; sealed IDs (R) are skipped before any file is opened;
  - cohort B (stratified_b) games outside the interesting pool are M, split by keyed hash
    into M-explore (30%) and M-test (70%) (tranches.m_side);
  - interesting-cohort games (surprise_v1) come from the 10% pool and go to M-explore only
    (FRAMING 9.4 rule 5): they are read, never used for an estimate;
  - everything else (C, F, audit) is never opened here.

The snapshot of 2026-10-13 was cut when cohort B had no downloads, so it lists today's B
games under C. The tranche rule puts them in M (tranches.assign checks cohort B before C),
and the C batch runner must recompute membership at unlock (CYCLE2_PREP.md).

Mediators are early-game quantities, from team 0's or each team's perspective, with fixed
windows. End-of-game totals and game length are never mediators (FRAMING 9.2).
"""
import argparse
import datetime as dt
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

from training.scilearn import config as C
from training.scilearn import tranches as T

VERSION = "rf1"
STORE = C.PRIVATE / "replay_features"
INTERESTING = {"surprise_v1"}
COHORT_CODES = {"stratified_b": 0, "surprise_v1": 1}
SIDES = {"M-explore": 0, "M-test": 1}
ISO_WINDOW_S, ISO_DIST = 15.0, 30.0       # isolated death: no other death this close in time and space
FIGHT_GAP_S = 15.0                         # deaths chained within 15 s form one fight

# Team-perspective mediators. kind: "lead" = own minus opponent (antisymmetric),
# "count" / "share" / "flag" = the team's own value. Windows in seconds of game time.
MEDIATORS = {
    "level_lead_5": ("lead", "team level lead at 5:00"),
    "level_lead_7": ("lead", "team level lead at 7:00"),
    "level_lead_10": ("lead", "team level lead at 10:00"),
    "level_lead_15": ("lead", "team level lead at 15:00"),
    "first_to_10": ("flag", "1 if the team reached level 10 first"),
    "xp_lead_10": ("lead", "total XP lead at 10:00 (thousands)"),
    "minion_xp_share_10": ("share", "minion XP share of the team's XP at 10:00"),
    "creep_xp_share_10": ("share", "merc-camp (creep) XP share of the team's XP at 10:00"),
    "hero_xp_share_10": ("share", "hero-kill XP share of the team's XP at 10:00"),
    "deaths_10": ("count", "team deaths before 10:00"),
    "isolated_deaths_10": ("count", "team deaths before 10:00 with no other death within 15 s and 30 units"),
    "first_blood": ("flag", "1 if the team got first blood"),
    "fights_won_10": ("count", "fights (2+ deaths chained within 15 s) won before 10:00"),
    "fights_won_post10": ("count", "fights won from 10:00 to 20:00"),
    "towers_lost_15": ("count", "towers lost by 15:00"),
    "forts_lost_15": ("count", "forts and keeps lost by 15:00"),
    "first_fort_lost": ("flag", "1 if the team lost the first fort or keep of the game"),
    "mercs_10": ("count", "merc camps captured before 10:00"),
}
MED_NAMES = list(MEDIATORS)


def code_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


# --------------------------------------------------------------------------- membership

def m_members(manifest_path=None):
    """{replay_id: (cohort, side, manifest row)} for M games. Reads no outcomes."""
    keys = T.load_keys()
    sealed = {int(x) for x in open(C.SEALED_IDS_FILE).read().split()}
    rows = {}
    for line in open(manifest_path or C.REPLAY_MANIFEST):
        r = json.loads(line)
        if r.get("status") not in ("ok", "linked") or r.get("cohort") not in COHORT_CODES:
            continue
        rid = int(r["replay_id"])
        if rid in sealed or str(r.get("path", "")).startswith("sealed/"):
            continue
        if not str(r.get("game_version", "")).startswith("2.57."):
            continue
        in_pool = T.khash(keys["pool"], rid) < C.POOL_FRAC
        if r["cohort"] in INTERESTING:
            if not in_pool:       # interesting cohorts draw only from the pool; anything else is a bug
                continue
            side = "M-explore"
        else:
            if in_pool:
                continue          # a pool game is never M through cohort B
            side = T.m_side(rid, keys)
        prev = rows.get(rid)
        # a game in both B and an interesting cohort: B's own split decides, unless it is a pool game
        if prev is None or (prev[0] in INTERESTING and r["cohort"] not in INTERESTING):
            rows[rid] = (r["cohort"], side, r)
    return rows


# --------------------------------------------------------------------------- per-game features

def _level_at(levels, t):
    lv = 1
    for L, tt in levels:
        if tt <= t:
            lv = max(lv, int(L))
    return lv


def _time_to(levels, L):
    for l, tt in levels:
        if int(l) >= L:
            return float(tt)
    return np.nan


def _xp_at(xp_periodic, team, t):
    best = None
    for e in xp_periodic:
        if int(e["team"]) == team and e["t"] <= t + 1e-6:
            if best is None or e["t"] >= best["t"]:
                best = e
    if best is None:
        return {"minion": 0.0, "creep": 0.0, "structure": 0.0, "hero": 0.0, "trickle": 0.0}
    return best


def game_features(d):
    """Mediators (2, K): row = team 0, team 1; team-perspective values."""
    team_of = {int(p["pid"]): int(p["team"]) for p in d["players"]}
    tl = {int(k): v for k, v in d["team_levels"].items()}
    out = np.full((2, len(MED_NAMES)), np.nan)
    col = {m: i for i, m in enumerate(MED_NAMES)}

    def setv(name, v0, v1):
        out[0, col[name]], out[1, col[name]] = v0, v1

    for t, name in ((300, "level_lead_5"), (420, "level_lead_7"), (600, "level_lead_10"), (900, "level_lead_15")):
        if d["length_s"] >= t:
            l0, l1 = _level_at(tl.get(0, []), t), _level_at(tl.get(1, []), t)
            setv(name, l0 - l1, l1 - l0)
    t0, t1 = _time_to(tl.get(0, []), 10), _time_to(tl.get(1, []), 10)
    if not (np.isnan(t0) and np.isnan(t1)):
        f0 = 1.0 if (np.isnan(t1) or (not np.isnan(t0) and t0 < t1)) else 0.0
        setv("first_to_10", f0, 1.0 - f0)
    if d["length_s"] >= 600:
        x = [_xp_at(d["xp_periodic"], k, 600) for k in (0, 1)]
        tot = [sum(float(x[k][s]) for s in ("minion", "creep", "structure", "hero", "trickle")) for k in (0, 1)]
        setv("xp_lead_10", (tot[0] - tot[1]) / 1000, (tot[1] - tot[0]) / 1000)
        for src, name in (("minion", "minion_xp_share_10"), ("creep", "creep_xp_share_10"), ("hero", "hero_xp_share_10")):
            setv(name, *(float(x[k][src]) / tot[k] if tot[k] > 0 else np.nan for k in (0, 1)))
    deaths = sorted(d["deaths"], key=lambda e: e["t"])
    vt = [team_of.get(int(e["victim"])) for e in deaths]
    if d["length_s"] >= 600:
        setv("deaths_10", *(sum(1 for e, v in zip(deaths, vt) if v == k and e["t"] < 600) for k in (0, 1)))
        iso = [0, 0]
        for i, (e, v) in enumerate(zip(deaths, vt)):
            if e["t"] >= 600 or v is None:
                continue
            lonely = True
            for j, f in enumerate(deaths):
                if j != i and abs(f["t"] - e["t"]) <= ISO_WINDOW_S and \
                        math_dist(e, f) <= ISO_DIST:
                    lonely = False
                    break
            iso[v] += lonely
        setv("isolated_deaths_10", iso[0], iso[1])
    if deaths and vt[0] is not None:
        fb = 1 - vt[0]
        setv("first_blood", 1.0 if fb == 0 else 0.0, 1.0 if fb == 1 else 0.0)
    # fights: chains of deaths with gaps <= 15 s; winner = the side with fewer deaths
    fights, cur = [], []
    for e, v in zip(deaths, vt):
        if v is None:
            continue
        if cur and e["t"] - cur[-1][0] > FIGHT_GAP_S:
            fights.append(cur)
            cur = []
        cur.append((e["t"], v))
    if cur:
        fights.append(cur)
    won = {"pre": [0, 0], "post": [0, 0]}
    for f in fights:
        if len(f) < 2:
            continue
        n0 = sum(1 for _, v in f if v == 0)
        n1 = len(f) - n0
        if n0 == n1:
            continue
        w = 0 if n0 < n1 else 1
        start = f[0][0]
        if start < 600:
            won["pre"][w] += 1
        elif start < 1200:
            won["post"][w] += 1
    if d["length_s"] >= 600:
        setv("fights_won_10", *won["pre"])
    if d["length_s"] >= 1200:
        setv("fights_won_post10", *won["post"])
    if d["length_s"] >= 900:
        tw, fk = [0, 0], [0, 0]
        for s in d["structures"]:
            if s.get("ev") != "died" or s["t"] > 900 or s.get("team") not in (0, 1):
                continue
            ty = str(s.get("type", ""))
            if ty.startswith("TownCannonTower"):
                tw[int(s["team"])] += 1
            elif ty.startswith("TownTownHall"):
                fk[int(s["team"])] += 1
        setv("towers_lost_15", *tw)
        setv("forts_lost_15", *fk)
    firsts = [s for s in sorted(d["structures"], key=lambda s: s["t"])
              if s.get("ev") == "died" and str(s.get("type", "")).startswith("TownTownHall") and s.get("team") in (0, 1)]
    if firsts:
        lost = int(firsts[0]["team"])
        setv("first_fort_lost", 1.0 if lost == 0 else 0.0, 1.0 if lost == 1 else 0.0)
    if d["length_s"] >= 600:
        setv("mercs_10", *(sum(1 for m in d["mercs"] if int(m.get("team", -1)) == k and m["t"] < 600) for k in (0, 1)))
    return out


def math_dist(e, f):
    try:
        return float(np.hypot(float(e["x"]) - float(f["x"]), float(e["y"]) - float(f["y"])))
    except (KeyError, TypeError, ValueError):
        return 0.0


# --------------------------------------------------------------------------- build

def build(snap_id=None, out_dir=None):
    from training.scilearn import data as D
    _, meta, _, _ = D.load(snap_id, ["Dpre"])
    hidx = {h: i for i, h in enumerate(meta["heroes"])}
    midx = {m: i for i, m in enumerate(meta["maps"])}
    bidx = {b: i for i, b in enumerate(meta["builds"])}
    mem = m_members()
    rows, skipped = [], {"no_timeline": 0, "bad_heroes": 0, "unknown_map": 0, "unreadable": 0}
    for rid in sorted(mem):
        cohort, side, r = mem[rid]
        p = timeline_path(rid, r)
        if not p.exists():
            skipped["no_timeline"] += 1
            continue
        try:
            d = json.load(gzip.open(p))
        except Exception:
            skipped["unreadable"] += 1
            continue
        heroes = [[], []]
        for pl in sorted(d["players"], key=lambda x: int(x["pid"])):
            heroes[int(pl["team"])].append(hidx.get(pl["hero"], -1))
        if any(len(t) != 5 for t in heroes) or min(heroes[0] + heroes[1]) < 0:
            skipped["bad_heroes"] += 1
            continue
        if r["game_map"] not in midx:
            skipped["unknown_map"] += 1
            continue
        ts = int(dt.datetime.fromisoformat(str(r["game_date"]).replace("Z", "+00:00")).replace(
            tzinfo=dt.timezone.utc).timestamp()) if r.get("game_date") else 0
        rows.append({"replay_id": rid, "cohort": COHORT_CODES[cohort], "side": SIDES[side],
                     "map": midx[r["game_map"]], "tier": meta["tiers"].index(r["skill_tier"]) if r.get("skill_tier") in meta["tiers"] else 3,
                     "build": bidx.get(r["game_version"], len(meta["builds"]) - 1), "game_ts": ts,
                     "heroes": heroes[0] + heroes[1], "y0": 1 if int(d["winner_team"]) == 0 else 0,
                     "med": game_features(d), "path": str(p.relative_to(C.REPLAYS_ROOT))})
    n = len(rows)
    arr = {
        "replay_id": np.array([r["replay_id"] for r in rows], np.int64),
        "cohort": np.array([r["cohort"] for r in rows], np.int8),
        "side": np.array([r["side"] for r in rows], np.int8),
        "map": np.array([r["map"] for r in rows], np.int64),
        "tier": np.array([r["tier"] for r in rows], np.int64),
        "build": np.array([r["build"] for r in rows], np.int64),
        "game_ts": np.array([r["game_ts"] for r in rows], np.int64),
        "heroes": np.array([r["heroes"] for r in rows], np.int64).reshape(n, 10),
        "med": np.array([r["med"] for r in rows], float).reshape(n, 2, len(MED_NAMES)),
    }
    labels = {"replay_id": arr["replay_id"], "y0": np.array([r["y0"] for r in rows], np.int8)}
    out = Path(out_dir or STORE / VERSION)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "features.npz", **arr)
    np.savez(out / "labels.npz", **labels)
    (out / "paths.json").write_text(json.dumps({str(r["replay_id"]): r["path"] for r in rows}))
    # Mediator-to-win correlation on cohort-B M-explore games only (the surprise cohort was
    # selected with outcomes). Q3 charges 0.16 rho^2 bits per released number; 0.5 until
    # at least 200 such games exist (FRAMING 5.2 default).
    rho = {}
    ex = (arr["side"] == SIDES["M-explore"]) & (arr["cohort"] == COHORT_CODES["stratified_b"])
    for j, m in enumerate(MED_NAMES):
        x = arr["med"][ex, 0, j]
        yy = labels["y0"][ex].astype(float)
        ok = ~np.isnan(x)
        rho[m] = float(abs(np.corrcoef(x[ok], yy[ok])[0, 1])) if ok.sum() >= 200 and x[ok].std() > 0 else 0.5
    man = {"version": VERSION, "code_hash": code_hash(), "built": dt.datetime.now(dt.timezone.utc).isoformat(),
           "snapshot_vocab": meta["snap_id"], "n_games": n,
           "by_side": {s: int((arr["side"] == c).sum()) for s, c in SIDES.items()},
           "by_cohort": {k: int((arr["cohort"] == c).sum()) for k, c in COHORT_CODES.items()},
           "m_test_games": int((arr["side"] == SIDES["M-test"]).sum()),
           "skipped": skipped, "members": len(mem), "mediators": MEDIATORS, "rho": rho,
           "rules": {"isolated": [ISO_WINDOW_S, ISO_DIST], "fight_gap_s": FIGHT_GAP_S}}
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    for f in out.iterdir():
        os.chmod(f, 0o600)
    return man


def load(version=VERSION):
    d = STORE / version
    f = dict(np.load(d / "features.npz"))
    lab = np.load(d / "labels.npz")
    assert np.array_equal(lab["replay_id"], f["replay_id"])
    man = json.loads((d / "manifest.json").read_text())
    return f, lab["y0"].astype(np.int8), man


def shuffle_index(map_, tier, build, key):
    """Permutation of whole M records within (map, tier, build) blocks (shuffled arm):
    record i's draft is paired with record perm[i]'s trajectory and outcome."""
    seed = int(hashlib.sha256(f"{key}:shuffle:M".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    blk = (map_.astype(np.int64) * 10 + tier) * 1000 + build
    perm = np.arange(len(blk))
    order = np.argsort(blk, kind="stable")
    sb = blk[order]
    starts = np.flatnonzero(np.r_[True, sb[1:] != sb[:-1]])
    ends = np.r_[starts[1:], len(sb)]
    for s, e in zip(starts, ends):
        idx = order[s:e]
        perm[idx] = idx[rng.permutation(len(idx))]
    return perm


def timeline_path(rid, r):
    return C.TIMELINES / str(r.get("build") or str(r["game_version"]).split(".")[-1]) / f"{rid}.json.gz"


def status():
    mem = m_members()
    have = {}
    for rid, (cohort, side, r) in mem.items():
        p = timeline_path(rid, r)
        k = (cohort, side)
        have.setdefault(k, [0, 0])
        have[k][0] += 1
        have[k][1] += p.exists()
    return {f"{c} {s}": {"members": v[0], "parsed": v[1]} for (c, s), v in sorted(have.items())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "status"])
    a = ap.parse_args()
    print(json.dumps(build() if a.cmd == "build" else status(), indent=1))
