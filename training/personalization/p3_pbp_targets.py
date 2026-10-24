"""
P3 play-by-play study (results/P3_PLAY_BY_PLAY.md): prediction targets and the
non-replay baseline inputs.

Stages (from training/, system python3, nice 19, cores 48-63):
  targets   every 2.57 Storm League game in the local research DB with ten
            replay_players rows, sealed holdout ids removed by id (their rows
            are never selected). Per slot: identity, hero, team, win, party,
            map, length, the DB scoreboard (model metrics and extras) and the
            talent names by tier (linuxbrew python3, psycopg2).
            Stage wp: population draft WP per game (the
            d2c_cumprev ensemble of p3_dr_core.WPEval, latest stats it has).
            -> cache/pbp/targets.npz
  wp        adds the draft WP column (system python3 with torch).
  fullhist  the scoreboard direct channel's inputs as of 2026-09-29 from the
            full_1017 history store (cache/hr/aug.npz, a_shrink z): per
            player x hero and per player sums and counts of the 15 metric z.
            -> cache/pbp/fullhist.npz
Both files hold player keys and stay in the gitignored cache.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

CACHE = os.path.join(HERE, "cache", "pbp")
TARGETS = os.path.join(CACHE, "targets.npz")
FULLHIST = os.path.join(CACHE, "fullhist.npz")
SEALED = os.path.join(HERE, "..", "replays", "cohorts", "sealed_ids.txt")

SB15 = ["hero_damage", "siege_damage", "structure_damage", "healing", "self_healing", "protection_allies",
        "damage_taken", "deaths", "kills", "assists", "experience_contribution", "merc_camp_captures",
        "outnumbered_deaths", "escapes", "regen_globes"]
SBX = ["stunning_enemies", "rooting_enemies", "silencing_enemies", "teamfight_hero_damage",
       "teamfight_damage_taken", "teamfight_healing", "minion_damage", "creep_damage", "time_spent_dead",
       "physical_damage", "spell_damage", "summon_damage", "takedowns", "highest_kill_streak", "town_kills",
       "vengeance", "teamfight_escapes", "clutch_heals", "watch_tower_captures", "multikill"]
TIERS = ["1", "4", "7", "10", "13", "16", "20"]


def db():
    import psycopg2
    env = {}
    for line in open(os.path.join(HERE, "..", "..", ".env")):
        if "=" in line and not line.startswith("#"):
            k, v = line.strip().split("=", 1)
            env[k] = v.strip().strip('"').strip("'")
    url = env["DATABASE_URL_RESEARCH"]
    assert "neon" not in url.lower()
    c = psycopg2.connect(url)
    c.set_session(readonly=True)
    return c


def sealed_ids():
    s = set(int(x) for x in open(SEALED) if x.strip())
    assert len(s) >= 5000
    return s


def targets():
    t0 = time.time()
    sealed = sealed_ids()
    cur = db().cursor()
    cur.execute("""select replay_id from replay_draft_data where game_version like '2.57%%'""")
    ids = [r[0] for r in cur.fetchall()]
    keep_ids = sorted(i for i in ids if i not in sealed)
    n_sealed = len(ids) - len(keep_ids)
    rows = []
    for k in range(0, len(keep_ids), 5000):
        chunk = keep_ids[k:k + 5000]
        cur.execute("""select r.replay_id, extract(epoch from r.game_date), r.game_map, r.region, r.game_length,
                              r.skill_tier, r.league_tier, r.game_version,
                              p.region, p.blizz_id, p.hero, p.team, p.winner, p.party, p.scoreboard, p.talents
                       from replay_draft_data r join replay_players p using (replay_id)
                       where r.replay_id = any(%s)""", (chunk,))
        rows.extend(cur.fetchall())
    rid = np.array([r[0] for r in rows], np.int64)
    assert not np.isin(rid, np.fromiter(sealed, np.int64)).any()
    u, inv = np.unique(rid, return_inverse=True)
    full = np.bincount(inv)[inv] == 10
    sel = np.flatnonzero(full)
    R = [rows[i] for i in sel]
    from p3_keys import player_keys
    out = {
        "replay_id": rid[sel], "unix": np.array([float(r[1]) for r in R]),
        "map": np.array([r[2] or "" for r in R]), "game_region": np.array([r[3] or 0 for r in R], np.int32),
        "minutes": np.array([max((r[4] or 0) / 60.0, 1.0) for r in R]),
        "skill_tier": np.array([r[5] or "" for r in R]), "league_tier": np.array([-1 if r[6] is None else r[6] for r in R], np.int32),
        "version": np.array([r[7] or "" for r in R]),
        "region": np.array([r[8] or 0 for r in R], np.int32), "toon_id": np.array([r[9] for r in R], np.int64),
        "hero": np.array([r[10] or "" for r in R]), "team": np.array([r[11] for r in R], np.int32),
        "win": np.array([bool(r[12]) for r in R], np.int32),
        "party": np.array([0 if r[13] is None else int(r[13]) for r in R], np.int64),
    }
    out["pkey"] = player_keys(out["region"], out["toon_id"], out["game_region"], "pbp targets")
    for f in SB15 + SBX:
        out["sb_" + f] = np.array([np.nan if (r[14] or {}).get(f) is None else float(r[14][f]) for r in R])
    for t in TIERS:
        out["tal_" + t] = np.array([str((r[15] or {}).get(t) or "") for r in R])
    out["n_sealed_excluded_games"] = np.int64(n_sealed)
    # games with ten rows sorted by game then team for the WP step
    o = np.lexsort((out["team"], out["replay_id"]))
    out = {k: (v[o] if isinstance(v, np.ndarray) and v.ndim and len(v) == len(o) else v) for k, v in out.items()}
    print(f"targets: {len(out['win']):,} slots, {len(np.unique(out['replay_id'])):,} games, "
          f"sealed excluded {n_sealed}; {time.time() - t0:.0f}s", flush=True)
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(TARGETS, **out)
    print(f"wrote {TARGETS}; {time.time() - t0:.0f}s", flush=True)


def draft_wp(t):
    """Population WP (team 0 wins) per slot (same value for the game's ten slots); NaN if a hero
    or the map is unknown to the WP model."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    torch.set_num_threads(2)
    import p3_dr_core as D
    from shared import MAPS
    s = np.load(os.path.join(HERE, "cache", "hr", "state_0928.npz"), allow_pickle=True)
    hn = [str(h) for h in s["hero_names"]]
    hix = {h: i for i, h in enumerate(hn)}
    W = D.WPEval(hn)
    ug, first = np.unique(t["replay_id"], return_index=True)
    wp = np.full(len(ug), np.nan)
    tiers = {"low": "low", "mid": "mid", "high": "high"}
    groups = {}
    miss_h, miss_m = set(), 0
    for gi, f in enumerate(first):
        sl = slice(f, f + 10)
        tm = t["team"][sl]
        hs = t["hero"][sl]
        mp = str(t["map"][f])
        tier = tiers.get(str(t["skill_tier"][f]).lower(), "mid")
        if mp not in MAPS:
            miss_m += 1
            continue
        if any(h not in hix for h in hs):
            miss_h |= {h for h in hs if h not in hix}
            continue
        a = tuple(hix[h] for h, x in zip(hs, tm) if x == 0)
        b = tuple(hix[h] for h, x in zip(hs, tm) if x == 1)
        if len(a) != 5 or len(b) != 5:
            continue
        groups.setdefault((mp, tier), []).append((gi, a, b))
    for (mp, tier), L in groups.items():
        for k in range(0, len(L), 2000):
            part = L[k:k + 2000]
            p = W.wp([(a, b) for _, a, b in part], 10 ** 6, mp, tier)
            wp[[gi for gi, _, _ in part]] = p
    print(f"draft WP: {np.isfinite(wp).mean():.3f} of games; unknown heroes {sorted(miss_h)}; "
          f"unknown maps {miss_m}", flush=True)
    gi = np.searchsorted(ug, t["replay_id"])
    return wp[gi]


def add_wp():
    """Separate stage: the WP ensemble needs torch (system python3), the DB pull psycopg2."""
    t = dict(np.load(TARGETS))
    t["wp_pop"] = draft_wp(t)
    np.savez_compressed(TARGETS, **t)
    print(f"wrote wp_pop into {TARGETS}", flush=True)


def fullhist():
    import p3_hs_core as C
    from p3_keys import key_blizz, key_region
    t0 = time.time()
    z = np.load(os.path.join(HERE, "cache", "hr", "aug.npz"))
    day = z["day"]
    keep = day < C.day_of("2026-09-29")
    pid = z["pid"][keep].astype(np.int64)
    hero = z["hero"][keep].astype(np.int64)
    Z = z["Z"][keep].astype(np.float32)
    keys = z["player_keys"]
    metrics = [str(m) for m in z["metrics"]]
    print(f"fullhist: {keep.sum():,} rows; metrics {metrics}; {time.time() - t0:.0f}s", flush=True)
    cell = pid * 90 + hero
    uc, ci = np.unique(cell, return_inverse=True)
    ok = np.isfinite(Z)
    S = np.zeros((len(uc), Z.shape[1]), np.float32)
    N = np.zeros((len(uc), Z.shape[1]), np.float32)
    for m in range(Z.shape[1]):
        S[:, m] = np.bincount(ci, weights=np.where(ok[:, m], Z[:, m], 0), minlength=len(uc))
        N[:, m] = np.bincount(ci, weights=ok[:, m], minlength=len(uc))
    G = np.bincount(ci, minlength=len(uc)).astype(np.float32)
    cp = uc // 90
    out = dict(cell_key=keys[cp].astype(np.int64), cell_hero=(uc % 90).astype(np.int8), S=S, N=N, G=G,
               metrics=np.array(metrics), hero_names=z["hero_names"],
               cell_region=key_region(keys[cp]), cell_blizz=key_blizz(keys[cp]).astype(np.int64))
    np.savez(FULLHIST, **out)
    print(f"wrote {FULLHIST}: {len(uc):,} player x hero cells; {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    {"targets": targets, "wp": add_wp, "fullhist": fullhist}[sys.argv[1]]()
