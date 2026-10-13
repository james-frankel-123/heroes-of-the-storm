"""Build an "interesting" replay cohort by two-phase surprise sampling (SELECTION_RULES.md §C).

  python training/replays/build_interesting.py --model-dir <A0 job out dir> --seed <int> \
      --name surprise_v1 [--dry-run]

Rule (FRAMING §9.4):
  1. Population: the scilearn interesting pool (keyed-hash 10% of non-sealed 2.57 games),
     read from the tranche manager's snapshot pool file. Never C, F or R.
  2. Prior: frozen NN-free A0 (training/scilearn), p0 = P(team 0 wins) from the draft.
  3. Trajectory surprise when the HP XP curve has a minute-10 sample and the game ran at
     least 12 minutes: s = -logit(p0) * r10, r10 = team-0 XP lead at minute 10 divided by
     the mean team XP then. Otherwise outcome surprise: s = -log P_A0(actual winner).
  4. Top quintile per surprise type; pi = 0.2 + 0.4 * top. Each game is drawn independently
     with u = sha256("<seed>:<replay_id>") as a uniform; selected when u < pi.
Outputs (oldest replayID first): <name>.txt (ids file for the downloader) and
<name>_design.tsv (every pool game: pi, surprise, type, quintile flag, model version, seed,
selected). Read-only on the research DB; sealed IDs are excluded in SQL and never read.
Imports training/scilearn read-only.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import psycopg2

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from training.scilearn import config as C  # noqa: E402
from training.scilearn import fit as F  # noqa: E402
from training.scilearn import terms as TM  # noqa: E402
from training.scilearn import tranches as T  # noqa: E402
from training.scilearn.snapshot import research_dsn  # noqa: E402

OUT_DIR = REPO / "training" / "replays" / "cohorts" / "interesting"
MIN_LEN_TRAJ = 720          # DB game_length seconds; shorter games use outcome surprise
XP_MINUTE = 10              # HP x_axis label of the periodic sample (replay time 10:38)
PI_BASE, PI_TOP, TOP_Q = 0.2, 0.4, 0.8


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def draw_u(seed, rid):
    return int.from_bytes(hashlib.sha256(f"{seed}:{rid}".encode()).digest()[:8], "big") / 2**64


def load_pool(snap):
    d = C.SNAPSHOTS / snap
    meta = json.loads((d / "manifest.json").read_text())
    pool = dict(np.load(d / "pool.npz"))
    assert (pool["tranche"] == T.CODES["pool"]).all()
    return d, meta, pool


def verify_pool(cur, pool_ids, sealed, keys, snap_as_of):
    """Recompute membership from the DB with the tranche manager's own assign()."""
    cur.execute("""SELECT replay_id, game_version, extract(epoch from game_date)::bigint
                   FROM replay_draft_data WHERE game_version LIKE '2.57.%%' AND game_date < %s
                   AND NOT (replay_id = ANY(%s))""", (snap_as_of, list(sealed)))
    ids = {int(r) for r, v, ts in cur.fetchall()
           if T.assign(int(r), v, int(ts), sealed=sealed, cohort_b=set(), keys=keys) == "pool"}
    if ids != set(pool_ids.tolist()):
        raise SystemExit(f"pool mismatch: snapshot {len(pool_ids)} vs DB {len(ids)}")


def pool_players(cur, pool, meta):
    """Player keys for pool games, same slot and privacy rules as the snapshot."""
    hidx = {h: i for i, h in enumerate(meta["heroes"])}
    private = set()
    cur.execute("SELECT battletag, region FROM player_privacy WHERE state = 'private'")
    for bt, reg in cur.fetchall():
        private.add((bt, int(reg)))
    pos = {int(r): i for i, r in enumerate(pool["replay_id"])}
    n = len(pos)
    pk = np.full((n, 10), -1, np.int64)
    party = np.zeros((n, 10), np.int64)
    cur.execute("""SELECT replay_id, blizz_id, battletag, coalesce(region, -1), hero, team, coalesce(party, 0)
                   FROM replay_players WHERE replay_id = ANY(%s)""", (list(pos),))
    for rid, blz, bt, reg, hero, team, prt in cur.fetchall():
        i = pos[int(rid)]
        if hero not in hidx:
            continue
        base = 0 if int(team) == 0 else 5
        hit = np.nonzero(pool["heroes"][i, base:base + 5] == hidx[hero])[0]
        if len(hit) == 0:
            continue
        s = base + int(hit[0])
        pk[i, s] = -1 if (bt, int(reg)) in private else (int(blz) if blz else -1)
        party[i, s] = int(prt)
    return pk, party


def a0_logit(model_dir, snap_dir, meta, pool, pk, party):
    """A0 logit for pool games. Player-history features come from snapshot rows (draft-time
    columns only, no labels) that share a (player, hero) key with a pool slot."""
    g = dict(np.load(snap_dir / "games.npz"))
    pl = np.load(snap_dir / "players.npz")
    assert np.array_equal(pl["replay_id"], g["replay_id"])
    want = np.unique((pk * 128 + pool["heroes"])[pk >= 0])
    hk = np.where(pl["player_key"] >= 0, pl["player_key"] * 128 + g["heroes"], -1)
    keep = np.isin(hk, want).any(1)
    cols = ("replay_id", "game_ts", "first_seen_ts", "build", "map", "tier", "heroes")
    games = {k: np.concatenate([g[k][keep], pool[k]]) for k in cols}
    players = {"player_key": np.concatenate([pl["player_key"][keep], pk]),
               "party": np.concatenate([pl["party"][keep], party])}
    root = TM.Ctx(games, meta, players=players)
    idx = np.arange(int(keep.sum()), root.n)
    model = F.Model.load(model_dir / "A0", meta)
    eta = model.logit(root.subset(idx))
    return np.asarray(eta, float), model, int(keep.sum())


def xp_minute10(cur, ids):
    """Team-0 relative XP lead at the HP minute-10 sample, the DB game length and winner.
    HP team 0 is checked against team0_heroes through replay_players."""
    cur.execute("""SELECT d.replay_id, d.game_length, d.winner, e.extras->'experience_breakdown'
                   FROM replay_draft_data d LEFT JOIN replay_extras e USING (replay_id)
                   WHERE d.replay_id = ANY(%s)""", (list(map(int, ids)),))
    out = {}
    for rid, gl, win, xb in cur.fetchall():
        r10 = None
        try:
            teams = {int(t["team"]): t["data"] for t in xb["data"]}
            v = []
            for tm in (0, 1):
                rows = teams[tm]
                if len(rows) <= XP_MINUTE:          # sample 10 must not be the final one
                    raise KeyError
                row = rows[XP_MINUTE - 1]
                if not row["TimeSpan"][11:19].startswith(f"00:{XP_MINUTE:02d}:"):
                    raise KeyError
                v.append(float(row["TotalXP"]))
            r10 = (v[0] - v[1]) / max((v[0] + v[1]) / 2, 1.0)
        except (KeyError, TypeError, ValueError, IndexError):
            r10 = None
        out[int(rid)] = (r10, int(gl or 0), None if win is None else int(win))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--snap", default="snap_20261013")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--name", default="surprise_v1")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    model_dir = Path(a.model_dir)

    snap_dir, meta, pool = load_pool(a.snap)
    sealed = {int(x) for x in open(C.SEALED_IDS_FILE).read().split()}
    assert not np.isin(pool["replay_id"], list(sealed)).any()
    keys = T.load_keys()
    conn = psycopg2.connect(research_dsn())
    conn.set_session(readonly=True)
    cur = conn.cursor()
    verify_pool(cur, pool["replay_id"], sealed, keys, meta["as_of"])

    pk, party = pool_players(cur, pool, meta)
    eta, model, n_hist = a0_logit(model_dir, snap_dir, meta, pool, pk, party)
    res = json.loads((model_dir / "result.json").read_text())
    mhash = model.model_hash()
    assert mhash == res["models"]["A0"]["model_hash"], "A0 hash differs from its job result"
    version = f"A0@{res['job_id']}:{mhash[:16]}"

    xp = xp_minute10(cur, pool["replay_id"])
    conn.close()

    n = len(pool["replay_id"])
    surprise = np.zeros(n)
    stype = np.empty(n, object)
    r10s = np.full(n, np.nan)
    for i, rid in enumerate(pool["replay_id"].tolist()):
        r10, gl, win = xp[rid]
        if r10 is not None and gl >= MIN_LEN_TRAJ:
            surprise[i] = -eta[i] * r10
            stype[i] = "trajectory"
            r10s[i] = r10
        else:
            y0 = 1 if win == 0 else 0
            p = 1 / (1 + math.exp(-eta[i]))
            surprise[i] = -math.log(p if y0 else 1 - p)
            stype[i] = "outcome"
    top = np.zeros(n, bool)
    thr = {}
    for t in ("trajectory", "outcome"):
        m = stype == t
        if m.any():
            thr[t] = float(np.quantile(surprise[m], TOP_Q))
            top[m] = surprise[m] >= thr[t]
    pi = PI_BASE + PI_TOP * top
    u = np.array([draw_u(a.seed, r) for r in pool["replay_id"].tolist()])
    sel = u < pi

    order = np.argsort(pool["replay_id"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = "" if not a.dry_run else "_dryrun"
    ids_path = OUT_DIR / f"{a.name}{tag}.txt"
    tsv_path = OUT_DIR / f"{a.name}{tag}_design.tsv"
    with open(ids_path, "w") as f:
        f.write(f"# {a.name}: two-phase surprise draw from the scilearn pool; see SELECTION_RULES.md §C\n")
        for i in order:
            if sel[i]:
                f.write(f"{pool['replay_id'][i]}\tpi={pi[i]:.1f} {stype[i]} top={int(top[i])}\n")
    with open(tsv_path, "w") as f:
        f.write("replay_id\tpi\tselected\tsurprise_type\tsurprise\ttop_quintile\tr10\ta0_logit\tmodel_version\tseed\n")
        for i in order:
            f.write(f"{pool['replay_id'][i]}\t{pi[i]:.1f}\t{int(sel[i])}\t{stype[i]}\t{surprise[i]:.6f}\t"
                    f"{int(top[i])}\t{'' if np.isnan(r10s[i]) else f'{r10s[i]:.6f}'}\t{eta[i]:.6f}\t"
                    f"{version}\t{a.seed}\n")
    summary = {
        "name": a.name, "snap": a.snap, "pool_file_sha256": sha_file(snap_dir / "pool.npz"),
        "pool_size": n, "pool_replay_id_range": [int(pool["replay_id"].min()), int(pool["replay_id"].max())],
        "history_rows_used": n_hist, "model_version": version, "model_hash": mhash,
        "model_train_tranches": json.loads((C.PRIVATE / "jobs" / res["job_id"] / "spec.json").read_text())["fit_tranches"]
        if (C.PRIVATE / "jobs" / res["job_id"] / "spec.json").exists() else None,
        "seed": a.seed, "surprise_type_counts": {t: int((stype == t).sum()) for t in ("trajectory", "outcome")},
        "top_thresholds": thr, "top_quintile": int(top.sum()),
        "selected": int(sel.sum()), "selected_by_pi": {"0.2": int((sel & ~top).sum()), "0.6": int((sel & top).sum())},
        "expected_selected": float(pi.sum()),
        "ht_check_sum_1_over_pi_selected": float((1 / pi[sel]).sum()),
        "ids_file": str(ids_path.relative_to(REPO)), "design_file": str(tsv_path.relative_to(REPO)),
        "ids_file_sha256": sha_file(ids_path), "design_file_sha256": sha_file(tsv_path),
    }
    (OUT_DIR / f"{a.name}{tag}_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
