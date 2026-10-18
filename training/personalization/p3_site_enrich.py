"""
Corpus scoreboard evidence for players looked up on the site
(docs/personalization-site-design.md, "Enrichment").

The site's lookup computes skill cards from Heroes Profile match history (wins
and losses). The main model also reads scoreboard stats, which only our corpus
has. For each account the site flagged (personal_profiles.wants_metric), this
job reads the account's corpus games from the local store, joins their
normalized scoreboard rows (the a_shrink card, cache/rcf/sb_a_shrink.npz),
reduces them to the model's factor-space evidence (6 + 3 numbers,
p3_site_export.evidence) and writes that one small row to Neon. It never
writes match rows, battletags or anything else to Neon, and it skips accounts
that are private in the local player_privacy table.

Reads:  DATABASE_URL_RESEARCH (local, read-only role): replay_players, player_privacy
Writes: NEON_DATABASE_URL: personal_metric_evidence (upsert), personal_profiles.wants_metric
Runs:   on the main box, after privacy-sync (light: a few seconds per account)

Usage (from training/, source .env first):
  python3 personalization/p3_site_enrich.py [--limit 200] [--dry-run]
  python3 personalization/p3_site_enrich.py --player <region>:<blizz_id> --dry-run
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SB = os.path.join(HERE, "cache", "rcf", "sb_a_shrink.npz")
IDX = os.path.join(HERE, "cache", "rcf", "site_sb_by_blizz")
PARAMS = os.path.join(HERE, "results", "rcf_full", "params", "full_ms.npz")
ARTIFACT = os.path.join(ROOT, "src", "lib", "personal", "model-artifact.json")
H = 90


def sb_index():
    """The a_shrink scoreboard rows sorted by blizz_id, as memory-mapped .npy
    files (built once from sb_a_shrink.npz, rebuilt when it changes)."""
    stamp = os.path.join(IDX, "source_mtime.txt")
    src_m = str(int(os.path.getmtime(SB)))
    if not (os.path.exists(stamp) and open(stamp).read() == src_m):
        t0 = time.time()
        z = np.load(SB)
        bl = z["blizz_id"].astype(np.int64)
        o = np.argsort(bl, kind="stable")
        os.makedirs(IDX, exist_ok=True)
        np.save(os.path.join(IDX, "blizz_id.npy"), bl[o])
        np.save(os.path.join(IDX, "replay_id.npy"), z["replay_id"].astype(np.int64)[o])
        np.save(os.path.join(IDX, "Z.npy"), z["Z"][o])
        np.save(os.path.join(IDX, "W.npy"), z["W"][o])
        with open(stamp, "w") as f:
            f.write(src_m)
        print(f"built scoreboard index ({len(o):,} rows, {time.time() - t0:.0f}s)", flush=True)
    ld = lambda k: np.load(os.path.join(IDX, f"{k}.npy"), mmap_mode="r")
    return {k: ld(k) for k in ("blizz_id", "replay_id", "Z", "W")}


def player_sums(ix, games, blizz_id, hidx):
    """X, C (H, M) for one account: games = [(replay_id, hero name)] from the
    local store (region-specific), joined to the scoreboard rows of blizz_id."""
    a = np.searchsorted(ix["blizz_id"], blizz_id, side="left")
    b = np.searchsorted(ix["blizz_id"], blizz_id, side="right")
    M = ix["Z"].shape[1]
    X = np.zeros((H, M))
    C = np.zeros((H, M))
    if b <= a:
        return X, C, 0
    hero_of = {int(r): hidx.get(h, -1) for r, h in games}
    rid = np.asarray(ix["replay_id"][a:b])
    hh = np.array([hero_of.get(int(r), -1) for r in rid])
    ok = hh >= 0
    if not ok.any():
        return X, C, 0
    Z = np.asarray(ix["Z"][a:b])[ok].astype(np.float64)
    W = np.asarray(ix["W"][a:b])[ok].astype(np.float64)
    hh = hh[ok]
    for k in range(M):
        x = Z[:, k]
        g = ~np.isnan(x)
        X[:, k] = np.bincount(hh[g], weights=(W * x)[g], minlength=H)
        C[:, k] = np.bincount(hh[g], weights=W[g], minlength=H)
    return X, C, int(ok.sum())


def main():
    import types
    import psycopg2
    from p3_site_export import evidence
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--player", help="region:blizz_id (skips the Neon queue; use with --dry-run)")
    a = ap.parse_args()
    art = json.load(open(ARTIFACT))
    version = art["version"]
    hidx = {n: i for i, n in enumerate(art["heroNames"])}
    # the fitted main model (p3_rcf.Params fields; loaded directly so this job
    # needs neither numba nor torch)
    pz = np.load(PARAMS)
    par = types.SimpleNamespace(**{k: pz[k].astype(np.float64) for k in ("U", "G", "c", "rho2", "eps2")})
    assert par.U.shape[1] == 3

    local = psycopg2.connect(os.environ["DATABASE_URL_RESEARCH"])
    local.set_session(readonly=True, autocommit=True)
    lc = local.cursor()
    neon = None
    if a.player:
        r, b = a.player.split(":")
        todo = [(int(r), int(b), None)]
    else:
        neon = psycopg2.connect(os.environ["NEON_DATABASE_URL"])
        nc = neon.cursor()
        nc.execute("SELECT to_regclass('public.personal_profiles') IS NOT NULL")
        if not nc.fetchone()[0]:
            print("personal_profiles does not exist on Neon yet; nothing to do")
            return
        nc.execute("""SELECT p.region, p.blizz_id, p.battletag FROM personal_profiles p
                      WHERE p.wants_metric
                        AND NOT EXISTS (SELECT 1 FROM personal_metric_evidence e
                                        WHERE e.region = p.region AND e.blizz_id = p.blizz_id
                                          AND e.model_version = %s)
                      ORDER BY p.profile_at DESC NULLS LAST LIMIT %s""", (version, a.limit))
        todo = nc.fetchall()
    print(f"{len(todo)} account(s) to enrich (model {version})", flush=True)
    if not todo:
        return
    ix = sb_index()
    done = 0
    for region, blizz_id, tag in todo:
        if tag is not None:
            lc.execute("SELECT state FROM player_privacy WHERE battletag = %s AND region = %s", (tag, region))
            st = lc.fetchone()
            if st and st[0] == "private":
                print(f"  skip {region}:{blizz_id} (private)", flush=True)
                if neon and not a.dry_run:
                    nc.execute("UPDATE personal_profiles SET wants_metric = false WHERE region = %s AND blizz_id = %s",
                               (region, blizz_id))
                    neon.commit()
                continue
        lc.execute("SELECT replay_id, hero FROM replay_players WHERE blizz_id = %s AND region = %s",
                   (blizz_id, region))
        games = lc.fetchall()
        X, C, n = player_sums(ix, games, blizz_id, hidx)
        Lam, eta = evidence(par, X, C)
        lam6 = [Lam[0, 0], Lam[0, 1], Lam[0, 2], Lam[1, 1], Lam[1, 2], Lam[2, 2]]
        print(f"  {region}:{blizz_id}: {len(games)} corpus games, {n} scoreboard rows; "
              f"lam {np.round(lam6, 3).tolist()} eta {np.round(eta, 3).tolist()}", flush=True)
        if a.dry_run or neon is None:
            continue
        if n > 0:
            nc.execute("""INSERT INTO personal_metric_evidence (region, blizz_id, lam, eta, n_rows, model_version, updated_at)
                          VALUES (%s, %s, %s, %s, %s, %s, now())
                          ON CONFLICT (region, blizz_id) DO UPDATE SET lam = excluded.lam, eta = excluded.eta,
                            n_rows = excluded.n_rows, model_version = excluded.model_version, updated_at = now()""",
                       (region, blizz_id, [float(x) for x in lam6], [float(x) for x in eta], n, version))
        # no corpus rows: stop asking; the card stays on wins and losses only
        nc.execute("UPDATE personal_profiles SET wants_metric = false WHERE region = %s AND blizz_id = %s",
                   (region, blizz_id))
        neon.commit()
        done += 1
    print(f"enriched {done} account(s)", flush=True)


if __name__ == "__main__":
    main()
