"""N at A0's cutoff: paper 2's drift-aware MLP (d2c_cumprev) retrained on S+Dpre+G.

Same architecture and recipe as the frozen N (drift2026/train_drift_wp.py, regime "all",
features "cumulative_prev", seeds 42/123/777, replay-level 2% validation slice drawn with
RandomState(42), BCE, AdamW 5e-4 / 5e-3, cosine T_max 100, batch 4096, <= 200 epochs,
patience 25, best-validation weights). Only the training set changes: every game of the
scilearn S, Dpre and G tranches on a 2.55 build, i.e. exactly A0 SDG's fit rows minus the
7,666 S games on builds before 2.55 (the recipe's features are defined on 2.55 builds only).

Features follow the frozen N's convention: a game of build b uses cumulative statistics
through the previous 2.55 build, from the 2026-05-22 snapshot (drift2026 patch_stats),
legacy tiers. Games already in drift2026's feature cache reuse their cached rows; the rest
(late uploads of S and Dpre builds, and G after 05-22) get the same features from
neural_n.drift_stats. `feats --check` recomputes cached games and compares.

  ids        (local)   scilearn fit rows -> fair/n_ids.npz (cache membership)
  extract    (local)   DB rows for fit games outside the cache (late S uploads) -> rows_Slate.pkl.gz
  feats      (remote)  features for rows outside the cache -> feats_<tag>.npz
  assemble   (local)   one training file: X (n, 283) float32, y, rid, is_val
  train      (remote)  one seed on the GPU -> d2c_cumprev_sdg_s<seed>.pt
"""
import argparse
import gzip
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

FAIR = Path(os.environ.get("SCILEARN_PRIVATE", "/home/max/scilearn-private")) / "fair"
TRAINING = Path(os.environ.get("NEURAL_N_TRAINING") or Path(__file__).resolve().parents[2])
CACHE_NPZ = TRAINING / "drift2026" / "feature_cache" / "features_cumulative_prev.npz"
VAL_FRAC = 0.02
SEEDS = (42, 123, 777)


def _nn():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines"))
    import neural_n as NN
    return NN


# --------------------------------------------------------------------------- local steps

def cmd_ids(_):
    from training.scilearn import data as D
    from training.scilearn import tranches as T
    g, meta, y, _ = D.load(tranches=["S", "Dpre", "G"], labels=True)
    b = np.array(meta["builds"])[g["build"]].astype(str)
    is255 = np.char.startswith(b, "2.55.")
    rid = g["replay_id"][is255]
    cache_rid = np.load(CACHE_NPZ)["replay_ids"][0::2]
    in_cache = np.isin(rid, cache_rid)
    FAIR.mkdir(parents=True, exist_ok=True)
    np.savez(FAIR / "n_ids.npz", replay_id=rid, in_cache=in_cache, tranche=g["tranche"][is255],
             build=b[is255], y0=y[is255])
    print(f"fit games on 2.55: {len(rid):,}; in cache {in_cache.sum():,}; outside {(~in_cache).sum():,}; "
          f"dropped pre-2.55 S games {(~is255).sum():,}")
    for t in ("S", "Dpre", "G"):
        m = g["tranche"][is255] == T.CODES[t]
        print(f"  {t}: {m.sum():,} games, {(m & ~in_cache).sum():,} outside the cache")


def cmd_extract(_):
    """DB rows for S games outside the drift cache (late uploads). Dpre and G rows already
    exist in fits/neural_n (rows_Dpre, rows_G)."""
    from training.scilearn import tranches as T
    NN = _nn()
    z = np.load(FAIR / "n_ids.npz")
    want = z["replay_id"][(~z["in_cache"]) & (z["tranche"] == T.CODES["S"])]
    conn = NN._db()
    cur = conn.cursor(name="fair_slate")
    cur.itersize = 50000
    cur.execute("""SELECT replay_id, game_version, EXTRACT(EPOCH FROM game_date)::bigint,
                          game_map, league_tier, avg_mmr, team0_heroes, team1_heroes,
                          team0_bans, team1_bans, winner
                   FROM replay_draft_data WHERE replay_id = ANY(%s)""", ([int(x) for x in want],))
    sealed = NN.sealed_ids()
    rows = []
    for rid, v, ts, gm, lt, am, t0, t1, b0, b1, w in cur:
        lst = lambda x: (json.loads(x) if isinstance(x, str) else x) or []
        t0, t1 = lst(t0), lst(t1)
        if rid in sealed or len(t0) != 5 or len(t1) != 5 or w not in (0, 1):
            continue
        assert v.startswith("2.55.") and NN.tranche_of(v) == "S", v
        rows.append({"replay_id": int(rid), "game_version": v, "ts": int(ts), "game_map": gm,
                     "league_tier": lt, "avg_mmr": None if am is None else float(am),
                     "team0_heroes": t0, "team1_heroes": t1,
                     "team0_bans": lst(b0), "team1_bans": lst(b1), "winner": int(w)})
    rows.sort(key=lambda g: (g["ts"], g["replay_id"]))
    with gzip.open(FAIR / "rows_Slate.pkl.gz", "wb") as f:
        pickle.dump(rows, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"S late uploads: wanted {len(want):,}, got {len(rows):,}")


# --------------------------------------------------------------------------- remote: features

def _stats_for(NN, build):
    from drift2026 import common
    builds = common.load_patch_index()["builds"]
    i = builds.index(build)
    first255 = min(j for j, v in enumerate(builds) if v.startswith("2.55."))
    if i == first255:
        return common.empty_stats()      # cumulative_prev: position 0 gets empty stats
    return NN.drift_stats(build, [])


def features(NN, rows, nproc):
    """(X (2n, 283) float32 rows [orig, swap] per game, rid (2n,), y (2n,) with y = team0 wins)."""
    import multiprocessing as mp
    from drift2026.train_drift_wp import enriched_cols
    mod = NN._wp_post()
    ecols = enriched_cols()
    by_b = {}
    for g in rows:
        by_b.setdefault(g["game_version"], []).append(g)
    Xs, rids, ys = [], [], []
    for b in sorted(by_b, key=NN.bkey):
        rr = by_b[b]
        mod._STATS = _stats_for(NN, b)
        feats = [{"team0_heroes": g["team0_heroes"], "team1_heroes": g["team1_heroes"],
                  "game_map": g["game_map"],
                  "skill_tier": NN.legacy_tier(g.get("league_tier"), g.get("avg_mmr")),
                  "winner": 0, "avg_mmr": g.get("avg_mmr"), "replay_id": i} for i, g in enumerate(rr)]
        chunks = [feats[k:k + 2000] for k in range(0, len(feats), 2000)]
        with mp.get_context("fork").Pool(nproc) as pool:
            parts = pool.map(mod._feat, chunks)
        parts = [p for p in parts if len(p[0])]
        B = np.concatenate([p[0] for p in parts])
        E = np.concatenate([p[1] for p in parts])
        pos = np.concatenate([p[2] for p in parts])
        Xs.append(np.concatenate([B, E[:, ecols]], axis=1).astype(np.float32))
        rid = np.array([rr[i]["replay_id"] for i in pos], np.int64)
        y0 = np.array([float(rr[i]["winner"] == 0) for i in pos[0::2]], np.float32)
        y = np.empty(len(pos), np.float32)
        y[0::2], y[1::2] = y0, 1 - y0
        rids.append(rid); ys.append(y)
        print(f"  {b}: {len(rr):,} games, {len(rr) - len(pos) // 2} failed", flush=True)
    return np.concatenate(Xs), np.concatenate(rids), np.concatenate(ys)


def cmd_feats(a):
    NN = _nn()
    rows = []
    for p in a.rows:
        rows += NN.guard(NN.load_rows(p), "rows")
    keep = None
    if a.ids:
        z = np.load(a.ids)
        keep = set(z["replay_id"][~z["in_cache"]].tolist()) if not a.check else None
    if a.check:
        cache_rid = np.load(CACHE_NPZ)["replay_ids"][0::2]
        cs = set(cache_rid.tolist())
        cand = [g for g in rows if g["replay_id"] in cs]
        rng = np.random.default_rng(0)
        rows = [cand[i] for i in sorted(rng.choice(len(cand), min(a.check, len(cand)), replace=False))]
    elif keep is not None:
        rows = [g for g in rows if g["replay_id"] in keep]
    t0 = time.time()
    X, rid, y = features(NN, rows, a.nproc)
    if a.check:
        from drift2026.train_drift_wp import enriched_cols
        z = np.load(CACHE_NPZ)
        crid = z["replay_ids"]
        idx = {}
        for i, r in enumerate(crid):
            idx.setdefault(int(r), i)
        ii = np.array([idx[int(r)] + (k % 2) for k, r in enumerate(rid)])
        CX = np.concatenate([z["bases"][ii], z["enricheds"][ii][:, enriched_cols()]], 1)
        d = np.abs(CX - X)
        print(f"check on {len(rid) // 2:,} cached games: max abs diff {d.max():.2e}, "
              f"mean {d.mean():.2e}, rows with diff > 1e-4: {(d.max(1) > 1e-4).mean():.4f}; "
              f"labels agree {(z['labels'][ii] == y).mean():.4f}")
        return
    np.savez(a.out, X=X, rid=rid, y=y)
    print(f"wrote {a.out}: {X.shape} in {time.time() - t0:.0f}s")


# --------------------------------------------------------------------------- local: assemble

def cmd_assemble(a):
    from drift2026.train_drift_wp import enriched_cols
    z = np.load(FAIR / "n_ids.npz")
    want = z["replay_id"]
    c = np.load(CACHE_NPZ, mmap_mode="r")
    crid = np.asarray(c["replay_ids"])
    m = np.isin(crid, want[z["in_cache"]])
    ci = np.flatnonzero(m)
    cols = enriched_cols()
    parts_X = [np.concatenate([np.asarray(c["bases"])[ci], np.asarray(c["enricheds"])[ci][:, cols]], 1)]
    parts_r = [crid[ci]]
    parts_y = [np.asarray(c["labels"])[ci]]
    for f in a.feats:
        f = np.load(f)
        parts_X.append(f["X"]); parts_r.append(f["rid"]); parts_y.append(f["y"])
    X = np.concatenate(parts_X).astype(np.float32)
    rid = np.concatenate(parts_r)
    y = np.concatenate(parts_y).astype(np.float32)
    u, cnt = np.unique(rid, return_counts=True)
    assert (cnt == 2).all(), "each game must have exactly an original and a swapped row"
    missing = np.setdiff1d(want, u)
    extra = np.setdiff1d(u, want)
    assert len(extra) == 0
    # label check against the harness snapshot (team-0 orientation is the first row)
    y0 = dict(zip(z["replay_id"].tolist(), z["y0"].tolist()))
    first = np.r_[True, rid[1:] != rid[:-1]]
    agree = np.mean([y[i] == y0[int(rid[i])] for i in np.flatnonzero(first)[:200000]])
    tr_rids = np.unique(rid)
    rng = np.random.RandomState(42)
    val_ids = tr_rids[rng.permutation(len(tr_rids))[:max(1, int(len(tr_rids) * VAL_FRAC))]]
    is_val = np.isin(rid, val_ids)
    np.save(FAIR / "n_train_X.npy", X)
    np.savez(FAIR / "n_train_meta.npz", rid=rid, y=y, is_val=is_val)
    print(f"assembled {len(u):,} games ({len(X):,} rows); missing {len(missing):,} of {len(want):,} "
          f"(feature or row failures); label agreement with snapshot y0 {agree:.5f}; "
          f"val games {len(val_ids):,}")


# --------------------------------------------------------------------------- remote: train

def train_model(model, Xtr, ytr, Xva, yva, device, name=""):
    """drift2026/train_drift_wp.train_model, regime 'all' (no weights, no embedding), verbatim."""
    import torch
    import torch.nn as nn
    lr, max_epochs = 5e-4, 200
    model.to(device)
    criterion = nn.BCELoss(reduction="none")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=min(1e-5, lr))
    best_loss, best_acc, best_state, pat = float("inf"), 0.0, None, 0
    n = len(Xtr)
    for ep in range(max_epochs):
        model.train()
        pm = torch.randperm(n, device=device)
        for i in range(0, n, 4096):
            idx = pm[i:i + 4096]
            loss = criterion(model(Xtr[idx]), ytr[idx]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sch.step()
        model.eval()
        with torch.no_grad():
            vp = torch.cat([model(Xva[i:i + 65536]) for i in range(0, len(Xva), 65536)])
            vl = criterion(vp, yva).mean().item()
            va = ((vp > 0.5).float() == yva).float().mean().item() * 100
        if vl < best_loss:
            best_loss, best_acc = vl, va
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= 25:
                break
        if (ep + 1) % 10 == 0:
            print(f"  {name} ep{ep + 1}: val_loss={vl:.5f} val_acc={va:.2f}% best={best_acc:.2f}%", flush=True)
    model.load_state_dict(best_state)
    model.eval()
    return model, best_acc, best_loss, ep + 1


def cmd_train(a):
    import torch
    sys.path.insert(0, str(TRAINING))
    from sweep_enriched_wp import WinProbEnrichedModel
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device("cuda")
    X = np.load(Path(a.data) / "n_train_X.npy", mmap_mode="r")
    meta = np.load(Path(a.data) / "n_train_meta.npz")
    is_val, y = meta["is_val"], meta["y"]

    def to_gpu(mask):
        idx = np.flatnonzero(mask)
        out = torch.empty((len(idx), X.shape[1]), dtype=torch.float32, device=device)
        for k in range(0, len(idx), 200000):
            out[k:k + 200000] = torch.from_numpy(np.ascontiguousarray(X[idx[k:k + 200000]]))
        return out, torch.tensor(y[idx], dtype=torch.float32, device=device)

    # the frozen recipe keeps rows in feature-cache order (by build, then replay); same here
    Xtr, ytr = to_gpu(~is_val)
    Xva, yva = to_gpu(is_val)
    print(f"seed {a.seed}: train={len(Xtr):,} val={len(Xva):,} rows", flush=True)
    t0 = time.time()
    model = WinProbEnrichedModel(X.shape[1], [256, 128], dropout=0.3)
    model, va, vl, ep = train_model(model, Xtr, ytr, Xva, yva, device, name=f"s{a.seed}")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "input_dim": int(X.shape[1]), "arch": [256, 128], "dropout": 0.3},
               out / f"d2c_cumprev_sdg_s{a.seed}.pt")
    (out / f"d2c_cumprev_sdg_s{a.seed}.json").write_text(json.dumps(
        {"seed": a.seed, "val_acc": va, "val_bce": vl, "epochs": ep, "n_train_rows": len(Xtr),
         "n_val_rows": len(Xva), "minutes": round((time.time() - t0) / 60, 1)}))
    print(f"seed {a.seed}: val_acc {va:.2f}% val_bce {vl:.5f} epochs {ep} "
          f"in {(time.time() - t0) / 60:.1f} min", flush=True)


def cmd_predict(a):
    """Per-game team-0 win probability on the training matrix (every fit game), each seed
    swap-symmetrized, (p(orig) + 1 - p(swap)) / 2, then the mean over seeds, as neural_n."""
    import torch
    sys.path.insert(0, str(TRAINING))
    from sweep_enriched_wp import WinProbEnrichedModel
    device = torch.device("cuda")
    X = np.load(Path(a.data) / "n_train_X.npy", mmap_mode="r")
    rid = np.load(Path(a.data) / "n_train_meta.npz")["rid"]
    first = np.flatnonzero(np.r_[True, rid[1:] != rid[:-1]])
    assert len(first) * 2 == len(rid) and np.all(rid[first] == rid[first + 1])
    ps = []
    for ck_path in a.models:
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        m = WinProbEnrichedModel(ck["input_dim"], ck["arch"], dropout=ck["dropout"])
        m.load_state_dict(ck["state_dict"])
        m = m.to(device).eval()
        out = np.empty(len(rid))
        with torch.no_grad():
            for k in range(0, len(rid), 262144):
                xb = torch.from_numpy(np.ascontiguousarray(X[k:k + 262144])).to(device)
                out[k:k + 262144] = m(xb).view(-1).double().cpu().numpy()
        ps.append((out[first] + 1.0 - out[first + 1]) / 2.0)
        print(f"{ck_path}: done", flush=True)
    ps = np.array(ps)
    np.savez(a.out, replay_id=rid[first], wp=ps.mean(0), wp_seeds=ps.astype(np.float32))
    print(f"wrote {a.out}: {len(first):,} games")


def cmd_score_strict(a):
    """Dpost scores under a strict as-of rule for N's statistics. The standard convention
    (neural_n.predict_drift) gives every game of build b the statistics of all games of the
    builds before b, including games of those builds that started after the scored game or
    were uploaded later. Here a game g of build b on UTC day d uses the snapshot counts
    through 2.55.16.96881 plus only those games of builds strictly between 96881 and b that
    were first seen before 00:00 UTC of day d (first_seen_ts from the harness extract), so
    every statistic comes from games first seen before g started."""
    import copy
    import multiprocessing as mp
    import torch
    sys.path.insert(0, str(TRAINING))
    from sweep_enriched_wp import WinProbEnrichedModel
    from drift2026.build_patch_stats import count_chunk, merge_cell, _new_cell
    from drift2026.train_drift_wp import enriched_cols
    NN = _nn()
    mod = NN._wp_post()
    ecols = enriched_cols()
    ex = np.load(a.extract)
    fs_of = dict(zip(ex["replay_id"].tolist(), ex["first_seen_ts"].tolist()))
    ts_of = dict(zip(ex["replay_id"].tolist(), ex["game_ts"].tolist()))
    rows = NN.guard(NN.load_rows(a.rows), "rows")
    hist = []
    seen = set()
    for p in list(a.history) + [a.rows]:
        for g in NN.guard(NN.load_rows(p), "history"):
            if g["replay_id"] in seen or g["replay_id"] not in fs_of:
                continue
            if g.get("winner") not in (0, 1) or len(g["team0_heroes"]) != 5 or len(g["team1_heroes"]) != 5:
                continue
            if NN.bkey(g["game_version"]) <= NN.bkey(NN.DRIFT_BASE_BUILD):
                continue
            seen.add(g["replay_id"])
            hist.append(g)
    hist.sort(key=lambda g: fs_of[g["replay_id"]])
    hfs = np.array([fs_of[g["replay_id"]] for g in hist])
    sets = {}
    for tag, pat in (("frozen", a.frozen), ("matched", a.matched)):
        ms = []
        for sd in SEEDS:
            ck = torch.load(pat.format(seed=sd), map_location="cpu", weights_only=False)
            m = WinProbEnrichedModel(ck["input_dim"], ck["arch"], dropout=ck["dropout"])
            m.load_state_dict(ck["state_dict"])
            ms.append(m.eval())
        sets[tag] = ms
    base = NN._base_counts()
    per_build = {}
    hp = 0
    rid = np.array([g["replay_id"] for g in rows], np.int64)
    gts = np.array([ts_of[int(r)] for r in rid])
    day = gts // 86400
    out = {t: np.full((len(SEEDS), len(rows)), np.nan) for t in sets}
    n_hist_used = np.zeros(len(rows), np.int64)
    for d in np.unique(day):
        start = int(d) * 86400
        e = int(np.searchsorted(hfs, start, side="left"))     # first_seen < day start
        if e > hp:
            new = hist[hp:e]
            by_b = {}
            for g in new:
                by_b.setdefault(g["game_version"], []).append(
                    (0, NN.legacy_tier(g.get("league_tier"), g.get("avg_mmr")), g["game_map"],
                     tuple(g["team0_heroes"]), tuple(g["team1_heroes"]),
                     tuple(g.get("team0_bans") or []) + tuple(g.get("team1_bans") or []), g["winner"]))
            for b, slim in by_b.items():
                cells = per_build.setdefault(b, {})
                for (_, tier), cell in count_chunk(slim).items():
                    merge_cell(cells.setdefault(tier, _new_cell()), cell)
                per_build[b + "#n"] = per_build.get(b + "#n", 0) + len(slim)
            hp = e
        idx_d = np.flatnonzero(day == d)
        for b in sorted({rows[i]["game_version"] for i in idx_d}, key=NN.bkey):
            ii = [int(i) for i in idx_d if rows[i]["game_version"] == b]
            cum = copy.deepcopy(base)
            nh = 0
            for bb, cells in per_build.items():
                if bb.endswith("#n") or not (NN.bkey(NN.DRIFT_BASE_BUILD) < NN.bkey(bb) < NN.bkey(b)):
                    continue
                nh += per_build[bb + "#n"]
                for tier, cell in cells.items():
                    merge_cell(cum.setdefault(tier, _new_cell()), cell)
            mod._STATS = mod.stats_from_counts_rounded(cum)
            feats = [{"team0_heroes": rows[i]["team0_heroes"], "team1_heroes": rows[i]["team1_heroes"],
                      "game_map": rows[i]["game_map"],
                      "skill_tier": NN.legacy_tier(rows[i].get("league_tier"), rows[i].get("avg_mmr")),
                      "winner": 0, "avg_mmr": rows[i].get("avg_mmr"), "replay_id": i} for i in ii]
            chunks = [feats[k:k + 2000] for k in range(0, len(feats), 2000)]
            if a.nproc > 1 and len(chunks) > 1:
                with mp.get_context("fork").Pool(a.nproc) as pool:
                    parts = pool.map(mod._feat, chunks)
            else:
                parts = [mod._feat(c) for c in chunks]
            parts = [p for p in parts if len(p[0])]
            B = np.concatenate([p[0] for p in parts])
            E_ = np.concatenate([p[1] for p in parts])
            pos = np.concatenate([p[2] for p in parts])[0::2]
            X = torch.from_numpy(np.concatenate([B, E_[:, ecols]], axis=1))
            with torch.no_grad():
                for t, ms in sets.items():
                    for s_, m in enumerate(ms):
                        p = m(X).view(-1).double().numpy()
                        out[t][s_, pos] = (p[0::2] + 1.0 - p[1::2]) / 2.0
            n_hist_used[pos] = nh
        print(f"day {int(d)}: {len(idx_d):,} games, history first seen before the day {hp:,}", flush=True)
    for t in sets:
        miss = np.isnan(out[t][0])
        out[t][:, miss] = 0.5
        np.savez(Path(a.out_dir) / f"n_{t}_strict_Dpost.npz", replay_id=rid, p=out[t].mean(0),
                 p_seeds=out[t].astype(np.float32), n_hist=n_hist_used)
        print(f"{t}: wrote {len(rid):,} games ({miss.sum()} failed)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ids")
    sub.add_parser("extract")
    f = sub.add_parser("feats")
    f.add_argument("--rows", nargs="+", required=True)
    f.add_argument("--ids")
    f.add_argument("--out")
    f.add_argument("--check", type=int, default=0)
    f.add_argument("--nproc", type=int, default=4)
    s = sub.add_parser("assemble")
    s.add_argument("--feats", nargs="+", required=True)
    t = sub.add_parser("train")
    t.add_argument("--seed", type=int, required=True)
    t.add_argument("--data", required=True)
    t.add_argument("--out-dir", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--data", required=True)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--out", required=True)
    q = sub.add_parser("score_strict")
    q.add_argument("--rows", required=True)
    q.add_argument("--history", nargs="*", default=[])
    q.add_argument("--extract", required=True)
    q.add_argument("--frozen", required=True)
    q.add_argument("--matched", required=True)
    q.add_argument("--out-dir", required=True)
    q.add_argument("--nproc", type=int, default=4)
    a = ap.parse_args()
    {"score_strict": cmd_score_strict, "predict": cmd_predict, "ids": cmd_ids, "extract": cmd_extract, "feats": cmd_feats, "assemble": cmd_assemble,
     "train": cmd_train}[a.cmd](a)


if __name__ == "__main__":
    main()
