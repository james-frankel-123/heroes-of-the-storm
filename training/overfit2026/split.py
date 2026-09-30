"""
Within-snapshot split experiment: proxy side A, gold side B.

The pinned snapshot (1,949,087 replays) is split replay-level by a salted
hash (data.half) into A (proxy side) and B (gold side). Nothing computed from
B ever enters a proxy, its statistics, its early stopping, or the search that
optimizes it; nothing computed from A enters a gold reference.

Statistics are built from raw replays in the frozen_stats schema (so the
unchanged MCTS worker can load them through WP_STATS_PATH). comp_wr keeps the
paper's external role-composition table (src/lib/data/compositions.json),
identical for proxies and judges.

Statistics modes for training rows
  leak  statistics over the same games the model trains on (paper-1 design:
        every row's hero / pair / comp statistics include its own outcome)
  oof   2-fold out-of-fold: rows in fold 1 get statistics from fold 2 and vice
        versa (no row's features contain its own outcome); the deployed model
        uses statistics over the whole training subset, like the leaky one.

Subcommands
  stats     write stats JSONs (cache/split_stats/*.json)
  train     train WP models from a spec list (models/*.pt, models/*.json)
  heldout   evaluate every trained model on the other half's real games
"""
import os
import sys
import json
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)

import numpy as np

from overfit2026 import data, feats

STATS_DIR = os.path.join(HERE, "cache", "split_stats")
FEAT_DIR = os.path.join(HERE, "cache", "split_feats")
MODEL_DIR = os.path.join(HERE, "models")


# ── subsets ─────────────────────────────────────────────────────────────

def side_of(rid):
    return "A" if data.half(rid) == 0 else "B"


def fold_of(rid):
    return data.half(rid, salt=1)          # 0/1, independent of the A/B split


def is_val(rid):
    return data.frac_bucket(rid, 50, salt=5) == 0   # 2% replay-level early-stop set


def subset(games, side, eighths=8):
    """side A/B, first `eighths`/8 of that side (nested: 1 < 2 < 4 < 8).
    side N = the no-drift post-snapshot gold games (backfill + T97), used whole."""
    if side in ("N", "S"):
        return [g for g in games if data.frac_bucket(g[0], 8) < eighths]
    return [g for g in games if side_of(g[0]) == side
            and data.frac_bucket(g[0], 8) < eighths]


def games_for(side):
    if side == "N":
        fut, bf = data.load_future(), data.load_backfill()
        return [g for v in bf.values() for g in v] + list(fut["2.55.16.97039"])
    return data.load_snapshot()


# ── statistics ──────────────────────────────────────────────────────────

def counts_for(games):
    from drift2026.build_patch_stats import count_chunk, merge_cell, _new_cell
    merged = {}
    for i in range(0, len(games), 20000):
        chunk = [(0, t, m, a, b, bans, w) for _, t, m, a, b, bans, w in games[i:i + 20000]]
        for (_, tier), cell in count_chunk(chunk).items():
            dst = merged.get(tier)
            if dst is None:
                dst = merged[tier] = _new_cell()
            merge_cell(dst, cell)
    return merged


def frozen_json(merged, meta):
    """make_snapshot.py schema/thresholds, undecayed."""
    hero_stats, hero_map_stats, pairwise_stats = [], [], []
    for tier, cell in merged.items():
        total = cell["games"]
        for h, (g, w) in cell["hero"].items():
            if g < 20:
                continue
            hero_stats.append({"hero": h, "tier": tier, "games": g,
                               "win_rate": round(100.0 * w / g, 3),
                               "pick_rate": round(100.0 * g / total, 3),
                               "ban_rate": round(100.0 * cell["bans"].get(h, 0) / total, 3)})
        for (m, h), (g, w) in cell["hmap"].items():
            if g < 5:
                continue
            hero_map_stats.append({"hero": h, "map": m, "tier": tier, "games": g,
                                   "win_rate": round(100.0 * w / g, 3)})
        for (a, b), (g, w) in cell["with"].items():
            if g < 10:
                continue
            wr = round(100.0 * w / g, 3)
            for x, y in ((a, b), (b, a)):
                pairwise_stats.append({"hero_a": x, "hero_b": y, "tier": tier,
                                       "relationship": "with", "win_rate": wr, "games": g})
        for (a, b), (g, wa) in cell["against"].items():
            if g < 10:
                continue
            wr_a = round(100.0 * wa / g, 3)
            pairwise_stats.append({"hero_a": a, "hero_b": b, "tier": tier,
                                   "relationship": "against", "win_rate": wr_a, "games": g})
            pairwise_stats.append({"hero_a": b, "hero_b": a, "tier": tier,
                                   "relationship": "against", "win_rate": round(100.0 - wr_a, 3),
                                   "games": g})
    return {"_meta": dict(meta, snapshot_date="2026-05-22", patch="2.55"),
            "hero_stats": hero_stats, "hero_map_stats": hero_map_stats,
            "pairwise_stats": pairwise_stats}


def stats_path(name):
    return os.path.join(STATS_DIR, f"{name}.json")


def stats_names():
    """name -> game-filter description."""
    out = {}
    for side in "ABNS":
        for e in ((8,) if side in "NS" else (1, 2, 4, 8)):
            base = f"{side}{e}"
            out[base] = (side, e, None)
            out[f"{base}f0"] = (side, e, 0)
            out[f"{base}f1"] = (side, e, 1)
    return out


_GAMES = None


def _build_one(args):
    name, (side, e, fold) = args
    games = [g for g in _GAMES if (g[0] > data.SNAPSHOT_BOUND) == (side == "N")]
    sel = [g for g in subset(games, side, e) if fold is None or fold_of(g[0]) == fold]
    js = frozen_json(counts_for(sel), {"subset": name, "games_used": len(sel)})
    with open(stats_path(name), "w") as f:
        json.dump(js, f)
    return name, len(sel)


def cmd_stats(args):
    import multiprocessing as mp
    os.makedirs(STATS_DIR, exist_ok=True)
    global _GAMES
    _GAMES = data.load_snapshot() + games_for("N")
    todo = [(n, spec) for n, spec in stats_names().items()
            if not os.path.exists(stats_path(n))]
    ctx = mp.get_context("fork")
    with ctx.Pool(min(24, max(1, len(todo)))) as pool:
        for n, k in pool.imap_unordered(_build_one, todo):
            print(f"  stats {n}: {k:,} games", flush=True)


# ── features ────────────────────────────────────────────────────────────

def load_stats(name):
    return feats.stats_from_json(stats_path(name))


def rows_of(games):
    return [(g[3], g[4], g[2], g[1]) for g in games]


def feature_set(side, e, mode):
    """Training features for subset (side, e) under stats mode. Returns
    (Xf, Xs, y, rids) with the row order of subset(); cached to disk."""
    os.makedirs(FEAT_DIR, exist_ok=True)
    path = os.path.join(FEAT_DIR, f"{side}{e}_{mode}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["Xf"], z["Xs"], z["y"], z["rid"]
    games = subset(games_for(side), side, e)
    if mode == "leak":
        Xf, Xs = feats.featurize(rows_of(games), load_stats(f"{side}{e}"))
    elif mode == "oof":
        Xf = np.zeros((len(games), 283), np.float32)
        Xs = np.zeros_like(Xf)
        folds = np.array([fold_of(g[0]) for g in games])
        for k in (0, 1):
            idx = np.where(folds == k)[0]
            st = load_stats(f"{side}{e}f{1 - k}")       # the OTHER fold's statistics
            a, b = feats.featurize([rows_of([games[i]])[0] for i in idx], st)
            Xf[idx], Xs[idx] = a, b
    else:
        raise ValueError(mode)
    y = np.array([1.0 if g[6] == 0 else 0.0 for g in games], np.float32)
    rid = np.array([g[0] for g in games])
    np.savez(path, Xf=Xf, Xs=Xs, y=y, rid=rid)
    return Xf, Xs, y, rid


def eval_features(side, e_stats, games_side):
    """All games of `games_side` (full half) featurized with the DEPLOY stats
    of the model trained on (side, e_stats) — i.e. how that model sees games
    it never trained on."""
    os.makedirs(FEAT_DIR, exist_ok=True)
    path = os.path.join(FEAT_DIR, f"eval_{games_side}_with_{side}{e_stats}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["Xf"], z["Xs"], z["y"], z["rid"]
    games = subset(data.load_snapshot(), games_side, 8)
    Xf, Xs = feats.featurize(rows_of(games), load_stats(f"{side}{e_stats}"))
    y = np.array([1.0 if g[6] == 0 else 0.0 for g in games], np.float32)
    rid = np.array([g[0] for g in games])
    np.savez(path, Xf=Xf, Xs=Xs, y=y, rid=rid)
    return Xf, Xs, y, rid


# ── training ────────────────────────────────────────────────────────────

# name: (side, eighths, stats_mode, arch, dropout, weight_decay, max_epochs,
#        early_stop, seed, input) ; input "enriched" (283) or "naive" (197)
def spec_table():
    S = {}

    def add(name, side="A", e=8, mode="leak", arch=(256, 128), dropout=0.3, wd=5e-3,
            epochs=200, es=True, seed=42, inp="enriched"):
        S[name] = dict(side=side, e=e, mode=mode, arch=list(arch), dropout=dropout, wd=wd,
                       epochs=epochs, es=es, seed=seed, inp=inp)
    # proxies (side A)
    for mode in ("leak", "oof"):
        for e in (1, 2, 4, 8):
            add(f"pA{e}_{mode}", e=e, mode=mode)
        for s in (1, 2, 3):
            add(f"pA8_{mode}_s{s}", mode=mode, seed=s)
    add("pA8_leak_w64", arch=(64, 32))
    add("pA8_leak_w1024", arch=(1024, 512))
    add("pA8_leak_deep", arch=(512, 256, 128))
    add("pA8_leak_noreg", dropout=0.0, wd=0.0)
    for ep in (1, 3, 10, 40):
        add(f"pA8_leak_ep{ep}", epochs=ep, es=False)
        add(f"pA8_leak_noreg_ep{ep}", epochs=ep, es=False, dropout=0.0, wd=0.0)
    add("pA8_naive", inp="naive")
    # gold judges (side B)
    for s in (0, 1, 2):
        add(f"gB8_oof_s{s}", side="B", mode="oof", seed=100 + s)
    add("gB8_leak", side="B", mode="leak", seed=100)
    add("gB8_naive", side="B", inp="naive", seed=100)
    # gold judges for the paper's own agents: no-drift post-snapshot games
    for s in (0, 1, 2):
        add(f"gN8_oof_s{s}", side="N", mode="oof", seed=200 + s)
    add("gN8_naive", side="N", inp="naive", seed=200)
    # full-snapshot proxies (paper scale): paper design vs leakage-free
    add("pS8_leak", side="S", mode="leak", seed=42)
    add("pS8_oof", side="S", mode="oof", seed=42)
    return S


def model_path(name):
    return os.path.join(MODEL_DIR, f"{name}.pt")


def train_one(name, spec, device):
    import torch
    import torch.nn as nn
    from sweep_enriched_wp import WinProbEnrichedModel
    Xf, Xs, y, rid = feature_set(spec["side"], spec["e"], spec["mode"])
    dim = 283 if spec["inp"] == "enriched" else 197
    val = np.array([is_val(r) for r in rid])
    X = np.concatenate([Xf, Xs])[:, :dim]
    Y = np.concatenate([y, 1 - y])
    V = np.concatenate([val, val])
    tX = torch.tensor(X[~V], device=device)
    tY = torch.tensor(Y[~V], device=device)
    vX = torch.tensor(X[V], device=device)
    vY = torch.tensor(Y[V], device=device)
    torch.manual_seed(spec["seed"])
    np.random.seed(spec["seed"])
    model = WinProbEnrichedModel(dim, spec["arch"], dropout=spec["dropout"]).to(device)
    crit = nn.BCELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=spec["wd"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=1e-5)
    best, best_state, pat, hist = float("inf"), None, 0, []
    n = len(tX)
    for ep in range(spec["epochs"]):
        model.train()
        pm = torch.randperm(n, device=device)
        for i in range(0, n, 4096):
            idx = pm[i:i + 4096]
            loss = crit(model(tX[idx]).view(-1), tY[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sch.step()
        model.eval()
        with torch.no_grad():
            vp = model(vX).view(-1)
            vl = crit(vp, vY).item()
            va = ((vp > 0.5).float() == vY).float().mean().item()
        hist.append((ep + 1, vl, va))
        if not spec["es"]:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            continue
        if vl < best:
            best, pat = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
            if pat >= 25:
                break
    torch.save(best_state, model_path(name))
    meta = dict(spec, name=name, input_dim=dim, n_train_rows=int(n), history=hist,
                best_val_loss=min(h[1] for h in hist))
    with open(model_path(name)[:-3] + ".json", "w") as f:
        json.dump(meta, f)
    print(f"  {name}: epochs={len(hist)} best_val_loss={meta['best_val_loss']:.5f} "
          f"final_val_acc={hist[-1][2]:.4f}", flush=True)


def cmd_train(args):
    import torch
    os.makedirs(MODEL_DIR, exist_ok=True)
    specs = spec_table()
    names = [n for n in specs if (not args.only or any(o in n for o in args.only.split(",")))]
    dev = torch.device("cuda")
    for n in names:
        if os.path.exists(model_path(n)) and not args.force:
            continue
        t0 = time.time()
        train_one(n, specs[n], dev)
        print(f"    ({time.time() - t0:.0f}s)", flush=True)


def load_model(name):
    meta = json.load(open(model_path(name)[:-3] + ".json"))
    return feats.load_wp(model_path(name), dim=meta["input_dim"], arch=meta["arch"],
                         dropout=meta["dropout"]), meta


def cmd_heldout(args):
    """Held-out real-game quality of every model on the OTHER half (with the
    model's deploy statistics), plus calibration slope."""
    import torch
    specs = spec_table()
    out = {}
    for n, sp in specs.items():
        if not os.path.exists(model_path(n)):
            continue
        other = "A" if sp["side"] == "B" else "B"
        Xf, Xs, y, rid = eval_features(sp["side"], sp["e"], other)
        m, meta = load_model(n)
        d = meta["input_dim"]
        p = feats.predict_sym(m, Xf[:, :d], Xs[:, :d], "cuda")
        ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        acc = float(np.mean((p > .5) == (y > .5)))
        z = np.log(p / (1 - p))
        from overfit2026.gold import fit_logistic
        c, _ = fit_logistic(z[:, None], y)
        out[n] = {"acc": acc, "logloss": ll, "calib_slope": float(c[1]),
                  "mean_abs_logit": float(np.abs(z).mean()), "n": len(y)}
        print(f"  {n:24s} acc={acc:.4f} ll={ll:.5f} slope={c[1]:.3f} |z|={np.abs(z).mean():.3f}",
              flush=True)
    with open(os.path.join(HERE, "results", "split_heldout.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--only", default="")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    {"stats": cmd_stats, "train": cmd_train, "heldout": cmd_heldout}[a.cmd](a)
