"""
Q5 — PARTIAL-REFRESH ABLATION on the d2c_cumprev value-function checkpoints.

Inference-time stats swap, NO retraining: the enriched WP models
(models/d2c_cumprev_s{42,123,777}.pt) are evaluated on the strictly-future
test builds with their enriched-feature aggregates sourced per signal class:

  all_refresh   hero family + pair/comp both cumulative-through-previous-build
                (= the existing d2c_cumprev eval; positive control ~57.08)
  none_refresh  everything cumulative-through-TRAIN_CUTOFF_BUILD
                (cutoff-frozen stats; negative control ~56.2)
  hero_only     ONLY the hero-WR family refreshed (hero WR / pick+ban rate /
                hero-map WR <- cumprev), pairwise + comp-WR cutoff-frozen
  pair_only     ONLY pairwise/composition refreshed (counter / synergy /
                comp-WR <- cumprev), hero family cutoff-frozen

Signal classes follow the W3(d) hybrid split (w3_build_hybrid.py):
  hero family  = PatchStats.{hero_wr, hero_meta, hero_map_wr}
  pair/comp    = PatchStats.{pairwise, comp_data}
Composition is attribute-level: the StatsCache getters only read these five
dicts, so a hybrid PatchStats is exact.

NOTE (inherent to the ablation): counter/synergy features are NORMALIZED by
per-hero WRs (sweep_enriched_wp._normalized_counter/_synergy), so in
pair_only the refreshed pairwise WRs are normalized against frozen hero WRs
(and vice versa in hero_only). That is the correct reading of "only this
signal class refreshed".

Controls are cross-checked against the cached feature passes
(feature_cache/features_cumulative_prev.npz / features_cutoff.npz): the
recomputed all_refresh / none_refresh accuracies must match the cache-based
evaluation of the same checkpoints to < 0.03 pp before cells 3-4 are trusted.

Outputs: results/partial_refresh.json, results/PARTIAL_REFRESH.md.
Existing results files are not touched.

Usage:
    CUDA_VISIBLE_DEVICES=2 python drift2026/phase_q5_partial_refresh.py
"""
import os
import sys
import json
import time
import argparse
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drift2026 import common

common.setup()

import numpy as np
import torch

NUM_WORKERS = min(mp.cpu_count(), 32)
CHUNK = 4000

# cell -> (hero_family_source, pair_comp_source); "refresh" = cumulative
# through the row's previous build (cumprev), "cutoff" = cumulative through
# TRAIN_CUTOFF_BUILD.
CELLS = {
    "all_refresh":  ("refresh", "refresh"),
    "none_refresh": ("cutoff", "cutoff"),
    "hero_only":    ("refresh", "cutoff"),
    "pair_only":    ("cutoff", "refresh"),
}
CELL_ORDER = ["all_refresh", "none_refresh", "hero_only", "pair_only"]
SEEDS = common.SEEDS


def compose(hero_src, pair_src):
    """Hybrid PatchStats: hero-WR family from one source, pair/comp from
    another (W3d per-signal split, at attribute level)."""
    return common.PatchStats(hero_src.hero_wr, hero_src.hero_meta,
                             hero_src.hero_map_wr,
                             pair_src.pairwise, pair_src.comp_data)


# ── feature extraction workers ──

_WORKER_STATS = {}


def _get_cum(key):
    st = _WORKER_STATS.get(key)
    if st is None:
        st = common.load_patch_stats("cumulative", key)
        _WORKER_STATS[key] = st
        while len(_WORKER_STATS) > 4:
            _WORKER_STATS.pop(next(iter(_WORKER_STATS)))
    return st


def feat_chunk(args):
    """(slim_rows, refresh_key, cutoff_key) -> per-cell aligned feature arrays
    (2 rows per replay: original + team swap, matching build_drift_features)."""
    from sweep_enriched_wp import extract_features, _swap_features, FEATURE_GROUPS
    rows, refresh_key, cutoff_key = args
    refreshed = _get_cum(refresh_key)
    cut = _get_cum(cutoff_key)
    stats_by_cell = {
        cell: compose(refreshed if h == "refresh" else cut,
                      refreshed if p == "refresh" else cut)
        for cell, (h, p) in CELLS.items()}
    all_mask = [True] * len(FEATURE_GROUPS)
    out = {cell: ([], []) for cell in CELLS}      # bases, enricheds
    labels, bidx, failed = [], [], 0
    for d in rows:
        try:
            feats = {}
            for cell, stats in stats_by_cell.items():
                feats[cell] = extract_features(d, stats, all_mask)
        except Exception:
            failed += 1
            continue
        y = float(d["winner"] == 0)
        labels += [y, 1.0 - y]
        bidx += [d["build_idx"]] * 2
        for cell, (b, e) in feats.items():
            bs, es = _swap_features(b, e)
            out[cell][0].extend([b, bs])
            out[cell][1].extend([e, es])
    packed = {cell: (np.asarray(bs, dtype=np.float32),
                     np.asarray(es, dtype=np.float32))
              for cell, (bs, es) in out.items()}
    return (packed, np.asarray(labels, dtype=np.float32),
            np.asarray(bidx, dtype=np.int16), failed)


# ── evaluation helpers ──

@torch.no_grad()
def batch_acc(model, X, y):
    preds = []
    for i in range(0, len(X), 65536):
        preds.append(model(X[i:i + 65536]))
    pred = torch.cat(preds)
    return ((pred > 0.5).float() == y).float().mean().item() * 100


def load_checkpoint(seed, device):
    from sweep_enriched_wp import WinProbEnrichedModel
    path = os.path.join(common.MODELS_DIR, f"d2c_cumprev_s{seed}.pt")
    ck = torch.load(path, map_location="cpu")
    model = WinProbEnrichedModel(ck["input_dim"], ck["arch"], ck["dropout"])
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval()


def eval_arrays(model, X, y, bidx, builds, device):
    """future / sizable / per-build accuracies for one (model, feature set)."""
    Xt = torch.tensor(X, device=device)
    yt = torch.tensor(y, device=device)
    res = {"future": round(batch_acc(model, Xt, yt), 3), "per_build": {}}
    sizable_mask = np.zeros(len(bidx), dtype=bool)
    for bi in np.unique(bidx):
        b = builds[int(bi)]
        m = bidx == bi
        res["per_build"][b] = round(batch_acc(model, Xt[m], yt[m]), 3)
        if b in common.SIZABLE_TEST_BUILDS:
            sizable_mask |= m
    res["sizable"] = round(batch_acc(model, Xt[sizable_mask], yt[sizable_mask]), 3)
    del Xt, yt
    return res


def cache_control_eval(pass_name, models_by_seed, cutoff, builds, device):
    """Evaluate the checkpoints on the TEST rows of an existing feature cache
    (exact reproduction path for the controls)."""
    from drift2026.train_drift_wp import enriched_cols
    z = np.load(os.path.join(common.CACHE_DIR, f"features_{pass_name}.npz"))
    bidx = z["build_idx"].astype(np.int64)
    test = bidx > cutoff
    cols = enriched_cols()
    X = np.concatenate([z["bases"][test], z["enricheds"][test][:, cols]], axis=1)
    y = z["labels"][test]
    tb = bidx[test].astype(np.int16)
    out = {}
    for seed, model in models_by_seed.items():
        out[seed] = eval_arrays(model, X, y, tb, builds, device)
    del X, y, z
    return out


def sanity_for(model, stats, device):
    from drift2026.train_drift_wp import sanity_suite, enriched_cols
    from experiment_synthetic_augmentation import make_eval_fn
    return sanity_suite(make_eval_fn(model, enriched_cols(), stats, device))


def mean_sd(vals):
    a = np.asarray(vals, dtype=np.float64)
    return round(float(a.mean()), 3), round(float(a.std(ddof=1)), 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-sanity", action="store_true")
    ap.add_argument("--skip-cache-controls", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})")

    idx = common.load_patch_index()
    builds = idx["builds"]
    cutoff = common.cutoff_idx(builds)

    models_by_seed = {s: load_checkpoint(s, device) for s in SEEDS}
    print(f"loaded {len(models_by_seed)} checkpoints (d2c_cumprev)")

    # ── control path A: cache-based eval (exact plumbing of D2) ──
    cache_controls = {}
    if not args.skip_cache_controls:
        for pass_name, cell in [("cumulative_prev", "all_refresh"),
                                ("cutoff", "none_refresh")]:
            print(f"[cache control] features_{pass_name}.npz ...", flush=True)
            cache_controls[cell] = cache_control_eval(
                pass_name, models_by_seed, cutoff, builds, device)
            for s in SEEDS:
                print(f"  {cell} (cache) s{s}: future="
                      f"{cache_controls[cell][s]['future']:.3f}")

    # ── recompute path: test rows, 4 cells, hybrid stats ──
    rows, _ = common.load_data_with_patches()
    slim_keys = ("team0_heroes", "team1_heroes", "game_map", "skill_tier",
                 "winner", "avg_mmr", "replay_id", "build_idx", "date_days")
    by_build = {}
    n_all = 0
    for r in rows:
        n_all += 1
        if r["build_idx"] > cutoff:
            by_build.setdefault(r["build_idx"], []).append(
                {k: r[k] for k in slim_keys})
    present_all = sorted({r["build_idx"] for r in rows})
    del rows
    test_present = sorted(by_build)
    n_test_replays = sum(len(v) for v in by_build.values())
    print(f"{n_test_replays:,} test replays across {len(test_present)} builds "
          f"(corpus {n_all:,} rows, {len(present_all)} builds)")

    # refresh key per test build = cumulative through the PREVIOUS present
    # build (cumprev convention from build_drift_features)
    pos_of = {bi: i for i, bi in enumerate(present_all)}
    refresh_key = {bi: builds[present_all[pos_of[bi] - 1]] for bi in test_present}
    for bi in test_present:
        print(f"  {builds[bi]}: refresh <- cumulative through {refresh_key[bi]}, "
              f"frozen <- cumulative through {common.TRAIN_CUTOFF_BUILD}")

    tasks = []
    for bi in test_present:
        rws = by_build[bi]
        for i in range(0, len(rws), CHUNK):
            tasks.append((rws[i:i + CHUNK], refresh_key[bi],
                          common.TRAIN_CUTOFF_BUILD))
    print(f"extracting features: {len(tasks)} chunks x 4 cells, "
          f"{NUM_WORKERS} workers", flush=True)

    parts = {cell: ([], []) for cell in CELLS}
    labels_parts, bidx_parts, failed = [], [], 0
    te = time.time()
    with mp.Pool(NUM_WORKERS) as pool:
        for i, (packed, lab, bx, f) in enumerate(pool.imap(feat_chunk, tasks)):
            for cell, (b, e) in packed.items():
                parts[cell][0].append(b)
                parts[cell][1].append(e)
            labels_parts.append(lab)
            bidx_parts.append(bx)
            failed += f
            if (i + 1) % 10 == 0:
                print(f"  chunk {i+1}/{len(tasks)} ({time.time()-te:.0f}s)",
                      flush=True)
    labels = np.concatenate(labels_parts)
    bidx = np.concatenate(bidx_parts)
    print(f"extracted {len(labels):,} rows ({failed} replays failed) "
          f"in {(time.time()-te)/60:.1f} min")

    from drift2026.train_drift_wp import enriched_cols
    cols = enriched_cols()

    results = {}
    for cell in CELL_ORDER:
        bases = np.concatenate(parts[cell][0])
        enr = np.concatenate(parts[cell][1])
        X = np.concatenate([bases, enr[:, cols]], axis=1)
        del bases, enr
        parts[cell] = None
        results[cell] = {"seeds": {}}
        for s in SEEDS:
            r = eval_arrays(models_by_seed[s], X, labels, bidx, builds, device)
            results[cell]["seeds"][s] = r
            print(f"{cell} s{s}: future={r['future']:.3f} "
                  f"sizable={r['sizable']:.3f}", flush=True)
        del X
        fu = [results[cell]["seeds"][s]["future"] for s in SEEDS]
        sz = [results[cell]["seeds"][s]["sizable"] for s in SEEDS]
        results[cell]["future_mean"], results[cell]["future_sd"] = mean_sd(fu)
        results[cell]["sizable_mean"], results[cell]["sizable_sd"] = mean_sd(sz)

    # ── control verification ──
    control_check = {"pass": True, "tolerance_pp": 0.03, "detail": {}}
    for cell in ("all_refresh", "none_refresh"):
        if cell not in cache_controls:
            continue
        for s in SEEDS:
            a = results[cell]["seeds"][s]["future"]
            b = cache_controls[cell][s]["future"]
            d = round(abs(a - b), 3)
            control_check["detail"][f"{cell}_s{s}"] = {
                "recomputed": a, "cache": b, "abs_diff": d}
            if d > control_check["tolerance_pp"]:
                control_check["pass"] = False
    print(f"control check: {'PASS' if control_check['pass'] else 'FAIL'} "
          f"{json.dumps(control_check['detail'])}")

    # ── sanity suite (deployment-time stats per cell) ──
    if not args.skip_sanity:
        builds_255 = [b for b in builds if b.startswith("2.55")]
        deploy_refresh = common.load_patch_stats("cumulative", builds_255[-2])
        deploy_cut = common.load_patch_stats("cumulative",
                                             common.TRAIN_CUTOFF_BUILD)
        for cell in CELL_ORDER:
            h, p = CELLS[cell]
            stats = compose(deploy_refresh if h == "refresh" else deploy_cut,
                            deploy_refresh if p == "refresh" else deploy_cut)
            for s in SEEDS:
                sn = sanity_for(models_by_seed[s], stats, device)
                results[cell]["seeds"][s]["sanity"] = sn
                print(f"{cell} s{s}: sanity {sn['passed_28']}/28 "
                      f"({sn['passed_21']}/21)", flush=True)
            p28 = [results[cell]["seeds"][s]["sanity"]["passed_28"] for s in SEEDS]
            results[cell]["sanity28_mean"], results[cell]["sanity28_sd"] = mean_sd(p28)

    payload = {
        "_meta": {
            "phase": "Q5 partial-refresh ablation",
            "checkpoints": [f"d2c_cumprev_s{s}" for s in SEEDS],
            "protocol": "inference-time stats swap, no retraining; "
                        "test = 7 builds after " + common.TRAIN_CUTOFF_BUILD,
            "signal_split": {
                "hero_family": ["hero_wr", "hero_meta(pick/ban)", "hero_map_wr"],
                "pair_comp": ["pairwise with/against", "comp_data"]},
            "n_test_replays": n_test_replays,
            "n_test_rows": int(len(labels)),
            "failed_replays": failed,
            "minutes": round((time.time() - t0) / 60, 1),
        },
        "cells": results,
        "cache_controls": {c: {s: v for s, v in d.items()}
                           for c, d in cache_controls.items()},
        "control_check": control_check,
    }
    common.write_json(os.path.join(common.RESULTS_DIR, "partial_refresh.json"),
                      payload)
    print(f"done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
