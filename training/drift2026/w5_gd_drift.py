"""
W5 — GD (opponent/behavior model) drift arm. Max-identified gap: every drift
experiment so far varied only the WP value function while the GD behavioral
model stayed frozen (rerun2026 full-snapshot GD, which has seen the future
period). But pick/ban behavior drifts too (W3b: pick/ban changepoints track
the community's ADAPTATION, recall 0.68) — so the opponent model itself may
go stale. Same W2 temporal cutoff (train <= 2.55.14.95918, test = last 7
builds).

(a) Behavioral prediction under drift: train GD_cutoff (GenericDraftModel
    protocol of rerun2026 reused by import — train_single_model unchanged,
    replay-level 98/2 split, MODEL_VARIANTS 0..1 = 2 seeds) on data <= cutoff,
    plus a GD_win6 recency arm (last 6 builds through the cutoff, same
    window definition as d2b_win6) and a GD_win30d WITHIN-BUILD arm (rolling
    ~1 month of play time ending at the cutoff, calendar-based via the
    sidecar's date_days — measures whether behavioral meta drift WITHIN a
    minor patch is significant). Compare next-pick top-1/top-5 accuracy on
    FUTURE-build drafts vs the frozen full-snapshot GD (upper bound — it has
    seen the future) and the W4 future-GD (leaky oracle reference, trained ON
    the future period).
(b) Policy-level: rerun the W2b greedy eval for the best WP arm (d2c_cumprev)
    with GD_cutoff opponents/completions instead of the frozen future-seeing
    pool (eval_policy_w2b.py --gd-pool cutoff). d2b_allhist + GD reference
    rerun too, for spread context.
(c) MCTS arm, ONLY IF (a)/(b) show material effects (trigger below): W4-style
    200 sims / 300K episodes, 3 seeds, cumprev WP VF (w4_vf_d2c_cumprev.pt),
    training opponents = GD_cutoff instead of frozen GD; benchmark vs
    future-GD opponents (identical protocol to W4, so the existing
    w4_d2c_cumprev_s* runs ARE the frozen-opponent arm).

Trigger for (c): material = (frozen top-1 minus GD_cutoff top-1 on the future
set >= 1.0 pp) OR (|delta healer| or |delta degen| >= 3 pp for d2c_cumprev
between opponent pools — the W2b/W4 3 pp convention).

Usage:
  nohup python3 -u drift2026/w5_gd_drift.py > drift2026/logs/w5_gd_drift.log 2>&1 &
  python3 drift2026/w5_gd_drift.py --dry-run
  python3 drift2026/w5_gd_drift.py --stage train-one --pool cutoff --variant 0
  python3 drift2026/w5_gd_drift.py --stage summarize   # re-render md only
Monitor:
  tail -f drift2026/logs/w5_gd_drift.log drift2026/logs/w5_*.log
"""
import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drift2026 import common

common.setup()

import numpy as np

from rerun2026 import common as rerun_common

ME = os.path.abspath(__file__)
EVAL_W2B = os.path.join(common.DRIFT_DIR, "eval_policy_w2b.py")
RERUN_MODELS = os.path.join(common.TRAINING_DIR, "rerun2026", "models")
GD_FUTURE_DIR = os.path.join(common.MODELS_DIR, "gd_future")
MCTS_RUNS_DIR = os.path.join(common.DRIFT_DIR, "mcts_runs")
WORKER = os.path.join(common.TRAINING_DIR, "train_mcts_worker.py")

EVAL_JSON = os.path.join(common.RESULTS_DIR, "w5_gd_eval.json")
TRIGGER_JSON = os.path.join(common.RESULTS_DIR, "w5_gd_trigger.json")
MCTS_JSON = os.path.join(common.RESULTS_DIR, "w5_gd_mcts.json")
SUMMARY_MD = os.path.join(common.RESULTS_DIR, "W3_GD_DRIFT.md")

WIN6_BUILDS = 6            # same window as the d2b_win6 regime (pos > cut-6)
WIN_DAYS = 30              # within-build arm: rolling calendar window (days)
GD_VARIANTS = [0, 1]       # rerun2026 MODEL_VARIANTS 0/1 = seeds 42/123
TRAIN_POOLS = ["cutoff", "win6", "win30d"]
W2B_CELLS = ["d2c_cumprev", "d2b_allhist"]
MCTS_SEEDS = 3
EVAL_SLAB = 131072

CACHES = {n: os.path.join(common.CACHE_DIR, f"gd_w5_{n}")
          for n in ["cutoff_train", "cutoff_val", "win6_train", "win6_val",
                    "win30d_train", "win30d_val", "future_test"]}


def gd_path(pool, i):
    if pool == "frozen":
        return os.path.join(RERUN_MODELS, f"generic_draft_{i}.pt")
    if pool == "future_oracle":
        return os.path.join(GD_FUTURE_DIR, f"generic_draft_{i}.pt")
    return os.path.join(common.MODELS_DIR, f"gd_{pool}", f"generic_draft_{i}.pt")


POOLS = {  # name -> (dir pool key, variant indices)
    "frozen": ("frozen", list(range(5))),
    "cutoff": ("cutoff", GD_VARIANTS),
    "win6": ("win6", GD_VARIANTS),
    "win30d": ("win30d", GD_VARIANTS),
    "future_oracle": ("future_oracle", list(range(5))),
}


# ── stage: caches ─────────────────────────────────────────────────────────

def _gd_rows_chunk(chunk):
    """GD next-pick samples + build index per sample (top-level for mp)."""
    import numpy as np
    from train_generic_draft import replay_to_training_samples
    xs, ys, bs = [], [], []
    for r in chunk:
        for x, y, _m in replay_to_training_samples(r):
            xs.append(x)
            ys.append(y)
            bs.append(r["build_idx"])
    if not xs:
        return (np.zeros((0, 289), np.float32), np.zeros(0, np.int64),
                np.zeros(0, np.int64))
    return (np.array(xs, np.float32), np.array(ys, np.int64),
            np.array(bs, np.int64))


def stage_cache():
    todo = [n for n, d in CACHES.items()
            if not os.path.exists(os.path.join(d, "meta.json"))]
    if not todo:
        print("caches: all present, skipping")
        return
    from shared import split_data
    from rerun2026.phase0_features import _pool_stream, _chunks
    rows, builds = common.load_data_with_patches()
    cut = common.cutoff_idx(builds)
    past = [r for r in rows if r["build_idx"] <= cut]
    win6 = [r for r in past if r["build_idx"] > cut - WIN6_BUILDS]
    last_day = max(r["date_days"] for r in past)
    win30d = [r for r in past if r["date_days"] > last_day - WIN_DAYS]
    future = [r for r in rows if r["build_idx"] > cut]
    print(f"pools: past={len(past):,} win6={len(win6):,} "
          f"win30d={len(win30d):,} (last {WIN_DAYS}d of train period) "
          f"future={len(future):,}")
    plans = {}
    for name, pool in (("cutoff", past), ("win6", win6), ("win30d", win30d)):
        tr, va = split_data(pool, test_frac=0.02, seed=42)
        plans[f"{name}_train"], plans[f"{name}_val"] = tr, va
    plans["future_test"] = future
    for name in todo:
        data = plans[name]
        print(f"building {CACHES[name]} ({len(data):,} replays)")
        rerun_common.memmap_write(
            CACHES[name], _pool_stream(_gd_rows_chunk, _chunks(data), name),
            289, [("actions", "int64"), ("builds", "int64")])


# ── stage: GD training (pool job; rerun2026 protocol reused unchanged) ────

def stage_train_one(args):
    import torch
    import train_generic_draft as tgd
    from rerun2026.train_jobs import MemmapGDDataset
    out_dir = os.path.join(common.MODELS_DIR, f"gd_{args.pool}")
    os.makedirs(out_dir, exist_ok=True)
    tgd.__file__ = os.path.join(out_dir, "x.py")   # redirect all saves
    train_ds = MemmapGDDataset(CACHES[f"{args.pool}_train"])
    val_ds = MemmapGDDataset(CACHES[f"{args.pool}_val"])
    print(f"gd_{args.pool} variant {args.variant}: "
          f"{len(train_ds):,} train / {len(val_ds):,} val samples")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss = tgd.train_single_model(args.variant, tgd.MODEL_VARIANTS[args.variant],
                                  train_ds, val_ds, device)
    print(f"gd_{args.pool} variant {args.variant}: best val loss {loss:.4f}")


def train_jobs():
    jobs = []
    for pool in TRAIN_POOLS:
        for v in GD_VARIANTS:
            jobs.append(common.Job(
                f"w5_gd_{pool}_v{v}",
                [ME, "--stage", "train-one", "--pool", pool, "--variant", str(v)],
                [gd_path(pool, v)]))
    return jobs


# ── stage: (a) next-pick accuracy on future drafts ────────────────────────

def _eval_model(path, mm, device):
    import torch
    from train_generic_draft import GenericDraftModel
    model = GenericDraftModel()
    model.load_state_dict(torch.load(path, weights_only=True,
                                     map_location="cpu"))
    model.to(device).eval()
    X, y, b = mm["states"], mm["actions"], mm["builds"]
    n = len(y)
    top1 = np.zeros(n, bool)
    top5 = np.zeros(n, bool)
    is_pick = np.zeros(n, bool)
    with torch.no_grad():
        for i in range(0, n, EVAL_SLAB):
            x = torch.from_numpy(np.asarray(X[i:i + EVAL_SLAB],
                                            np.float32)).to(device)
            yy = torch.from_numpy(np.asarray(y[i:i + EVAL_SLAB])).to(device)
            mask = 1.0 - x[:, :270].reshape(len(x), 3, 90).sum(1).clamp(0, 1)
            logits = model(x, mask)
            top1[i:i + len(x)] = (logits.argmax(1) == yy).cpu().numpy()
            t5 = logits.topk(5, 1).indices
            top5[i:i + len(x)] = (t5 == yy.unsqueeze(1)).any(1).cpu().numpy()
            is_pick[i:i + len(x)] = (x[:, 288] > 0.5).cpu().numpy()
    builds_np = np.asarray(b)
    per_build = {}
    for bi in np.unique(builds_np):
        m = builds_np == bi
        per_build[int(bi)] = [round(float(top1[m].mean() * 100), 2),
                              int(m.sum())]
    return {
        "top1": round(float(top1.mean() * 100), 2),
        "top5": round(float(top5.mean() * 100), 2),
        "top1_pick": round(float(top1[is_pick].mean() * 100), 2),
        "top1_ban": round(float(top1[~is_pick].mean() * 100), 2),
        "per_build": per_build,
    }


def stage_eval(force=False):
    if os.path.exists(EVAL_JSON) and not force:
        print(f"eval: {EVAL_JSON} exists, skipping")
        with open(EVAL_JSON) as f:
            return json.load(f)
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    future = rerun_common.memmap_open(CACHES["future_test"])
    val = rerun_common.memmap_open(CACHES["cutoff_val"])
    out = {"pools": {}, "n_future_samples": int(future["meta"]["n"]),
           "n_val_samples": int(val["meta"]["n"])}
    for name, (key, variants) in POOLS.items():
        per = []
        for v in variants:
            p = gd_path(key, v)
            r = {"variant": v,
                 "future": _eval_model(p, future, device),
                 "train_val": _eval_model(p, val, device)}
            per.append(r)
            print(f"  {name} v{v}: future top1={r['future']['top1']} "
                  f"top5={r['future']['top5']} | val top1="
                  f"{r['train_val']['top1']}", flush=True)
        agg = {
            "n_models": len(per),
            "future_top1": round(float(np.mean([r["future"]["top1"]
                                                for r in per])), 2),
            "future_top1_sd": round(float(np.std([r["future"]["top1"]
                                                  for r in per])), 3),
            "future_top5": round(float(np.mean([r["future"]["top5"]
                                                for r in per])), 2),
            "future_top1_pick": round(float(np.mean(
                [r["future"]["top1_pick"] for r in per])), 2),
            "future_top1_ban": round(float(np.mean(
                [r["future"]["top1_ban"] for r in per])), 2),
            "val_top1": round(float(np.mean([r["train_val"]["top1"]
                                             for r in per])), 2),
            "per_build_top1": {
                bi: round(float(np.mean([r["future"]["per_build"][bi][0]
                                         for r in per])), 2)
                for bi in per[0]["future"]["per_build"]},
            "per_build_n": {bi: per[0]["future"]["per_build"][bi][1]
                            for bi in per[0]["future"]["per_build"]},
            "models": per,
        }
        out["pools"][name] = agg
    common.write_json(EVAL_JSON, out)
    return out


# ── stage: (b) W2b policy eval with GD_cutoff opponents ───────────────────

def w2b_out(cell, seed):
    return os.path.join(common.RESULTS_DIR, "w2b",
                        f"{cell}_s{seed}_gdcutoff.json")


def w2b_jobs():
    jobs = []
    for cell in W2B_CELLS:
        for seed in common.SEEDS:
            jobs.append(common.Job(
                f"w5_w2b_{cell}_s{seed}_gdcut",
                [EVAL_W2B, "--cell", cell, "--seed", str(seed),
                 "--drafts", "500", "--gd-pool", "cutoff"],
                [w2b_out(cell, seed)], weight="light"))
    jobs.append(common.Job(
        "w5_w2b_gd_s42_gdcut",
        [EVAL_W2B, "--cell", "gd", "--seed", "42", "--drafts", "1500",
         "--gd-pool", "cutoff"],
        [w2b_out("gd", 42)], weight="light"))
    return jobs


def load_w2b(cell, pool):
    """Mean/sd over seeds of the W2b metrics for one cell under one pool."""
    rows = []
    seeds = [42] if cell == "gd" else common.SEEDS
    for seed in seeds:
        suffix = "" if pool == "frozen" else "_gdcutoff"
        p = os.path.join(common.RESULTS_DIR, "w2b",
                         f"{cell}_s{seed}{suffix}.json")
        if os.path.exists(p):
            with open(p) as f:
                rows.append(json.load(f))
    if not rows:
        return None
    keys = ["healer_rate", "degen_rate", "counter_future", "synergy_future"]
    out = {k: (round(float(np.mean([r[k] for r in rows])), 3),
               round(float(np.std([r[k] for r in rows])), 3)) for k in keys}
    out["n"] = len(rows)
    if "hero_entropy" in rows[0]:
        out["hero_entropy"] = (round(float(np.mean([r["hero_entropy"]
                                                    for r in rows])), 3),
                               round(float(np.std([r["hero_entropy"]
                                                   for r in rows])), 3))
    return out


# ── trigger + (c) MCTS arm ────────────────────────────────────────────────

def compute_trigger(ev):
    beh_gap = (ev["pools"]["frozen"]["future_top1"]
               - ev["pools"]["cutoff"]["future_top1"])
    recency = (ev["pools"]["win6"]["future_top1"]
               - ev["pools"]["cutoff"]["future_top1"])
    within = (ev["pools"]["win30d"]["future_top1"]
              - ev["pools"]["cutoff"]["future_top1"])
    fr = load_w2b("d2c_cumprev", "frozen")
    cu = load_w2b("d2c_cumprev", "cutoff")
    d_healer = d_degen = None
    if fr and cu:
        d_healer = round(cu["healer_rate"][0] - fr["healer_rate"][0], 2)
        d_degen = round(cu["degen_rate"][0] - fr["degen_rate"][0], 2)
    material_a = beh_gap >= 1.0
    material_b = (d_healer is not None
                  and (abs(d_healer) >= 3.0 or abs(d_degen) >= 3.0))
    trig = {
        "behavior_gap_pp": round(beh_gap, 2),
        "recency_effect_pp": round(recency, 2),
        "within_build_effect_pp": round(within, 2),
        "d2c_cumprev_delta_healer_pp": d_healer,
        "d2c_cumprev_delta_degen_pp": d_degen,
        "material_a_accuracy": bool(material_a),
        "material_b_policy": bool(material_b),
        "mcts_arm_fires": bool(material_a or material_b),
        "criteria": "a: frozen-cutoff future top1 >= 1.0pp; "
                    "b: |d healer| or |d degen| >= 3pp (d2c_cumprev)",
    }
    common.write_json(TRIGGER_JSON, trig)
    return trig


def mcts_jobs():
    from drift2026 import phase_w4_mcts as w4
    stats_build = w4.cell_stats_build("d2c_cumprev")
    vf = os.path.join(common.MODELS_DIR, "w4_vf_d2c_cumprev.pt")
    assert os.path.exists(vf), vf
    exclude = w4.build_exclude_ids()
    jobs = []
    for seed in range(MCTS_SEEDS):
        name = f"w5_cumprev_gdcut_s{seed}"
        save_dir = os.path.join(MCTS_RUNS_DIR, name)
        os.makedirs(save_dir, exist_ok=True)
        env = {
            "MCTS_SAVE_DIR": save_dir,
            "MCTS_WP_MODEL": "enriched_full",
            "MCTS_WP_PATH": vf,
            "MCTS_GD_PATH": gd_path("cutoff", 0),   # <- the drifted opponent
            "MCTS_NUM_EPISODES": "300000",
            "MCTS_NUM_SIMS": "200",
            "MCTS_BATCH_EPISODES": "128",
            "MCTS_FRESH": "1",
            "MCTS_POLICY_HEAD": "linear",
            "MCTS_NET_SIZE": "base",
            "MCTS_STATS_BUILD": stats_build,
            "MCTS_EXCLUDE_IDS": exclude,
            "WANDB_RUN_NAME": f"drift2026_{name}",
        }
        jobs.append(common.Job(f"w5_mcts_{name}", [WORKER],
                               [os.path.join(save_dir, "draft_policy.pt")],
                               weight="light", env=env))
    return jobs


def stage_mcts_bench():
    """W4 benchmark protocol (same code by import): kernel drafts vs
    future-GD opponents, metrics vs future-truth stats, neutral judge."""
    from drift2026 import phase_w4_mcts as w4
    results = {}
    if os.path.exists(MCTS_JSON):
        with open(MCTS_JSON) as f:
            results = json.load(f)
    todo = [s for s in range(MCTS_SEEDS)
            if f"w5_cumprev_gdcut_s{s}" not in results]
    if not todo:
        print("mcts bench: all present, skipping")
        return results
    kernel = w4.load_kernel_module()
    deps = w4.load_bench_deps(["d2c_cumprev"])
    for seed in todo:
        name = f"w5_cumprev_gdcut_s{seed}"
        ckpt = os.path.join(MCTS_RUNS_DIR, name, "draft_policy.pt")
        if not os.path.exists(ckpt):
            print(f"  missing checkpoint {name}, skipping")
            continue
        r = w4.benchmark_run("d2c_cumprev", name, ckpt, deps, kernel)
        r["arm"] = "d2c_cumprev_gdcut"
        results[name] = r
        common.write_json(MCTS_JSON, results)
        print(f"  BENCH {name}: judgeWP={r['judge_wp']:.4f} "
              f"ctr={r['counter_future']:+.3f} syn={r['synergy_future']:+.3f} "
              f"healer={r['healer']:.0f}% degen={r['degen']:.0f}%", flush=True)
    return results


# ── summary ───────────────────────────────────────────────────────────────

def _mcts_rows():
    """(frozen-opponent arm = existing w4_d2c_cumprev_s*, gdcut arm = w5)."""
    rows = {}
    w4_path = os.path.join(common.RESULTS_DIR, "w4_mcts_results.json")
    if os.path.exists(w4_path):
        with open(w4_path) as f:
            w4r = json.load(f)
        rows["frozen GD opponents (W4)"] = [
            r for r in w4r.values() if r["arm"] == "d2c_cumprev"]
    if os.path.exists(MCTS_JSON):
        with open(MCTS_JSON) as f:
            rows["GD_cutoff opponents (W5)"] = list(json.load(f).values())
    return rows


def stage_summarize():
    with open(EVAL_JSON) as f:
        ev = json.load(f)
    trig = compute_trigger(ev)
    # NOTE: sidecar build_idx indexes the FULL ordered build list (92 entries
    # incl. pre-2.55), not the 2.55-only sublist — index the full list here.
    builds = common.load_patch_index()["builds"]

    L = ["# W3/W5 — GD (opponent/behavior model) drift arm", "",
         "Gap (Max): all drift arms varied only the WP value function while "
         "the GD behavioral model stayed frozen (rerun2026 full-snapshot GD, "
         "which has seen the future period). W3b showed pick/ban changepoints "
         "track community ADAPTATION — behavior drifts too. Same W2 temporal "
         "cutoff (train <= 2.55.14.95918, test = last 7 builds).", "",
         "## (a) Next-pick prediction accuracy on FUTURE-build drafts", "",
         f"{ev['n_future_samples']:,} next-pick samples (7 post-cutoff "
         "builds); train-val = 2% replay-level slice of the <=cutoff period. "
         "GenericDraftModel protocol reused unchanged (train_single_model); "
         "cutoff/win6 = 2 variants (seeds 42/123), frozen/future = 5.", "",
         "| GD pool | trained on | n | future top-1 | future top-5 | "
         "train-val top-1 | pick top-1 | ban top-1 |",
         "|---|---|---|---|---|---|---|---|"]
    desc = {
        "frozen": "full snapshot incl. future (paper-1 pool; non-causal bound)",
        "cutoff": "all builds <= cutoff (deployable)",
        "win6": f"last {WIN6_BUILDS} builds <= cutoff (deployable, recency)",
        "win30d": f"last {WIN_DAYS} calendar days <= cutoff (within-build "
                  "rolling window)",
        "future_oracle": "post-cutoff builds only (leaky oracle ref)",
    }
    for name in ["frozen", "cutoff", "win6", "win30d", "future_oracle"]:
        a = ev["pools"][name]
        L.append(f"| {name} | {desc[name]} | {a['n_models']} | "
                 f"{a['future_top1']:.2f} ± {a['future_top1_sd']} | "
                 f"{a['future_top5']:.2f} | {a['val_top1']:.2f} | "
                 f"{a['future_top1_pick']:.2f} | {a['future_top1_ban']:.2f} |")
    L += ["",
          f"- Behavioral-drift gap (frozen - cutoff, future top-1): "
          f"**{trig['behavior_gap_pp']:+.2f} pp**",
          f"- Recency effect (win6 - cutoff, future top-1): "
          f"**{trig['recency_effect_pp']:+.2f} pp**",
          f"- Within-build effect (win{WIN_DAYS}d - cutoff, future top-1): "
          f"**{trig['within_build_effect_pp']:+.2f} pp**", "",
          "Per-build future top-1 (columns = pools):", "",
          "| build | n samples | frozen | cutoff | win6 | win30d | "
          "future_oracle |",
          "|---|---|---|---|---|---|---|"]
    pb = {n: ev["pools"][n]["per_build_top1"] for n in ev["pools"]}
    pbn = ev["pools"]["cutoff"]["per_build_n"]
    for bi in sorted(pbn, key=int):
        L.append(f"| {builds[int(bi)]} | {pbn[bi]:,} | "
                 + " | ".join(f"{pb[n].get(bi, float('nan')):.2f}"
                              for n in ["frozen", "cutoff", "win6", "win30d",
                                        "future_oracle"]) + " |")

    L += ["", "## (b) Policy-level: W2b greedy eval with GD_cutoff "
          "opponents/completions", "",
          "Same W2b protocol (3 seeds x 500 drafts; counter/synergy vs "
          "future-truth stats); only the GD pool changes.", "",
          "| cell | GD pool | n | healer % | degen % | counter | synergy |",
          "|---|---|---|---|---|---|---|"]
    for cell in W2B_CELLS + ["gd"]:
        for pool in ["frozen", "cutoff"]:
            r = load_w2b(cell, pool)
            if r is None:
                L.append(f"| {cell} | {pool} | - | (missing) | | | |")
                continue
            L.append(f"| {cell} | {pool} | {r['n']} | "
                     f"{r['healer_rate'][0]:.1f} ± {r['healer_rate'][1]:.1f} | "
                     f"{r['degen_rate'][0]:.1f} ± {r['degen_rate'][1]:.1f} | "
                     f"{r['counter_future'][0]:+.2f} ± {r['counter_future'][1]:.2f} | "
                     f"{r['synergy_future'][0]:+.2f} ± {r['synergy_future'][1]:.2f} |")
    if trig["d2c_cumprev_delta_healer_pp"] is not None:
        L += ["", f"d2c_cumprev deltas (cutoff - frozen pool): healer "
              f"{trig['d2c_cumprev_delta_healer_pp']:+.2f} pp, degen "
              f"{trig['d2c_cumprev_delta_degen_pp']:+.2f} pp."]

    L += ["", "## (c) MCTS arm (conditional)", "",
          f"Trigger ({trig['criteria']}):",
          f"- material (a) accuracy: {trig['material_a_accuracy']} "
          f"(gap {trig['behavior_gap_pp']:+.2f} pp)",
          f"- material (b) policy: {trig['material_b_policy']}",
          f"- **fires: {trig['mcts_arm_fires']}**", ""]
    rows = _mcts_rows()
    if trig["mcts_arm_fires"] and rows.get("GD_cutoff opponents (W5)"):
        L += ["cumprev WP VF, 200 sims / 300K episodes; training opponents "
              "differ (frozen GD vs GD_cutoff); benchmark identical to W4 "
              "(vs future-GD opponents, future-truth metrics, neutral judge).",
              "",
              "| training opponents | n | judge WP | counter | synergy | "
              "healer % | degen % | R_early | R_late |",
              "|---|---|---|---|---|---|---|---|---|"]
        for label, rs in rows.items():
            if not rs:
                continue
            def m(k):
                return float(np.mean([r[k] for r in rs]))
            def sd(k):
                return float(np.std([r[k] for r in rs]))
            L.append(f"| {label} | {len(rs)} | {m('judge_wp'):.4f} ± "
                     f"{sd('judge_wp'):.3f} | {m('counter_future'):+.3f} | "
                     f"{m('synergy_future'):+.3f} | {m('healer'):.1f} | "
                     f"{m('degen'):.1f} | {m('resil_early'):+.3f} | "
                     f"{m('resil_late'):+.3f} |")
    elif not trig["mcts_arm_fires"]:
        L.append("Trigger did NOT fire — (a)/(b) effects immaterial by the "
                 "pre-registered thresholds; MCTS arm skipped.")
    else:
        L.append("Trigger fired; MCTS runs pending (see logs/w5_mcts_*.log).")

    with open(SUMMARY_MD, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"Wrote {SUMMARY_MD}")
    return trig


# ── driver ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", default="all",
                    choices=["all", "cache", "train-one", "eval", "w2b",
                             "mcts", "summarize"])
    ap.add_argument("--pool", default="cutoff", choices=TRAIN_POOLS)
    ap.add_argument("--variant", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force-eval", action="store_true")
    args = ap.parse_args()

    if args.stage == "train-one":
        stage_train_one(args)
        return
    if args.stage == "cache":
        stage_cache()
        return
    if args.stage == "eval":
        stage_eval(force=args.force_eval)
        return
    if args.stage == "summarize":
        stage_summarize()
        return

    if args.dry_run:
        for j in train_jobs() + w2b_jobs():
            print(f"  JOB {j.name} -> {j.outputs[0]}")
        print(f"  then: eval (a), trigger, conditional MCTS x{MCTS_SEEDS}, "
              f"summarize -> {SUMMARY_MD}")
        return

    t0 = time.time()
    if args.stage in ("all",):
        stage_cache()
        if common.run_pool(train_jobs()):
            sys.exit(1)
        ev = stage_eval()
        if common.run_pool(w2b_jobs()):
            sys.exit(1)
        trig = compute_trigger(ev)
        print(f"TRIGGER: {json.dumps(trig, indent=2)}")
        if trig["mcts_arm_fires"] or args.stage == "mcts":
            if common.run_pool(mcts_jobs()):
                sys.exit(1)
            stage_mcts_bench()
        stage_summarize()
    elif args.stage == "w2b":
        if common.run_pool(w2b_jobs()):
            sys.exit(1)
    elif args.stage == "mcts":
        if common.run_pool(mcts_jobs()):
            sys.exit(1)
        stage_mcts_bench()
        stage_summarize()
    print(f"w5_gd_drift done in {(time.time() - t0) / 3600:.1f} h")


if __name__ == "__main__":
    main()
