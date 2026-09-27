"""
W4: the MCTS arm of the drift paper — retrain MCTS draft policies under the
BEST vs WORST drift-regime value functions and benchmark them against
future-build opponents. Runs unconditionally (2026-07-07 decision); the W2b
trigger only matters for the paper's framing.

Protocol = rerun2026 phase4 "light" cell (200 sims, 300K episodes, linear
head, base net, MCTS_BATCH_EPISODES=128), 5 seeds per arm, with three
drift-specific changes:
  1. The kernel's WP value function is the wave-1 drift checkpoint of the
     chosen regime cell (best seed by future acc), exported as a plain 283-d
     WinProbEnrichedModel state_dict (patch-embedding cells are baked to the
     cutoff build's embedding — exact first-layer fold, verified numerically).
  2. The kernel's enriched-feature lookup tables use the ARM'S DEPLOYMENT
     STATS (MCTS_STATS_BUILD override in train_mcts_worker.py: cumulative
     through the cutoff for d2b_* cells; through the last completed build for
     d2c_cumprev), never the frozen end-of-time snapshot.
  3. Value-head pretraining excludes ALL post-cutoff replays
     (w4_exclude_ids.json = rerun2026 pre-2.55 exclusions + future builds).

Benchmark (after training, same driver): 200 kernel drafts per checkpoint vs
FUTURE-GD opponents (5 GenericDraft models retrained on post-cutoff replays
only — the future meta), healer/degen + counter/synergy/resilience scored
against future-period ground-truth stats, terminal WP judged by the neutral
rerun2026 wp_enriched_256 (trained on the full corpus incl. the future
period, identical for both arms).

Usage:
  nohup python3 -u drift2026/phase_w4_mcts.py > drift2026/logs/phase_w4.log 2>&1 &
  python3 drift2026/phase_w4_mcts.py --dry-run
  python3 drift2026/phase_w4_mcts.py --stage gd-one --variant 0   # (pool job)
Monitor:
  tail -f drift2026/logs/phase_w4.log drift2026/logs/w4_mcts_*.log
"""
import os
import sys
import json
import gzip
import time
import pickle
import random
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drift2026 import common

common.setup()
sys.path.insert(0, os.path.join(common.TRAINING_DIR, "cuda_mcts"))

import numpy as np
import torch

RERUN_DIR = os.path.join(common.TRAINING_DIR, "rerun2026")
RERUN_MODELS = os.path.join(RERUN_DIR, "models")
WORKER = os.path.join(common.TRAINING_DIR, "train_mcts_worker.py")
MCTS_RUNS_DIR = os.path.join(common.DRIFT_DIR, "mcts_runs")
GD_FUTURE_DIR = os.path.join(common.MODELS_DIR, "gd_future")
EXCLUDE_PATH = os.path.join(common.DRIFT_DIR, "w4_exclude_ids.json")
RESULTS_PATH = os.path.join(common.RESULTS_DIR, "w4_mcts_results.json")

W2B_CELLS = ["d2b_allhist", "d2b_win6", "d2b_decay90", "d2b_embed",
             "d2c_cumprev"]
SEEDS_PER_ARM = 5
EPISODES = 300000
SIMS = 200
BENCH_DRAFTS = 200


def cell_stats_build(cell):
    builds = [b for b in common.load_patch_index()["builds"]
              if b.startswith("2.55")]
    if cell == "d2c_cumprev":
        return builds[-2]                     # last completed build
    return common.TRAIN_CUTOFF_BUILD          # cutoff-features cells


# ── arm selection & VF export ──

def wave1_future_acc(cell, seed):
    p = os.path.join(common.RESULTS_DIR, "d2", f"{cell}_s{seed}.json")
    with open(p) as f:
        return json.load(f)["test_acc_future"]


def select_arms(args):
    if args.best and args.worst:
        return args.best, args.worst
    means = {c: np.mean([wave1_future_acc(c, s) for s in common.SEEDS])
             for c in W2B_CELLS}
    best = max(means, key=means.get)
    worst = min(means, key=means.get)
    print(f"arm selection (wave-1 future acc): best={best} ({means[best]:.2f}) "
          f"worst={worst} ({means[worst]:.2f})")
    return best, worst


def export_vf(cell):
    """Plain 283-d WinProbEnrichedModel state_dict for the cell's best seed
    (embed cells: fold the fixed cutoff embedding into layer-0 bias)."""
    from sweep_enriched_wp import WinProbEnrichedModel
    out = os.path.join(common.MODELS_DIR, f"w4_vf_{cell}.pt")
    best_seed = max(common.SEEDS, key=lambda s: wave1_future_acc(cell, s))
    ck = torch.load(os.path.join(common.MODELS_DIR, f"{cell}_s{best_seed}.pt"),
                    weights_only=True, map_location="cpu")
    if not ck.get("embed"):
        sd = ck["state_dict"]
    else:
        inner = {k[len("net."):]: v.clone() for k, v in ck["state_dict"].items()
                 if k.startswith("net.")}
        e = ck["state_dict"]["embed.weight"][ck["n_patches"] - 1]
        W = inner["net.0.weight"]
        inner["net.0.weight"] = W[:, :ck["input_dim"]].contiguous()
        inner["net.0.bias"] = inner["net.0.bias"] + W[:, ck["input_dim"]:] @ e
        sd = inner
        # numeric check vs the wrapped model
        from drift2026.train_drift_wp import WPWithPatchEmbed
        wrap = WPWithPatchEmbed(ck["input_dim"], ck["n_patches"],
                                ck["embed_dim"])
        wrap.load_state_dict(ck["state_dict"])
        wrap.eval()
        flat = WinProbEnrichedModel(ck["input_dim"], list(ck["arch"]),
                                    ck["dropout"])
        flat.load_state_dict(sd)
        flat.eval()
        x = torch.randn(64, ck["input_dim"])
        p = torch.full((64,), ck["n_patches"] - 1, dtype=torch.long)
        with torch.no_grad():
            assert torch.allclose(wrap(x, p), flat(x), atol=1e-5), \
                "embed baking mismatch"
    torch.save(sd, out)
    print(f"exported VF {cell} (seed {best_seed}, "
          f"future={wave1_future_acc(cell, best_seed):.2f}) -> {out}")
    return out


def build_exclude_ids():
    """rerun2026 pre-2.55 exclusions + every post-cutoff replay_id (value-head
    pretraining must not see the future period)."""
    if os.path.exists(EXCLUDE_PATH):
        return EXCLUDE_PATH
    from rerun2026 import common as rc
    with open(rc.EXCLUDE_IDS_PATH) as f:
        ids = set(json.load(f))
    rids, bidx, _, builds = common.load_sidecar()
    cut = builds.index(common.TRAIN_CUTOFF_BUILD)
    future = rids[bidx > cut]
    ids.update(int(r) for r in future)
    with open(EXCLUDE_PATH, "w") as f:
        json.dump(sorted(ids), f)
    print(f"wrote {EXCLUDE_PATH}: {len(ids)} ids "
          f"({len(future)} post-cutoff + pre-2.55)")
    return EXCLUDE_PATH


# ── future-GD opponents ──

def load_future_rows():
    rows, builds = common.load_data_with_patches()
    cut = builds.index(common.TRAIN_CUTOFF_BUILD)
    future = [r for r in rows if r["build_idx"] > cut]
    print(f"future rows: {len(future)}")
    return future


def stage_gd_one(args):
    """Pool job: train ONE future-GD variant on post-cutoff replays."""
    import train_generic_draft as tgd
    from shared import split_data
    os.makedirs(GD_FUTURE_DIR, exist_ok=True)
    tgd.__file__ = os.path.join(GD_FUTURE_DIR, "x.py")  # redirect ckpt dir
    future = load_future_rows()
    train, test = split_data(future, test_frac=0.02, seed=42)
    train_ds = tgd.DraftDataset(train)
    test_ds = tgd.DraftDataset(test)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v = args.variant
    loss = tgd.train_single_model(v, tgd.MODEL_VARIANTS[v], train_ds, test_ds,
                                  device)
    print(f"future-GD variant {v}: best test loss {loss:.4f}")


def gd_future_path(i):
    return os.path.join(GD_FUTURE_DIR, f"generic_draft_{i}.pt")


# ── benchmark ──

def future_truth_stats():
    from drift2026.build_patch_stats import _new_cell, merge_cell
    with gzip.open(common.COUNTS_PKL, "rb") as f:
        data = pickle.load(f)
    cut = data["builds"].index(common.TRAIN_CUTOFF_BUILD)
    merged = {}
    for bidx, by_tier in data["per_build"].items():
        if bidx <= cut:
            continue
        for tier, plain in by_tier.items():
            cell = merged.get(tier)
            if cell is None:
                cell = merged[tier] = _new_cell()
            merge_cell(cell, plain)
    return common.stats_from_counts(merged, pair_min=10)


def load_kernel_module():
    import importlib.util
    so_dir = os.path.join(common.TRAINING_DIR, "cuda_mcts")
    so = [f for f in os.listdir(so_dir)
          if f.startswith("cuda_mcts_kernel") and f.endswith(".so")]
    spec = importlib.util.spec_from_file_location(
        "cuda_mcts_kernel", os.path.join(so_dir, so[0]))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_bench_deps(arms):
    """Future-GD flats, per-arm WP kernel configs (arm VF + arm LUT stats),
    neutral judge WP, future-truth metric stats."""
    from shared import (NUM_HEROES, HEROES, HERO_TO_IDX, MAPS, SKILL_TIERS,
                        HERO_ROLE_FINE, is_degenerate)
    from sweep_enriched_wp import (WinProbEnrichedModel, FEATURE_GROUPS,
                                   compute_group_indices, extract_features)
    from train_generic_draft import GenericDraftModel
    from extract_weights import (extract_gd_weights, extract_wp_weights,
                                 build_wp_net_offsets, extract_lookup_tables)
    from experiment_synthetic_augmentation import ENRICHED_GROUPS

    gd_flats = []
    for i in range(5):
        gd = GenericDraftModel()
        gd.load_state_dict(torch.load(gd_future_path(i), weights_only=True,
                                      map_location="cpu"))
        gd.eval()
        gd_flats.append(extract_gd_weights(gd))

    gi = compute_group_indices()
    wp_cols = []
    for g in ENRICHED_GROUPS:
        s, e = gi[g]
        wp_cols.extend(range(s, e))

    def load_283(path):
        m = WinProbEnrichedModel(283, [256, 128], dropout=0.3)
        m.load_state_dict(torch.load(path, weights_only=True,
                                     map_location="cpu"))
        m.eval()
        return m

    arm_cfgs = {}
    for cell in arms:
        wm = load_283(os.path.join(common.MODELS_DIR, f"w4_vf_{cell}.pt"))
        wf, wn = extract_wp_weights(wm)
        wo = build_wp_net_offsets(wm, wn, 283)
        lut = extract_lookup_tables(
            common.load_patch_stats("cumulative", cell_stats_build(cell)))
        arm_cfgs[cell] = (wf, wo, lut)

    judge = load_283(os.path.join(RERUN_MODELS, "wp_enriched_256.pt"))
    from rerun2026 import common as rc
    judge_stats = rc.stats_cache()   # frozen end-of-time for judge features
    truth = future_truth_stats()

    return {"gd_flats": gd_flats, "arm_cfgs": arm_cfgs, "judge": judge,
            "judge_stats": judge_stats, "truth": truth, "wp_cols": wp_cols,
            "all_mask": [True] * len(FEATURE_GROUPS),
            "extract_features": extract_features, "NUM_HEROES": NUM_HEROES,
            "HEROES": HEROES, "HERO_TO_IDX": HERO_TO_IDX, "MAPS": MAPS,
            "SKILL_TIERS": SKILL_TIERS, "HERO_ROLE_FINE": HERO_ROLE_FINE,
            "is_degenerate": is_degenerate}


def benchmark_run(cell, name, ckpt, deps, kernel, device_id=0):
    """phase4 benchmark protocol vs future-GD opponents, metrics vs
    future-truth stats, terminal WP by the neutral judge."""
    from train_draft_policy import AlphaZeroDraftNet
    from extract_weights import extract_policy_weights
    from experiment_draft_quality import draft_resilience, draft_counter_quality
    from rebench_with_resilience import reconstruct_pick_steps

    wf, wo, lut = deps["arm_cfgs"][cell]
    policy = AlphaZeroDraftNet(size="base", policy_head_type="linear")
    policy.load_state_dict(torch.load(ckpt, weights_only=True,
                                      map_location="cpu"))
    policy.eval()
    pf, po = extract_policy_weights(policy)

    NUM_HEROES, HEROES = deps["NUM_HEROES"], deps["HEROES"]
    MAPS, SKILL_TIERS = deps["MAPS"], deps["SKILL_TIERS"]
    truth, judge = deps["truth"], deps["judge"]
    extract_features = deps["extract_features"]
    HERO_TO_IDX = deps["HERO_TO_IDX"]
    healers = set(h for h, r in deps["HERO_ROLE_FINE"].items() if r == "healer")
    is_degenerate = deps["is_degenerate"]

    def judge_wp(t0h, t1h, gm, tier):
        d = {"team0_heroes": sorted(t0h, key=lambda h: HERO_TO_IDX.get(h, 0)),
             "team1_heroes": sorted(t1h, key=lambda h: HERO_TO_IDX.get(h, 0)),
             "game_map": gm, "skill_tier": tier, "winner": 0}
        b, e = extract_features(d, deps["judge_stats"], deps["all_mask"])
        x = torch.tensor(np.concatenate([b, e[deps["wp_cols"]]]),
                         dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return judge(x).item()

    def ctr_d(ha, hb, tier):
        r = truth.get_counter(ha, hb, tier)
        if r is None:
            return None
        return r - (truth.get_hero_wr(ha, tier)
                    + (100 - truth.get_hero_wr(hb, tier)) - 50)

    def syn_d(ha, hb, tier):
        r = truth.get_synergy(ha, hb, tier)
        if r is None:
            return None
        return r - (50 + (truth.get_hero_wr(ha, tier) - 50)
                    + (truth.get_hero_wr(hb, tier) - 50))

    N = BENCH_DRAFTS
    random.seed(42)
    np.random.seed(42)
    cfgs = [(random.choice(range(len(MAPS))), 1, random.randint(0, 1))
            for _ in range(N)]
    ca = np.array(cfgs, dtype=np.int32)

    aw, ac, asn, ah, ad = [], [], [], [], []
    a_re, a_rl, a_cq = [], [], []
    heroes_seen = set()
    for bs in range(0, N, 128):
        bc = ca[bs:min(bs + 128, N)]
        gf, go = deps["gd_flats"][(bs // 128) % len(deps["gd_flats"])]
        eng = kernel.MCTSKernelEngine(pf, gf, wf, po, go, wo, lut,
                                      max_concurrent=min(len(bc), 128),
                                      device_id=device_id)
        res = eng.run_episodes(bc, SIMS, 2.0, 42 + bs)
        del eng
        for i, (wp, examples, terminal_state, ep_our_team) in enumerate(res):
            t = np.array(terminal_state)
            mi, ti, ou = cfgs[bs + i]
            gm, tier = MAPS[mi], SKILL_TIERS[ti]
            t0h = [HEROES[j] for j in range(NUM_HEROES) if t[j] > 0.5]
            t1h = [HEROES[j] for j in range(NUM_HEROES)
                   if t[NUM_HEROES + j] > 0.5]
            oh = t0h if ou == 0 else t1h
            op = t1h if ou == 0 else t0h
            heroes_seen.update(oh)
            wv = judge_wp(t0h, t1h, gm, tier)
            aw.append(wv if ou == 0 else 1 - wv)
            cd = [d for o in op for h in oh
                  for d in [ctr_d(h, o, tier)] if d is not None]
            ac.append(np.mean(cd) if cd else 0)
            sy = [d for j, h1 in enumerate(oh) for h2 in oh[j + 1:]
                  for d in [syn_d(h1, h2, tier)] if d is not None]
            asn.append(np.mean(sy) if sy else 0)
            ah.append(any(h in healers for h in oh))
            ad.append(is_degenerate(oh))
            pick_steps = reconstruct_pick_steps(examples, t, ou)
            rm = draft_resilience(pick_steps, truth, tier)
            a_re.append(rm["early_pick_resilience"])
            a_rl.append(rm["late_pick_resilience"])
            a_cq.append(draft_counter_quality(pick_steps, truth,
                                              tier)["avg_counter"])

    return {"name": name, "arm": cell,
            "judge_wp": float(np.mean(aw)),
            "counter_future": float(np.mean(ac)),
            "synergy_future": float(np.mean(asn)),
            "healer": float(np.mean(ah) * 100),
            "degen": float(np.mean(ad) * 100),
            "resil_early": float(np.mean(a_re)),
            "resil_late": float(np.mean(a_rl)),
            "counter_quality": float(np.mean(a_cq)),
            "distinct_heroes": len(heroes_seen)}


# ── driver ──

def mcts_jobs(arms, vf_paths, exclude_path):
    jobs = []
    for cell in arms:
        for seed in range(SEEDS_PER_ARM):
            name = f"w4_{cell}_s{seed}"
            save_dir = os.path.join(MCTS_RUNS_DIR, name)
            os.makedirs(save_dir, exist_ok=True)
            env = {
                "MCTS_SAVE_DIR": save_dir,
                "MCTS_WP_MODEL": "enriched_full",
                "MCTS_WP_PATH": vf_paths[cell],
                "MCTS_GD_PATH": os.path.join(RERUN_MODELS, "generic_draft_0.pt"),
                "MCTS_NUM_EPISODES": str(EPISODES),
                "MCTS_NUM_SIMS": str(SIMS),
                "MCTS_BATCH_EPISODES": "128",
                "MCTS_FRESH": "1",
                "MCTS_POLICY_HEAD": "linear",
                "MCTS_NET_SIZE": "base",
                "MCTS_STATS_BUILD": cell_stats_build(cell),
                "MCTS_EXCLUDE_IDS": exclude_path,
                "WANDB_RUN_NAME": f"drift2026_{name}",
                # Bulk variant sweeps stay off wandb cloud: no run-finished
                # emails, no dashboard clutter (local logs unaffected).
                "WANDB_MODE": "offline",
            }
            jobs.append(common.Job(f"w4_mcts_{name}", [WORKER],
                                   [os.path.join(save_dir, "draft_policy.pt")],
                                   weight="light", env=env))
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", default="all",
                    choices=["all", "gd-one", "bench-only"])
    ap.add_argument("--variant", type=int, default=0, help="gd-one variant idx")
    ap.add_argument("--best", default=None)
    ap.add_argument("--worst", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.stage == "gd-one":
        stage_gd_one(args)
        return

    best, worst = select_arms(args)
    arms = [best, worst]
    if args.dry_run:
        print(f"W4 plan: arms={arms}, {SEEDS_PER_ARM} seeds each, "
              f"{SIMS} sims, {EPISODES} episodes; future-GD x5; "
              f"benchmark {BENCH_DRAFTS} drafts vs future-GD")
        return

    t0 = time.time()
    vf_paths = {c: export_vf(c) for c in arms}
    exclude_path = build_exclude_ids()

    if args.stage != "bench-only":
        me = os.path.abspath(__file__)
        gd_jobs = [common.Job(f"w4_gd_future_{i}",
                              [me, "--stage", "gd-one", "--variant", str(i)],
                              [gd_future_path(i)], weight="light")
                   for i in range(5)]
        if common.run_pool(gd_jobs):
            sys.exit(1)
        if common.run_pool(mcts_jobs(arms, vf_paths, exclude_path)):
            sys.exit(1)

    # benchmark
    kernel = load_kernel_module()
    deps = load_bench_deps(arms)
    results = {}
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            results = json.load(f)
    for cell in arms:
        for seed in range(SEEDS_PER_ARM):
            name = f"w4_{cell}_s{seed}"
            ckpt = os.path.join(MCTS_RUNS_DIR, name, "draft_policy.pt")
            if not os.path.exists(ckpt):
                print(f"  missing checkpoint {name}, skipping")
                continue
            r = benchmark_run(cell, name, ckpt, deps, kernel)
            results[name] = r
            with open(RESULTS_PATH, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  BENCH {name}: judgeWP={r['judge_wp']:.4f} "
                  f"ctr={r['counter_future']:+.3f} syn={r['synergy_future']:+.3f} "
                  f"healer={r['healer']:.0f}% degen={r['degen']:.0f}%", flush=True)

    # summary
    lines = ["# W4 — MCTS policies under best vs worst drift-regime VFs",
             "",
             f"{SEEDS_PER_ARM} seeds/arm, {SIMS} sims, {EPISODES} episodes; "
             "opponents/benchmark = future-GD (post-cutoff meta); metrics vs "
             "future-truth stats; terminal WP by neutral wp_enriched_256.",
             "",
             "| arm | n | judge WP | counter | synergy | healer % | degen % | "
             "R_early | R_late |",
             "|---|---|---|---|---|---|---|---|---|"]
    for cell in arms:
        rs = [r for r in results.values() if r["arm"] == cell]
        if not rs:
            continue
        def m(k):
            return float(np.mean([r[k] for r in rs]))
        def sd(k):
            return float(np.std([r[k] for r in rs]))
        lines.append(f"| {cell} | {len(rs)} | {m('judge_wp'):.4f} ± "
                     f"{sd('judge_wp'):.3f} | {m('counter_future'):+.3f} | "
                     f"{m('synergy_future'):+.3f} | {m('healer'):.1f} | "
                     f"{m('degen'):.1f} | {m('resil_early'):+.3f} | "
                     f"{m('resil_late'):+.3f} |")
    path = os.path.join(common.RESULTS_DIR, "W4_SUMMARY.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {path} ({(time.time()-t0)/3600:.1f} h total)")


if __name__ == "__main__":
    main()
