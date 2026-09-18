#!/usr/bin/env python3
"""
rerun2026 phase 4: MCTS self-play campaign (Table VI tab:mcts + abstract claims).

Modeled on training/run_all_experiments.py, with the MANIFEST phase-4 changes:
  - 15 configs x 15 seeds = 225 runs (L_800sim_4M dropped per 2026-07-03 decision)
  - relational vs. absolute enrichment ablations at B_fullwp settings:
      M_relational / N_absolute — INVALID first attempt (kept for
        reproducibility of their rows): enriched_full WP weights with the
        ablated groups zeroed at inference. The model was TRAINED WITH those
        features, so zeroing feeds it off-manifold inputs; do not use these
        rows to attribute the feature gap.
      M2_relational / N2_absolute — CORRECTED (2026-07-06): WP value functions
        genuinely RETRAINED without the ablated groups
        (models/wp_{relational,absolute}_256.pt, train_jobs.py wp presets;
        M2 drops the ABSOLUTE groups meta_strength/comp_wr/team_avg_wr, N2
        drops the RELATIONAL groups counter_detail/pairwise_counters/
        pairwise_synergies). The reduced-input checkpoints are expanded to the
        kernel's 283-dim layout with zero first-layer columns at the ablated
        positions (exact; verified) so the enriched_full kernel path is reused;
        group-exclusive LUT tables are zeroed as before.
  - WP value functions / GD opponents come from rerun2026/models/
    (snapshot-retrained), never the historical training/*.pt:
      enriched_full -> wp_enriched_256.pt   augmented -> wp_aug_v2_256.pt
      enriched -> partial_wp.pt             base -> base_partial_wp.pt
      true_base -> wp_true_base.pt          GD pool -> generic_draft_{0..4}.pt
  - GPU pool: NUM_GPUS env (default 4); the thermally-throttled GPU
    (NUM_GPUS-1, i.e. GPU 3) may ONLY take 200-sim/<=300K-episode jobs;
    400/600/800-sim and 1M-episode jobs go to the other GPUs.
  - skip-if-done: checkpoint in rerun2026/mcts_runs/<name>/draft_policy.pt AND
    entry in rerun2026/results/mcts_experiment_results.json.
  - benchmark (200 drafts, run_all_experiments protocol) evaluates against the
    NEW wp_enriched_256 terminal evaluator and adds resilience metrics
    (experiment_draft_quality.draft_resilience / draft_counter_quality over
    rebench_with_resilience.reconstruct_pick_steps pick orders).
  - value-head pretraining inside the worker is pinned to the 2.55-filtered
    snapshot via MCTS_EXCLUDE_IDS.

Usage:
    python rerun2026/phase4_mcts.py --dry-run          # print job list
    python rerun2026/phase4_mcts.py --verify-zeroing   # M/N ablation assertions only
    python rerun2026/phase4_mcts.py [--only SUBSTR]    # run campaign
"""
import os
import sys
import json
import time
import random
import argparse
import subprocess
import importlib.util

RERUN_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(RERUN_DIR)
sys.path.insert(0, TRAINING_DIR)
sys.path.insert(0, os.path.join(TRAINING_DIR, "cuda_mcts"))

from rerun2026 import common

common.setup()

import numpy as np
import torch

WORKER = os.path.join(TRAINING_DIR, "train_mcts_worker.py")
MODELS_DIR = common.MODELS_DIR
MCTS_RUNS_DIR = common.MCTS_RUNS_DIR
RESULTS_PATH = os.path.join(common.RESULTS_DIR, "mcts_experiment_results.json")
SEEDS_PER_EXPERIMENT = 15
NUM_GPUS = common.NUM_GPUS
SLOW_GPU = common.SLOW_GPU  # thermal throttle: 200-sim/<=300K-episode jobs only
BENCH_DEVICE = int(os.environ.get("CUDA_DEVICE", "0"))

# ── Experiment definitions (run_all_experiments.py EXPERIMENTS + M/N) ──

EXPERIMENTS = {
    "A_partial":    {"wp": "enriched",      "episodes": 300000,  "head": "linear", "size": "base",  "sims": 200},
    "B_fullwp":     {"wp": "enriched_full", "episodes": 300000,  "head": "linear", "size": "base",  "sims": 200},
    "G_base":       {"wp": "base",          "episodes": 300000,  "head": "linear", "size": "base",  "sims": 200},
    "H_augmented":  {"wp": "augmented",     "episodes": 300000,  "head": "linear", "size": "base",  "sims": 200},
    "K_truebase":   {"wp": "true_base",     "episodes": 300000,  "head": "linear", "size": "base",  "sims": 200},
    "C_large":      {"wp": "enriched_full", "episodes": 300000,  "head": "linear", "size": "large", "sims": 200},
    "D_deep":       {"wp": "enriched_full", "episodes": 300000,  "head": "deep",   "size": "base",  "sims": 200},
    "E_1M":         {"wp": "enriched_full", "episodes": 1000000, "head": "linear", "size": "base",  "sims": 200},
    "F_400sim":     {"wp": "enriched_full", "episodes": 300000,  "head": "linear", "size": "base",  "sims": 400},
    "I_600sim":     {"wp": "enriched_full", "episodes": 300000,  "head": "linear", "size": "base",  "sims": 600},
    "J_800sim":     {"wp": "enriched_full", "episodes": 300000,  "head": "linear", "size": "base",  "sims": 800},
    # MANIFEST "Phase 4 addition": relational vs. absolute enrichment ablation.
    # M/N (INVALID methodology: enriched_full weights with groups zeroed at
    # inference — the model was trained WITH those features) are kept only so
    # their historical rows/checkpoints remain reproducible.
    "M_relational": {"wp": "relational",    "episodes": 300000,  "head": "linear", "size": "base",  "sims": 200},
    "N_absolute":   {"wp": "absolute",      "episodes": 300000,  "head": "linear", "size": "base",  "sims": 200},
    # CORRECTED ablation (2026-07-06): WP value functions genuinely RETRAINED
    # without the ablated feature groups (train_jobs.py wp --groups
    # relational/absolute, best of 3 seeds, 256->128 — the wp_enriched_256
    # protocol). Reduced-input checkpoints are expanded to the kernel's 283-dim
    # layout with zero first-layer columns at the ablated positions
    # (mathematically identical; see ensure_expanded_checkpoints).
    "M2_relational": {"wp": "relational_trained", "episodes": 300000, "head": "linear", "size": "base", "sims": 200},
    "N2_absolute":   {"wp": "absolute_trained",   "episodes": 300000, "head": "linear", "size": "base", "sims": 200},
}

SUMMARY_ORDER = ["J_800sim", "I_600sim", "F_400sim", "E_1M", "B_fullwp", "A_partial",
                 "H_augmented", "C_large", "D_deep",
                 "M2_relational", "N2_absolute", "M_relational", "N_absolute",
                 "G_base", "K_truebase"]

# Enriched groups zeroed at extraction per ablation (MANIFEST 2026-07-02).
# For the *_trained types the WP first-layer columns are already zero (the
# model never had those inputs); the entry still zeroes the group-exclusive
# LUT tables and keeps launch()/benchmark wiring uniform.
ABLATION_ZERO_GROUPS = {
    "relational": ["meta_strength", "comp_wr", "team_avg_wr"],           # M: zero ABSOLUTE
    "absolute": ["counter_detail", "pairwise_counters", "pairwise_synergies"],  # N: zero RELATIONAL
    "relational_trained": ["meta_strength", "comp_wr", "team_avg_wr"],
    "absolute_trained": ["counter_detail", "pairwise_counters", "pairwise_synergies"],
}

# Corrected-ablation sources: wp type -> (reduced-dim checkpoint, expanded 283d
# checkpoint written next to it). Reduced models come from train_jobs.py cmd_wp.
TRAINED_ABLATION_SOURCES = {
    "relational_trained": ("wp_relational_256.pt", "wp_relational_256_expanded283.pt"),
    "absolute_trained": ("wp_absolute_256.pt", "wp_absolute_256_expanded283.pt"),
}

# wp type -> rerun2026/models checkpoint (train_jobs.py cmd_wp/cmd_partial/cmd_wp_aug outputs)
WP_CHECKPOINTS = {
    "enriched": "partial_wp.pt",
    "enriched_full": "wp_enriched_256.pt",
    "base": "base_partial_wp.pt",
    "true_base": "wp_true_base.pt",
    "augmented": "wp_aug_v2_256.pt",
    "relational": "wp_enriched_256.pt",  # enriched_full weights, groups zeroed
    "absolute": "wp_enriched_256.pt",
    # corrected ablations: 283d-expanded copies of the retrained reduced models
    "relational_trained": "wp_relational_256_expanded283.pt",
    "absolute_trained": "wp_absolute_256_expanded283.pt",
}
# wp type as the worker understands it (ablations ride the enriched_full path)
WORKER_WP_TYPE = {"relational": "enriched_full", "absolute": "enriched_full",
                  "relational_trained": "enriched_full",
                  "absolute_trained": "enriched_full"}


def ensure_expanded_checkpoints():
    """Build the 283-dim expanded checkpoints for the TRAINED-WITHOUT ablation
    WP models (M2_relational / N2_absolute) from the reduced-input models that
    train_jobs.py produced.

    The kernel always computes the full 86-dim enriched vector in
    KERNEL_ENRICHED_LAYOUT order, so a model trained on a group subset (whose
    kept columns are in the same relative order — train_jobs.py WP_GROUP_PRESETS
    preserves ENRICHED_GROUPS order) embeds exactly: first-layer weight columns
    go to the kept positions, ablated positions get zeros. All later layers /
    BN / bias are unchanged, so outputs are bit-for-bit the reduced model's
    (verified numerically here before saving)."""
    from sweep_enriched_wp import WinProbEnrichedModel
    from extract_weights import kernel_enriched_group_cols

    for wp_type, (src_name, out_name) in TRAINED_ABLATION_SOURCES.items():
        src = os.path.join(MODELS_DIR, src_name)
        out = os.path.join(MODELS_DIR, out_name)
        if not os.path.exists(src):
            raise RuntimeError(
                f"Missing retrained ablation WP model: {src} — run "
                f"'python rerun2026/train_jobs.py wp --name {src_name[:-3]} "
                f"--groups {wp_type.replace('_trained', '')} --arch 256,128 --seeds 3'")
        drop_cols = [197 + c for c in
                     kernel_enriched_group_cols(ABLATION_ZERO_GROUPS[wp_type])]
        keep_cols = [c for c in range(283) if c not in drop_cols]
        sd = torch.load(src, weights_only=True, map_location="cpu")
        red_dim = sd["net.0.weight"].shape[1]
        if red_dim != len(keep_cols):
            raise RuntimeError(
                f"{src_name}: input dim {red_dim} != expected {len(keep_cols)} "
                f"(197 + kept enriched dims for {wp_type})")

        if os.path.exists(out):
            esd = torch.load(out, weights_only=True, map_location="cpu")
            w = esd["net.0.weight"]
            if (w.shape[1] == 283 and torch.equal(w[:, keep_cols], sd["net.0.weight"])
                    and not w[:, drop_cols].any()):
                print(f"  expanded checkpoint up to date: {out_name}")
                continue
            print(f"  expanded checkpoint stale (source changed), rebuilding: {out_name}")

        esd = {k: v.clone() for k, v in sd.items()}
        W = torch.zeros(sd["net.0.weight"].shape[0], 283,
                        dtype=sd["net.0.weight"].dtype)
        W[:, keep_cols] = sd["net.0.weight"]
        esd["net.0.weight"] = W

        # Numeric equivalence: expanded(283 input) == reduced(kept cols)
        m_red = WinProbEnrichedModel(red_dim, [256, 128], dropout=0.3)
        m_red.load_state_dict(sd); m_red.eval()
        m_exp = WinProbEnrichedModel(283, [256, 128], dropout=0.3)
        m_exp.load_state_dict(esd); m_exp.eval()
        torch.manual_seed(0)
        x = torch.randn(256, 283)
        with torch.no_grad():
            a, b = m_red(x[:, keep_cols]), m_exp(x)
        if not torch.allclose(a, b, atol=1e-6):
            raise RuntimeError(f"{out_name}: expanded model is not numerically "
                               f"identical to the reduced model")
        torch.save(esd, out)
        print(f"  wrote {out_name} (reduced {red_dim}d -> 283d, "
              f"{len(drop_cols)} zero columns, equivalence verified)")


def wp_ckpt_path(wp_type):
    return os.path.join(MODELS_DIR, WP_CHECKPOINTS[wp_type])


def is_light(cfg):
    """GPU-3-eligible: 200 sims and <=300K episodes (~2h jobs)."""
    return cfg["sims"] == 200 and cfg["episodes"] <= 300000


def run_name(exp, seed):
    return f"{exp}_s{seed}"


def checkpoint_path(exp, seed):
    return os.path.join(MCTS_RUNS_DIR, run_name(exp, seed), "draft_policy.pt")


def has_checkpoint(exp, seed):
    return os.path.exists(checkpoint_path(exp, seed))


def launch(name, cfg, gpu_id):
    save_dir = os.path.join(MCTS_RUNS_DIR, name)
    os.makedirs(save_dir, exist_ok=True)
    log = os.path.join(common.LOGS_DIR, f"mcts_{name}.log")
    wp_type = cfg["wp"]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(common.GPU_IDS[gpu_id])
    env["MCTS_SAVE_DIR"] = save_dir
    env["MCTS_WP_MODEL"] = WORKER_WP_TYPE.get(wp_type, wp_type)
    env["MCTS_WP_PATH"] = wp_ckpt_path(wp_type)
    env["MCTS_GD_PATH"] = os.path.join(MODELS_DIR, "generic_draft_0.pt")
    env["MCTS_NUM_EPISODES"] = str(cfg["episodes"])
    env["MCTS_NUM_SIMS"] = str(cfg["sims"])
    env["MCTS_BATCH_EPISODES"] = "128"
    env["MCTS_FRESH"] = "1"
    env["MCTS_POLICY_HEAD"] = cfg["head"]
    env["MCTS_NET_SIZE"] = cfg["size"]
    env["WANDB_RUN_NAME"] = f"rerun2026_{name}"
    if wp_type in ABLATION_ZERO_GROUPS:
        env["MCTS_WP_ZERO_GROUPS"] = ",".join(ABLATION_ZERO_GROUPS[wp_type])
    if os.path.exists(common.EXCLUDE_IDS_PATH):
        env["MCTS_EXCLUDE_IDS"] = common.EXCLUDE_IDS_PATH
    lf = open(log, "w")
    p = subprocess.Popen([sys.executable, "-u", WORKER], cwd=TRAINING_DIR,
                         env=env, stdout=lf, stderr=subprocess.STDOUT)
    return p, lf, log


# ── Benchmark (run_all_experiments.benchmark_run + resilience) ──

def load_benchmark_deps():
    """All benchmark dependencies, built from rerun2026/models/ checkpoints."""
    from shared import (NUM_HEROES, HEROES, HERO_TO_IDX, MAPS, SKILL_TIERS,
                        HERO_ROLE_FINE, is_degenerate)
    from sweep_enriched_wp import (WinProbEnrichedModel, FEATURE_GROUPS,
                                   compute_group_indices, FEATURE_GROUP_DIMS,
                                   extract_features)
    from train_draft_policy import AlphaZeroDraftNet
    from train_generic_draft import GenericDraftModel
    from train_partial_wp import PartialStateWP
    from extract_weights import (extract_policy_weights, extract_gd_weights,
                                 extract_wp_weights, build_wp_net_offsets,
                                 extract_lookup_tables, KERNEL_ENRICHED_LAYOUT,
                                 kernel_enriched_group_cols, zero_wp_input_columns,
                                 zero_lookup_table_groups)
    from experiment_synthetic_augmentation import ENRICHED_GROUPS
    from experiment_draft_quality import draft_resilience, draft_counter_quality
    from rebench_with_resilience import reconstruct_pick_steps

    # The kernel's 86-dim enriched layout must match the Python group tables
    assert [g for g, _ in KERNEL_ENRICHED_LAYOUT] == list(ENRICHED_GROUPS)
    assert all(FEATURE_GROUP_DIMS[g] == d for g, d in KERNEL_ENRICHED_LAYOUT)

    stats = common.stats_cache()  # pinned to frozen_stats_2026-05-19.json
    gi = compute_group_indices()
    wp_cols = []
    for g in ENRICHED_GROUPS:
        s, e = gi[g]
        wp_cols.extend(range(s, e))
    wp_input_dim = 197 + sum(FEATURE_GROUP_DIMS[g] for g in ENRICHED_GROUPS)
    all_mask = [True] * len(FEATURE_GROUPS)

    # GD opponent pool: rerun2026/models/generic_draft_{0..4}.pt
    gd_flats = []
    for i in range(5):
        gd = GenericDraftModel()
        gd.load_state_dict(torch.load(os.path.join(MODELS_DIR, f"generic_draft_{i}.pt"),
                                      weights_only=True, map_location="cpu"))
        gd.eval()
        gd_flats.append(extract_gd_weights(gd))

    def load_enriched_256(fname):
        m = WinProbEnrichedModel(wp_input_dim, [256, 128], dropout=0.3)
        m.load_state_dict(torch.load(os.path.join(MODELS_DIR, fname),
                                     weights_only=True, map_location="cpu"))
        m.eval()
        return m

    # Terminal evaluator: NEW snapshot-retrained enriched WP
    wp_eval = load_enriched_256(WP_CHECKPOINTS["enriched_full"])

    wp_kernel_configs = {}
    base_lut = extract_lookup_tables(stats)

    # enriched (partial, step-conditioned 283+8)
    ckpt = torch.load(wp_ckpt_path("enriched"), weights_only=True, map_location="cpu")
    wm = PartialStateWP(input_dim=wp_input_dim, step_embed_dim=8, hidden=(256, 128))
    wm.load_state_dict(ckpt["model_state_dict"]); wm.eval()
    wf, wn = extract_wp_weights(wm)
    wo = build_wp_net_offsets(wm, wn, wp_input_dim + 8)
    se = wm.step_embed.weight.data.cpu().numpy()
    wp_kernel_configs["enriched"] = (wf, wo, extract_lookup_tables(stats, step_embed_weights=se))

    # enriched_full
    wm2 = load_enriched_256(WP_CHECKPOINTS["enriched_full"])
    wf2, wn2 = extract_wp_weights(wm2)
    wo2 = build_wp_net_offsets(wm2, wn2, wp_input_dim)
    wp_kernel_configs["enriched_full"] = (wf2, wo2, base_lut)

    # true_base (197d, no enriched, no step embed)
    wm_tb = WinProbEnrichedModel(197, [256, 128], dropout=0.3)
    wm_tb.load_state_dict(torch.load(wp_ckpt_path("true_base"),
                                     weights_only=True, map_location="cpu"))
    wm_tb.eval()
    wf_tb, wn_tb = extract_wp_weights(wm_tb)
    wo_tb = build_wp_net_offsets(wm_tb, wn_tb, 197, use_enriched=False)
    wp_kernel_configs["true_base"] = (wf_tb, wo_tb, base_lut)

    # base (197+8 step embed, no enriched)
    ckpt_b = torch.load(wp_ckpt_path("base"), weights_only=True, map_location="cpu")
    wm3 = PartialStateWP(input_dim=197, step_embed_dim=8, hidden=(256, 128))
    wm3.load_state_dict(ckpt_b["model_state_dict"]); wm3.eval()
    wf3, wn3 = extract_wp_weights(wm3)
    wo3 = build_wp_net_offsets(wm3, wn3, 197 + 8, use_enriched=False)
    se3 = wm3.step_embed.weight.data.cpu().numpy()
    wp_kernel_configs["base"] = (wf3, wo3, extract_lookup_tables(stats, step_embed_weights=se3))

    # augmented (aug_v2, 283d)
    wm4 = load_enriched_256(WP_CHECKPOINTS["augmented"])
    wf4, wn4 = extract_wp_weights(wm4)
    wo4 = build_wp_net_offsets(wm4, wn4, wp_input_dim)
    wp_kernel_configs["augmented"] = (wf4, wo4, base_lut)

    # Ablations. relational/absolute (INVALID, historical): enriched_full
    # weights with group columns zeroed at inference. relational_trained/
    # absolute_trained (corrected M2/N2): genuinely retrained reduced models,
    # expanded to 283d — zero_wp_input_columns must then be a no-op (asserted),
    # since the expanded checkpoints already carry zero columns there.
    for abl, zg in ABLATION_ZERO_GROUPS.items():
        wm_a = load_enriched_256(WP_CHECKPOINTS[abl])
        wf_a, wn_a = extract_wp_weights(wm_a)
        wo_a = build_wp_net_offsets(wm_a, wn_a, wp_input_dim)
        zero_cols = [197 + c for c in kernel_enriched_group_cols(zg)]
        wf_zeroed = zero_wp_input_columns(wf_a, wn_a, wm_a, zero_cols)
        if abl in TRAINED_ABLATION_SOURCES:
            assert np.array_equal(wf_zeroed, wf_a), \
                f"{abl}: expanded checkpoint had nonzero weights at ablated columns"
        lut_a = zero_lookup_table_groups(base_lut, zg)
        wp_kernel_configs[abl] = (wf_zeroed, wo_a, lut_a)

    verify_ablation_zeroing(wp_kernel_configs)

    return {
        "NUM_HEROES": NUM_HEROES, "HEROES": HEROES, "HERO_TO_IDX": HERO_TO_IDX,
        "MAPS": MAPS, "SKILL_TIERS": SKILL_TIERS, "HERO_ROLE_FINE": HERO_ROLE_FINE,
        "is_degenerate": is_degenerate,
        "stats": stats, "wp_cols": wp_cols, "all_mask": all_mask,
        "wp_eval": wp_eval, "wp_input_dim": wp_input_dim,
        "gd_flats": gd_flats, "wp_kernel_configs": wp_kernel_configs,
        "extract_features": extract_features,
        "extract_policy_weights": extract_policy_weights,
        "AlphaZeroDraftNet": AlphaZeroDraftNet,
        "draft_resilience": draft_resilience,
        "draft_counter_quality": draft_counter_quality,
        "reconstruct_pick_steps": reconstruct_pick_steps,
    }


def verify_ablation_zeroing(wp_kernel_configs):
    """M/N kernel inputs must differ from both B_fullwp (enriched_full) and
    K_truebase — on the lookup-table bytes AND the WP weight bytes — and from
    each other."""
    b_wf, _, b_lut = wp_kernel_configs["enriched_full"]
    k_wf, _, k_lut = wp_kernel_configs["true_base"]
    for abl in ABLATION_ZERO_GROUPS:
        a_wf, _, a_lut = wp_kernel_configs[abl]
        assert a_lut.tobytes() != b_lut.tobytes(), \
            f"{abl}: LUT identical to enriched_full (B_fullwp)"
        assert a_lut.tobytes() != k_lut.tobytes(), \
            f"{abl}: LUT identical to true_base (K_truebase)"
        assert a_wf.tobytes() != b_wf.tobytes(), \
            f"{abl}: WP weights identical to enriched_full (B_fullwp)"
        assert a_wf.tobytes() != k_wf.tobytes(), \
            f"{abl}: WP weights identical to true_base (K_truebase)"
        n_lut = int(np.count_nonzero(a_lut != b_lut))
        n_wf = int(np.count_nonzero(a_wf != b_wf))
        print(f"  verify {abl}: {n_lut} LUT bytes and {n_wf} WP floats differ "
              f"from enriched_full — OK")
    kinds = list(ABLATION_ZERO_GROUPS)
    for i, a in enumerate(kinds):
        for b in kinds[i + 1:]:
            assert (wp_kernel_configs[a][0].tobytes()
                    != wp_kernel_configs[b][0].tobytes()), \
                f"{a} and {b} WP weights identical"
            same_zg = ABLATION_ZERO_GROUPS[a] == ABLATION_ZERO_GROUPS[b]
            lut_same = (wp_kernel_configs[a][2].tobytes()
                        == wp_kernel_configs[b][2].tobytes())
            # same zero-group set (e.g. relational vs relational_trained) must
            # share the LUT ablation; different sets must not.
            assert lut_same == same_zg, \
                f"{a} vs {b}: LUT sharing inconsistent with zero-group sets"
    print(f"  verify: pairwise distinctness across {kinds} — OK")


def benchmark_run(exp_name, seed, deps, kernel, device_id=0):
    """MCTS inference benchmark on one checkpoint (run_all_experiments protocol,
    200 drafts, mid tier) + resilience metrics. Returns metrics dict or None."""
    name = run_name(exp_name, seed)
    ckpt_path = checkpoint_path(exp_name, seed)
    if not os.path.exists(ckpt_path):
        return None

    cfg = EXPERIMENTS[exp_name]
    wf, wo, lut = deps["wp_kernel_configs"][cfg["wp"]]

    AlphaZeroDraftNet = deps["AlphaZeroDraftNet"]
    policy = AlphaZeroDraftNet(size=cfg["size"], policy_head_type=cfg["head"])
    sd = torch.load(ckpt_path, weights_only=True, map_location="cpu")
    if any(k.startswith("res_block1.") for k in sd):
        sd = {k.replace("res_block1.", "res_blocks.0.").replace("res_block2.", "res_blocks.1.")
              .replace("res_block3.", "res_blocks.2."): v for k, v in sd.items()}
    policy.load_state_dict(sd); policy.eval()
    pf, po = deps["extract_policy_weights"](policy)

    NUM_HEROES = deps["NUM_HEROES"]; HEROES = deps["HEROES"]
    MAPS = deps["MAPS"]; SKILL_TIERS = deps["SKILL_TIERS"]
    stats = deps["stats"]; wp_eval = deps["wp_eval"]
    wp_cols = deps["wp_cols"]; all_mask = deps["all_mask"]
    extract_features = deps["extract_features"]
    HERO_TO_IDX = deps["HERO_TO_IDX"]
    healer_heroes = set(h for h, r in deps["HERO_ROLE_FINE"].items() if r == "healer")
    is_degenerate = deps["is_degenerate"]
    draft_resilience = deps["draft_resilience"]
    draft_counter_quality = deps["draft_counter_quality"]
    reconstruct_pick_steps = deps["reconstruct_pick_steps"]

    def eval_wp(t0h, t1h, gm, tier):
        d = {"team0_heroes": sorted(t0h, key=lambda h: HERO_TO_IDX.get(h, 0)),
             "team1_heroes": sorted(t1h, key=lambda h: HERO_TO_IDX.get(h, 0)),
             "game_map": gm, "skill_tier": tier, "winner": 0}
        b, e = extract_features(d, stats, all_mask)
        x = torch.tensor(np.concatenate([b, e[wp_cols]]), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return wp_eval(x).item()

    def ctr_d(ha, hb, tier):
        r = stats.get_counter(ha, hb, tier)
        if r is None: return None
        return r - (stats.get_hero_wr(ha, tier) + (100 - stats.get_hero_wr(hb, tier)) - 50)

    def syn_d(ha, hb, tier):
        r = stats.get_synergy(ha, hb, tier)
        if r is None: return None
        return r - (50 + (stats.get_hero_wr(ha, tier) - 50) + (stats.get_hero_wr(hb, tier) - 50))

    N = 200
    random.seed(42); np.random.seed(42)
    cfgs = [(random.choice(range(len(MAPS))), 1, random.randint(0, 1)) for _ in range(N)]
    ca = np.array(cfgs, dtype=np.int32)

    aw, ac, asn, ah, ad = [], [], [], [], []
    a_res_avg, a_res_early, a_res_late, a_cq = [], [], [], []
    hero_counter = {}
    for bs in range(0, N, 128):
        be = min(bs + 128, N); bc = ca[bs:be]
        gi = (bs // 128) % len(deps["gd_flats"])
        gf, go = deps["gd_flats"][gi]
        eng = kernel.MCTSKernelEngine(pf, gf, wf, po, go, wo, lut,
                                      max_concurrent=min(len(bc), 128), device_id=device_id)
        res = eng.run_episodes(bc, 200, 2.0, 42 + bs)
        del eng
        for i, (wp, examples, terminal_state, ep_our_team) in enumerate(res):
            t = np.array(terminal_state); c = cfgs[bs + i]
            mi, ti, ou = c; gm = MAPS[mi]; tier = SKILL_TIERS[ti]
            t0h = [HEROES[j] for j in range(NUM_HEROES) if t[j] > 0.5]
            t1h = [HEROES[j] for j in range(NUM_HEROES) if t[NUM_HEROES + j] > 0.5]
            oh = t0h if ou == 0 else t1h; op = t1h if ou == 0 else t0h
            for h in oh:
                hero_counter[h] = hero_counter.get(h, 0) + 1
            wv = eval_wp(t0h, t1h, gm, tier); ow = wv if ou == 0 else 1 - wv; aw.append(ow)
            cd = [d for o in op for h in oh for d in [ctr_d(h, o, tier)] if d is not None]
            ac.append(np.mean(cd) if cd else 0)
            sd2 = []
            for j, h1 in enumerate(oh):
                for h2 in oh[j + 1:]:
                    d = syn_d(h1, h2, tier)
                    if d is not None: sd2.append(d)
            asn.append(np.mean(sd2) if sd2 else 0)
            ah.append(1 if any(h in healer_heroes for h in oh) else 0)
            ad.append(1 if is_degenerate(oh) else 0)

            # Resilience / counter quality from reconstructed pick order
            pick_steps = reconstruct_pick_steps(examples, t, ou)
            rm = draft_resilience(pick_steps, stats, tier)
            cq = draft_counter_quality(pick_steps, stats, tier)
            a_res_avg.append(rm["avg_resilience"])
            a_res_early.append(rm["early_pick_resilience"])
            a_res_late.append(rm["late_pick_resilience"])
            a_cq.append(cq["avg_counter"])

    return {
        "name": name, "experiment": exp_name, "seed": seed,
        "wp": float(np.mean(aw)), "wr": float(np.mean([1 if w > 0.5 else 0 for w in aw]) * 100),
        "counter": float(np.mean(ac)), "synergy": float(np.mean(asn)),
        "healer": float(np.mean(ah) * 100), "degen": float(np.mean(ad) * 100),
        "resil_avg": float(np.mean(a_res_avg)),
        "resil_early": float(np.mean(a_res_early)),
        "resil_late": float(np.mean(a_res_late)),
        "counter_quality": float(np.mean(a_cq)),
        "distinct_heroes": len(hero_counter),
    }


# ── Results bookkeeping ──

def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {}


def save_results(results):
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)


def print_bench_line(prefix, r):
    print(f"  {prefix} {r['name']}: WP={r['wp']:.4f} WR={r['wr']:.1f}% "
          f"Ctr={r['counter']:+.4f} Syn={r['synergy']:+.4f} "
          f"RE={r['resil_early']:+.3f} RL={r['resil_late']:+.3f} "
          f"Hlr={r['healer']:.0f}% Deg={r['degen']:.0f}%", flush=True)


def print_summary(results):
    print(f"\n{'='*120}")
    print(f"{'Experiment':<14} {'N':>3} {'Avg WP':>8} {'±σ':>6} {'WR%':>6} "
          f"{'Counter':>9} {'Synergy':>9} {'R_early':>8} {'R_late':>8} "
          f"{'Hlr%':>5} {'Deg%':>5} {'Div':>4}")
    print(f"{'-'*120}")
    for exp in SUMMARY_ORDER:
        seeds = [v for v in results.values() if v["experiment"] == exp]
        if not seeds:
            continue
        print(f"{exp:<14} {len(seeds):>3} "
              f"{np.mean([s['wp'] for s in seeds]):>8.4f} "
              f"{np.std([s['wp'] for s in seeds]):>6.3f} "
              f"{np.mean([s['wr'] for s in seeds]):>5.1f}% "
              f"{np.mean([s['counter'] for s in seeds]):>+9.4f} "
              f"{np.mean([s['synergy'] for s in seeds]):>+9.4f} "
              f"{np.mean([s.get('resil_early', 0) for s in seeds]):>+8.3f} "
              f"{np.mean([s.get('resil_late', 0) for s in seeds]):>+8.3f} "
              f"{np.mean([s['healer'] for s in seeds]):>4.0f}% "
              f"{np.mean([s['degen'] for s in seeds]):>4.0f}% "
              f"{np.mean([s.get('distinct_heroes', 0) for s in seeds]):>4.0f}")
    print(f"{'='*120}")


def load_kernel_module():
    so_dir = os.path.join(TRAINING_DIR, "cuda_mcts")
    so_files = [f for f in os.listdir(so_dir)
                if f.startswith("cuda_mcts_kernel") and f.endswith(".so")]
    if not so_files:
        raise RuntimeError("cuda_mcts_kernel not built")
    spec = importlib.util.spec_from_file_location(
        "cuda_mcts_kernel", os.path.join(so_dir, so_files[0]))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Main ──

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="print job list and exit")
    ap.add_argument("--only", default=None,
                    help="substring filter on run names (e.g. B_fullwp_s0)")
    ap.add_argument("--verify-zeroing", action="store_true",
                    help="build kernel configs, run M/N ablation assertions, exit")
    args = ap.parse_args()

    # Only the WP checkpoints the selected runs use are required (a namespace
    # rerun of one config must not need the whole phase1 suite).
    needed_wp = {EXPERIMENTS[e]["wp"] for e in EXPERIMENTS
                 for s in range(SEEDS_PER_EXPERIMENT)
                 if not args.only or args.only in run_name(e, s)}
    if needed_wp & {"relational_trained", "absolute_trained"}:
        # Corrected M2/N2 ablations: expand the retrained reduced-input WP
        # models to the kernel's 283-dim layout (skip-if-current, verified).
        ensure_expanded_checkpoints()

    for wp_type in sorted(needed_wp):
        p = wp_ckpt_path(wp_type)
        if not os.path.exists(p):
            raise RuntimeError(f"Missing rerun2026 WP checkpoint for '{wp_type}': {p}")
    for i in range(5):
        p = os.path.join(MODELS_DIR, f"generic_draft_{i}.pt")
        if not os.path.exists(p):
            raise RuntimeError(f"Missing rerun2026 GD checkpoint: {p}")
    os.makedirs(MCTS_RUNS_DIR, exist_ok=True)

    if args.verify_zeroing:
        load_benchmark_deps()
        print("Zeroing verification passed.")
        return

    results = load_results()

    # Build run list; skip-if-done = checkpoint AND benchmark entry
    all_runs = []
    for exp in EXPERIMENTS:
        for seed in range(SEEDS_PER_EXPERIMENT):
            name = run_name(exp, seed)
            if args.only and args.only not in name:
                continue
            if name in results and has_checkpoint(exp, seed):
                continue  # done
            all_runs.append((exp, seed))

    need_training = [(e, s) for e, s in all_runs if not has_checkpoint(e, s)]
    need_benchmark_only = [(e, s) for e, s in all_runs if has_checkpoint(e, s)]
    # Heavy jobs first so GPUs 0..SLOW_GPU-1 start on the long ones while the
    # throttled GPU works through the 200-sim queue.
    need_training.sort(key=lambda r: (is_light(EXPERIMENTS[r[0]]), r[0], r[1]))

    n_total = len(EXPERIMENTS) * SEEDS_PER_EXPERIMENT
    print(f"phase4: {len(EXPERIMENTS)} configs x {SEEDS_PER_EXPERIMENT} seeds = {n_total} runs"
          + (f" (filter: {args.only})" if args.only else ""))
    print(f"  Done (ckpt + benchmark): {n_total - len(all_runs) if not args.only else '-'}")
    print(f"  Need training: {len(need_training)}")
    print(f"  Need benchmark only: {len(need_benchmark_only)}")
    print(f"  GPUs: {NUM_GPUS} (GPU {SLOW_GPU} restricted to 200-sim/<=300K-episode jobs)")

    if args.dry_run:
        for e, s in need_benchmark_only:
            print(f"  BENCH-ONLY {run_name(e, s)}")
        for e, s in need_training:
            cfg = EXPERIMENTS[e]
            tag = "light" if is_light(cfg) else "heavy"
            print(f"  TRAIN {run_name(e, s):<18} [{tag}] wp={cfg['wp']}"
                  f" ({WP_CHECKPOINTS[cfg['wp']]}) sims={cfg['sims']}"
                  f" episodes={cfg['episodes']} size={cfg['size']} head={cfg['head']}"
                  + (f" zero={ABLATION_ZERO_GROUPS[cfg['wp']]}"
                     if cfg["wp"] in ABLATION_ZERO_GROUPS else ""))
        print("(dry run: nothing executed)")
        return

    kernel_mod = load_kernel_module()
    deps = None  # lazy

    def get_deps():
        nonlocal deps
        if deps is None:
            deps = load_benchmark_deps()
        return deps

    # Benchmark already-trained checkpoints first
    if need_benchmark_only:
        print(f"\nBenchmarking {len(need_benchmark_only)} already-completed runs...")
        for exp, seed in need_benchmark_only:
            r = benchmark_run(exp, seed, get_deps(), kernel_mod, device_id=BENCH_DEVICE)
            if r:
                results[r["name"]] = r
                save_results(results)
                print_bench_line("BENCH", r)
        print_summary(results)

    # Train with GPU pool: poll, refill (run_all_experiments pattern), with the
    # strict GPU-3 constraint.
    if need_training:
        print(f"\nTraining {len(need_training)} runs across {NUM_GPUS} GPUs (pool mode)...")
        run_queue = list(need_training)
        active = [None] * NUM_GPUS  # gpu -> (exp, seed, name, proc, logfile, logpath)
        pending_bench = []
        completed = 0
        total_training = len(need_training)

        def start_next(gpu_id):
            idx = None
            for i, (e, s) in enumerate(run_queue):
                # Strict thermal rule (only meaningful with >1 GPU): the slow
                # GPU never takes 400/600/800-sim or 1M-episode jobs.
                if NUM_GPUS > 1 and gpu_id == SLOW_GPU and not is_light(EXPERIMENTS[e]):
                    continue
                idx = i
                break
            if idx is None:
                return False
            exp, seed = run_queue.pop(idx)
            name = run_name(exp, seed)
            p, lf, log = launch(name, EXPERIMENTS[exp], gpu_id)
            active[gpu_id] = (exp, seed, name, p, lf, log)
            print(f"  START {name} on GPU {gpu_id} (PID {p.pid})  "
                  f"[{len(run_queue)} queued]", flush=True)
            return True

        def finish_gpu(gpu_id):
            nonlocal completed
            exp, seed, name, p, lf, log = active[gpu_id]
            lf.close()
            last = "empty"
            try:
                with open(log) as f:
                    lines = f.readlines()
                    if lines:
                        last = lines[-1].strip()
            except OSError:
                pass
            ok = p.returncode == 0 and has_checkpoint(exp, seed)
            completed += 1
            print(f"  DONE {name}: {'OK' if ok else 'FAIL'} — {last}  "
                  f"[{completed}/{total_training}]", flush=True)
            active[gpu_id] = None
            if ok:
                pending_bench.append((exp, seed))

        def run_pending_benchmarks():
            while pending_bench:
                exp, seed = pending_bench.pop(0)
                r = benchmark_run(exp, seed, get_deps(), kernel_mod, device_id=BENCH_DEVICE)
                if r:
                    results[r["name"]] = r
                    save_results(results)
                    print_bench_line("BENCH", r)

        for gpu_id in range(NUM_GPUS):
            start_next(gpu_id)

        while any(a is not None for a in active) or run_queue:
            time.sleep(10)
            for gpu_id in range(NUM_GPUS):
                if active[gpu_id] is not None and active[gpu_id][3].poll() is not None:
                    finish_gpu(gpu_id)
                if active[gpu_id] is None:
                    start_next(gpu_id)
            # Benchmark while GPUs are busy training (bench mostly CPU + GPU 0)
            if all(a is not None for a in active) or not run_queue:
                had = bool(pending_bench)
                run_pending_benchmarks()
                if had and any(a is not None for a in active):
                    print_summary(results)
            if not run_queue and all(a is None for a in active):
                break

        run_pending_benchmarks()

    print("\n\nALL DONE.")
    print_summary(results)


if __name__ == "__main__":
    main()
