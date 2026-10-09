#!/usr/bin/env python3
"""
MCTS training worker — fully optimized.

1. Pre-allocated numpy ring buffer (zero allocation per batch)
2. Async pipelined: GPU generates batch N+1 while CPU trains on batch N
3. 128 episodes per kernel launch
4. Training on GPU (same device, interleaved with generation)
5. Weight sync every 512 episodes
6. Eval every 5000 episodes, 50 drafts only
"""
import os
import sys
import time
import random
import threading
import queue
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib.util

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cuda_mcts'))

from train_draft_policy import (
    AlphaZeroDraftNet, pretrain_value_head, bootstrap_from_generic_draft,
    STATE_DIM, NUM_HEROES, HEROES,
)
from train_generic_draft import GenericDraftModel
from extract_weights import extract_policy_weights, extract_gd_weights, extract_wp_weights, build_wp_net_offsets, extract_lookup_tables
from shared import MAPS, SKILL_TIERS, heroes_to_multi_hot, map_to_one_hot, tier_to_one_hot

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# Load kernel module. HOTS_HERO_SET=v2 (production: 91 heroes, 15 maps) uses
# the separate build in cuda_mcts/h91/ (cuda_mcts/build_h91.py); v1 the
# research build in cuda_mcts/.
from shared import HERO_SET as _HERO_SET
so_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cuda_mcts')
# MCTS_KERNEL_DIR: load cuda_mcts_kernel.*.so from another build directory (opt-in)
if os.environ.get("MCTS_KERNEL_DIR"):
    so_dir = os.environ["MCTS_KERNEL_DIR"]
elif _HERO_SET == "v2":
    so_dir = os.path.join(so_dir, "h91")
    if not os.path.isdir(so_dir):
        raise RuntimeError("HOTS_HERO_SET=v2 needs the h91 kernel: python cuda_mcts/build_h91.py")
so_files = [f for f in os.listdir(so_dir) if f.startswith('cuda_mcts_kernel.') and f.endswith('.so')]
if len(so_files) != 1:
    raise RuntimeError(f"expected one cuda_mcts_kernel.*.so in {so_dir}, found {so_files}")
spec = importlib.util.spec_from_file_location('cuda_mcts_kernel', os.path.join(so_dir, so_files[0]))
kernel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kernel)

# Config
SAVE_DIR = os.environ.get("MCTS_SAVE_DIR", os.path.dirname(__file__))
NUM_EPISODES = int(os.environ.get("MCTS_NUM_EPISODES", "300000"))
NUM_SIMS = int(os.environ.get("MCTS_NUM_SIMS", "200"))
FRESH = os.environ.get("MCTS_FRESH", "1") == "1"
RUN_NAME = os.environ.get("WANDB_RUN_NAME", "mcts_run")
BATCH_EPISODES = int(os.environ.get("MCTS_BATCH_EPISODES", "128"))
NET_SIZE = os.environ.get("MCTS_NET_SIZE", "base")  # base, large, xlarge
POLICY_HEAD = os.environ.get("MCTS_POLICY_HEAD", "linear")  # linear, deep, step, deep_step
# rerun2026 overrides (all optional; defaults preserve historical behavior):
#   MCTS_WP_PATH        checkpoint path override for the selected MCTS_WP_MODEL type
#   MCTS_GD_PATH        GenericDraft checkpoint override (kernel opponent + bootstrap)
#   MCTS_WP_ZERO_GROUPS comma-separated enriched feature groups to zero at
#                       extraction (M_relational / N_absolute ablations)
#   MCTS_EXCLUDE_IDS    JSON list of replay_ids to exclude from value-head
#                       pretraining (rerun2026 2.55 filter)
WP_PATH_OVERRIDE = os.environ.get("MCTS_WP_PATH", "")
GD_PATH_OVERRIDE = os.environ.get("MCTS_GD_PATH", "")
WP_ZERO_GROUPS = [g for g in os.environ.get("MCTS_WP_ZERO_GROUPS", "").split(",") if g]
EXCLUDE_IDS_PATH = os.environ.get("MCTS_EXCLUDE_IDS", "")
# MCTS_VALUE_PRETRAIN=0 skips value-head pretraining on replay outcomes (default 1,
# historical behavior). Production sets 0: with the backbone frozen at random init
# the head collapses to a constant 0.5 (loss 0.2500, measured 2026-10-03), and the
# search scores leaves with GD rollouts + WP, not the value head.
VALUE_PRETRAIN = os.environ.get("MCTS_VALUE_PRETRAIN", "1") != "0"
# Search algorithm of the CUDA kernel (X2 fix, audits/CONSOLIDATED_AUDIT_2026-10-01.md):
#   chance   (default) opponent turns are chance nodes; the tree reaches later
#            own picks and bans (cuda_mcts/search_v2.cuh)
#   legacy   the pre-fix kernel, bit-for-bit (tree limited to the current
#            own-pick block); use only to reproduce runs made before 2026-10-01
#   rollfwd  open-loop variant (opponent steps re-sampled every visit)
SEARCH_MODE_NAME = os.environ.get("MCTS_SEARCH_MODE", "chance")
SEARCH_MODE = {"legacy": 0, "chance": 1, "rollfwd": 2}[SEARCH_MODE_NAME]
PW_K = float(os.environ.get("MCTS_PW_K", "1.0"))
PW_ALPHA = float(os.environ.get("MCTS_PW_ALPHA", "0.5"))
# Pause-safe checkpointing (opt-in; both unset = historical behavior):
#   MCTS_CKPT_EVERY_SEC  >0: every N seconds (checked at batch boundaries) write
#                        SAVE_DIR/resume_state.pt: network, optimizer, scheduler,
#                        counters, ring buffer, produced-but-unconsumed batches,
#                        engine weights and RNG states. With MCTS_FRESH=0 the run
#                        continues from it instead of the best-eval checkpoint.
#   MCTS_PAUSE_FILE      if this file exists at a batch boundary (or SIGTERM was
#                        received), write resume_state.pt and exit with code 75.
CKPT_EVERY_SEC = float(os.environ.get("MCTS_CKPT_EVERY_SEC", "0"))
PAUSE_FILE = os.environ.get("MCTS_PAUSE_FILE", "")
RESUMABLE = CKPT_EVERY_SEC > 0 or bool(PAUSE_FILE)
PAUSE_EXIT_CODE = 75
WEIGHT_SYNC_INTERVAL = 512
EVAL_INTERVAL = 5000
EVAL_DRAFTS = 50
BUFFER_SIZE = 150_000
BATCH_SIZE = 512


def main():
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Run: {RUN_NAME}")
    print(f"Config: episodes={NUM_EPISODES}, sims={NUM_SIMS}, batch={BATCH_EPISODES}, "
          f"train_device={device}, fresh={FRESH}")
    print(f"GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'none')}")
    if not hasattr(kernel, "SEARCH_CHANCE"):
        if SEARCH_MODE != 0:
            raise RuntimeError("cuda_mcts_kernel build predates the X2 fix; rebuild "
                               "training/cuda_mcts (setup.py) or set MCTS_SEARCH_MODE=legacy")
    print(f"Search: {SEARCH_MODE_NAME} (mode {SEARCH_MODE}, pw_k={PW_K}, pw_alpha={PW_ALPHA})")

    # Load GD
    gd = GenericDraftModel()
    gd_path = GD_PATH_OVERRIDE
    if not gd_path:
        gd_path = os.path.join(os.path.dirname(__file__), "generic_draft_0.pt")
        if not os.path.exists(gd_path):
            gd_path = os.path.join(os.path.dirname(__file__), "generic_draft.pt")
    print(f"GD model: {gd_path}")
    gd.load_state_dict(torch.load(gd_path, weights_only=True, map_location="cpu"))
    gd.eval()
    gd_flat, gd_offsets = extract_gd_weights(gd)

    # Load WP model for CUDA kernel leaf evaluation
    from sweep_enriched_wp import StatsCache, WinProbEnrichedModel, FEATURE_GROUP_DIMS
    WP_GROUPS = ['role_counts', 'team_avg_wr', 'map_delta',
                 'pairwise_counters', 'pairwise_synergies', 'counter_detail',
                 'meta_strength', 'draft_diversity', 'comp_wr']
    enriched_dim = sum(FEATURE_GROUP_DIMS[g] for g in WP_GROUPS)
    from sweep_enriched_wp import INPUT_DIM_BASE
    wp_input_dim = INPUT_DIM_BASE + enriched_dim  # v1: 197 + 86 = 283

    WP_MODEL_TYPE = os.environ.get("MCTS_WP_MODEL", "enriched")
    if WP_MODEL_TYPE == "base":
        # Base WP (no enriched features, 197+8 step embed = 205 input)
        from train_partial_wp import PartialStateWP
        wp_path = WP_PATH_OVERRIDE or os.path.join(os.path.dirname(__file__), "base_partial_wp.pt")
        ckpt = torch.load(wp_path, weights_only=True, map_location="cpu")
        wp_model = PartialStateWP(input_dim=197, step_embed_dim=8, hidden=(256, 128))
        wp_model.load_state_dict(ckpt['model_state_dict'])
        wp_model.eval()
        wp_flat, wp_name_offsets = extract_wp_weights(wp_model)
        wp_offsets = build_wp_net_offsets(wp_model, wp_name_offsets, 197 + 8, use_enriched=False)
        step_embed = wp_model.step_embed.weight.data.cpu().numpy()
        print(f"WP model: base (197+8d, no enriched features)")
    elif WP_MODEL_TYPE == "true_base":
        # True base WP (no enriched features, no step embedding, 197→256→128→1)
        wp_path = WP_PATH_OVERRIDE or os.path.join(os.path.dirname(__file__), "wp_true_base.pt")
        wp_model = WinProbEnrichedModel(197, [256, 128], dropout=0.3)
        wp_model.load_state_dict(torch.load(wp_path, weights_only=True, map_location="cpu"))
        wp_model.eval()
        wp_flat, wp_name_offsets = extract_wp_weights(wp_model)
        wp_offsets = build_wp_net_offsets(wp_model, wp_name_offsets, 197, use_enriched=False)
        step_embed = None
        print(f"WP model: true base (197d, no enriched, no step embed)")
    elif WP_MODEL_TYPE == "enriched_full":
        # Full enriched WP (no step embedding, 283→256→128→1)
        wp_path = WP_PATH_OVERRIDE or os.path.join(os.path.dirname(__file__), "wp_experiment_enriched.pt")
        wp_model = WinProbEnrichedModel(wp_input_dim, [256, 128], dropout=0.3)
        wp_model.load_state_dict(torch.load(wp_path, weights_only=True, map_location="cpu"))
        wp_model.eval()
        wp_flat, wp_name_offsets = extract_wp_weights(wp_model)
        wp_offsets = build_wp_net_offsets(wp_model, wp_name_offsets, wp_input_dim)
        step_embed = None  # zeros in LUT — kernel appends step embed but model ignores it (283 input)
        print(f"WP model: enriched full ({wp_input_dim}d, no step embed)")
    elif WP_MODEL_TYPE == "augmented":
        # Augmented enriched WP (synthetic degenerate data, 283→256→128→1)
        wp_path = WP_PATH_OVERRIDE or os.path.join(os.path.dirname(__file__), "wp_augmented.pt")
        wp_model = WinProbEnrichedModel(wp_input_dim, [256, 128], dropout=0.3)
        wp_model.load_state_dict(torch.load(wp_path, weights_only=True, map_location="cpu"))
        wp_model.eval()
        wp_flat, wp_name_offsets = extract_wp_weights(wp_model)
        wp_offsets = build_wp_net_offsets(wp_model, wp_name_offsets, wp_input_dim)
        step_embed = None
        print(f"WP model: augmented ({wp_input_dim}d, synthetic degenerate data)")
    else:
        # Partial WP (step-conditioned, 291→256→128→1)
        from train_partial_wp import PartialStateWP
        wp_path = WP_PATH_OVERRIDE or os.path.join(os.path.dirname(__file__), "partial_wp.pt")
        ckpt = torch.load(wp_path, weights_only=True, map_location="cpu")
        wp_model = PartialStateWP(input_dim=wp_input_dim, step_embed_dim=8, hidden=(256, 128))
        wp_model.load_state_dict(ckpt['model_state_dict'])
        wp_model.eval()
        wp_flat, wp_name_offsets = extract_wp_weights(wp_model)
        wp_offsets = build_wp_net_offsets(wp_model, wp_name_offsets, wp_input_dim + 8)  # 291
        step_embed = wp_model.step_embed.weight.data.cpu().numpy()  # (16, 8)
        print(f"WP model: partial ({wp_input_dim}+8d step embed)")

    print(f"  WP weights: {len(wp_flat)} floats, {wp_offsets['num_layers']} layers")
    # drift2026 override: build the enriched-feature lookup tables from a
    # patch-cumulative stats file instead of the frozen end-of-time snapshot
    # (MCTS_STATS_BUILD=<build>, e.g. "2.55.14.95918"). Default preserved.
    stats_build = os.environ.get("MCTS_STATS_BUILD", "")
    if stats_build:
        from drift2026 import common as drift_common
        # MCTS_STATS_KIND selects the patch_stats/<kind>/ family (default
        # "cumulative"; e.g. "decayed90k100" for the Q7 champion config).
        stats_kind = os.environ.get("MCTS_STATS_KIND", "cumulative")
        wp_stats = drift_common.load_patch_stats(stats_kind, stats_build)
        print(f"LUT stats: drift2026 {stats_kind}@{stats_build}")
    else:
        wp_stats = StatsCache()
    lut_blob = extract_lookup_tables(wp_stats, step_embed_weights=step_embed)

    # Feature-group ablation (rerun2026 M_relational / N_absolute): zero the
    # selected enriched groups at extraction. Exact zeroing is done by zeroing
    # the WP first-layer weight columns for those input dims (identical to
    # zeroing the features themselves); LUT tables read exclusively by ablated
    # groups are zeroed as well.
    if WP_ZERO_GROUPS:
        if WP_MODEL_TYPE in ("base", "true_base"):
            raise ValueError(f"MCTS_WP_ZERO_GROUPS requires an enriched WP model, "
                             f"got MCTS_WP_MODEL={WP_MODEL_TYPE}")
        from extract_weights import (kernel_enriched_group_cols,
                                     zero_wp_input_columns, zero_lookup_table_groups)
        zero_cols = [197 + c for c in kernel_enriched_group_cols(WP_ZERO_GROUPS)]
        wp_flat = zero_wp_input_columns(wp_flat, wp_name_offsets, wp_model, zero_cols)
        lut_blob = zero_lookup_table_groups(lut_blob, WP_ZERO_GROUPS)
        print(f"WP ablation: zeroed groups {WP_ZERO_GROUPS} "
              f"({len(zero_cols)} input columns)")

    # Init network on GPU for training
    network = AlphaZeroDraftNet(size=NET_SIZE, policy_head_type=POLICY_HEAD).to(device)
    print(f"Policy ({NET_SIZE}, head={POLICY_HEAD}): {sum(p.numel() for p in network.parameters()):,} params on {device}")

    os.makedirs(SAVE_DIR, exist_ok=True)
    ckpt_path = os.path.join(SAVE_DIR, "draft_policy_checkpoint.pt")
    weights_path = os.path.join(SAVE_DIR, "draft_policy.pt")

    start_episode = 0
    best_eval_wp = 0.0
    resume_path = os.path.join(SAVE_DIR, "resume_state.pt")
    resume = None

    if RESUMABLE and not FRESH and os.path.exists(resume_path):
        resume = torch.load(resume_path, weights_only=False, map_location="cpu")  # RNG states must stay on CPU
        # a run must not change search algorithm mid-way (states saved before
        # the X2 fix carry no key and were produced by the legacy kernel)
        _saved = resume.get('search', {"search_mode": "legacy"})
        _now = {"search_mode": SEARCH_MODE_NAME, "pw_k": PW_K, "pw_alpha": PW_ALPHA}
        if _saved.get("search_mode") != SEARCH_MODE_NAME or (
                SEARCH_MODE_NAME != "legacy" and _saved != _now):
            raise RuntimeError(f"resume_state.pt was written with search {_saved}, this run is {_now}; "
                               "set MCTS_SEARCH_MODE/MCTS_PW_K/MCTS_PW_ALPHA to match")
        network.load_state_dict(resume['model_state_dict'])
        start_episode = resume['episode']
        best_eval_wp = resume['best_eval_wp']
        print(f"Resumed from resume_state.pt: episode {start_episode}, best_wp={best_eval_wp:.4f}, "
              f"{len(resume['pending'])} pending batches")
    elif not FRESH and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
        network.load_state_dict(ckpt['model_state_dict'])
        start_episode = ckpt.get('episode', 0)
        best_eval_wp = ckpt.get('best_eval_wp', 0.0)
        print(f"Resumed: episode {start_episode}, best_wp={best_eval_wp:.4f}")
    else:
        if EXCLUDE_IDS_PATH:
            # rerun2026: pin value-head pretraining to the 2.55-filtered snapshot
            import json as _json
            import train_draft_policy as _tdp
            with open(EXCLUDE_IDS_PATH) as _f:
                _exclude = set(_json.load(_f))
            _orig_loader = _tdp.load_replay_data

            def _filtered_loader(*a, **kw):
                rows = _orig_loader(*a, **kw)
                kept = [r for r in rows if r.get("replay_id") not in _exclude]
                print(f"Pretrain data filter: {len(rows)} -> {len(kept)} replays "
                      f"({len(rows) - len(kept)} excluded)")
                return kept

            _tdp.load_replay_data = _filtered_loader
        bootstrap_from_generic_draft(network, device, gd_path=GD_PATH_OVERRIDE or None)
        if VALUE_PRETRAIN:
            pretrain_value_head(network, device)
        else:
            print("Value-head pretraining skipped (MCTS_VALUE_PRETRAIN=0)")

    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150000, eta_min=1e-5)

    if resume is not None:
        optimizer.load_state_dict(resume['optimizer_state_dict'])
        scheduler.load_state_dict(resume['scheduler_state_dict'])
        print("Restored optimizer + scheduler (resume_state)")
    elif not FRESH and os.path.exists(ckpt_path) and start_episode > 0:
        try:
            ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            print("Restored optimizer + scheduler")
        except Exception:
            print("Could not restore optimizer/scheduler")

    # Create kernel engine (with WP model + lookup tables for in-kernel evaluation)
    policy_flat, policy_offsets = extract_policy_weights(network)
    engine = kernel.MCTSKernelEngine(
        policy_flat, gd_flat, wp_flat,
        policy_offsets, gd_offsets, wp_offsets,
        lut_blob,
        max_concurrent=BATCH_EPISODES, device_id=0)
    print(f"CUDA kernel engine (batch={BATCH_EPISODES}, WP in-kernel)")
    # pre-fix builds take no search arguments (legacy is then the only mode)
    search_args = (SEARCH_MODE, PW_K, PW_ALPHA) if hasattr(kernel, "SEARCH_CHANCE") else ()
    try:
        import json as _json
        import subprocess as _sp
        os.makedirs(SAVE_DIR, exist_ok=True)
        _rev = _sp.run(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)),
                       capture_output=True, text=True).stdout.strip()
        import hashlib as _hl
        _so = os.path.join(so_dir, so_files[0])
        with open(_so, "rb") as _f:
            _so_sha = _hl.sha256(_f.read()).hexdigest()
        _bi = os.path.join(so_dir, "BUILD_INFO.json")
        _build = _json.load(open(_bi)) if os.path.exists(_bi) else None
        _json.dump({"search_mode": SEARCH_MODE_NAME, "pw_k": PW_K, "pw_alpha": PW_ALPHA,
                    "num_sims": NUM_SIMS, "kernel_so": _so, "so_sha256": _so_sha,
                    "hero_set": _HERO_SET, "build_info": _build,
                    "build_info_matches_so": (None if _build is None
                                              else _build.get("so_sha256") == _so_sha),
                    "git_head": _rev, "time": time.strftime("%Y-%m-%d %H:%M:%S")},
                   open(os.path.join(SAVE_DIR, "kernel_info.json"), "w"), indent=1)
    except Exception as _e:  # provenance only; never block training
        print(f"kernel_info.json not written: {_e}")

    if HAS_WANDB:
        wandb.init(project="hots-draft-policy", name=RUN_NAME,
                   config={"episodes": NUM_EPISODES, "sims": NUM_SIMS,
                           "batch": BATCH_EPISODES, "engine": "cuda_kernel_v3"})

    # ── Pre-allocated ring buffer (zero allocation per batch) ──
    buf_states = np.zeros((BUFFER_SIZE, STATE_DIM), dtype=np.float32)
    buf_policies = np.zeros((BUFFER_SIZE, NUM_HEROES), dtype=np.float32)
    buf_masks = np.zeros((BUFFER_SIZE, NUM_HEROES), dtype=np.float32)
    buf_values = np.zeros(BUFFER_SIZE, dtype=np.float32)
    buf_write_idx = 0
    buf_size = 0
    if resume is not None:
        buf_states[:] = resume['buf_states']
        buf_policies[:] = resume['buf_policies']
        buf_masks[:] = resume['buf_masks']
        buf_values[:] = resume['buf_values']
        buf_write_idx = resume['gen_write_idx']
        buf_size = resume['buf_size']
    print(f"Ring buffer: {BUFFER_SIZE} entries, "
          f"{(buf_states.nbytes + buf_policies.nbytes + buf_masks.nbytes + buf_values.nbytes) / 1024/1024:.0f} MB")

    n_maps = len(MAPS)
    n_tiers = len(SKILL_TIERS)
    episodes_since_weight_sync = 0
    last_eval_episode = start_episode
    if resume is not None:
        episodes_since_weight_sync = resume['episodes_since_weight_sync']
        last_eval_episode = resume['last_eval_episode']

    train_start = time.time()
    episode = start_episode

    def make_configs(n, ep):
        return np.array([
            [random.randint(0, n_maps-1), random.randint(0, n_tiers-1), (ep+i) % 2]
            for i in range(n)
        ], dtype=np.int32)

    # ── Initial weight sync ──
    policy_flat, _ = extract_policy_weights(network)
    engine_flat = [policy_flat]  # weights the engine currently holds (resume state)
    if resume is not None:
        engine_flat[0] = resume['engine_flat']
    engine.update_weights(engine_flat[0])

    # ── Generation thread for pipelining ──
    gen_queue = queue.Queue(maxsize=2)
    gen_running = True
    gen_seed = [episode]  # mutable for thread access
    gen_write_idx = [buf_write_idx]
    # Resume support: batches produced but not yet consumed are carried over
    # in `prefetched`; gen_hold parks the generator between batches.
    prefetched = []
    gen_hold = threading.Event()
    gen_idle = threading.Event()
    if resume is not None:
        gen_seed[0] = resume['gen_seed']
        prefetched = list(resume['pending'])
        random.setstate(resume['rng_python'])
        np.random.set_state(resume['rng_numpy'])
        torch.set_rng_state(resume['rng_torch'])
        if torch.cuda.is_available() and resume.get('rng_cuda') is not None:
            torch.cuda.set_rng_state_all(resume['rng_cuda'])
        del resume

    def generation_thread():
        while gen_running:
            if gen_hold.is_set():
                gen_idle.set()
                while gen_hold.is_set() and gen_running:
                    time.sleep(0.05)
                gen_idle.clear()
                continue
            batch_count = min(BATCH_EPISODES, NUM_EPISODES - gen_seed[0])
            if batch_count <= 0:
                gen_queue.put(None)
                break
            configs = make_configs(batch_count, gen_seed[0])
            result = engine.run_episodes_into_buffer(
                configs, NUM_SIMS, 2.0, gen_seed[0],
                buf_states, buf_policies, buf_masks, buf_values,
                gen_write_idx[0], BUFFER_SIZE, *search_args
            )
            n_written, wp_values = result
            gen_write_idx[0] = (gen_write_idx[0] + n_written) % BUFFER_SIZE
            gen_seed[0] += batch_count
            gen_queue.put((batch_count, n_written, np.array(wp_values)))

    gen_thread = threading.Thread(target=generation_thread, daemon=True)
    gen_thread.start()

    pause_requested = [False]
    if RESUMABLE:
        import signal
        signal.signal(signal.SIGTERM, lambda *_: pause_requested.__setitem__(0, True))
    last_ckpt_time = time.time()

    def save_resume_state(reason):
        # Park the generator after its in-flight batch; keep what it produced.
        gen_hold.set()
        while not gen_idle.is_set() and gen_thread.is_alive():
            try:
                prefetched.append(gen_queue.get(timeout=0.1))
            except queue.Empty:
                pass
        while True:
            try:
                prefetched.append(gen_queue.get_nowait())
            except queue.Empty:
                break
        state = {
            'model_state_dict': network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'episode': episode, 'best_eval_wp': best_eval_wp,
            'last_eval_episode': last_eval_episode,
            'episodes_since_weight_sync': episodes_since_weight_sync,
            'buf_states': buf_states, 'buf_policies': buf_policies,
            'buf_masks': buf_masks, 'buf_values': buf_values,
            'buf_size': buf_size, 'gen_seed': gen_seed[0], 'gen_write_idx': gen_write_idx[0],
            'pending': list(prefetched), 'engine_flat': engine_flat[0],
            'rng_python': random.getstate(), 'rng_numpy': np.random.get_state(),
            'rng_torch': torch.get_rng_state(),
            'rng_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'search': {"search_mode": SEARCH_MODE_NAME, "pw_k": PW_K, "pw_alpha": PW_ALPHA},
        }
        tmp = resume_path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, resume_path)
        import json as _json
        with open(resume_path[:-3] + ".json.tmp", "w") as f:
            _json.dump({"episode": episode, "reason": reason, "time": time.time(),
                        "pending": len(prefetched)}, f)
        os.replace(resume_path[:-3] + ".json.tmp", resume_path[:-3] + ".json")
        print(f"  Resume checkpoint @ {episode} ({reason}, {len(prefetched)} pending batches)", flush=True)

    # ══════════════════════════════════════════════════════
    # MAIN LOOP: pipelined generate (GPU thread) + train (main thread)
    # ══════════════════════════════════════════════════════
    last_wp = 0.0

    while episode < NUM_EPISODES:
        # Wait for next batch from generation thread
        item = prefetched.pop(0) if prefetched else gen_queue.get()
        if item is None:
            break
        batch_count, n_written, wp_values = item
        episode += batch_count
        episodes_since_weight_sync += batch_count

        # WP values are already written into buf_values by the C++ layer
        # (kernel computes symmetrized WP in-kernel, C++ writes to ring buffer)
        buf_size = min(buf_size + n_written, BUFFER_SIZE)
        last_wp = wp_values[-1] if len(wp_values) > 0 else 0.5

        # Train on GPU (while generation thread is already launching next batch)
        if buf_size >= BATCH_SIZE:
            network.train()
            n_train_steps = max(1, batch_count // 8)
            for _ in range(n_train_steps):
                indices = np.random.randint(0, buf_size, size=BATCH_SIZE)
                states_t = torch.from_numpy(buf_states[indices]).to(device)
                policies_t = torch.from_numpy(buf_policies[indices]).to(device)
                masks_t = torch.from_numpy(buf_masks[indices]).to(device)
                values_t = torch.from_numpy(buf_values[indices]).to(device)

                pred_logits, pred_values = network(states_t, masks_t)
                pred_log_probs = F.log_softmax(pred_logits, dim=1)
                policy_loss = -(policies_t * pred_log_probs).sum(dim=1).mean()
                value_loss = F.mse_loss(pred_values, values_t)
                loss = policy_loss + value_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(network.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

        # Weight sync
        if episodes_since_weight_sync >= WEIGHT_SYNC_INTERVAL:
            network.eval()
            policy_flat, _ = extract_policy_weights(network)
            engine.update_weights(policy_flat)
            engine_flat[0] = policy_flat
            episodes_since_weight_sync = 0

        # Logging
        elapsed = time.time() - train_start
        eps = (episode - start_episode) / elapsed if elapsed > 0 else 0
        eta = (NUM_EPISODES - episode) / eps / 3600 if eps > 0 else 0
        print(f"Episode {episode}: wp={last_wp:.4f} buffer={buf_size} "
              f"lr={scheduler.get_last_lr()[0]:.6f} [{eps:.1f} ep/s, ETA {eta:.1f}h]")
        if HAS_WANDB and wandb.run:
            wandb.log({"episode": episode, "last_wp": last_wp,
                       "buffer_size": buf_size, "eps": eps}, step=episode)

        # Eval (infrequent, lightweight)
        if episode - last_eval_episode >= EVAL_INTERVAL or episode >= NUM_EPISODES:
            last_eval_episode = episode
            network.eval()
            policy_flat, _ = extract_policy_weights(network)
            engine.update_weights(policy_flat)
            engine_flat[0] = policy_flat

            # Eval uses a temporary buffer (not the training ring buffer)
            eval_buf_s = np.zeros((EVAL_DRAFTS * 8, STATE_DIM), dtype=np.float32)
            eval_buf_p = np.zeros((EVAL_DRAFTS * 8, NUM_HEROES), dtype=np.float32)
            eval_buf_m = np.zeros((EVAL_DRAFTS * 8, NUM_HEROES), dtype=np.float32)
            eval_buf_v = np.zeros(EVAL_DRAFTS * 8, dtype=np.float32)
            eval_configs = make_configs(EVAL_DRAFTS, 99999)
            result = engine.run_episodes_into_buffer(
                eval_configs, NUM_SIMS // 2, 2.0, 99999,
                eval_buf_s, eval_buf_p, eval_buf_m, eval_buf_v, 0, EVAL_DRAFTS * 8,
                *search_args)
            _, eval_wps = result
            eval_wps = np.array(eval_wps)
            avg_wp = np.mean(eval_wps)
            win_rate = np.mean([1.0 if w > 0.5 else 0.0 for w in eval_wps])
            print(f"\n  EVAL @ {episode}: avg_wp={avg_wp:.4f} win_rate={win_rate:.1%}\n")

            if HAS_WANDB and wandb.run:
                wandb.log({"eval/avg_wp": avg_wp, "eval/win_rate": win_rate}, step=episode)

            if avg_wp > best_eval_wp:
                best_eval_wp = avg_wp
                torch.save(network.state_dict(), weights_path)
                torch.save({
                    'model_state_dict': network.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'episode': episode,
                    'best_eval_wp': best_eval_wp,
                }, ckpt_path)
                print(f"  New best! Saved to {SAVE_DIR}\n")

        # Pause / periodic resume checkpoint (opt-in, see MCTS_CKPT_EVERY_SEC)
        if RESUMABLE:
            pause = pause_requested[0] or (PAUSE_FILE and os.path.exists(PAUSE_FILE))
            periodic = CKPT_EVERY_SEC > 0 and time.time() - last_ckpt_time >= CKPT_EVERY_SEC
            if pause or periodic:
                save_resume_state("pause" if pause else "periodic")
                last_ckpt_time = time.time()
                if pause:
                    gen_running = False
                    print(f"PAUSED at episode {episode} (exit {PAUSE_EXIT_CODE})", flush=True)
                    sys.stdout.flush()
                    os._exit(PAUSE_EXIT_CODE)
                gen_hold.clear()

    gen_running = False
    gen_thread.join(timeout=10)

    if HAS_WANDB and wandb.run:
        wandb.finish()
    elapsed = time.time() - train_start
    total_eps = episode - start_episode
    print(f"Complete. {total_eps} episodes in {elapsed/3600:.1f}h ({total_eps/elapsed:.1f} ep/s)")


if __name__ == "__main__":
    main()
