"""Smoke test for a remote worker.
Imports the trainer stack, loads the extension modules, and runs the MCTS
worker's setup path on CPU up to (not including) value pretraining and
self-play: GD + leak-free leaf WP load, own deploy stats -> kernel LUTs,
policy-net init, kernel weight extraction. Also imports the paper1_revision
benchmark/tournament modules and checks their input files exist.
Run: python ~/hots/repo/training/remote_workers/smoke.py [--gpu]
--gpu also checks the device (tiny matmul + kernel engine construction).
(Do not hide the GPU with CUDA_VISIBLE_DEVICES="": on driver 576.52 WSL's
libcuda aborts in cuInit with a double free.)
"""
import os
import sys
import json
import importlib.util

GPU = "--gpu" in sys.argv
T = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [T, os.path.join(T, "cuda_mcts")]
os.chdir(T)

import torch
import numpy as np
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}), numpy {np.__version__}, python {sys.version.split()[0]}")
for m in ("psycopg2", "onnx", "onnxruntime", "filelock", "matplotlib", "scipy"):
    __import__(m)
print("third-party imports OK")


def load_so(d, pref):
    so = [f for f in os.listdir(d) if f.startswith(pref) and f.endswith(".so")]
    assert so, f"{pref} not built in {d}"
    spec = importlib.util.spec_from_file_location(pref.rstrip("."), os.path.join(d, so[0]))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


k = load_so(os.path.join(T, "cuda_mcts"), "cuda_mcts_kernel")
load_so(os.path.join(T, "cuda_mcts"), "cuda_mcts.")
load_so(os.path.join(T, "overfit2026", "cuda_ofit"), "ofit_kernel")
print("extension modules load OK:", [n for n in dir(k) if not n.startswith("_")][:8])

from paper1_revision import core
meta = json.load(open(os.path.join(core.MODEL_DIR, "enriched.json")))
wp_path = os.path.join(core.MODEL_DIR, f"enriched_s{meta['selected_seed']}.pt")
os.environ.setdefault("WP_STATS_PATH", core.stats_path("deploy"))
import sweep_enriched_wp as swp
swp.FROZEN_STATS_PATH = core.stats_path("deploy")
comp = core.comp_path("deploy")


def _load_compositions(self):
    raw = json.load(open(comp))
    self.comp_data = {t: {",".join(sorted(c["roles"])): (c["winRate"], c["games"]) for c in cs}
                      for t, cs in raw.items()}
swp.StatsCache._load_compositions = _load_compositions

from train_draft_policy import AlphaZeroDraftNet
from train_generic_draft import GenericDraftModel
from extract_weights import (extract_policy_weights, extract_gd_weights, extract_wp_weights,
                             build_wp_net_offsets, extract_lookup_tables)

gd = GenericDraftModel()
gd.load_state_dict(torch.load(os.path.join(T, "rerun2026", "models", "generic_draft_0.pt"),
                              weights_only=True, map_location="cpu"))
gd_flat, _gd_offs = extract_gd_weights(gd.eval())
WP_GROUPS = ['role_counts', 'team_avg_wr', 'map_delta', 'pairwise_counters', 'pairwise_synergies',
             'counter_detail', 'meta_strength', 'draft_diversity', 'comp_wr']
dim = 197 + sum(swp.FEATURE_GROUP_DIMS[g] for g in WP_GROUPS)
wp = swp.WinProbEnrichedModel(dim, [256, 128], dropout=0.3)
wp.load_state_dict(torch.load(wp_path, weights_only=True, map_location="cpu"))
wp_flat, names = extract_wp_weights(wp.eval())
offs = build_wp_net_offsets(wp, names, dim)
lut = extract_lookup_tables(swp.StatsCache(), step_embed_weights=None)
net = AlphaZeroDraftNet(size="base", policy_head_type="linear")
pol = extract_policy_weights(net)
print(f"worker setup path OK on CPU: GD {len(gd_flat)} floats, WP {os.path.basename(wp_path)} "
      f"{len(wp_flat)} floats/{offs['num_layers']} layers, LUT blob ok, "
      f"policy {sum(p.numel() for p in net.parameters()):,} params")

# value-pretraining corpus present (not parsed: 2.9 GB)
import shared
sp = shared._REPLAY_SNAPSHOT_PATH
print(f"snapshot {os.path.basename(sp)}: {os.path.getsize(sp)/1e9:.2f} GB")
assert os.path.exists(os.path.join(core.CACHE, "mcts_pretrain_exclude.json"))

# benchmark / tournament modules
from paper1_revision import bench_mcts, tournament, train_mcts  # noqa: F401
from overfit2026 import search  # noqa: F401
for p in [os.path.join(T, "rerun2026", "models", f"generic_draft_{i}.pt") for i in range(5)] + \
         [os.path.join(T, "rerun2026", "models", f) for f in bench_mcts.SUB_WP_FILES.values()] + \
         [os.path.join(T, "qm2026", "results", "qm_wp_v0.pt"),
          os.path.join(T, "overfit2026", "cache", "snapshot_games.pkl.gz")]:
    assert os.path.exists(p), p
print("paper1_revision bench/tournament imports + inputs OK")
if GPU:
    assert torch.cuda.is_available()
    x = torch.randn(1024, 1024, device="cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}, cap {torch.cuda.get_device_capability(0)}, "
          f"matmul ok ({(x @ x).sum().item():.1f})")
    pol_flat, pol_offs = pol
    eng = k.MCTSKernelEngine(pol_flat, gd_flat, wp_flat, pol_offs, _gd_offs, offs, lut,
                             max_concurrent=8, device_id=0)
    print("kernel engine constructed on GPU")
print("SMOKE OK")
