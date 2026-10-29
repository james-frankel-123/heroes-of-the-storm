"""
MCTS throughput of the fused CUDA kernel as used in the revision (chance
nodes at opponent turns, composition fallback mid_high_low), with the same
protocol as rerun2026/throughput_apples.py (the submission's 151 ep/s fused
and 68 ep/s host-pipeline measurements, RTX PRO 6000 Blackwell):
  * fused kernel: 128 concurrent episodes per launch, one warm-up launch, then
    6 timed launches; 200 simulations per move (and 400, the operating point);
    search_mode 0 (the submission's search) and 1 (the revision's);
  * host pipeline (mcts_apples, the tuned KataGo-style baseline): the
    equal-semantics configuration of the submission (leaf batch 1, 256
    threads, eval batch 1024, 2/2/2 streams). It implements only the
    submission's search (leaves expand at our turns), so it is compared with
    the fused kernel in mode 0.
Weights: the paper's operating-point agent F_oof_s0, the site-tier leak-free
enriched leaf WP with its deploy statistics and own composition table, GD 0 of
the p1site pool.

  P1_TIERS=site python3 paper1_revision/throughput_fixed_kernel.py <out.json>
"""
import os
import sys
import json
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING = os.path.dirname(HERE)
sys.path.insert(0, TRAINING)
sys.path.insert(0, os.path.join(TRAINING, "cuda_mcts"))
os.environ.setdefault("P1_TIERS", "site")
from paper1_revision import core  # noqa: E402

os.environ["WP_STATS_PATH"] = core.stats_path("deploy")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import sweep_enriched_wp as swp  # noqa: E402

COMP = core.comp_path("deploy")


def _load_compositions(self):
    raw = json.load(open(COMP))
    self.comp_data = {t: {",".join(sorted(c["roles"])): (c["winRate"], c["games"]) for c in cs}
                      for t, cs in raw.items()}


swp.StatsCache._load_compositions = _load_compositions
NUM_SIMS = 200
C_PUCT = 2.0
POLICY = os.path.join(core.MCTS_RUNS, "F_oof_s0", "draft_policy.pt")
GD = os.path.join(TRAINING, "rerun2026", "ns", "p1site", "models", "generic_draft_0.pt")
meta = json.load(open(os.path.join(core.MODEL_DIR, "enriched.json")))
WP = os.path.join(core.MODEL_DIR, f"enriched_s{meta['selected_seed']}.pt")
TS = os.path.join(core.MODEL_DIR, "_throughput_policy_traced.pt")


def build():
    from train_draft_policy import AlphaZeroDraftNet
    from train_generic_draft import GenericDraftModel
    from extract_weights import (extract_policy_weights, extract_gd_weights, extract_wp_weights,
                                 build_wp_net_offsets, extract_lookup_tables)
    net = AlphaZeroDraftNet()
    net.load_state_dict(torch.load(POLICY, weights_only=True, map_location="cpu"))
    net.eval()
    gd = GenericDraftModel()
    gd.load_state_dict(torch.load(GD, weights_only=True, map_location="cpu"))
    gd.eval()
    wp = swp.WinProbEnrichedModel(283, [256, 128], dropout=0.3)
    wp.load_state_dict(torch.load(WP, weights_only=True, map_location="cpu"))
    wp.eval()
    pf, po = extract_policy_weights(net)
    gf, go = extract_gd_weights(gd)
    wf, wn = extract_wp_weights(wp)
    wo = build_wp_net_offsets(wp, wn, 283)
    lut = extract_lookup_tables(swp.StatsCache())
    with torch.no_grad():
        torch.jit.trace(net, (torch.randn(4, 290), torch.ones(4, 90))).save(TS)
    return dict(pf=pf, po=po, gf=gf, go=go, wf=wf, wo=wo, lut=lut)


def fused(w, mode, sims):
    import cuda_mcts_kernel as k
    free0 = torch.cuda.mem_get_info(0)[0]
    t0 = time.time()
    eng = k.MCTSKernelEngine(w["pf"], w["gf"], w["wf"], w["po"], w["go"], w["wo"], w["lut"],
                             max_concurrent=128, device_id=0)
    startup = time.time() - t0
    cfgs = np.array([[i % 14, (i // 14) % 3, i % 2] for i in range(128)], dtype=np.int32)
    eng.run_episodes(cfgs, sims, C_PUCT, 42, 1.0, 0.3, 0.0, mode)
    torch.cuda.synchronize()
    t0 = time.time()
    for wv in range(6):
        eng.run_episodes(cfgs, sims, C_PUCT, 100 + wv, 1.0, 0.3, 0.0, mode)
    dt = time.time() - t0
    mem = (free0 - torch.cuda.mem_get_info(0)[0]) / 1e6
    del eng
    return {"search_mode": mode, "sims": sims, "eps_per_s": round(768 / dt, 1), "episodes": 768,
            "elapsed_s": round(dt, 2), "engine_startup_s": round(startup, 3), "gpu_memory_mb": round(mem, 1)}


def host(w):
    import mcts_apples
    eng = mcts_apples.ApplesEngine(TS, w["gf"], w["go"], w["wf"], w["wo"], w["lut"], 0)
    r = dict(eng.benchmark(n_threads=256, n_episodes=256 * 10, warmup_episodes=256 * 2, num_sims=NUM_SIMS,
                           leaf_batch=1, max_eval_batch=1024, policy_streams=2, gd_streams=2, wp_streams=2,
                           c_puct=C_PUCT, seed=42, virtual_loss=3.0))
    r.pop("profile", None)
    del eng
    return r


def main():
    import cuda_mcts_kernel as k
    out = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "kernel_comp_fallback_order": getattr(k, "COMP_FALLBACK_ORDER", None),
           "policy": os.path.relpath(POLICY, TRAINING), "wp": os.path.relpath(WP, TRAINING),
           "gd": os.path.relpath(GD, TRAINING), "protocol": __doc__.strip().splitlines()[0]}
    w = build()
    out["fused"] = [fused(w, m, s) for m, s in ((0, 200), (1, 200), (0, 400), (1, 400))]
    for r in out["fused"]:
        print(r, flush=True)
    try:
        out["host_equal_semantics_mode0"] = host(w)
        print(out["host_equal_semantics_mode0"], flush=True)
    except Exception as e:  # the host baseline module may be missing on a worker
        out["host_equal_semantics_mode0"] = {"error": repr(e)}
    f = {(r["search_mode"], r["sims"]): r["eps_per_s"] for r in out["fused"]}
    h = out["host_equal_semantics_mode0"].get("eps_per_s")
    out["summary"] = {"fused_chance_200": f[(1, 200)], "fused_legacy_200": f[(0, 200)],
                      "fused_chance_400": f[(1, 400)], "fused_legacy_400": f[(0, 400)],
                      "chance_over_legacy_200": round(f[(1, 200)] / f[(0, 200)], 3),
                      "host_mode0_200": h,
                      "fused_mode0_over_host": round(f[(0, 200)] / h, 2) if h else None}
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    print(json.dumps(out["summary"]))


if __name__ == "__main__":
    main()
