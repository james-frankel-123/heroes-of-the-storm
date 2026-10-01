"""
Re-measure the "C++ tree, batched virtual loss (K=32)" row of the supplement's
throughput table after fixing that engine's virtual-loss undo at the root
(training/cuda_mcts/mcts_engine.cpp; the undo was applied to the root node,
which never received virtual loss).

Same protocol as rerun2026/throughput_baseline.py historical_single_thread:
single thread, compile-time K=32, 200 simulations per move, c_puct 2.0,
J_800sim_s9 policy weights, generic_draft_0 opponent, 3 warm-up episodes then
N timed episodes. The fixed and the original module are measured on the same GPU in
alternating blocks of episodes (one process per block), so load from other
jobs affects both alike. Mean episode win probability (the engine's own
value estimate at the end of the episode) is reported as a search sanity check.

The fixed engine is built as a separate module (cuda_mcts_vlfix: same sources,
renamed PYBIND11 module) so the shared cuda_mcts build is not touched:
    cp training/cuda_mcts/{mcts_engine.cpp,fused_forward.cu} <dir>/
    sed -i 's/PYBIND11_MODULE(cuda_mcts, m)/PYBIND11_MODULE(cuda_mcts_vlfix, m)/' <dir>/mcts_engine.cpp
    (CUDAExtension build of those two files as cuda_mcts_vlfix)

Usage (from training/): CUDA_VISIBLE_DEVICES=<gpu> nice -n 19 taskset -c 48-53 \
    python3 paper1_revision/throughput_vl_fixed.py --fixed-dir <dir> [--episodes 120]
Output: paper1_revision/results/throughput_vl_fixed.json
"""
import os
import sys
import json
import time
import argparse
import importlib

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)
sys.path.insert(0, os.path.join(TRAINING_DIR, "cuda_mcts"))

import numpy as np


def measure(which, fixed_dir, start, n):
    """Child process: time n episodes of one module (the two pybind modules
    register the same C++ type names, so they cannot share a process)."""
    import torch
    torch.set_num_threads(4)
    from rerun2026.throughput_baseline import load_nets, flatten, ren, NUM_SIMS, C_PUCT
    if which == "fixed":
        sys.path.insert(0, fixed_dir)
    m = importlib.import_module("cuda_mcts_vlfix" if which == "fixed" else "cuda_mcts")
    net, gd = load_nets()
    pw, poff = flatten(net, ren)
    gw, goff = flatten(gd)
    eng = m.CUDAInferenceEngine(pw, gw, poff, goff, device_id=0)
    for i in range(3):
        m.run_episode(eng, i % 14, i % 3, i % 2, NUM_SIMS, C_PUCT, 42 + i)
    wp = []
    t0 = time.time()
    for i in range(start, start + n):
        w, _ = m.run_episode(eng, i % 14, i % 3, i % 2, NUM_SIMS, C_PUCT, 100 + i)
        wp.append(float(w))
    print(json.dumps({"secs": time.time() - t0, "wp": wp}))


def main():
    import subprocess
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed-dir", required=True)
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--block", type=int, default=40)
    ap.add_argument("--child", default=None)
    ap.add_argument("--start", type=int, default=0)
    a = ap.parse_args()
    if a.child:
        measure(a.child, a.fixed_dir, a.start, a.block)
        return
    t = {"original": 0.0, "fixed": 0.0}
    wp = {"original": [], "fixed": []}
    done = 0
    while done < a.episodes:
        for k in ("original", "fixed"):
            r = subprocess.run([sys.executable, os.path.abspath(__file__), "--fixed-dir", a.fixed_dir,
                                "--child", k, "--start", str(done), "--block", str(a.block)],
                               capture_output=True, text=True, check=True)
            d = json.loads(r.stdout.strip().splitlines()[-1])
            t[k] += d["secs"]
            wp[k] += d["wp"]
        done += a.block
    out = {"protocol": "single thread, K=32, 200 sims, c_puct 2.0, J_800sim_s9; "
                       "original and fixed alternate in blocks, one process each", "episodes": done}
    for k in t:
        out[k] = {"eps_per_s": done / t[k], "ms_per_episode": 1e3 * t[k] / done,
                  "mean_episode_wp": float(np.nanmean(wp[k])),
                  "nan_episodes": int(np.sum(~np.isfinite(wp[k])))}
    out["fixed_over_original"] = out["fixed"]["eps_per_s"] / out["original"]["eps_per_s"]
    json.dump(out, open(os.path.join(HERE, "results", "throughput_vl_fixed.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
