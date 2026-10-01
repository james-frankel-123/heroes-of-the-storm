"""
Record (or check) the pre-fix kernel's behaviour on fixed benchmark seeds.

  record: run a kernel build (default: the installed pre-fix .so) and save
          every output to x2_results/legacy_reference.npz
  check:  run another build with search_mode=0 and require bit-identical
          outputs (np.array_equal on every array, float bits included)

Cases (64 drafts each, configs seed 20261001, c_puct 2.0, policy prior
new:F_oof_s0, GD generic_draft_0, revision leak-free enriched WP):
  A  run_episodes, 200 sims, seed 1234, root T=1
  B  run_episodes, 400 sims, seed 1235, root T=0 (argmax)
  C  run_episodes, 100 sims, seed 1236, T=1, root Dirichlet eps 0.25
  D  run_episodes_into_buffer (training path), 200 sims, seed 777

Usage (from training/):
  python3 cuda_mcts/x2_record_legacy.py record [--so PATH]
  python3 cuda_mcts/x2_record_legacy.py check --so PATH
"""
import os
import sys
import json
import argparse
import hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np

import x2_common as C

OUT = os.path.join(HERE, "x2_results")
REF = os.path.join(OUT, "legacy_reference.npz")
N = 64
CASES = {"A": dict(sims=200, seed=1234, T=1.0, eps=0.0),
         "B": dict(sims=400, seed=1235, T=0.0, eps=0.0),
         "C": dict(sims=100, seed=1236, T=1.0, eps=0.25)}


def run_all(kernel, legacy_flag):
    eng = C.make_engine(kernel, "new:F_oof_s0", gd_idx=0, max_concurrent=N)
    cfg = C.configs(N, 20261001)
    out = {}
    for name, c in CASES.items():
        kw = {"search_mode": 0} if legacy_flag else {}
        res = eng.run_episodes(cfg, c["sims"], 2.0, c["seed"], c["T"], 0.3, c["eps"], **kw)
        for k, v in C.pack(res).items():
            out[f"{name}_{k}"] = v
    B = 8 * N
    bs, bp = np.zeros((B, 290), np.float32), np.zeros((B, 90), np.float32)
    bm, bv = np.zeros((B, 90), np.float32), np.zeros(B, np.float32)
    args = (cfg, 200, 2.0, 777, bs, bp, bm, bv, 0, B)
    nw, wps = eng.run_episodes_into_buffer(*args, search_mode=0) if legacy_flag \
        else eng.run_episodes_into_buffer(*args)
    out.update({"D_n": np.array([nw]), "D_wp": np.asarray(wps), "D_states": bs,
                "D_policies": bp, "D_masks": bm, "D_values": bv})
    return out


def digest(d):
    h = hashlib.sha256()
    for k in sorted(d):
        h.update(k.encode())
        h.update(np.ascontiguousarray(d[k]).tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["record", "check"])
    ap.add_argument("--so", default=None)
    a = ap.parse_args()
    kernel = C.load_kernel(a.so)
    os.makedirs(OUT, exist_ok=True)
    if a.cmd == "record":
        d = run_all(kernel, legacy_flag=False)
        np.savez_compressed(REF, **d)
        meta = {"so": os.path.realpath(a.so or [os.path.join(HERE, f) for f in os.listdir(HERE)
                                                  if f.startswith("cuda_mcts_kernel") and f.endswith(".so")][0]),
                "sha256_outputs": digest(d), "n": N, "cases": CASES,
                "policy": "new:F_oof_s0", "gd": "rerun2026 generic_draft_0",
                "wp": "paper1_revision leak-free enriched + own deploy stats",
                "configs_seed": 20261001}
        so = meta["so"]
        meta["so_sha256"] = hashlib.sha256(open(so, "rb").read()).hexdigest()
        json.dump(meta, open(os.path.join(OUT, "legacy_reference.json"), "w"), indent=1)
        print("recorded", meta["sha256_outputs"])
        for k in ("A", "B", "C"):
            print(k, "mean terminal WP", float(d[f"{k}_wp"].mean()))
    else:
        ref = dict(np.load(REF))
        d = run_all(kernel, legacy_flag=True)
        bad = [k for k in ref if not np.array_equal(ref[k], d[k])]
        print("arrays:", len(ref), "mismatched:", bad)
        print("ref digest", digest(ref))
        print("new digest", digest(d))
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
