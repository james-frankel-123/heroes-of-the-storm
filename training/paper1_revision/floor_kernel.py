"""
Support floor inside the CUDA self-play kernel (MCTS trained under the floor).

The -DP1_FLOOR kernel build (cuda_mcts/mcts_kernel.cu own_mask) restricts our own
picks at the root, at leaf expansion and in our rollout picks to the role mask
AND the triple floor of support_mitigation.py (every triple with our current
picks has >= FLOOR training teams). Tables (host -> device, p1_floor_set):
  bits       triple-supported bit per sorted (a, b, c), index (a*H + b)*H + c
  role_ok    base-6 role-count code (BLIZZ_ROLES order) -> completable to a valid
             final team with the remaining picks (constrained_search._feasible,
             availability not limited: the CPU mask also checks that enough
             heroes of a role are free, which never binds with 90 heroes and at
             most 15 taken)
  hero_role  constrained_search.HERO_BLIZZ_IDX

Subcommands (from training/, p1site env, on a GPU host):
  build OUTDIR          build the floor kernel (research build flags, -DP1_FLOOR)
  test OUTDIR           floor on: every recorded root mask equals the CPU mask
                        (floor_search.own_mask), visit targets stay inside it, our
                        final teams clear the floor; floor off: equals the stock
                        build's episodes bit for bit (run in two subprocesses)
"""
import os
import sys
import json
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)
import numpy as np

H = 90


def tables():
    from paper1_revision.support_mitigation import triple_counts, FLOOR
    from rerun2026.constrained_search import _feasible, HERO_BLIZZ_IDX
    c3 = np.asarray(triple_counts()).reshape(-1)
    assert c3.size == H ** 3
    ok = (c3 >= FLOOR).astype(np.uint8)
    nwords = (H ** 3 + 31) // 32
    packed = np.packbits(np.concatenate([ok, np.zeros(nwords * 32 - ok.size, np.uint8)]), bitorder="little")
    bits = np.ascontiguousarray(packed.view("<u4").astype(np.uint32))
    role_ok = np.zeros(46656, np.uint8)
    for code in range(46656):
        c = [(code // 6 ** r) % 6 for r in range(6)]
        n = sum(c)
        if n > 5:
            continue
        k = 5 - n
        role_ok[code] = 1 if _feasible(tuple(c), tuple([min(5, k)] * 6), k) else 0
    hero_role = np.ascontiguousarray(np.asarray(HERO_BLIZZ_IDX, np.int32))
    return bits, role_ok, hero_role


def load(outdir, on=1):
    from cuda_mcts import x2_common as X
    so = [f for f in os.listdir(outdir) if f.startswith("cuda_mcts_kernel.") and f.endswith(".so")]
    assert len(so) == 1, so
    k = X.load_kernel(os.path.join(outdir, so[0]))
    bits, role_ok, hero_role = tables()
    k.p1_floor_set(bits, role_ok, hero_role, int(on))
    return k


def build(outdir):
    env = dict(os.environ)
    src = os.path.join(TRAINING_DIR, "cuda_mcts")
    # research build flags (setup.py) + the floor
    code = f"""
import os, shutil, tempfile, sys
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
FILES = ["mcts_kernel.cu", "kernel_bindings.cpp", "hots_dims.h", "device_forward.cuh",
         "enriched_features.cuh", "search_v2.cuh", "search_v2_util.cuh", "tree_v2.h"]
work = tempfile.mkdtemp(prefix="kfloor_")
for f in FILES: shutil.copy(os.path.join({src!r}, f), work)
defs = ["-DP1_FLOOR"] + (["-DCOMP_FALLBACK_MID_HIGH_LOW"] if os.environ.get("HOTS_COMP_FALLBACK_MID_HIGH_LOW") == "1" else [])
os.makedirs({outdir!r}, exist_ok=True); os.chdir(work)
setup(name="cuda_mcts_kernel_floor",
      ext_modules=[CUDAExtension("cuda_mcts_kernel", ["mcts_kernel.cu", "kernel_bindings.cpp"],
                   extra_compile_args={{"cxx": ["-O3", "-std=c++17"] + defs,
                                        "nvcc": ["-O3", "--use_fast_math", "-std=c++17", "--expt-relaxed-constexpr"] + defs}})],
      cmdclass={{"build_ext": BuildExtension}},
      script_args=["build_ext", "--build-lib", {outdir!r}, "--build-temp", os.path.join(work, "tmp")])
print("BUILT", {outdir!r}, defs)
"""
    sys.exit(subprocess.call([sys.executable, "-c", code], env=env))


def episodes(so_dir, on, n=256, sims=64, seed=20261111):
    """Self-play episodes with the training settings (chance search, root temp 1, no noise)."""
    from cuda_mcts import x2_common as X
    if so_dir is None:
        k = X.load_kernel()
    else:
        k = load(so_dir, on)
    eng = X.make_engine(k, "new:F_oof_s0", gd_idx=0, max_concurrent=n)
    cf = X.configs(n, seed)
    return eng.run_episodes(cf, sims, 2.0, seed, root_temp=1.0, dir_alpha=0.3, dir_eps=0.0,
                            search_mode=X.MODES["chance"], pw_k=1.0, pw_alpha=0.5)


def test(outdir):
    from paper1_revision import floor_search as fs
    from paper1_revision.support_mitigation import triple_counts, FLOOR
    from itertools import combinations
    res = episodes(outdir, 1)
    c3 = np.asarray(triple_counts()).reshape(-1)
    n_ex = mism = outside = pick_ex = 0
    worst = []
    for wp, ex, term, our in res:
        for s, p, m in ex:
            x = np.asarray(s, np.float32)
            n_ex += 1
            step = int(round(x[287] * 15))
            from paper1_revision.search_actor import IS_PICK, free_mask
            ref = fs.own_mask(x, int(our), True) if IS_PICK[step] == 1.0 else free_mask(x)
            got = np.asarray(m) > 0.5
            if (got != ref).any():
                mism += 1
                if len(worst) < 5:
                    worst.append({"step": step, "kernel_only": np.flatnonzero(got & ~ref).tolist(),
                                  "cpu_only": np.flatnonzero(ref & ~got).tolist()})
            outside += int((np.asarray(p)[~got] > 0).any())
            pick_ex += int(IS_PICK[step] == 1.0)
        t = np.asarray(term)
        team = np.flatnonzero(t[(0 if our == 0 else 90):(90 if our == 0 else 180)] > 0.5)
        mn = min(c3[(a * H + b) * H + c] for a, b, c in combinations(sorted(team.tolist()), 3))
        worst.append({"min3": int(mn)}) if mn < FLOOR and len(worst) < 10 else None
    lows = sum(1 for wp, ex, term, our in res
               if (lambda tm: min(c3[(a * H + b) * H + c] for a, b, c in combinations(sorted(tm), 3)))(
                   np.flatnonzero(np.asarray(term)[(0 if our == 0 else 90):(90 if our == 0 else 180)] > 0.5).tolist()) < FLOOR)
    out = {"examples": n_ex, "pick_examples": pick_ex, "mask_mismatch": mism, "policy_outside_mask": outside,
           "episodes": len(res), "our_final_below_floor": lows, "examples_detail": worst}
    print(json.dumps(out))
    # floor off == stock build
    a = subprocess.run([sys.executable, __file__, "_dump", "-", "0"], capture_output=True, text=True, cwd=TRAINING_DIR)
    b = subprocess.run([sys.executable, __file__, "_dump", outdir, "0"], capture_output=True, text=True, cwd=TRAINING_DIR)
    same = a.stdout.strip().splitlines()[-1:] == b.stdout.strip().splitlines()[-1:] and a.returncode == 0 == b.returncode
    print(json.dumps({"floor_off_equals_stock": same, "stock": a.stdout.strip()[-200:], "floor_off": b.stdout.strip()[-200:],
                      "err": (a.stderr[-300:] if a.returncode else "") + (b.stderr[-300:] if b.returncode else "")}))


def _dump(so_dir, on):
    import hashlib
    res = episodes(None if so_dir == "-" else so_dir, int(on))
    h = hashlib.sha256()
    for wp, ex, term, our in res:
        h.update(np.float32(wp).tobytes())
        h.update(np.asarray(term, np.float32).tobytes())
        for s, p, m in ex:
            h.update(np.asarray(p, np.float32).tobytes())
    print(h.hexdigest())


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "build":
        build(sys.argv[2])
    elif c == "test":
        test(sys.argv[2])
    elif c == "_dump":
        _dump(sys.argv[2], sys.argv[3])
