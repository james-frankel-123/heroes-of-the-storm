"""Build a variant of the h91 production kernel into OUTDIR (outside cuda_mcts/h91), from the
current sources or from a git revision (needs a git checkout), optionally with the sampling
audit (-DP1_SAMPLE_AUDIT). Same flags as build_h91.py. For test_sampling_safe.py.
Usage: python cuda_mcts/build_variant.py OUTDIR [--rev REV | --src DIR] [--audit] [--heroes 91 --maps 15]"""
import argparse, os, shutil, subprocess, sys, tempfile
ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--rev"); ap.add_argument("--audit", action="store_true")
ap.add_argument("--src", help="take the sources from this directory instead (a host without git)")
ap.add_argument("--research", action="store_true", help="the v1 research build (setup.py's flags: hots_dims defaults, no extra defines)")
ap.add_argument("--heroes", type=int, default=91); ap.add_argument("--maps", type=int, default=15)
a = ap.parse_args()
SRC = os.path.dirname(os.path.abspath(__file__))
FILES = ["mcts_kernel.cu", "kernel_bindings.cpp", "hots_dims.h", "device_forward.cuh",
         "enriched_features.cuh", "search_v2.cuh", "search_v2_util.cuh", "tree_v2.h"]
work = tempfile.mkdtemp(prefix="kbuild_")
for f in FILES:
    if a.rev:
        data = subprocess.run(["git", "show", f"{a.rev}:training/cuda_mcts/{f}"], cwd=SRC, check=True, capture_output=True).stdout
        open(os.path.join(work, f), "wb").write(data)
    else:
        shutil.copy(os.path.join(a.src or SRC, f), work)
defs = ([] if a.research else [f"-DNUM_HEROES={a.heroes}", f"-DNUM_MAPS={a.maps}", "-DCOMP_FALLBACK_MID_HIGH_LOW"]) \
    + (["-DP1_SAMPLE_AUDIT"] if a.audit else [])
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
os.makedirs(a.out, exist_ok=True)
os.chdir(work)
setup(name="cuda_mcts_kernel_variant",
      ext_modules=[CUDAExtension("cuda_mcts_kernel", ["mcts_kernel.cu", "kernel_bindings.cpp"],
                                 extra_compile_args={"cxx": ["-O3", "-std=c++17"] + defs,
                                                     "nvcc": ["-O3", "--use_fast_math", "-std=c++17", "--expt-relaxed-constexpr"] + defs})],
      cmdclass={"build_ext": BuildExtension},
      script_args=["build_ext", "--build-lib", a.out, "--build-temp", os.path.join(work, "tmp")])
print("BUILT", a.out, defs, "rev", a.rev or "working tree")
