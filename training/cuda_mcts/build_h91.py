"""
Build the production v2 MCTS kernel (91 heroes, 15 maps; shared.py
HOTS_HERO_SET=v2) into cuda_mcts/h91/, next to but apart from the v1 research
build, and record provenance in h91/BUILD_INFO.json:

  source_commit   last commit touching training/cuda_mcts (git), or
                  $HOTS_SOURCE_COMMIT on machines without .git (remote sync)
  source_dirty    tracked kernel sources differ from that commit (git only)
  sources_sha256  digest of the kernel sources actually compiled
  so_sha256       digest of the built extension

train_mcts_worker.py loads h91/ when HOTS_HERO_SET=v2 and copies this record
(plus the .so digest it actually loaded) into each run's kernel_info.json.

Usage: python training/cuda_mcts/build_h91.py [--if-stale]
  --if-stale  rebuild only when the sources changed since the last build
"""
import datetime
import glob
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "h91")
INFO = os.path.join(OUT, "BUILD_INFO.json")
NUM_HEROES, NUM_MAPS = 91, 15
SOURCES = ["mcts_kernel.cu", "kernel_bindings.cpp", "hots_dims.h", "device_forward.cuh",
           "enriched_features.cuh", "search_v2.cuh", "search_v2_util.cuh", "tree_v2.h"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sources_digest():
    h = hashlib.sha256()
    for s in SOURCES:
        h.update(s.encode())
        h.update(open(os.path.join(HERE, s), "rb").read())
    return h.hexdigest()


def git_provenance():
    try:
        commit = subprocess.run(["git", "log", "-1", "--format=%H", "--", "."], cwd=HERE,
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--"]
                               + SOURCES, cwd=HERE, capture_output=True, text=True,
                               check=True).stdout.strip()
        if commit:
            return commit, bool(dirty)
    except (OSError, subprocess.CalledProcessError):
        pass
    return os.environ.get("HOTS_SOURCE_COMMIT", "unknown"), None


def so_path():
    so = glob.glob(os.path.join(OUT, "cuda_mcts_kernel*.so"))
    return so[0] if len(so) == 1 else None


def main():
    digest = sources_digest()
    if "--if-stale" in sys.argv and so_path() and os.path.exists(INFO):
        if json.load(open(INFO)).get("sources_sha256") == digest:
            print(f"h91 kernel up to date ({so_path()})")
            return
    from setuptools import setup
    from torch.utils.cpp_extension import CUDAExtension, BuildExtension
    # COMP_FALLBACK_MID_HIGH_LOW: composition fallback order of the Python WP
    # features (enriched_features.cuh); kernel parity failed without it
    defs = [f"-DNUM_HEROES={NUM_HEROES}", f"-DNUM_MAPS={NUM_MAPS}", "-DCOMP_FALLBACK_MID_HIGH_LOW"]
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT)
    tmp = os.path.join(HERE, "build_h91")
    cwd = os.getcwd()
    os.chdir(HERE)
    try:
        setup(name="cuda_mcts_kernel_h91",
              ext_modules=[CUDAExtension("cuda_mcts_kernel", ["mcts_kernel.cu", "kernel_bindings.cpp"],
                                         extra_compile_args={
                                             "cxx": ["-O3", "-std=c++17"] + defs,
                                             "nvcc": ["-O3", "--use_fast_math", "-std=c++17",
                                                      "--expt-relaxed-constexpr"] + defs})],
              cmdclass={"build_ext": BuildExtension},
              script_args=["build_ext", "--build-lib", OUT, "--build-temp", tmp])
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmp, ignore_errors=True)
    so = so_path()
    if so is None:
        sys.exit("h91 build produced no single cuda_mcts_kernel*.so")
    commit, dirty = git_provenance()
    info = {"hero_set": "v2", "num_heroes": NUM_HEROES, "num_maps": NUM_MAPS,
            "source_commit": commit, "source_dirty": dirty,
            "sources_sha256": digest, "so": os.path.basename(so), "so_sha256": sha256(so),
            "built_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "host": socket.gethostname()}
    json.dump(info, open(INFO, "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
