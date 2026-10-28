#!/bin/bash
# Build the four P3 kernels in place (personal, its population reference,
# prior, pgd) with the composition fallback in the WP's tier order, then
# check each module's COMP_FALLBACK_ORDER. Remote workers: sm_86.
# Usage (from training/): bash personalization/build_p3_kernels.sh
#   bash personalization/build_p3_kernels.sh pvnet   builds only cuda_pvnet
#   (learned personal value net); the other builds are left untouched.
#   bash personalization/build_p3_kernels.sh explore builds only cuda_explore
#   (ban-step prior bias, player-aware opponent in the search).
#   bash personalization/build_p3_kernels.sh ref     builds only the population reference.
#   TORCH_CUDA_ARCH_LIST from the env (8.6 default; the 5090's env.sh: 12.0).
set -euo pipefail
cd "$(dirname "$0")"
export HOTS_COMP_FALLBACK_MID_HIGH_LOW=1
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.6}
export MAX_JOBS=${MAX_JOBS:-2}
TARGET=${1:-all}
case "$TARGET" in
  all) DIRS="cuda_personal cuda_personal/ref cuda_prior cuda_pgd"
       MODS="cuda_personal:personal_kernel cuda_personal/ref:pop_ref_kernel cuda_prior:prior_kernel cuda_pgd:pgd_kernel" ;;
  pvnet) DIRS="cuda_pvnet"; MODS="cuda_pvnet:pvnet_kernel" ;;
  explore) DIRS="cuda_explore"; MODS="cuda_explore:explore_kernel" ;;
  ref) DIRS="cuda_personal/ref"; MODS="cuda_personal/ref:pop_ref_kernel" ;;
  *) echo "usage: build_p3_kernels.sh [all|pvnet|explore|ref]" >&2; exit 2 ;;
esac
echo "arch $TORCH_CUDA_ARCH_LIST target $TARGET"
for d in $DIRS; do
  echo "== build $d"
  (cd "$d" && rm -rf build && python setup.py build_ext --inplace > build.log 2>&1) || { tail -30 "$d/build.log"; exit 1; }
done
P3_MODS="$MODS" python - <<'PY'
import importlib.util, os, sys
bad = []
for dm in os.environ["P3_MODS"].split():
    d, name = dm.split(":")
    so = [f for f in os.listdir(d) if f.startswith(name) and f.endswith(".so")]
    import torch  # noqa: F401  (loads the CUDA runtime the extension links)
    spec = importlib.util.spec_from_file_location(name, os.path.join(d, so[0]))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    order = getattr(m, "COMP_FALLBACK_ORDER", None)
    print(f"{name}: {so[0]} COMP_FALLBACK_ORDER={order} CFG_LEN={getattr(m, 'CFG_LEN', '-')}")
    if order != "mid_high_low":
        bad.append(name)
if bad:
    sys.exit(f"wrong fallback order: {bad}")
print("BUILD OK")
PY
