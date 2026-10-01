#!/bin/bash
# Build an isolated HotS training env in ~/hots inside WSL (user space, no root,
# no GPU use). Usage: bash setup_env.sh <cu126|cu128>
# cu tag must be supported by the Windows NVIDIA driver (nvidia-smi on Windows):
# driver >= 570 -> cu128; driver 560-569 -> cu126.
set -euo pipefail
CU=${1:?usage: setup_env.sh cu126|cu128}
case "$CU" in
  cu126) CUDA_VER=12.6.3 ;;
  cu128) CUDA_VER=12.8.1 ;;
  *) echo "unknown cu tag $CU"; exit 1 ;;
esac
PY=3.14
TORCH=2.10.0
H=$HOME/hots
mkdir -p "$H"/{bin,logs,dl,cuda,runs}
export UV_CACHE_DIR=$H/.uv-cache UV_PYTHON_INSTALL_DIR=$H/.uv-python
export PATH=$H/bin:$PATH

# uv (installed into ~/hots/bin only; shell profiles untouched)
if [ ! -x "$H/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$H/bin" UV_NO_MODIFY_PATH=1 sh
fi
[ -d "$H/venv" ] || uv venv --python "$PY" "$H/venv"
VPY=$H/venv/bin/python
uv pip install --python "$VPY" "torch==$TORCH" --index-url "https://download.pytorch.org/whl/$CU"
uv pip install --python "$VPY" numpy psycopg2-binary onnx onnxruntime filelock matplotlib \
  scipy setuptools wheel ninja packaging
uv pip install --python "$VPY" numba || echo "WARN: numba not installable (only used by a few side scripts)"

# CUDA toolkit (nvcc + headers + libs) from NVIDIA redistributable tarballs,
# merged into one CUDA_HOME. Driver-independent; no root.
CH=$H/cuda/$CUDA_VER
if [ ! -x "$CH/bin/nvcc" ]; then
  BASE=https://developer.download.nvidia.com/compute/cuda/redist
  curl -fsSL "$BASE/redistrib_$CUDA_VER.json" -o "$H/dl/redistrib_$CUDA_VER.json"
  mkdir -p "$CH"
  for comp in cuda_nvcc cuda_cudart cuda_cccl cuda_nvrtc cuda_nvtx cuda_profiler_api \
              cuda_cupti libcublas libcusparse libcusolver libcurand libcufft libnvjitlink cuda_crt cuda_nvvm; do
    rel=$("$VPY" - "$H/dl/redistrib_$CUDA_VER.json" "$comp" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
c = d.get(sys.argv[2])
print(c["linux-x86_64"]["relative_path"] if c and "linux-x86_64" in c else "")
PYEOF
)
    [ -n "$rel" ] || { echo "skip $comp (not in $CUDA_VER manifest)"; continue; }
    f=$H/dl/$(basename "$rel")
    [ -f "$f" ] || curl -fsSL "$BASE/$rel" -o "$f"
    tar -xJf "$f" -C "$CH" --strip-components=1
    rm -f "$f"
  done
  [ -e "$CH/lib64" ] || ln -s lib "$CH/lib64"
fi

cat > "$H/env.sh" <<ENVEOF
# source ~/hots/env.sh  -- HotS training env (isolated; touches nothing outside ~/hots)
export HOTS=\$HOME/hots
export CUDA_HOME=$CH
export PATH=\$HOTS/venv/bin:\$CUDA_HOME/bin:\$HOTS/bin:\$PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:\$CUDA_HOME/lib64\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}
export TORCH_CUDA_ARCH_LIST=8.6
export MAX_JOBS=\${MAX_JOBS:-4}
export UV_CACHE_DIR=\$HOTS/.uv-cache UV_PYTHON_INSTALL_DIR=\$HOTS/.uv-python
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
ENVEOF
rm -rf "$UV_CACHE_DIR"
echo "SETUP DONE ($CU, CUDA $CUDA_VER)"
