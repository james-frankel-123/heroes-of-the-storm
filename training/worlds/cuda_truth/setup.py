# Truth-leaf build of paper 2's support kernel (copied from personalization/cuda_support; only change: -DP3_TRUTH
# makes wp_eval_symmetrized return the world truth, truth_eval.cuh). Build (3090, from this directory):
#   HOTS_COMP_FALLBACK_MID_HIGH_LOW=1 TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=4 python setup.py build_ext --inplace
from setuptools import setup
import os
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
_DEFS = ["-DP3_TRUTH"] + (["-DCOMP_FALLBACK_MID_HIGH_LOW"] if os.environ.get("HOTS_COMP_FALLBACK_MID_HIGH_LOW") == "1" else [])
setup(
    name='truth_kernel',
    ext_modules=[CUDAExtension('truth_kernel', ['truth_kernel.cu', 'truth_bindings.cpp'],
                               extra_compile_args={'cxx': ['-O3', '-std=c++17'] + _DEFS,
                                                   'nvcc': ['-O3', '--use_fast_math', '-std=c++17', '--expt-relaxed-constexpr'] + _DEFS})],
    cmdclass={'build_ext': BuildExtension},
)
