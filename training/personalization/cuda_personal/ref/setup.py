# Unmodified copy of overfit2026/cuda_ofit (population MCTS kernel with the
# tree-capacity guard), built under a different module name for the
# bit-identity check of the personalized kernel.
from setuptools import setup
import os
_DEFS = ["-DCOMP_FALLBACK_MID_HIGH_LOW"] if os.environ.get("HOTS_COMP_FALLBACK_MID_HIGH_LOW") == "1" else []
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='pop_ref_kernel',
    ext_modules=[
        CUDAExtension('pop_ref_kernel', ['mcts_kernel.cu', 'kernel_bindings.cpp'],
                      extra_compile_args={
                          'cxx': ['-O3', '-std=c++17'] + _DEFS,
                          'nvcc': ['-O3', '--use_fast_math', '-std=c++17',
                                   '--expt-relaxed-constexpr'] + _DEFS}),
    ],
    cmdclass={'build_ext': BuildExtension},
)
