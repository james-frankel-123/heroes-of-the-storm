# P3 support kernel (explore kernel + support floor / leaf penalty / GD prior mix). Build (from this directory):
#   HOTS_COMP_FALLBACK_MID_HIGH_LOW=1 MAX_JOBS=4 nice -n 19 python3 setup.py build_ext --inplace
# (remote workers: personalization/build_p3_kernels.sh support; sm_86 or HOTS_ARCH=12.0)
from setuptools import setup
import os
# HOTS_COMP_FALLBACK_MID_HIGH_LOW=1 builds the composition fallback in the
# Python WP's tier order (mid, high, low); unset keeps the index order.
_DEFS = ["-DCOMP_FALLBACK_MID_HIGH_LOW"] if os.environ.get("HOTS_COMP_FALLBACK_MID_HIGH_LOW") == "1" else []
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='support_kernel',
    ext_modules=[
        CUDAExtension('support_kernel', ['support_kernel.cu', 'support_bindings.cpp'],
                      extra_compile_args={
                          'cxx': ['-O3', '-std=c++17'] + _DEFS,
                          'nvcc': ['-O3', '--use_fast_math', '-std=c++17',
                                   '--expt-relaxed-constexpr'] + _DEFS}),
    ],
    cmdclass={'build_ext': BuildExtension},
)
