# P3 personalized MCTS kernel. Build (from this directory):
#   PATH=/usr/local/cuda/bin:$PATH MAX_JOBS=4 TORCH_CUDA_ARCH_LIST="12.0" \
#     nice -n 19 taskset -c 48-63 python3 setup.py build_ext --inplace
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='pgd_kernel',
    ext_modules=[
        CUDAExtension('pgd_kernel', ['pgd_kernel.cu', 'pgd_bindings.cpp'],
                      extra_compile_args={
                          'cxx': ['-O3', '-std=c++17'],
                          'nvcc': ['-O3', '--use_fast_math', '-std=c++17',
                                   '--expt-relaxed-constexpr']}),
    ],
    cmdclass={'build_ext': BuildExtension},
)
