from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='ofit_kernel',
    ext_modules=[
        CUDAExtension('ofit_kernel', ['mcts_kernel.cu', 'kernel_bindings.cpp'],
                      extra_compile_args={
                          'cxx': ['-O3', '-std=c++17'],
                          'nvcc': ['-O3', '--use_fast_math', '-std=c++17',
                                   '--expt-relaxed-constexpr']}),
    ],
    cmdclass={'build_ext': BuildExtension},
)
