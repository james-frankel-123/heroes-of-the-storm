from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='cuda_mcts_all',
    ext_modules=[
        # Original: host-launched kernels (for testing/comparison)
        CUDAExtension('cuda_mcts', [
            'fused_forward.cu',
            'mcts_engine.cpp',
        ], extra_compile_args={
            'cxx': ['-O3', '-std=c++17'],
            'nvcc': ['-O3', '--use_fast_math', '-std=c++17',
                     '--expt-relaxed-constexpr'],
        }),
        # New: full MCTS kernel (one block = one episode)
        CUDAExtension('cuda_mcts_kernel', [
            'mcts_kernel.cu',
            'kernel_bindings.cpp',
        ], extra_compile_args={
            'cxx': ['-O3', '-std=c++17'],
            'nvcc': ['-O3', '--use_fast_math', '-std=c++17',
                     '--expt-relaxed-constexpr',
                     '-maxrregcount=128'],
        }),
    ],
    cmdclass={'build_ext': BuildExtension},
)
