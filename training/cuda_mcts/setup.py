from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='cuda_mcts',
    ext_modules=[
        CUDAExtension('cuda_mcts', [
            'fused_forward.cu',
            'mcts_engine.cpp',
        ], extra_compile_args={
            'cxx': ['-O3', '-std=c++17'],
            'nvcc': ['-O3', '--use_fast_math', '-std=c++17',
                     '--expt-relaxed-constexpr'],
        }),
    ],
    cmdclass={'build_ext': BuildExtension},
)
