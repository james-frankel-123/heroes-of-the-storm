from setuptools import setup, Extension
import pybind11

ext = Extension(
    'mcts_core',
    sources=['mcts_core.cpp'],
    include_dirs=[pybind11.get_include()],
    language='c++',
    extra_compile_args=['-std=c++17', '-O3', '-march=native', '-ffast-math'],
)

setup(
    name='mcts_core',
    ext_modules=[ext],
)
