import os
import pybind11
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

os.environ["CC"] = "/opt/homebrew/bin/g++-15"
os.environ["CXX"] = "/opt/homebrew/bin/g++-15"

extra_compile_args = [
    "-O3",
    "-fopenmp",
    "-DEIGEN_VECTORIZE",
    "-arch", "arm64"
]

extra_link_args = [
    "-fopenmp",
    "-L/usr/local/opt/libomp/include",
    "-lomp",
    "-arch", "arm64"
]

ext_modules = [
    Extension(
        "numerical_methods",
        ["numerical_methods.cpp"],
        include_dirs=[
            pybind11.get_include(),
            "/opt/homebrew/include/eigen3",
            "/usr/local/opt/libomp/include"
        ],
        library_dirs=["/usr/local/opt/libomp/lib"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    )
]

setup(
    name="numerical_methods",
    version="0.1",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)

