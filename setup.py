import os
import pybind11
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

os.environ["CC"] = "gcc"
os.environ["CXX"] = "g++"

extra_compile_args = [
    "-O3",
    "-fopenmp",
    "-DEIGEN_VECTORIZE",
    "-march=x86-64",
]

extra_link_args = [
    "-fopenmp",
]

ext_modules = [
    Extension(
        "numerical_methods",
        ["numerical_methods.cpp"],
        include_dirs=[
            pybind11.get_include(),
            "/usr/include/eigen3",
        ],
        library_dirs=[],
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
