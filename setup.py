import os
import sys
import pybind11
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# Platform-specific compilation flags
if sys.platform == "darwin":  # macOS
    extra_compile_args = [
        "-O3",
        "-fopenmp",
        "-DEIGEN_VECTORIZE",
        "-arch", "arm64" if os.uname().machine == "arm64" else "x86_64"
    ]
    extra_link_args = [
        "-fopenmp",
        "-lomp"
    ]
    eigen_include = "/opt/homebrew/include/eigen3" if os.path.exists("/opt/homebrew/include/eigen3") else "/usr/local/include/eigen3"
else:  # Linux (Docker/cloud)
    extra_compile_args = [
        "-O3",
        "-fopenmp",
        "-DEIGEN_VECTORIZE"
    ]
    extra_link_args = [
        "-fopenmp"
    ]
    eigen_include = "/usr/include/eigen3"

ext_modules = [
    Extension(
        "numerical_methods",
        ["numerical_methods.cpp"],
        include_dirs=[
            pybind11.get_include(),
            eigen_include
        ],
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
    zip_safe=False,
)

