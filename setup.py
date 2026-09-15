from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


setup(
    ext_modules=[Pybind11Extension(
        "timbre._hnsw_native", ["src/timbre/_hnsw_native.cpp"],
        cxx_std=17, extra_compile_args=["-O3", "-ffp-contract=off"],
    )],
    cmdclass={"build_ext": build_ext},
)
