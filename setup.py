import sys

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext


class build_ext(_build_ext):
    def build_extension(self, ext):
        if self.debug:
            ext.extra_compile_args.append("-O0")
            if sys.implementation.name == "cpython":
                ext.define_macros.append(("CYTHON_TRACE_NOGIL", 1))
        _build_ext.build_extension(self, ext)


extensions = [
    Extension(
        "bilby_cython.geometry",
        ["bilby_cython/geometry.pyx"],
        include_dirs=[np.get_include()],
    ),
]

setup(
    name="bilby.cython",
    ext_modules=cythonize(extensions, language_level="3"),
    cmdclass={"build_ext": build_ext},
    packages=["bilby_cython"],
    install_requires=["numpy", "cython"],
    python_requires=">=3.8",
)
