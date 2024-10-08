import os

import numpy as np
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class LazyImportBuildExtCmd(build_ext):
    def finalize_options(self):
        from Cython.Build import cythonize
        from Cython.Compiler.Version import version as cython_version
        from packaging.version import Version

        compiler_directives = dict(
            language_level=3,
            boundscheck=False,
            wraparound=False,
            cdivision=True,
            initializedcheck=False,
            embedsignature=True,
        )
        if os.environ.get("CYTHON_COVERAGE"):
            compiler_directives["linetrace"] = True
            annotate = True
        else:
            annotate = False
        if Version(cython_version) >= Version("3.1.0a1"):
            compiler_directives["freethreading_compatible"] = True
        self.distribution.ext_modules = cythonize(
            self.distribution.ext_modules,
            compiler_directives=compiler_directives,
            annotate=annotate,
        )
        super(LazyImportBuildExtCmd, self).finalize_options()


if os.environ.get("CYTHON_COVERAGE"):
    macros = [
        ("CYTHON_TRACE", "1"),
        ("CYTHON_TRACE_NOGIL", "1"),
    ]
else:
    macros = list()
extensions = [
    Extension(
        "bilby_cython.geometry",
        ["bilby_cython/geometry.pyx"],
        include_dirs=[np.get_include()],
        define_macros=macros,
    ),
    Extension(
        "bilby_cython.time",
        ["bilby_cython/time.pyx"],
        include_dirs=[np.get_include()],
        define_macros=macros,
    ),
]

setup(
    cmdclass={"build_ext": LazyImportBuildExtCmd},
    ext_modules=extensions,
)
