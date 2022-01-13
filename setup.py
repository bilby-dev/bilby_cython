from setuptools import Extension, setup
from Cython.Build import cythonize

import numpy as np

extensions = [
    Extension("bilby_cython.geometry", ["bilby_cython/geometry.pyx"], include_dirs=[np.get_include()]),
]
setup(
    name="bilby.cython",
    ext_modules=cythonize(extensions, language_level="3"),
    packages=["bilby_cython"],
)
