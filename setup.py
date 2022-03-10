import os
import subprocess
import sys

import numpy as np
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext


python_version = sys.version_info
min_python_version = (3, 8)
min_python_version_str = f"{min_python_version[0]}.{min_python_version[1]}"
if python_version < min_python_version:
    sys.exit(f"Python < {min_python_version_str} is not supported, aborting setup")


class build_ext(_build_ext):
    def build_extension(self, ext):
        if self.debug:
            ext.extra_compile_args.append("-O0")
            if sys.implementation.name == "cpython":
                ext.define_macros.append(("CYTHON_TRACE_NOGIL", 1))
        _build_ext.build_extension(self, ext)


def write_version_file(version):
    """Writes a file with version information to be used at run time

    Parameters
    ----------
    version: str
        A string containing the current version information

    Returns
    -------
    version_file: str
        A path to the version file

    """
    try:
        git_log = subprocess.check_output(
            ["git", "log", "-1", "--pretty=%h %ai"]
        ).decode("utf-8")
        git_diff = (
            subprocess.check_output(["git", "diff", "."])
            + subprocess.check_output(["git", "diff", "--cached", "."])
        ).decode("utf-8")
        if git_diff == "":
            git_status = "(CLEAN) " + git_log
        else:
            git_status = "(UNCLEAN) " + git_log
    except Exception as e:
        print(f"Unable to obtain git version information, exception: {e}")
        git_status = "release"

    version_file = ".version"
    long_version_file = f"bilby_cython/{version_file}"
    if os.path.isfile(long_version_file) is False:
        with open(long_version_file, "w+") as f:
            f.write(f"{version}: {git_status}")

    return version_file


def get_long_description():
    """Finds the README and reads in the description"""
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, "README.md")) as f:
        long_description = f.read()
    return long_description


def get_requirements():
    with open("requirements.txt", "r") as ff:
        requirements = ff.readlines()
    return requirements


VERSION = "0.1.3"
version_file = write_version_file(VERSION)

extensions = [
    Extension(
        "bilby_cython.geometry",
        ["bilby_cython/geometry.pyx"],
        include_dirs=[np.get_include()],
    ),
]

setup(
    author="Colm Talbot",
    author_email="colm.talbot@ligo.org",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    cmdclass={"build_ext": build_ext},
    description="Optimized functionality for Bilby",
    ext_modules=extensions,
    install_requires=["numpy"],
    license="MIT",
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    name="bilby.cython",
    packages=["bilby_cython"],
    package_data=dict(bilby_cython=[version_file]),
    python_requires=f">={min_python_version_str}",
    setup_requires=["numpy", "cython"],
    url="https://git.ligo.org/colm.talbot/bilby-cython",
    version=VERSION,
)
