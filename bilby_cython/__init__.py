import os

from ._version import __version__
from . import geometry, time  # noqa

BACKEND = os.environ.get("BILBY_ARRAY_DEFAULT_BACKEND", "cython")
SUPPORTED_BACKENDS = ["cython", "jax"]


def set_backend(backend: str) -> None:
    """Set the backend to use

    Parameters
    ----------
    backend: str
        The backend to use, currently only 'cython' and 'jax' are supported
    """
    global BACKEND
    if backend == "jax":
        from jax import config

        config.update("jax_enable_x64", True)
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Backend {backend} not supported, must be one of {SUPPORTED_BACKENDS}"
        )
    else:
        BACKEND = backend


set_backend(BACKEND)
