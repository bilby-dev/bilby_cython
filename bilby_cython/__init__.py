from ._version import __version__

BACKEND = "cython"
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
        raise ValueError(f"Backend {backend} not supported, must be one of {SUPPORTED_BACKENDS}")
    else:
        BACKEND = backend


def __getattr__(name: str) -> object:
    if name == "geometry":
        import importlib
        return importlib.import_module(f"bilby_cython.{BACKEND}.{name}")
    elif name == "__version__":
        return __version__
    else:
        raise AttributeError(f"module {__name__} has no attribute {name} for backend {BACKEND}")