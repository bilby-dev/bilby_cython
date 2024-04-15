import importlib


def __getattr__(name: str) -> object:
    from . import BACKEND, __version__
    module = importlib.import_module(f"bilby_cython.{BACKEND}.time")
    return getattr(module, name)