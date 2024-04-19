import pytest
import bilby_cython


def test_unknown_backend_raises_error():
    with pytest.raises(ValueError):
        bilby_cython.set_backend("unknown")


def test_known_backends_set_correctly(backend):
    assert bilby_cython.BACKEND == backend
