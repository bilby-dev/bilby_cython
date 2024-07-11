import bilby_cython
import pytest


@pytest.fixture(params=bilby_cython.SUPPORTED_BACKENDS)
def backend(request):
    bilby_cython.set_backend(request.param)
    return request.param

