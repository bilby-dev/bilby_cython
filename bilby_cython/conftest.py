import pytest
import numpy as np


@pytest.fixture(params=[np.int64, np.int32, np.int16, np.float64, np.float32])
def types(request):
    return request.param


@pytest.fixture(params=["H1", "L1", "V1", "K1"])
def ifo(request):
    return request.param

@pytest.fixture(params=["plus", "cross", "x", "y", "breathing", "longitudinal"])
def mode(request):
    return request.param