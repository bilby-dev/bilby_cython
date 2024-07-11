import time

import lal
import numpy as np
import bilby_cython
import pytest
from astropy.time import Time


EPSILON = dict(
    cython=1e-10,
    jax=1e-8,
)


def test_gmst(backend):
    times = np.random.uniform(1325623903, 1345623903, 100000)
    diffs = list()
    for tt in times:
        diffs.append(
            bilby_cython.time.greenwich_mean_sidereal_time(tt)
            - lal.GreenwichMeanSiderealTime(tt)
        )
    assert max(np.abs(diffs)) < EPSILON[backend]


def test_gmst_vectorized(backend):
    times = np.random.uniform(1325623903, 1345623903, 100000)
    cy_gmst = bilby_cython.time.greenwich_mean_sidereal_time(times)
    lal_gmst = np.array([lal.GreenwichMeanSiderealTime(tt) for tt in times])
    assert max(np.abs(cy_gmst - lal_gmst)) < EPSILON[backend]


def test_gmt(backend):
    times = np.random.uniform(1325623903, 1345623903, 100000)
    equinoxes = np.random.uniform(0, 2 * np.pi, 100000)
    diffs = list()
    for tt, eq in zip(times, equinoxes):
        diffs.append(
            bilby_cython.time.greenwich_sidereal_time(tt, eq)
            - lal.GreenwichSiderealTime(tt, eq)
        )
    assert max(np.abs(diffs)) < EPSILON[backend]


def test_current_time(backend):
    """
    Test that the current GMST matches LAL and Astropy.
    This should ensure robustness against additional leap seconds being added.
    """
    now = float(lal.GPSTimeNow())
    lal_now = lal.GreenwichMeanSiderealTime(now) % (2 * np.pi)
    cython_now = bilby_cython.time.greenwich_mean_sidereal_time(now) % (2 * np.pi)
    astropy_now = Time(now, format="gps").sidereal_time("mean", 0.0).radian
    assert np.abs(cython_now - lal_now) < EPSILON[backend]
    assert np.abs(cython_now - astropy_now) < max(EPSILON[backend], 1e-5)


def test_datetime_repr():
    """
    Test that the minimal datetime implementation repr works as expected.
    """
    bilby_cython.set_backend("jax")
    from datetime import datetime as reference
    from bilby_cython.time import datetime as test

    assert (
        reference(2021, 3, 1, 11, 23, 5).strftime("%Y-%-m-%-d %-H:%-M:%-S")
        == test(2021, 3, 1, 11, 23, 5).__repr__()
    )
