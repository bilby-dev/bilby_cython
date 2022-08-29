import time

import lal
import numpy as np
from astropy.time import Time
from bilby_cython.time import (
    greenwich_sidereal_time,
    greenwich_mean_sidereal_time,
)


def test_gmst():
    times = np.random.uniform(1325623903, 1345623903, 100000)
    diffs = list()
    for tt in times:
        diffs.append(
            greenwich_mean_sidereal_time(tt)
            - lal.GreenwichMeanSiderealTime(tt)
        )
    assert max(np.abs(diffs)) < 1e-10


def test_gmt():
    times = np.random.uniform(1325623903, 1345623903, 100000)
    equinoxes = np.random.uniform(0, 2 * np.pi, 100000)
    diffs = list()
    for tt, eq in zip(times, equinoxes):
        diffs.append(
            greenwich_sidereal_time(tt, eq)
            - lal.GreenwichSiderealTime(tt, eq)
        )
    assert max(np.abs(diffs)) < 1e-10


def test_current_time():
    """
    Test that the current GMST matches LAL and Astropy.
    This should ensure robustness against additional leap seconds being added. 
    """
    now = float(lal.GPSTimeNow())
    lal_now = lal.GreenwichMeanSiderealTime(now) % (2 * np.pi)
    cython_now = greenwich_mean_sidereal_time(now) % (2 * np.pi)
    astropy_now = Time(now, format="gps").sidereal_time("mean", 0.0).radian
    assert np.abs(cython_now - lal_now) < 1e-10
    assert np.abs(cython_now - astropy_now) < 1e-5
