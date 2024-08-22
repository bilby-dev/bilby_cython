import numpy as np
import pytest
from bilby_cython.time import (
    greenwich_sidereal_time,
    greenwich_mean_sidereal_time,
    gps_time_to_utc,
    utc_to_julian_day,
)

def test_without_lal():
    assert greenwich_mean_sidereal_time(1e9) - 26930.069103915423 < 1e-5


def test_gps_time_to_julian_day(types):
    lal = pytest.importorskip("lal")
    times = np.random.uniform(1325623903, 1345623903, 100000).astype(types)
    diffs = list()
    for tt in times:
        diffs.append(
            utc_to_julian_day(gps_time_to_utc(tt))
            - lal.JulianDay(lal.GPSToUTC(int(tt)))
        )
    assert max(np.abs(diffs)) < 1e-10


def test_gmst(types):
    lal = pytest.importorskip("lal")
    times = np.random.uniform(1325623903, 1345623903, 100000).astype(types)
    diffs = list()
    for tt in times:
        diffs.append(
            greenwich_mean_sidereal_time(tt)
            - lal.GreenwichMeanSiderealTime(tt)
        )
    assert max(np.abs(diffs)) < 1e-10


def test_gmst_vectorized(types):
    lal = pytest.importorskip("lal")
    times = np.random.uniform(1325623903, 1345623903, 100000).astype(types)
    diffs = list()
    cy_gmst = greenwich_mean_sidereal_time(times)
    lal_gmst = np.array([lal.GreenwichMeanSiderealTime(tt) for tt in times])
    assert max(np.abs(cy_gmst - lal_gmst)) < 1e-10


def test_gmt(types):
    lal = pytest.importorskip("lal")
    times = np.random.uniform(1325623903, 1345623903, 100000).astype(types)
    equinoxes = np.random.uniform(0, 2 * np.pi, 100000)
    diffs = list()
    for tt, eq in zip(times, equinoxes):
        diffs.append(
            greenwich_sidereal_time(tt, eq)
            - lal.GreenwichSiderealTime(tt, eq)
        )
    assert max(np.abs(diffs)) < 1e-4


def test_current_time():
    """
    Test that the current GMST matches LAL and Astropy.
    This should ensure robustness against additional leap seconds being added. 
    """
    lal = pytest.importorskip("lal")
    pytest.importorskip("astropy")
    from astropy.time import Time
    now = float(lal.GPSTimeNow())
    lal_now = lal.GreenwichMeanSiderealTime(now) % (2 * np.pi)
    cython_now = greenwich_mean_sidereal_time(now) % (2 * np.pi)
    astropy_now = Time(now, format="gps").sidereal_time("mean", 0.0).radian
    assert np.abs(cython_now - lal_now) < 1e-10
    assert np.abs(cython_now - astropy_now) < 1e-5
