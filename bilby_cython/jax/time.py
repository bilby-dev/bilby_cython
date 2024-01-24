import jax.numpy as jnp

__all__ = [
    "gps_time_to_utc",
    "greenwich_mean_sidereal_time",
    "greenwich_sidereal_time",
    "n_leap_seconds",
    "utc_to_julian_day",
]

JULIAN_GPS_EPOCH = 1721013.5
EPOCH_J2000_0_JD = 2451545.0
DAYS_PER_CENTURY = 36525.0
SECONDS_PER_DAY = 86400.0


class datetime:
    """
    A barebones datetime class for use in the GPS to GMST conversion.
    """
    def __init__(
        self,
        year: int = 0,
        month: int = 0,
        day: int = 0,
        hour: int = 0,
        minute: int = 0,
        second: float = 0,
    ):
        self.year = year
        self.month = month
        self.day = day
        self.hour = hour
        self.minute = minute
        self.second = second
    
    def __repr__(self):
        return f"{self.year}-{self.month}-{self.day} {self.hour}:{self.minute}:{self.second}"

    def __add__(self, other):
        """
        Add two datetimes together.
        Note that this does not handle overflow and can lead to unphysical
        values for the various attributes.
        """
        return datetime(
            self.year + other.year,
            self.month + other.month,
            self.day + other.day,
            self.hour + other.hour,
            self.minute + other.minute,
            self.second + other.second,
        )

    @property
    def julian_day(self):
        return (
            367 * self.year
            - 7 * (self.year + (self.month + 9) // 12) // 4
            + 275 * self.month // 9
            + self.day
            + self.second / SECONDS_PER_DAY
            + JULIAN_GPS_EPOCH
        )


GPS_EPOCH = datetime(1980, 1, 6, 0, 0, 0)
LEAP_SECONDS = jnp.asarray([
    46828800,
    78364801,
    109900802,
    173059203,
    252028804,
    315187205,
    346723206,
    393984007,
    425520008,
    457056009,
    504489610,
    551750411,
    599184012,
    820108813,
    914803214,
    1025136015,
    1119744016,
    1167264017,
])


def n_leap_seconds(date):
    """
    Find the number of leap seconds required for the specified date.
    """
    return jnp.sum(date > LEAP_SECONDS[:, None], axis=0).squeeze()


def gps_time_to_utc(secs):
    """
    Convert from GPS time to UTC, this is a necessary intermediate step in
    converting from GPS to GMST.

    Add the number of seconds minus the leap seconds to the GPS epoch.

    Parameters
    ----------
    secs: float
        The time to convert in GPS time.
    """
    return GPS_EPOCH + datetime(second=secs - n_leap_seconds(secs))


def utc_to_julian_day(time):
    """
    Convert from UTC to Julian day, this is a necessary intermediate step in
    converting from GPS to GMST.

    Parameters
    ----------
    time: datetime
        The UTC time to convert
    """
    return time.julian_day


def greenwich_mean_sidereal_time(gps_time):
    """
    Compute the Greenwich mean sidereal time from the GPS time.

    Parameters
    ----------
    gps_time: double
        The GPS time to convert
    """
    return greenwich_sidereal_time(gps_time, 0)


def greenwich_sidereal_time(gps_time, equation_of_equinoxes):
    """
    Compute the Greenwich mean sidereal time from the GPS time and equation of
    equinoxes.

    Based on XLALGreenwichSiderealTime in lalsuite/lal/lib/XLALSiderealTime.c.

    Parameters
    ----------
    gps_time: float
        The GPS time to convert
    equation_of_equinoxes: float
        The equation of equinoxes
    """
    julian_day = utc_to_julian_day(gps_time_to_utc(gps_time // 1))
    t_hi = (julian_day - EPOCH_J2000_0_JD) / DAYS_PER_CENTURY
    t_lo = (gps_time % 1) / (DAYS_PER_CENTURY * SECONDS_PER_DAY)

    t = t_hi + t_lo

    sidereal_time = equation_of_equinoxes + (-6.2e-6 * t + 0.093104) * t**2 + 67310.54841
    sidereal_time += 8640184.812866 * t_lo
    sidereal_time += 3155760000.0 * t_lo
    sidereal_time += 8640184.812866 * t_hi
    sidereal_time += 3155760000.0 * t_hi

    return sidereal_time * 2 * jnp.pi / SECONDS_PER_DAY
