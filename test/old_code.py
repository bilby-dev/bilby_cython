from math import fmod

import numpy as np


def calculate_arm(arm_tilt, arm_azimuth, longitude, latitude):
    """
    Pulled from bilby v1.1.5
    """
    e_long = np.array([-np.sin(longitude), np.cos(longitude), 0])
    e_lat = np.array([-np.sin(latitude) * np.cos(longitude),
                      -np.sin(latitude) * np.sin(longitude), np.cos(latitude)])
    e_h = np.array([np.cos(latitude) * np.cos(longitude),
                    np.cos(latitude) * np.sin(longitude), np.sin(latitude)])

    return (np.cos(arm_tilt) * np.cos(arm_azimuth) * e_long +
            np.cos(arm_tilt) * np.sin(arm_azimuth) * e_lat +
            np.sin(arm_tilt) * e_h)


def time_delay_geocentric(detector1, detector2, ra, dec, time):
    """
    Copied from Bilby v1.1.4.

    Calculate time delay between two detectors in geocentric coordinates based on XLALArrivaTimeDiff in TimeDelay.c

    Parameters
    ==========
    detector1: array_like
        Cartesian coordinate vector for the first detector in the geocentric frame
        generated by the Interferometer class as self.vertex.
    detector2: array_like
        Cartesian coordinate vector for the second detector in the geocentric frame.
        To get time delay from Earth center, use detector2 = np.array([0,0,0])
    ra: float
        Right ascension of the source in radians
    dec: float
        Declination of the source in radians
    time: float
        GPS time in the geocentric frame

    Returns
    =======
    float: Time delay between the two detectors in the geocentric frame

    """
    from lal import GreenwichMeanSiderealTime
    speed_of_light = 299792458.0
    gmst = fmod(GreenwichMeanSiderealTime(time), 2 * np.pi)
    phi = ra - gmst
    theta = np.pi / 2 - dec
    omega = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    delta_d = detector2 - detector1
    return np.dot(omega, delta_d) / speed_of_light


def get_polarization_tensor(ra, dec, time, psi, mode):
    """
    Copied from Bilby v1.1.4.

    Calculate the polarization tensor for a given sky location and time

    See Nishizawa et al. (2009) arXiv:0903.0528 for definitions of the polarisation tensors.
    [u, v, w] represent the Earth-frame
    [m, n, omega] represent the wave-frame
    Note: there is a typo in the definition of the wave-frame in Nishizawa et al.

    Parameters
    ==========
    ra: float
        right ascension in radians
    dec: float
        declination in radians
    time: float
        geocentric GPS time
    psi: float
        binary polarisation angle counter-clockwise about the direction of propagation
    mode: str
        polarisation mode

    Returns
    =======
    array_like: A 3x3 representation of the polarization_tensor for the specified mode.

    """
    from lal import GreenwichMeanSiderealTime
    gmst = fmod(GreenwichMeanSiderealTime(time), 2 * np.pi)
    phi = ra - gmst
    theta = np.pi / 2 - dec
    u = np.array([np.cos(phi) * np.cos(theta), np.cos(theta) * np.sin(phi), -np.sin(theta)])
    v = np.array([-np.sin(phi), np.cos(phi), 0])
    m = -u * np.sin(psi) - v * np.cos(psi)
    n = -u * np.cos(psi) + v * np.sin(psi)

    if mode.lower() == 'plus':
        return np.einsum('i,j->ij', m, m) - np.einsum('i,j->ij', n, n)
    elif mode.lower() == 'cross':
        return np.einsum('i,j->ij', m, n) + np.einsum('i,j->ij', n, m)
    elif mode.lower() == 'breathing':
        return np.einsum('i,j->ij', m, m) + np.einsum('i,j->ij', n, n)

    # Calculating omega here to avoid calculation when model in [plus, cross, breathing]
    omega = np.cross(m, n)
    if mode.lower() == 'longitudinal':
        return np.einsum('i,j->ij', omega, omega)
    elif mode.lower() == 'x':
        return np.einsum('i,j->ij', m, omega) + np.einsum('i,j->ij', omega, m)
    elif mode.lower() == 'y':
        return np.einsum('i,j->ij', n, omega) + np.einsum('i,j->ij', omega, n)
    else:
        raise ValueError("{} not a polarization mode!".format(mode))


def get_polarization_tensor_multiple_modes(ra, dec, time, psi, modes):
    return [get_polarization_tensor(ra, dec, time, psi, mode) for mode in modes]


def antenna_response(detector_tensor, ra, dec, time, psi, mode):
    polarization_tensor = get_polarization_tensor(ra, dec, time, psi, mode)
    return np.einsum('ij,ij->', detector_tensor, polarization_tensor)


__cached_euler_matrix = None
__cached_delta_x = None


def euler_rotation(delta_x):
    """
    Calculate the rotation matrix mapping the vector (0, 0, 1) to delta_x
    while preserving the origin of the azimuthal angle.

    This is decomposed into three Euler angle, alpha, beta, gamma, which rotate
    about the z-, y-, and z- axes respectively.

    Parameters
    ==========
    delta_x: array-like (3,)
        Vector onto which (0, 0, 1) should be mapped.

    Returns
    =======
    total_rotation: array-like (3,3)
        Rotation matrix which maps vectors from the frame in which delta_x is
        aligned with the z-axis to the target frame.
    """
    global __cached_delta_x
    global __cached_euler_matrix

    delta_x = delta_x / np.sum(delta_x**2)**0.5
    if np.array_equal(delta_x, __cached_delta_x):
        return __cached_euler_matrix
    else:
        __cached_delta_x = delta_x
    alpha = np.arctan(- delta_x[1] * delta_x[2] / delta_x[0])
    beta = np.arccos(delta_x[2])
    gamma = np.arctan(delta_x[1] / delta_x[0])
    rotation_1 = np.array([
        [np.cos(alpha), -np.sin(alpha), 0], [np.sin(alpha), np.cos(alpha), 0],
        [0, 0, 1]])
    rotation_2 = np.array([
        [np.cos(beta), 0, - np.sin(beta)], [0, 1, 0],
        [np.sin(beta), 0, np.cos(beta)]])
    rotation_3 = np.array([
        [np.cos(gamma), -np.sin(gamma), 0], [np.sin(gamma), np.cos(gamma), 0],
        [0, 0, 1]])
    total_rotation = np.einsum(
        'ij,jk,kl->il', rotation_3, rotation_2, rotation_1)
    __cached_delta_x = delta_x
    __cached_euler_matrix = total_rotation
    return total_rotation


def zenith_azimuth_to_theta_phi(zenith, azimuth, ifo_1, ifo_2):
    """
    Convert from the 'detector frame' to the Earth frame.

    Parameters
    ==========
    zenith: float
        The zenith angle in the detector frame
    azimuth: float
        The azimuthal angle in the detector frame
    delta_x: list
        List of Interferometer objects defining the detector frame

    Returns
    =======
    theta, phi: float
        The zenith and azimuthal angles in the earth frame.
    """
    delta_x = ifo_1 - ifo_2
    midpoint = (ifo_1 + ifo_2) / 2
    rotation_matrix = euler_rotation(delta_x)
    temp = rotation_matrix.T @ midpoint
    azimuth -= np.arctan2(temp[0], temp[1])
    omega_prime = np.array([
        np.sin(zenith) * np.cos(azimuth),
        np.sin(zenith) * np.sin(azimuth),
        np.cos(zenith)])
    omega = np.dot(rotation_matrix, omega_prime)
    theta = np.arccos(omega[2])
    phi = np.arctan2(omega[1], omega[0]) % (2 * np.pi)
    return theta, phi


def greenwich_mean_sidereal_time(time):
    from lal import GreenwichMeanSiderealTime
    time = float(time)
    return GreenwichMeanSiderealTime(time)
