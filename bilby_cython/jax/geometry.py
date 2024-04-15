from functools import partial

import jax.numpy as np
from jax import jit

from .time import greenwich_mean_sidereal_time

__all__ = [
    "antenna_response",
    "calculate_arm",
    "detector_tensor",
    "get_polarization_tensor",
    "get_polarization_tensor_multiple_modes",
    "rotation_matrix_from_delta",
    "three_by_three_matrix_contraction",
    "time_delay_geocentric",
    "time_delay_from_geocenter",
    "zenith_azimuth_to_theta_phi",
]


@partial(jit, static_argnames=["mode"])
def antenna_response(detector_tensor, ra, dec, time, psi, mode):
    polarization_tensor = get_polarization_tensor(ra, dec, time, psi, mode)
    return three_by_three_matrix_contraction(detector_tensor, polarization_tensor)


@jit
def calculate_arm(arm_tilt, arm_azimuth, longitude, latitude):
    """
    Compute the unit-vector along an interferometer arm given the specified parameters.

    Parameters
    ----------
    arm_tilt: float
        The angle between the tangent to the Earth and the arm
    arm_azimuth: float
        The azimuthal angle of the arm in FIXME
    longitude: float
        The longitude of the vertex
    latitude: float
        The latitude of the vertex

    Returns
    -------
    output: array_like
        The unit-vector pointing along the interferometer arm

    """
    e_long = np.array([-np.sin(longitude), np.cos(longitude), longitude * 0])
    e_lat = np.array(
        [
            -np.sin(latitude) * np.cos(longitude),
            -np.sin(latitude) * np.sin(longitude),
            np.cos(latitude),
        ]
    )
    e_h = np.array(
        [
            np.cos(latitude) * np.cos(longitude),
            np.cos(latitude) * np.sin(longitude),
            np.sin(latitude),
        ]
    )

    return (
        np.cos(arm_tilt) * np.cos(arm_azimuth) * e_long
        + np.cos(arm_tilt) * np.sin(arm_azimuth) * e_lat
        + np.sin(arm_tilt) * e_h
    )


@jit
def detector_tensor(x, y):
    """
    Calculate the detector tensor for a given pair of arms

    Returns
    =======
    array_like: A 3x3 representation of the detector tensor

    """
    return (np.outer(x, x) - np.outer(y, y)) / 2


@partial(jit, static_argnames=["mode"])
def get_polarization_tensor(ra, dec, time, psi, mode):
    """
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
    from functools import partial

    gmst = greenwich_mean_sidereal_time(time) % (2 * np.pi)
    phi = ra - gmst
    theta = np.atleast_1d(np.pi / 2 - dec).squeeze()
    u = np.array(
        [
            np.cos(phi) * np.cos(theta),
            np.cos(theta) * np.sin(phi),
            -np.sin(theta) * phi**0,
        ]
    )
    v = np.array([-np.sin(phi), np.cos(phi), phi * 0])
    m = -u * np.sin(psi) - v * np.cos(psi)
    n = -u * np.cos(psi) + v * np.sin(psi)
    einsum_shape = "i...,j...->ij..."
    product = partial(np.einsum, einsum_shape)

    if mode.lower() == "plus":
        return product(m, m) - product(n, n)
    elif mode.lower() == "cross":
        return product(m, n) + product(n, m)
    elif mode.lower() == "breathing":
        return product(m, m) + product(n, n)

    # Calculating omega here to avoid calculation when model in [plus, cross, breathing]
    omega = np.cross(m, n, axis=0)
    if mode.lower() == "longitudinal":
        return product(omega, omega)
    elif mode.lower() == "x":
        return product(m, omega) + product(omega, m)
    elif mode.lower() == "y":
        return product(n, omega) + product(omega, n)
    else:
        raise ValueError(f"{mode} not a polarization mode!")


@partial(jit, static_argnames=["modes"])
def get_polarization_tensor_multiple_modes(ra, dec, time, psi, modes):
    """
    Calculate the polarization tensor for a given sky location and time with
    multiple modes

    See Nishizawa et al. (2009) arXiv:0903.0528 for definitions of the polarisation tensors.
    [u, v, w] represent the Earth-frame
    [m, n, omega] represent the wave-frame
    Note: there is a typo in the definition of the wave-frame in Nishizawa et al.

    Parameters
    ----------
    ra: float
        right ascension in radians
    dec: float
        declination in radians
    time: float
        geocentric GPS time
    psi: float
        binary polarisation angle counter-clockwise about the direction of propagation
    modes: tuple
        Tuple of the polarization modes. Note that this is a tuple to ensure it is hashable
        for the JIT compilation.

    Returns
    -------
    array_like: A 3x3 representation of the polarization_tensor for the specified mode.

    """
    return [get_polarization_tensor(ra, dec, time, psi, mode) for mode in modes]


@jit
def rotation_matrix_from_delta(delta_x):
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
    delta_x /= np.linalg.norm(delta_x)
    alpha = np.arctan2(-delta_x[1] * delta_x[2], delta_x[0])
    beta = np.arccos(delta_x[2])
    gamma = np.arctan2(delta_x[1], delta_x[0])
    rotation_1 = np.array(
        [
            [np.cos(alpha), -np.sin(alpha), np.zeros(alpha.shape)],
            [np.sin(alpha), np.cos(alpha), np.zeros(alpha.shape)],
            [np.zeros(alpha.shape), np.zeros(alpha.shape), np.ones(alpha.shape)],
        ]
    )
    rotation_2 = np.array(
        [
            [np.cos(beta), np.zeros(beta.shape), -np.sin(beta)],
            [np.zeros(beta.shape), np.ones(beta.shape), np.zeros(beta.shape)],
            [np.sin(beta), np.zeros(beta.shape), np.cos(beta)],
        ]
    ).T
    rotation_3 = np.array(
        [
            [np.cos(gamma), -np.sin(gamma), np.zeros(gamma.shape)],
            [np.sin(gamma), np.cos(gamma), np.zeros(gamma.shape)],
            [np.zeros(gamma.shape), np.zeros(gamma.shape), np.ones(gamma.shape)],
        ]
    )
    return rotation_3 @ rotation_2 @ rotation_1


@jit
def three_by_three_matrix_contraction(tensor1, tensor2):
    """
    Calculate the contraction of two 3x3 tensors

    Parameters
    ==========
    tensor1: array_like
        A 3x3 tensor
    tensor2: array_like
        A 3x3 tensor

    Returns
    =======
    array_like: The contraction of tensor1 and tensor2
    """
    return np.einsum("ij,ij", tensor1, tensor2)


@jit
def time_delay_from_geocenter(detector1, ra, dec, time):
    """
    Calculate time delay between a detectors and the geocenter
    based on XLALArrivalTimeDiff in TimeDelay.c

    Parameters
    ----------
    detector1: array_like
        Cartesian coordinate vector for the first detector in the geocentric frame
        generated by the Interferometer class as self.vertex.
    ra: float
        Right ascension of the source in radians
    dec: float
        Declination of the source in radians
    time: float
        GPS time in the geocentric frame

    Returns
    -------
    float: Time delay between the two detectors in the geocentric frame

    """
    return time_delay_geocentric(detector1, np.array([0, 0, 0]), ra, dec, time)


@jit
def time_delay_geocentric(detector1, detector2, ra, dec, time):
    """
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
    gmst = greenwich_mean_sidereal_time(time) % (2 * np.pi)
    speed_of_light = 299792458.0
    phi = ra - gmst
    theta = np.pi / 2 - dec
    omega = np.array(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    )
    delta_d = detector2 - detector1
    return np.dot(omega, delta_d) / speed_of_light


@jit
def zenith_azimuth_to_theta_phi(zenith, azimuth, delta_x):
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
    omega_prime = np.array(
        [
            np.sin(zenith) * np.cos(azimuth),
            np.sin(zenith) * np.sin(azimuth),
            np.cos(zenith),
        ]
    )
    rotation_matrix = rotation_matrix_from_delta(delta_x)
    omega = np.dot(rotation_matrix, omega_prime)
    theta = np.arccos(omega[2])
    phi = np.arctan2(omega[1], omega[0]) % (2 * np.pi)
    return theta, phi
