import numpy as np
cimport numpy as np
from libc.math cimport sin, cos, fmod, pi, acos, atan2, atan, pow
from .time import greenwich_mean_sidereal_time

cdef double CC = 299792458.0


cpdef time_delay_geocentric(
    np.ndarray[np.float64_t, ndim=1] detector1,
    np.ndarray[np.float64_t, ndim=1] detector2,
    double ra,
    double dec,
    double time,
):
    """
    Calculate time delay between two detectors in geocentric coordinates based on XLALArrivaTimeDiff in TimeDelay.c

    Parameters
    ----------
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
    -------
    float: Time delay between the two detectors in the geocentric frame

    """
    cdef double output, gmst, theta, phi, sintheta, costheta, sinphi, cosphi
    cdef double[:] detector_1_view = detector1
    cdef double[:] detector_2_view = detector2

    gmst = fmod(greenwich_mean_sidereal_time(time), 2 * pi)
    phi = ra - gmst
    theta = pi / 2 - dec
    sintheta = sin(theta)
    costheta = cos(theta)
    sinphi = sin(phi)
    cosphi = cos(phi)

    output = (
        (detector_2_view[0] - detector_1_view[0]) * sintheta * cosphi
        + (detector_2_view[1] - detector_1_view[1]) * sintheta * sinphi
        + (detector_2_view[2] - detector_1_view[2]) * costheta
    ) / CC
    return output


_GEOCENTER = np.zeros(3, dtype=float)


cpdef time_delay_from_geocenter(
    np.ndarray[np.float64_t, ndim=1] detector1,
    double ra,
    double dec,
    double time,
):
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
    return time_delay_geocentric(detector1, _GEOCENTER, ra, dec, time)


cdef _vectors_for_polarization_tensor(
    double phi,
    double theta,
    double psi,
    double[:] omega_view,
    double[:] m_view,
    double[:] n_view,
):
    r"""
    Compute the three vectors that can be used to construct the different
    population modes.

    .. math::

        m = (
            - \cos\theta \cos\phi \sin\psi + \sin\phi \cos\psi,
            - \cos\theta \sin\phi \sin\psi - \cos\phi \cos\psi,
            \sin\theta \sin\psi
        )

        n = (
            - \cos\theta \cos\phi \cos\psi - \sin\phi \sin\psi,
            - \cos\theta \sin\phi \cos\psi + \cos\phi \sin\psi,
            \sin\theta \cos\psi
        )
        
        \omega = m \cross n

    Parameters
    ----------
    phi
    theta
    psi

    Returns
    -------

    """
    cdef double cosphi, sinphi, costheta, sintheta, cospsi, sinpsi
    cdef int ii, jj

    cosphi = cos(phi)
    sinphi = sin(phi)
    costheta = cos(theta)
    sintheta = sin(theta)
    cospsi = cos(psi)
    sinpsi = sin(psi)
    m_view[0] = - costheta * cosphi * sinpsi + sinphi * cospsi
    m_view[1] = - costheta * sinphi * sinpsi - cosphi * cospsi
    m_view[2] = sintheta * sinpsi
    n_view[0] = - costheta * cosphi * cospsi - sinphi * sinpsi
    n_view[1] = - costheta * sinphi * cospsi + cosphi * sinpsi
    n_view[2] = sintheta * cospsi
    omega_view[0] = m_view[1] * n_view[2] - m_view[2] * n_view[1]
    omega_view[1] = m_view[2] * n_view[0] - m_view[0] * n_view[2]
    omega_view[2] = m_view[0] * n_view[1] - m_view[1] * n_view[0]


cpdef _polarization_tensor(
    double[:, :] output_view,
    str mode,
    double[:] omega_view,
    double[:] m_view,
    double[:] n_view,
):
    if mode == 'plus':
        _plus(output_view, m_view, n_view)
    elif mode == 'cross':
        _cross(output_view, m_view, n_view)
    elif mode == 'breathing':
        _breathing(output_view, m_view, n_view)
    elif mode == 'longitudinal':
        _longitudinal(output_view, omega_view)
    elif mode == 'x':
        _x(output_view, omega_view, m_view)
    elif mode == 'y':
        _y(output_view, omega_view, n_view)
    else:
        raise ValueError("{} not a polarization mode!".format(mode))


cpdef get_polarization_tensor(double ra, double dec, double time, double psi, str mode):
    """
    Calculate the polarization tensor for a given sky location and time

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
    mode: str
        polarisation mode

    Returns
    -------
    array_like: A 3x3 representation of the polarization_tensor for the specified mode.

    """
    cdef double gmst, phi, theta
    output = np.empty((3, 3))
    cdef double[:, :] output_view = output
    omega = np.empty(3)
    m = np.empty(3)
    n = np.empty(3)
    cdef double[:] omega_view = omega
    cdef double[:] m_view = m
    cdef double[:] n_view = n

    gmst = fmod(greenwich_mean_sidereal_time(time), 2 * pi)
    phi = ra - gmst
    theta = pi / 2 - dec
    _vectors_for_polarization_tensor(phi, theta, psi, omega_view, m_view, n_view)

    _polarization_tensor(output_view, mode, omega_view, m_view, n_view)

    return output


cpdef get_polarization_tensor_multiple_modes(double ra, double dec, double time, double psi, list modes):
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
    modes: list
        List of the polarization modes

    Returns
    -------
    array_like: A 3x3 representation of the polarization_tensor for the specified mode.

    """
    cdef double gmst, phi, theta
    cdef double[:, :] output_view
    omega = np.empty(3)
    m = np.empty(3)
    n = np.empty(3)
    cdef double[:] omega_view
    cdef double[:] m_view
    cdef double[:] n_view
    output = list()

    gmst = fmod(greenwich_mean_sidereal_time(time), 2 * pi)
    phi = ra - gmst
    theta = pi / 2 - dec
    _vectors_for_polarization_tensor(phi, theta, psi, omega_view, m_view, n_view)

    for mode in modes:
        tensor = np.zeros((3, 3))
        output_view = tensor
        _polarization_tensor(output_view, mode, omega_view, m_view, n_view)
        output.append(tensor)
    return output


cdef _plus(double[:, :] output, double[:] m_view, double[:] n_view):
    cdef int ii, jj

    for ii in range(3):
        output[ii][ii] = m_view[ii] * m_view[ii] - n_view[ii] * n_view[ii]
        for jj in range(ii):
            output[ii][jj] = m_view[ii] * m_view[jj] - n_view[ii] * n_view[jj]
            output[jj][ii] = output[ii][jj]


cdef _breathing(double[:, :] output, double[:] m_view, double[:] n_view):
    cdef int ii, jj

    for ii in range(3):
        output[ii][ii] = m_view[ii] * m_view[ii] + n_view[ii] * n_view[ii]
        for jj in range(ii):
            output[ii][jj] = m_view[ii] * m_view[jj] + n_view[ii] * n_view[jj]
            output[jj][ii] = output[ii][jj]


cdef _longitudinal(double[:, :] output, double[:] omega_view):
    cdef int ii, jj

    for ii in range(3):
        output[ii][ii] = omega_view[ii] * omega_view[ii]
        for jj in range(ii):
            output[ii][jj] = omega_view[ii] * omega_view[jj]
            output[jj][ii] = output[ii][jj]


cdef _symmetric_response(double[:, :] output, double[:] input_1, double[:] input_2):
    cdef int ii, jj

    for ii in range(3):
        output[ii][ii] = 2 * input_1[ii] * input_2[ii]
        for jj in range(ii):
            output[ii][jj] = input_1[ii] * input_2[jj] + input_1[jj] * input_2[ii]
            output[jj][ii] = output[ii][jj]


cdef _cross(double[:, :] output, double[:] m_view, double[:] n_view):
    _symmetric_response(output, m_view, n_view)


cdef _x(double[:, :] output, double[:] omega_view, double[:] m_view):
    _symmetric_response(output, m_view, omega_view)


cdef _y(double[:, :] output, double[:] omega_view, double[:] n_view):
    _symmetric_response(output, n_view, omega_view)


cpdef three_by_three_matrix_contraction(np.ndarray[np.float64_t, ndim=2] x, np.ndarray[np.float64_t, ndim=2] y):
    """
    Doubly contract two 3x3 input matrices following Einstein summation.
    
    ..math::

        output = x_{ij} y_{ij}
    
    Parameters
    ----------
    x: array_like
        First input matrix
    y: array_like
        Second input matrix

    Returns
    -------
    output: float
        The contracted value

    """
    cdef double output = 0
    cdef double[:, :] x_view = x
    cdef double[:, :] y_view = y

    for ii in range(3):
        for jj in range(3):
            output += x_view[ii, jj] * y_view[ii, jj]
    return output


cpdef detector_tensor(np.ndarray[np.float64_t, ndim=1] x, np.ndarray[np.float64_t, ndim=1] y):
    r"""
    Compute the detector tensor given the two unit arm vectors.
    
    .. math::

        d_{ij} = \frac{x_{i} x_{j} - y_{i} y_{j}}{2}

    Parameters
    ----------
    x: array_like
        The x-arm vector
    y: array_like
        The y-arm vector

    Returns
    -------
    output: array_like
        The 3x3 detector tensor

    """
    output = np.empty((3, 3))
    cdef double[:] x_view = x
    cdef double[:] y_view = y
    cdef double[:, :] output_ = output

    for ii in range(3):
        for jj in range(3):
            output_[ii, jj] = (x_view[ii] * x_view[jj] - y_view[ii] * y_view[jj]) / 2
    return output



cpdef calculate_arm(double arm_tilt, double arm_azimuth, double longitude, double latitude):
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
    output = np.empty(3)
    cdef double[:] output_ = output
    cdef double cos_tilt, sin_tilt
    cdef double cos_azimuth, sin_azimuth
    cdef double cos_longitude, sin_longitude
    cdef double cos_latitude, sin_latitude
    cos_tilt = cos(arm_tilt)
    sin_tilt = sin(arm_tilt)
    cos_azimuth = cos(arm_azimuth)
    sin_azimuth = sin(arm_azimuth)
    cos_longitude = cos(longitude)
    sin_longitude = sin(longitude)
    cos_latitude = cos(latitude)
    sin_latitude = sin(latitude)

    output_[0] = (
        - sin_longitude * cos_tilt * cos_azimuth
        - sin_latitude * cos_longitude * cos_tilt * sin_azimuth
        + cos_latitude * cos_longitude * sin_tilt
    )
    output_[1] = (
        cos_longitude * cos_tilt * cos_azimuth
        - sin_latitude * sin_longitude * cos_tilt * sin_azimuth
        + cos_latitude * sin_longitude * sin_tilt
    )
    output_[2] = cos_latitude * cos_tilt * sin_azimuth + sin_latitude * sin_tilt
    return output


cdef euler_rotation(double[:] delta_x, double[:, :] rotation):
    """
    Calculate the rotation matrix mapping the vector (0, 0, 1) to delta_x
    while preserving the origin of the azimuthal angle.

    This is decomposed into three Euler angles, alpha, beta, gamma, which rotate
    about the z-, y-, and z- axes respectively.
    """
    cdef double alpha, gamma, norm
    cdef double cos_alpha, sin_alpha, cos_beta, sin_beta, cos_gamma, sin_gamma

    norm = pow(delta_x[0] * delta_x[0] + delta_x[1] * delta_x[1] + delta_x[2] * delta_x[2], 0.5)
    cos_beta = delta_x[2] / norm
    sin_beta = pow(1 - cos_beta**2, 0.5)

    alpha = atan2(- delta_x[1] * cos_beta, delta_x[0])
    gamma = atan2(delta_x[1], delta_x[0])

    cos_alpha = cos(alpha)
    sin_alpha = sin(alpha)
    cos_gamma = cos(gamma)
    sin_gamma = sin(gamma)

    rotation[0][0] = cos_alpha * cos_beta * cos_gamma - sin_alpha * sin_gamma
    rotation[1][0] = cos_alpha * cos_beta * sin_gamma + sin_alpha * cos_gamma
    rotation[2][0] = -cos_alpha * sin_beta
    rotation[0][1] = -sin_alpha * cos_beta * cos_gamma - cos_alpha * sin_gamma
    rotation[1][1] = -sin_alpha * cos_beta * sin_gamma + cos_alpha * cos_gamma
    rotation[2][1] = sin_alpha * sin_beta
    rotation[0][2] = sin_beta * cos_gamma
    rotation[1][2] = sin_beta * sin_gamma
    rotation[2][2] = cos_beta


cpdef rotation_matrix_from_delta(delta_x):
    """
    Calculate the rotation matrix mapping the vector (0, 0, 1) to delta_x
    while preserving the origin of the azimuthal angle.

    This is decomposed into three Euler angles, alpha, beta, gamma, which rotate
    about the z-, y-, and z- axes respectively.

    Parameters
    ==========
    delta_x: array-like (3,)
        Vector onto which (0, 0, 1) should be mapped.
    rotation: array-like (3,3)

    Returns
    =======
    total_rotation: array-like (3,3)
        Rotation matrix which maps vectors from the frame in which delta_x is
        aligned with the z-axis to the target frame.
    """
    cdef double[:] delta
    cdef double[:, :] rotation_
    rotation = np.empty((3, 3))
    delta = delta_x
    rotation_ = rotation
    euler_rotation(delta, rotation_)
    return rotation


cpdef zenith_azimuth_to_theta_phi(double zenith, double azimuth, np.ndarray[np.float64_t, ndim=1] delta_x):
    """
    Convert from the 'detector frame' to the Earth frame.

    Parameters
    ==========
    zenith: float
        The zenith angle in the detector frame
    azimuth: float
        The azimuthal angle in the detector frame
    delta_x: array_like
        The separation vector for the two detectors defining the frame

    Returns
    =======
    theta, phi: float
        The zenith and azimuthal angles in the earth frame.
    """
    rotation = np.empty((3, 3))
    cdef double sin_azimuth, cos_azimuth
    cdef double sin_zenith, cos_zenith
    cdef double[:] delta_
    cdef double[:, :] rotation_
    sin_azimuth = sin(azimuth)
    cos_azimuth = cos(azimuth)
    sin_zenith = sin(zenith)
    cos_zenith = cos(zenith)

    delta_ = delta_x
    rotation_ = rotation
    euler_rotation(delta_, rotation_)

    theta = acos(
        rotation_[2][0] * sin_zenith * cos_azimuth
        + rotation_[2][1] * sin_zenith * sin_azimuth
        + rotation_[2][2] * cos_zenith
    )
    phi = fmod(
        atan2(
            rotation_[1][0] * sin_zenith * cos_azimuth
            + rotation_[1][1] * sin_zenith * sin_azimuth
            + rotation_[1][2] * cos_zenith,
            rotation_[0][0] * sin_zenith * cos_azimuth
            + rotation_[0][1] * sin_zenith * sin_azimuth
            + rotation_[0][2] * cos_zenith
        ) + 2 * pi,
        (2 * pi)
    )
    return theta, phi
