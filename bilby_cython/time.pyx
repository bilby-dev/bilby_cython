import numpy as np
cimport numpy as np
import cython
from libc.math cimport sin, cos, M_PI as pi


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cpdef time_shift(double delta_frequency, double minimum_frequency, double maximum_frequency, double shift):
    cdef int n_frequencies = int((maximum_frequency - minimum_frequency) // delta_frequency) + 1
    cdef double current_real, current_imag, delta_real, delta_imag, temp

    output_real = np.empty(n_frequencies)
    output_imag = np.empty(n_frequencies)
    cdef double[:] output_real_ = output_real
    cdef double[:] output_imag_ = output_imag

    delta_imag = - sin(2 * pi * shift * delta_frequency)
    delta_real = - 2 * sin(pi * shift * delta_frequency) * sin(pi * shift * delta_frequency)
    current_real = cos(2 * pi * shift * minimum_frequency)
    current_imag = -sin(2 * pi * shift * minimum_frequency)
    for ii in range(n_frequencies):
        output_real_[ii] = current_real
        output_imag_[ii] = -current_imag
        temp = current_real + current_real * delta_real - current_imag * delta_imag
        current_imag += current_real * delta_imag + current_imag * delta_real
        current_real = temp

    return output_real + 1j * output_imag
