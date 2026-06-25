# cython: language_level=3
import cython
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef public void gpu_kernel(double w, double *pi_value_ptr, int n):
    cdef double pi_value = pi_value_ptr[0]
    cdef int i
    cdef double local_x
    cdef int OMP4PY_MARKER_23324916967872 = 0

    for i in range(n):
        local_x = (i + 0.5) * w
        pi_value += 4.0 / (1.0 + local_x * local_x)

    pi_value_ptr[0] = pi_value