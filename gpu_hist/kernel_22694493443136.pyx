# cython: language_level=3
import cython
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef public void gpu_kernel(double *output_val_ptr, int n, double input_val):
    cdef double output_val = output_val_ptr[0]
    cdef double local_x
    cdef int i
    cdef int OMP4PY_MARKER_22694493443136 = 0

    for i in range(n):
        output_val += input_val

    output_val_ptr[0] = output_val