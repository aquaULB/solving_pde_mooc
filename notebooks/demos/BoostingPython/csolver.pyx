import sys
import os.path

cimport cython

cimport libc.stdlib as lib

import numpy as np
cimport numpy as np

# UNSAFE! __file__ is not defined in all Python distributions.
HOME = os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0, os.path.join(HOME, '../../modules'))

import norms

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.wraparound(False)
def c_gauss_seidel(np.ndarray[DTYPE_t, ndim=2] p, np.ndarray[DTYPE_t, ndim=2] b,
np.ndarray[DTYPE_t, ndim=1] tol_hist_gs, DTYPE_t dx, DTYPE_t tol, int max_it):
    if p.shape[0] != b.shape[0] or p.shape[1] != b.shape[1]:
        raise ValueError("p and b must have the same shape.")
    if tol_hist_gs.size != max_it:
        raise ValueError("tol_hist_gs must have lenght equal to max_it.")

    cdef int nx = p.shape[0]
    cdef int ny = p.shape[1]
    cdef DTYPE_t diff = np.abs(tol) * 10

    cdef np.ndarray[DTYPE_t, ndim=2] pnew = p.copy()

    cdef int it = 0
    while diff > tol:
        it += 1
        if it > max_it:
            break

        for i in range(1, nx-1):
            for j in range(1, ny-1):
                pnew[i, j] = (0.25*(pnew[i-1, j] + p[i+1, j] + pnew[i, j-1]
                           + p[i, j+1] - b[i, j]*dx**2))

        diff = norms.l2_diff(pnew, p)
        tol_hist_gs[i-1] = diff

        np.copyto(p, pnew)
    else:
        return it

    return it - 1