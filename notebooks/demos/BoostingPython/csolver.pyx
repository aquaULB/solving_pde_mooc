cimport cython

cimport libc.stdlib as lib

import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.wraparound(False)
def c_gauss_seidel(DTYPE_t[:, :] p, DTYPE_t[:, :] b,
DTYPE_t dx, DTYPE_t tol, int max_it):
    # if p.shape[0] != b.shape[0] or p.shape[1] != b.shape[1]:
    #     raise ValueError("p and b must have the same shape.")
    # if tol_hist_gs.size != max_it:
    #     raise ValueError("tol_hist_gs must have lenght equal to max_it.")

    cdef Py_ssize_t nx = p.shape[0]
    cdef Py_ssize_t ny = p.shape[1]
    cdef DTYPE_t diff = np.abs(tol) * 10

    cdef np.ndarray[DTYPE_t, ndim=1] tol_hist_gs = np.empty(max_it, dtype=DTYPE)
    cdef DTYPE_t[:] tol_hist_gs_view = tol_hist_gs

    cdef np.ndarray[DTYPE_t, ndim=2] p_new = np.empty((nx, ny), dtype=DTYPE)
    cdef DTYPE_t[:, :] p_new_view = p_new

    cdef int it = 0
    cdef Py_ssize_t i, j
    while diff > tol:
        it += 1
        if it > max_it:
            break

        for i in range(1, nx-1):
            for j in range(1, ny-1):
                p_new_view[i, j] = (0.25*(p_new_view[i-1, j] + p[i+1, j]
                                 + p_new_view[i, j-1] + p[i, j+1] - b[i, j]*dx**2))

        diff = 0.0
        for i in range(nx):
            for j in range(ny):
                diff += (p_new_view[i, j] - p[i, j])**2
        diff = np.sqrt(diff) / (nx+ny)

        tol_hist_gs_view[it-1] = diff

        p[:, :] = p_new_view
    else:
        return True, p, tol_hist_gs

    return False, p, tol_hist_gs