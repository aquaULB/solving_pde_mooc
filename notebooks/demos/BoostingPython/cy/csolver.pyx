cimport cython

import numpy as np
cimport numpy as np

# We try importing tqdm for progress bars. If it is not found, do nothing.
try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    pass

# We set up double precision for the floats. Each numpy datatype has C ana-
# logy with suffix _t.
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False) # Deactivate bounds checking for arrays
@cython.wraparound(False)  # Deactivate treatment for negative indexing
def c_gauss_seidel(DTYPE_t[:, :] p, DTYPE_t[:, :] b,
                   DTYPE_t dx, DTYPE_t tol, int max_it):
    '''Performs Gauss-Seidel iterations until convergence

    Parameters
    ----------
    p: memoryview
        Initial guess for solution
    b: memoryview
        RHS of equation
    dx: numpy.float64
        Grid step on uniform grid
    tol: numpy.float64
        Convergence precision
    max_it: int
        Maximal amount of iterations in convergence study

    Returns
    -------
    bool
        Whether solution converged
    p.base: numpy.ndarray
        Final solution (Base of Cython memoryview)
    tol_hist_gs: numpy.ndarray
        Array filled with L2 differences between previous and current solutions
        computed after each Gauss-Seidel iteration
    '''
    # According to official docs: p.shape is now a C array, so it's not possible
    # to compare it simply by using == without a for-loop. To be able to compare
    # it to b.shape easily, we convert them both to Python tuples.
    if tuple(p.shape) != tuple(b.shape):
        raise ValueError("p and b must have the same shape")

    # Here you can see that the C arrays p.shape and b.shape contain data of type
    # Py_ssize_t (not int like in NumPy).
    cdef Py_ssize_t nx = p.shape[0]
    cdef Py_ssize_t ny = p.shape[1]

    cdef int isize = nx * ny

    cdef DTYPE_t diff = np.abs(tol) * 10

    # Creation of memoryview from numpy.ndarray.
    cdef np.ndarray[DTYPE_t, ndim=1] tol_hist_gs = np.zeros(max_it, dtype=DTYPE)
    cdef DTYPE_t[:] tol_hist_gs_view = tol_hist_gs

    # p is a memoryview. memoryviews in Cython also have copy methods.
    cdef DTYPE_t[:, :] p_new_view = p.copy()

    try:
        pbar = tqdm(total=max_it)
        pbar.set_description("it / max_it")
    except NameError:
        pass

    # Declare iteration variables. Note that as we are going to loop in
    # range(nx) and range(ny), which are of type Py_ssize_t, i and j must also
    # be of type Py_ssize_t.
    cdef int it = 0
    cdef Py_ssize_t i, j
    while diff > tol:
        it += 1
        if it > max_it:
            break

        try:
            pbar.update(1)
        except NameError:
            pass

        # In a loop we manipulate the memoryviews, not numpy arrays. This pro-
        # vides the actual speed up.
        for j in range(1, ny-1):
            for i in range(1, nx-1):
                p_new_view[i, j] = (0.25*(p_new_view[i-1, j] + p[i+1, j]
                    + p_new_view[i, j-1] + p[i, j+1] - b[i, j]*dx**2))

        # We cannot use l2_diff anymore - it is suited for numpy arrays but not the
        # memoryviews.
        diff = 0.0
        for i in range(nx):
            for j in range(ny):
                diff += (p_new_view[i, j] - p[i, j])**2
        # Most numpy functions can take memryview as a parameter.
        diff = np.sqrt(diff) / isize

        # Not a list like in Python implementation - has to be accessed by the
        # index.
        tol_hist_gs_view[it-1] = diff

        p[:, :] = p_new_view
    else:
        print(f'\nSolution converged after {it} iterations')

        try:
            pbar.close()
        except NameError:
            pass

        # Note that we only return the section of tol_hist_gs that has been
        # filled when iterating.
        return True, p.base, tol_hist_gs[:it].base

    print('\nSolution did not converged within the maximum '
          'number of iterations')

    try:
        pbar.close()
    except NameError:
        pass

    # Note that we only return the section of tol_hist_gs that has been
    # filled when iterating.
    return False, p.base, tol_hist_gs[:it-1].base