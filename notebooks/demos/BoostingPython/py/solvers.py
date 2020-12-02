import sys
import os.path

from numba import jit
import numpy as np

# We try importing tqdm for progress bars. If it is not found, do nothing.
try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    pass

# UNSAFE! __file__ is not always defined.
HOME = os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0, os.path.join(HOME, '../../../modules'))

import norms

# We will now create two functions that perform Gauss-Seidel iteration. One
# of them we decorate with numba.jit decorator.

def py_gauss_seidel_step(p, pnew, b, nx, ny, dx):
    '''Performs one Gauss-Seidel iteration'''
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            pnew[i, j] = (0.25*(pnew[i-1, j] + p[i+1, j] + pnew[i, j-1]
                       + p[i, j+1] - b[i, j]*dx**2))
    return pnew

@jit(nopython=True)
def gauss_seidel_step(p, pnew, b, nx, ny, dx):
    '''Performs one Gauss-Seidel iteration'''
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            pnew[i, j] = (0.25*(pnew[i-1, j] + p[i+1, j] + pnew[i, j-1]
                       + p[i, j+1] - b[i, j]*dx**2))
    return pnew


def gauss_seidel(p, b, dx, tol, max_it=1000, use_numba=False):
    '''Performs Gauss-Seidel iterations until convergence

    Parameters
    ----------
    p: numpy.ndarray
        Initial guess for solution
    b: numpy.ndarray
        RHS of equation
    dx: float
        Grid step on uniform grid
    tol: float
        Convergence precision
    max_it: int
        Maximal amount of iterations in convergence study
    use_numba: bool
        Compile with Numba or not

    Returns
    -------
    bool
        Whether solution converged
    p: numpy.ndarray
        Final solution
    tol_hist_gs: list
        List filled with L2 differences between previous and current solutions
        computed after each Gauss-Seidel iteration
    '''
    if p.shape != b.shape:
        raise ValueError('p and b must have the same shape')
    nx, ny = p.shape

    tol_hist_gs = []

    # Ensure that it's always larger than tol before entering the loop.
    diff = np.abs(tol) * 10

    # Try starting the progress bar. If not found (because not imported),
    # just proceed further.
    try:
        pbar = tqdm(total=max_it)
        pbar.set_description("it / max_it")
    except NameError:
        pass

    it = 0
    pnew = p.copy()
    while diff > tol:
        it += 1
        if it > max_it: break

        # Try updating progress bar if it's available.
        try:
            pbar.update(1)
        except NameError:
            pass

        # If use_numba flag is set to True, do Gauss-Seidel step with decorated
        # function. norms module also contains two versions of l2_diff: decora-
        # ted one and non-decorated one.
        if use_numba:
            pnew = gauss_seidel_step(p, pnew, b, nx, ny, dx)
            diff = norms.l2_diff(pnew, p)
        else:
            pnew = py_gauss_seidel_step(p, pnew, b, nx, ny, dx)
            diff = norms.py_l2_diff(pnew, p)

        tol_hist_gs.append(diff)

        # Memory leak safe deepcopy (garbage collector sometimes has troubles
        # under certain circumstances when in a loop). Alternative: numpy.copyto.
        p[:, :] = pnew
    else:
        print(f'\nSolution converged after {it} iterations')

        # If progress bar is active, needs to be finalized.
        try:
            pbar.close()
        except NameError:
            pass

        return True, p, tol_hist_gs

    print('\nSolution did not converged within the maximum '
          'number of iterations')

    try:
        pbar.close()
    except NameError:
        pass

    return False, p, tol_hist_gs