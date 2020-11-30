import sys
import os.path

from numba import jit
import numpy as np

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    pass

# UNSAFE! __file__ is not defined in all Python distributions.
HOME = os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0, os.path.join(HOME, '../../modules'))

import norms

def py_gauss_seidel_step(p, pnew, b, nx, ny, dx):
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            pnew[i, j] = (0.25*(pnew[i-1, j] + p[i+1, j] + pnew[i, j-1]
                       + p[i, j+1] - b[i, j]*dx**2))
    return pnew

@jit
def gauss_seidel_step(p, pnew, b, nx, ny, dx):
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            pnew[i, j] = (0.25*(pnew[i-1, j] + p[i+1, j] + pnew[i, j-1]
                       + p[i, j+1] - b[i, j]*dx**2))
    return pnew


def gauss_seidel(p, b, dx, tol, max_it, use_numba=False):
    if p.shape != b.shape:
        raise ValueError('p and b must have the same shape')
    nx, ny = p.shape

    tol_hist_gs = []

    diff = np.abs(tol) * 10

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

        try:
            pbar.update(1)
        except NameError:
            pass

        if use_numba:
            pnew = gauss_seidel_step(p, pnew, b, nx, ny, dx)
            diff = norms.l2_diff(pnew, p)
        else:
            pnew = py_gauss_seidel_step(p, pnew, b, nx, ny, dx)
            diff = norms.py_l2_diff(pnew, p)

        tol_hist_gs.append(diff)

        # Memory leak safe deepcopy. Alternative: numpy.copyto.
        p[:, :] = pnew
    else:
        print(f'\nSolution converged after {it} iterations')

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