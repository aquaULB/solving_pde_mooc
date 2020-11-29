import sys
import os.path

import numpy as np

# UNSAFE! __file__ is not defined in all Python distributions.
HOME = os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0, os.path.join(HOME, '../../modules'))

import norms

def py_gauss_seidel(p, b, dx, tol, max_it, tol_hist_gs=[]):
    if p.shape != b.shape:
        raise ValueError('p and b must have the same shape')
    nx, ny = p.shape

    # If tol_hist_gs is not a list, make it one.
    if not isinstance(tol_hist_gs, list):
        tol_hist_gs = []

    diff = np.abs(tol) * 10

    it = 0
    pnew = p.copy()
    while diff > tol:
        it += 1
        if it > max_it: break

        for i in range(1, nx-1):
            for j in range(1, ny-1):
                pnew[i, j] = (0.25*(pnew[i-1, j] + p[i+1, j] + pnew[i, j-1]
                           + p[i, j+1] - b[i, j]*dx**2))

        diff = norms.l2_diff(pnew, p)
        tol_hist_gs.append(diff)

        print('\r', f'diff: {diff:5.2e}', end='')

        # Memory leak safe deepcopy. Alternative: numpy.copyto.
        p[:, :] = pnew
    else:
        print(f'\nSolution converged after {it} iterations')
        return True

    print('\nSolution did not converged within the maximum '
          'number of iterations')
    return False