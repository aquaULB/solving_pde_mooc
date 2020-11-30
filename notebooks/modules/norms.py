from numba import njit
import numpy as np

def py_l2_diff(f1, f2):
    """
    Computes the l2-norm of the difference
    between a function f1 and a function f2

    Parameters
    ----------
    f1 : array of floats
        function 1
    f2 : array of floats
        function 2

    Returns
    -------
    diff : float
        The l2-norm of the difference.
    """
    l2_diff = np.sqrt(np.sum((f1 - f2)**2))/f1.size

    return l2_diff

@njit
def l2_diff(f1, f2):
    """
    Computes the l2-norm of the difference
    between a function f1 and a function f2

    Parameters
    ----------
    f1 : array of floats
        function 1
    f2 : array of floats
        function 2

    Returns
    -------
    diff : float
        The l2-norm of the difference.
    """
    l2_diff = np.sqrt(np.sum((f1 - f2)**2)) / f1.size

    return l2_diff
