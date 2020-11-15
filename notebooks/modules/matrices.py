import numpy as np
from scipy.sparse import diags

def d2_mat_dirichlet(nx, dx):
    """
    Constructs the centered second-order accurate second-order derivative for
    Dirichlet boundary conditions.

    Parameters
    ----------
    nx : integer
        number of grid points
    dx : float
        grid spacing

    Returns
    -------
    d2mat : numpy.ndarray
        matrix to compute the centered second-order accurate first-order deri-
        vative with Dirichlet boundary conditions on both side of the interval
    """
    # We construct a sequence of main diagonal elements,
    diagonals = [[1.], [-2.], [1.]]
    # and a sequence of positions of the diagonal entries relative to the main
    # diagonal.
    offsets = [-1, 0, 1]

    # Call to the diags routine; note that diags return a representation of the
    # array; to explicitly obtain its ndarray realisation, the call to .toarray()
    # is needed. Note how the matrix has dimensions (nx-2)*(nx-2).
    d2mat = diags(diagonals, offsets, shape=(nx-2,nx-2)).toarray()

    # Return the final array divided by the grid spacing **2.
    return d2mat / dx**2
