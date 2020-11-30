import numpy as np

def p_exact_2d(X, Y):
    """
    Computes the exact solution of the Poisson equation in the domain 
    [0, 1]x[-0.5, 0.5] with rhs:
    b = (np.sin(np.pi * X) * np.cos(np.pi * Y) +
     np.sin(5.0 * np.pi * X) * np.cos(5.0 * np.pi * Y))
    
    Parameters
    ----------
    X : numpy.ndarray
        array of x coordinates for all grid points
    Y : numpy.ndarray
        array of x coordinates for all grid points

    Returns
    -------
    sol : numpy.ndarray
        exact solution of the Poisson equation
    """
    
    sol = ( -1.0 / (2.0*np.pi**2) * np.sin(np.pi * X) * np.cos(np.pi * Y) + 
     -1.0 / (50.0*np.pi**2) * np.sin(5.0 * np.pi * X) * np.cos(5.0 * np.pi * Y) )
    
    return sol

def rhs_2d(X, Y):
    """
    Computes the right-hand side of the Poisson equation in the domain 
    [0, 1]x[-0.5, 0.5]:
    b = (np.sin(np.pi * X) * np.cos(np.pi * Y) +
     np.sin(5.0 * np.pi * X) * np.cos(5.0 * np.pi * Y))
    
    Parameters
    ----------
    X : numpy.ndarray
        array of x coordinates for all grid points
    Y : numpy.ndarray
        array of x coordinates for all grid points

    Returns
    -------
    rhs : numpy.ndarray
        exact solution of the Poisson equation
    """
    
    rhs = (np.sin(np.pi * X) * np.cos(np.pi * Y) +
     np.sin(5.0 * np.pi * X) * np.cos(5.0 * np.pi * Y))
    
    return rhs
