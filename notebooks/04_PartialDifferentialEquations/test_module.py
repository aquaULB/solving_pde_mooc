import numpy as np


def triangle_area(base, height):
    """Returns the area of a triangle

    Parameters
    ----------
    base : float
        length of the triangle's base
    height : function
        height of the triangle

    Returns
    -------
    area : float
        area of the triangle
    """

    area = 0.5*base*height

    return area


def twos(shape):
    """Returns a 1D array filled with the value 2

    Parameters
    ----------
    shape : integer
        size requested for the output vector

    Returns
    -------
    myarray : numpy 1D array
        array of size shape, filled with the value 2
    """

    myarray = 2.*np.ones(shape)

    return myarray
