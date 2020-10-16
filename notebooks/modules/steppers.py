import numpy as np

def euler_step(u, f, dt, *args):
    """Returns the solution at the next time-step using 
    the forward Euler method.
    
    Parameters
    ----------
    u : array of float
        solution at the current time-step.
    f : function
        function to compute the right hand-side of the system of equation.
    dt : float
        time-increment.
    
    Returns
    -------
    u_n_plus_1 : array of float
        approximate solution at the next time step.
    """
    
    return u + dt * f(u, *args)