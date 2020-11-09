import numpy as np

def euler_step(u, f, dt, *args):
    """Returns the solution at the next time-step using 
    the forward Euler method.
    
    Parameters
    ----------
    u : array of floats
        solution at the current time-step.
    f : function
        function to compute the right hand-side of the system of equation.
    dt : float
        time-increment.
    args : optional arguments for the rhs function
    
    Returns
    -------
    unp1 : array of floats
        approximate solution at the next time step.
    """
    unp1 = u + dt * f(u, *args)
    return unp1

def rk4_step(u, f, dt, *args):
    """Returns the solution at the next time-step using 
    the RK4 method. Assumes that f is time independent
    
    Parameters
    ----------
    u : array of floats
        solution at the current time-step.
    f : function
        function to compute the right hand-side of the system of equation.
    dt : float
        time-increment.
    args : optional arguments for the rhs function
    
    Returns
    -------
    u_n_plus_1 : array of floats
        approximate solution at the next time step.
    """
    
    #substep 1
    k1 = f(u, *args)
    
    #substep 2
    k2 = f(u + dt/2.*k1, *args)
    
    #substep 3
    k3 = f(u + dt/2.*k2, *args)
    
    #substep 4
    k4 = f(u + k3*dt, *args)
    
    unp1 = u + dt/6.*(k1 + 2.0*k2 + 2.0*k3 + k4)
    
    return unp1