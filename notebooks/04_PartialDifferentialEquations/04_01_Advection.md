---
jupytext:
  formats: ipynb,md:myst
  notebook_metadata_filter: toc
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.12
    jupytext_version: 1.6.0
kernelspec:
  display_name: Python 3
  language: python
  name: python3
toc:
  base_numbering: 1
  nav_menu: {}
  number_sections: true
  sideBar: true
  skip_h1_title: true
  title_cell: Table of Contents
  title_sidebar: Contents
  toc_cell: true
  toc_position: {}
  toc_section_display: true
  toc_window_display: false
---

<h1 style="text-align: center">Partial differential Equation I</h1>

+++ {"toc": true}

<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Python-modules" data-toc-modified-id="Python-modules-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Python modules</a></span></li><li><span><a href="#Advection-equation" data-toc-modified-id="Advection-equation-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Advection equation</a></span><ul class="toc-item"><li><span><a href="#Forward-Euler,-forward-finite-difference" data-toc-modified-id="Forward-Euler,-forward-finite-difference-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Forward Euler, forward finite difference</a></span></li></ul></li><li><span><a href="#Summary" data-toc-modified-id="Summary-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Summary</a></span></li></ul></div>

+++

## Introduction

For convenience, we start with importing some modules needed below:

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use('../styles/mainstyle.use')
```

In Chapter 2 and 3 of this course, we described respectively the time integration of ordinary differential equations and the discretization of differential operators using finite difference formulas.

Here we combine these tools to address the numerical solution of partial differential equations. We mainly focus on the first order wave equation (all symbols are properly defined in the corresponding sections of the notebooks),

$$
   \frac{\partial u}{\partial t}+c\frac{\partial u}{\partial x}=0,\label{eq:advection}
$$

and the heat equation,

$$
   \partial_t T(x,t) = \alpha \frac{d^2 T} {dx^2}(x,t) + \sigma (x,t).
$$

To solve these equations we will transform them into systems of coupled ordinary differential equations using a semi-discretization technique. In that framework, our model equations are approximated as,

\begin{equation}
    \frac{du_i}{dt}=f_i (u_0, u_1, \ldots, u_{nx-1}), \; \; i=0,\ldots, nx-1 \label{eq:semiDiscrete}
\end{equation}

To achieve this, we need to choose a time integration scheme for the left-hand side *and* a spatial discretization scheme for the right-hand side. The combination of these two choices determines the success or failure of the whole numerical method. 

As usual, we need to introduce a grid and grid points at which we evaluate our physical quantities. We consider again a uniform grid consisting of $nx$ grid points with coordinates,

\begin{equation}
 x_i = i \Delta x, \; \; 0\leq i \leq nx-1
\end{equation}

where $\Delta x$ is the grid spacing.

For the time integration scheme we use a constant time step $dt$ so that all the intermediate times at which we compute our variables are,

\begin{equation}
 t^n = t_0 + n dt, \; \; 0\leq n \leq nt.
\end{equation}

The discrete quantities we want to evaluate are therefore,

\begin{align}
u^n_i = u(x_i, t^n).
\end{align}

But before we start digging into the theory and show some examples, we introduce below the concept of Python modules.

+++

## Python modules

Up to now, we have always written all the necessary Python code parts for our examples and test cases inside each notebooks. This has the advantage of making all the necessary statements and steps visible to the reader. But in the long run, this approach also has its limitations. For example, if we need a function to compute the first order derivative, we need to define it in each notebook and this is a repetitive task. Gladly, Python allows to overcome this through the use of *modules*.

From the official Python documentation: 

> [A module][21] is a file containing Python definitions and statements. The file name is the module name with the suffix .py appended.

A Python module may contain any Python code. This code can then be imported into your main program or other modules. In fact, we have already used modules. When writing,

> `from scipy.linalg import inv`

we are *importing* the `inv` function, defined in the module `scipy.linalg` to make it available in our notebook. This is very convenient, we don't need to copy/paste the Python code contained in the `inv` function to use it.

So let us create our first module. If you look into the directory in which this notebook is stored on your computer, you will notice a file called `test_module.py`. Take a look at this file now. Its content is reproduced here:

```python
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

```

[21]: <https://docs.python.org/3/tutorial/modules.html> "Modules documentation"

+++

It contains the definition of two functions. Notice how we import `numpy` at the beginning of the file. Without this import, the second function would not work.

To use the function `triangle_area` in the current notebook, we need to import it using:

```{code-cell} ipython3
from test_module import triangle_area
```

The function can then be called as usual:

```{code-cell} ipython3
area = triangle_area(4, 3)

print(f'The area of the triangle is: {area}.')
```

At this stage we may not yet use the function `two` because it has not been imported. If it's needed, we may type

```{code-cell} ipython3
from test_module import twos
```

and use it afterwards:

```{code-cell} ipython3
a = twos(4)
print(a)
```

When more than one item of a module are needed, we can import several or all of them at the same time. Here are different ways to do this:

```{code-cell} ipython3
# Imports the functions separated by a comma
from test_module import triangle_area, twos

# Import every item contained in test_module
from test_module import *

# Import all of test_module and make its content 
# available by using its name followed by a dot.
# Sample usage would be: test_module.twos(4)
import test_module

# Import all of test_module and create a nickname
# to use its content.
# Sample usage would be: tm.twos(4)
import test_module as tm
```

The choice between the different ways of loading modules is somewhat a matter of taste and often depends on the context (there are plenty of tutorials on the Internet discussing this). For this course, we adopt the following convention:

- For `numpy` we use `import numpy as np` to import the whole content of the package and we use the shortcut `np`.

- For `scipy`, we import the desired functions from the submodules of the package, e.g `from scipy.linalg import inv`

- For our own modules, we use the `from module import item1, item2, ...` construct to be explicit about what we want to import.

+++

If the module file is located in the same folder as your notebook or main program, all the commands described above will work. If not, the module file needs to be located in one of the default search paths of your Python distribution. These paths can be listed by executing the following commands:

```python
import sys
print(path.sys)
```

+++

In a specific notebook, module or program, you may add extra search paths to locate your modules. For this course, we store our custom modules in a folder called *modules*. It is located in the *notebooks* folder. Therefore, its relative path to any notebook is `../modules` (the `..` represents the parent directory). Look now in your local repository and you should see this folder.

To add this path to the default search path, you may run the following cell:

```{code-cell} ipython3
import sys
sys.path.insert(0, '../modules')
```

Remember that you need to add these lines in all the notebooks in which you want to use the modules located in this folder.

Now that we know how to create and use modules, let's return to the main topic of this chapter: partial differential equations.

+++

## Advection equation

+++

We first consider advection with a constant velocity $c$. This process is described by equation \eqref{eq:advection}. If $u$ is our unknown, the solution to the equation is:

$$
u(x,t) = u_0(x-ct),
$$

where $u_0(x) = u(x,0)$ is the initial condition. At time $t$, the mathematical solution to the problem is therefore the initial condition shifted by an mount $ct$. To obtain this solution we don't need a computer, so why bother trying to solve it numerically? It turns out that it constitutes a rich and interesting laboratory for developing general methods and analyze their shortcomings.

+++

### Forward Euler, forward finite difference

For our first attempt at solving equation \eqref{eq:advection}, we choose the forward Euler method for the time integration and the first-order accurate forward finite difference formula for the derivative.


The discretization then proceeds as follows:

\begin{align}
\frac{d u^n_i}{dt} &\approx \frac{u^{n+1}_i - u^n_i}{dt}, \; \; \; \frac{\partial u}{\partial x}(x_i, t^n) \approx \frac{u^n_{i+1} - u^n_i}{\Delta x} \\
& \Rightarrow \;
u^{n+1}_i = u^n_i -cdt \frac{u^n_{i+1} - u^n_i}{\Delta x}
\end{align}

Note that this discretization is explicit as $u^{n+1}_i$ is directly expressed in terms of quantities known at the previous time step.

+++

For the sake of the example, we solve this equation in the interval $x\in [0, 1]$ with the following initial condition:

\begin{equation}
u(x,0) = e^{-200 (x-0.25)^2}
\end{equation}

+++

Let us now write a Python code to compute the solution. We first import a function we have defined in our custom module named `steppers`:

```{code-cell} ipython3
from steppers import euler_step
```

You should now look at the definition of this function and understand what it is doing. You may obtain its documentation by typing:

```{code-cell} ipython3
%pinfo euler_step
```

In summary, it just performs the operation:

\begin{align}
u^{n+1}_i = u^n_i + dt f_i(u^n)
\end{align}

where $f$ is any function supplied as an argument. If $f$ contains specific parameters, these can be passed to the `euler_step` function as optional arguments. In our case, the function $f$ evaluated at grid point $x_i$ is given by,

\begin{align}
f_i = -c \frac{u^n_{i+1} - u^n_i}{\Delta x}
\end{align}

We still need to choose our boundary conditions. Here the initial condition is nearly equal to zero at the boundaries. As we wont let the wave reach the right boundary (we choose a positive value for $c$), we can safely maintain the conditions $u(x_0,t)=u(x_n,t)=0$ for all times. This can be achieved by setting $f[0]=f[-1]=0$ or by limiting our unkowns to `u[1:-1]`. Here we choose the former option. 

We thus impose Dirichlet boundary conditions, even at the right boundary node where a boundary condition is in principle not required in the original partial differential equation. This is a feature we need to introduce in our numerical solution because we cannot afford a semi-infinite domain.

The desired function $f$ can then be numerically implemented as follows:

```{code-cell} ipython3
def rhs_forward(u, dx, c):
    """Returns the right-hand side of the wave
    equation based on forward finite differences
    
    Parameters
    ----------
    u : array of float
        solution at the current time-step.
    dx : float
        grid spacing
    c : float
        advection velocity
    
    Returns
    -------
    f : array of float
        right-hand side of the wave equation with
        boundary conditions implemented (initial values
        of u are kept constant)
    """
    nx = u.shape[0]
    f = np.empty(nx)
    f[1:-1] = -c*(u[2:]-u[1:-1]) / dx
    f[0] = 0
    f[-1] = 0
    
    return f
```

Parameters for the simulation:

```{code-cell} ipython3
c=1.          # wave or advection speed
lx = 1.       # length of the computational domain
t_final = 0.2 # final time of for the computation (assuming t0=0)
```

```{code-cell} ipython3
dt = 0.005                   # time step
nt = int(t_final / dt)       # number of time steps

nx = 101                     # number of grid points 
dx = lx / (nx-1)             # grid spacing
x = np.linspace(0., lx, nx)  # coordinates of grid points
```

```{code-cell} ipython3
# initial condition
u0 = np.exp(-200 * (x-0.25)**2)
```

```{code-cell} ipython3
# create an array to store the solution
u = np.empty((nt+1, nx))
# copy the initial condition in the solution array
u[0] = u0.copy()
```

```{code-cell} ipython3
# perform nt time steps using the forward Euler method
# with first-order forward finite difference
for n in range(nt):
    u[n+1] = euler_step(u[n], rhs_forward, dt, dx, c)
```

```{code-cell} ipython3
# plot the solution at several times
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(x, u[0], label='Initial condition')
ax.plot(x, u[int(0.10 / dt)], lw=1.5, color='green', label='t=0.10')
ax.plot(x, u[int(0.15 / dt)], lw=1.5, color='indigo', label='t=0.15')
ax.plot(x, u[int(t_final / dt)], lw=1.5, color='brown', label=f't={t_final}')

ax.set_xlabel('$x$')
ax.set_ylabel('$u$')
ax.set_title('Advection with forward Euler scheme - forward finite differences')
ax.legend()
```

What is happening ? The solution rapidly deteriorates; it is not peacefully translated at constant wave speed $c$: the maximum increases and some wiggles appear at the trailing edge. If you were to run the simulation just a bit longer, the solution would completely blow up. Try it !

+++

We have already observed such behaviors when discussing integration schemes. We saw that some of them have a limited domain of stability and we should suspect that a similar limitation appears here. We will discuss this point thoroughly in the next notebook. 

Here we just try a few other things to see what happens. Let us begin by replacing the forward finite difference scheme with the backward finite difference scheme. The only change we need to make is in the discretization of the right-hand side of the equation. We replace it with the following function (make sure you understand the change):

```{code-cell} ipython3
def rhs_backward(u, dx, c):
    """Returns the right-hand side of the wave
    equation based on backward finite differences
    
    Parameters
    ----------
    u : array of float
        solution at the current time-step.
    dx : float
        grid spacing
    c : float
        advection velocity
    
    Returns
    -------
    f : array of float
        right-hand side of the wave equation with
        boundary conditions implemented (initial values
        of u are kept constant)
    """
    nx = u.shape[0]
    f = np.empty(nx)
    f[1:-1] = -c*(u[1:-1]-u[:-2]) / dx
    f[0] = 0
    f[-1] = 0
    
    return f
```

We can now rerun our simulation with the same parameters.

```{code-cell} ipython3
# create an array to store the solution
u = np.empty((nt+1, nx))
# copy the initial condition in the solution array
u[0] = u0.copy()
```

```{code-cell} ipython3
# perform nt time steps using the forward Euler method
# with first-order backward finite difference
for n in range(nt):
    u[n+1] = euler_step(u[n], rhs_backward, dt, dx, c)
```

```{code-cell} ipython3
# plot the solution at several times
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(x, u[0], label='Initial condition')
ax.plot(x, u[int(0.10 / dt)], lw=1.5, color='green', label='t=0.10')
ax.plot(x, u[int(0.15 / dt)], lw=1.5, color='indigo', label='t=0.15')
ax.plot(x, u[int(t_final / dt)], lw=1.5, color='brown', label=f't={t_final}')

ax.set_xlabel('$x$')
ax.set_ylabel('$u$')
ax.set_title('Advection with forward Euler scheme - backward finite differences')
ax.legend()
```

This time, the solution does not seem to be unstable, but it's still not what we would like: the maximum decreases as the wave is advected and close inspection reveals that the packet widens. This will also be explained in the next notebook. You should however rerun this version of the discretization with a lager time step, for example $dt=0.05$. What happens in that case?

Before digging more on the stability and accuracy of the numerical schemes used so far, we introduce two other topics: one theoretical (on periodic boundary condition) and one computational (on how to create animations to visualize our simulation results).

```{code-cell} ipython3
import matplotlib.animation as animation
from IPython.display import HTML

fig, ax = plt.subplots()

plt.close()

line, = ax.plot(x, u0)



def animate(i):
    line.set_ydata(u[i])  # update the data.
    return line,


ani = animation.FuncAnimation(
    fig, animate, interval=100, frames=40, blit=True)



HTML(ani.to_jshtml())
```

+++ {"cell_style": "center"}

## Summary

...

```{code-cell} ipython3
from IPython.core.display import HTML
css_file = '../styles/notebookstyle.css'
HTML(open(css_file, 'r').read())
```
