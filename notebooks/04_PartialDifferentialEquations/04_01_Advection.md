---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.6.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<h1 style="text-align: center">Partial differential Equation I</h1>

<!-- #region toc=true -->
<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Python-modules" data-toc-modified-id="Python-modules-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Python modules</a></span></li><li><span><a href="#Advection-equation" data-toc-modified-id="Advection-equation-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Advection equation</a></span></li><li><span><a href="#Summary" data-toc-modified-id="Summary-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Summary</a></span></li></ul></div>
<!-- #endregion -->

## Introduction

For convenience, we start with importing some modules needed below:

```python
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use('../styles/mainstyle.use')
```

In Chapter 2 and 3 of this course, we described respectively the time integration of ordinary differential equations and the discretization of differential operators using finite difference formulas.

Here we combine the tools learned in these two chapters to address the numerical solution of partial differential equations. We mainly focus on the first order wave equation (all symbols are properly defined in the corresponding sections of the notebooks),

$$
   \frac{\partial u}{\partial t}+c\frac{\partial u}{\partial x}=0,\label{eq:advection}
$$

and the heat equation,

$$
   \partial_t T(x,t) = \alpha \frac{d^2 T} {dx^2}(x,t) + \sigma (x,t).
$$

But before we start digging into the theory, we begin this chapter by introducing the concept of Python modules.

<!-- #region -->
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
<!-- #endregion -->

It contains the definition of two functions. Notice how we import `numpy` at the beginning of the file. Without this import, the second function would not work.

To use the function `triangle_area` in the current notebook, we need to import it using:

```python
from test_module import triangle_area
```

The function can then be called as usual:

```python
area = triangle_area(4, 3)

print(f'The area of the triangle is: {area}.')
```

At this stage we may not yet use the function `two` because it has not been imported. If it's needed, we may type

```python
from test_module import twos
```

and use it afterwards:

```python
a = twos(4)
print(a)
```

When more than one item of a module are needed, we can import several or all of them at the same time. Here are different ways to do this:

```python
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

The choice between the different ways of loading modules is somewhat a matter of taste and often depends on the context (there are plenty of tutorials on the internet discussing this). For this course, we adopt the following convention:

- For `numpy` we use `import numpy as np` to import the whole content of the package and we use the shortcut `np`.

- For `scipy`, we import the desired functions from the submodules of the package, e.g `from scipy.linalg import inv`

- For our own modules, we use the `from module import item1, item2, ...` construct to be explicit about what we want to import.

<!-- #region -->
If the module file is located in the same folder as your notebook or main program, all the commands described above will work. If not, the module file needs to be located in one of the default search paths of your Python distribution. These paths can be listed by executing the following commands:

```python
import sys
print(path.sys)
```

<!-- #endregion -->

In a specific notebook, module or program, you may add extra search paths to locate your modules. For this course, we store our custom modules in a folder called *modules*. It is located in the *notebooks* folder. Therefore, its relative path to any notebook is `../modules` (the `..` represents the parent directory). Look now in your local repository and you should see this folder.

To add this path to the default search path, you may run the following cell:

```python
import sys
sys.path.insert(0, '../modules')
```

Remember that you need to add these lines in all the notebooks in which you want to use the modules located in this folder.

Now that we know how to create and use modules, let's return to the main topic of this chapter: partial differential equations.


## Advection equation


We first consider advection with a constant velocity $c$. This process is described by equation \eqref{eq:advection}. If $u$ is our unknown, the solution to the equation is:

$$
u(x,t) = u_0(x-ct),
$$

where $u_0(x) = u(x,0)$ is the initial condition. At time $t$, the mathematical solution to the problem is just the initial condition shifted by an mount $ct$. To obtain this solution we don't need a computer, so why bother trying to solve it numerically? It turns out that it constitutes a rich and interesting laboratory for developing general methods and analyze their shortcomings.

```python
from steppers import euler_step
```

```python
def rhs(u, dx, c):
    
    nx = u.shape[0]
    f = np.empty(nx)
    f[1:-1] = -c*(u[2:]-u[:-2]) / (2*dx)
    f[0] = -c*(u[1]-u[-2]) / (2*dx)#0
    f[-1] = f[0]
    
    return f
```

```python
t_final = 8.
dt = 0.001
nt = int(t_final / dt)
c = 1.
nx = 1001
lx = 10.
dx = lx / (nx-1)
x = np.linspace(0., lx, nx)
print(nt)
```

```python
# create the initial condition and plot it
u0 = np.exp(-200 * (x-0.25)**2)
```

```python
u = np.empty((nt+1, nx))
u[0] = u0.copy()
```

```python
for n in range(nt):
    u[n+1] = euler_step(u[n], rhs, dt, dx, c)
```

```python
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(x, u[0], label='Initial condition')
ax.plot(x, u[int(0.12 / dt)], lw=1.5, color='green', label='t=0.12')
ax.plot(x, u[int(0.25 / dt)], lw=1.5, color='indigo', label='t=0.25')
ax.plot(x, u[int(t_final / dt)], lw=1.5, color='brown', label='t=t_final')

ax.set_xlabel('$x$')
ax.set_ylabel('$u$')
ax.set_title('Advection with forward Euler scheme')
ax.legend()
```

```python
from steppers import rk4_step
```

```python
for n in range(nt):
    u[n+1] = rk4_step(u[n], rhs, dt, dx, c)
```

```python
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(x, u[0], label='Initial condition')
ax.plot(x, u[int(0.12 / dt)], lw=1.5, color='green', label='t=0.12')
ax.plot(x, u[int(0.25 / dt)], lw=1.5, color='indigo', label='t=0.25')
ax.plot(x, u[int(t_final / dt)], lw=1.5, color='brown', label='t=0.38')

ax.set_xlabel('$x$')
ax.set_ylabel('$u$')
ax.set_title('Advection with fourth-order Runge-Kutta scheme')
ax.legend()
```

```python
import matplotlib.animation as animation
from IPython.display import HTML

fig, ax = plt.subplots()

plt.close()

line, = ax.plot(x, u0)



def animate(i):
    line.set_ydata(u[i*8])  # update the data.
    return line,


ani = animation.FuncAnimation(
    fig, animate, interval=5, frames=1000, blit=True)



HTML(ani.to_jshtml())

```

<!-- #region cell_style="center" -->
## Summary

...
<!-- #endregion -->

```python
from IPython.core.display import HTML
css_file = '../styles/notebookstyle.css'
HTML(open(css_file, 'r').read())
```

