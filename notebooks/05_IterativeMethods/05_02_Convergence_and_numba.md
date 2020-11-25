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

<div class="copyright" property="vk:rights">&copy;
  <span property="vk:dateCopyrighted">2020</span>
  <span property="vk:publisher">B. Knaepen & Y. Velizhanina</span>
</div>

+++

# Convergence and numba acceleration

+++ {"toc": true}

<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Gauss-Seidel-with-numba" data-toc-modified-id="Gauss-Seidel-with-numba-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Gauss-Seidel with numba</a></span></li></ul></div>

+++

## Introduction

For convenience, we start by importing some modules needed below:

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../modules')
# Function to compute an error in L2 norm
from norms import l2_diff

from iter_module import p_exact_2d

%matplotlib inline

plt.style.use('../styles/mainstyle.use')
```

In the previous chapter we have ...

+++

## Boosting Python

+++

Python has plenty of appeal to the programming community: it's simple, interactive and free. But Fortran, C, C++ dominate high-performance programming. Why? Python is *slow*. There are two major reasons for that: **Python is a dynamically typed language** and **Python is an interpreted language**.

There is *a lot* of reading you can do on this topic. In case you are interested, you might consider the following sources to begin with:

* [Interpreted vs Compiled Programming Languages: What's the Difference?][1]
* [Why Python is Slow: Looking Under the Hood][2]

It doesn't make Python *a bad* programming language. On the contrary, Python is a great tool for various tasks that do not require running expensive simulations (web-development, scripting). Python also dominates the data-science due to availability of such packages as NumPy and SciPy.

As we discussed earlier, NumPy and SciPy integrate optimized and precompiled C code into Python and, therefore, might provide significant speed up. Though, there are serious limitations to the optimization that can be done by using NumPy and SciPy. We have already encountered situations when it was not possible to avoid running Python loops that easily end up being painfully slow.

In this subsection we are going to discuss tools designed to provide C-like performance to the Python code: *Cython* and *Numba*. What are they, what are they good for and why people still use C/C++ with much more complicated syntax?

But before proceeding to the discussion, let us introduce the concept Numba strongly relies on - *Python decorators*.

[1]: <https://www.freecodecamp.org/news/compiled-versus-interpreted-languages/> "Compiled VS interpreted"
[2]: <http://jakevdp.github.io/blog/2014/05/09/why-python-is-slow/> "Why Python is slow?"

+++

### Python decorators

+++

So called wrapper functions are widely used in various programming languages. They take function or method as parameter extending it behaviour. Wrapper functions usually intend to *abstract* the code. They, therefore, might shorten and simplify it *by a lot*. Python decorator is unique to Python - it is basically a shortcut (syntactic sugar) for calling a wrapper function. Consider implementation of sample decorator function:

```{code-cell} ipython3
def decorator(func):
    def wrapper(*args, **kwargs):
        print('Preprocessing...')
        
        res = func(*args, **kwargs)
        
        print('Postprocessing...')
        
        return res
    return wrapper
```

This decorator does nothing but simply prints something before and after the internal function is called. We propose you perceive it as an abstraction for some computations. Note that a decorator function returns *a function*. When decorating functions, you will ultimately want to normally return what the internal function returns *but* also perform certain computations before and after it executes.

If we proceed without using Python decorators, two more principal steps are required from us. First, we must implement the function we want to decorate. Let's go for something trivial:

```{code-cell} ipython3
def print_identity(name, age):
    print(f'Your name is {name}. You are {age} year(s) old.')
```

Second, we have to perform actual decoration:

```{code-cell} ipython3
print_identity_and_more = decorator(print_identity)
```

Obviously, `print_identity_and_more` is a function that accepts the same parameters as `print_identity` and prints certain string before and after it executes.

```{code-cell} ipython3
print_identity_and_more('Marichka', 42)
```

Python decorator decorates the function in a single step:

```{code-cell} ipython3
@decorator
def print_identity(name, age):
    print(f'Your name is {name}. You are {age} year(s) old.')
```

We can simply call `print_identity` now:

```{code-cell} ipython3
print_identity('Mao', 33)
```

Let's consider slightly less trivial but a very useful example. Until now, whenever we needed to time execution of the code, we were using `time` or `timeit` magic. Magic commands is a nice tool but they are unique to IPython and usage of IPython is quite limited. The programmer is paying the price of lowered performance for the graphical interface. So, after all IPython is great for debugging, testing and visualization but in the optimized code you will have it disabled. Let's then implement a *decorator* that will be a timer for arbitrary function:

```{code-cell} ipython3
from timeit import default_timer

def timing(func):
    def wrapper(*args, **kwargs):
        t_init = default_timer()
        res = func(*args, **kwargs)
        t_fin = default_timer()
        
        print(f'Time elapsed: {t_fin-t_init} s')
        
        return res
    return wrapper
```

```{code-cell} ipython3
@timing
def print_identity(name, age):
    print(f'Your name is {name}. You are {age} year(s) old.')
```

```{code-cell} ipython3
print_identity('Mark', 21)
```

It is possible to wrap the function in multiple decorators if necessary. Note that the outer decorator must go before the inner one:

```{code-cell} ipython3
@timing
@decorator
def print_identity(name, age):
    print(f'Your name is {name}. You are {age} year(s) old.')
```

```{code-cell} ipython3
print_identity('Jacques', 9)
```

In the end of the day Python decorators are simply functions, so it is possible to pass parameters to the decorator. The syntax for that, though, is a bit different and requires additional layer of decoration called the *decorator factory*. Decorator factory is a decorator function that accepts certain parameters and returns the actual decorator. Consider the example where the timing is made optional using the decorator factory:

```{code-cell} ipython3
def timing(timing_on=True):
    def inner_timing(func):
        def wrapper(*args, **kwargs):
            if not timing_on:
                return func(*args, **kwargs)
            
            t_init = default_timer()
            res = func(*args, **kwargs)
            t_fin = default_timer()

            print(f'Time elapsed: {t_fin-t_init} s')

            return res
        
        return wrapper
    return inner_timing
```

We can now have timing enabled or disabled with the same decorator:

```{code-cell} ipython3
@timing()
def time_printing_identity(name, age):
    print(f'Your name is {name}. You are {age} year(s) old.')

@timing(False)
def print_identity(name, age):
    print(f'Your name is {name}. You are {age} year(s) old.')

time_printing_identity('Donald', 74)

print('\n')

print_identity('Joe', 78)
```

Implementation of the timing function is accessible in `modules/timers.py` by the name `dummy_timer`. You are free to use it as an alternative to `time` magic. Note that it does not implement the decorator factory, as in the above example, and does not provide functionality of the `timeit` magic that estimated the time averaged among $n$ runs. Consider example using `dummy_timer`:

```{code-cell} ipython3
import timers

@timers.dummy_timer
def print_identity(name, age):
    print(f'Your name is {name}. You are {age} year(s) old.')

print_identity('Jacob', 65)
```

## Gauss-Seidel with numba

```{code-cell} ipython3
from numba import njit
```

```{code-cell} ipython3
# Grid parameters.
nx = 61                  # number of points in the x direction
ny = 61                  # number of points in the y direction
xmin, xmax = 0.0, 1.0     # limits in the x direction
ymin, ymax = -0.5, 0.5    # limits in the y direction
lx = xmax - xmin          # domain length in the x direction
ly = ymax - ymin          # domain length in the y direction
dx = lx / (nx - 1)        # grid spacing in the x direction
dy = ly / (ny - 1)        # grid spacing in the y direction

# Create the gridline locations and the mesh grid;
# see notebook 02_02_Runge_Kutta for more details
x = np.linspace(xmin, xmax, nx)
y = np.linspace(ymin, ymax, ny)
X, Y = np.meshgrid(x, y, indexing ='ij')

# Compute the rhs
b = (np.sin(np.pi * X / lx) * np.cos(np.pi * Y / ly) +
     np.sin(5.0 * np.pi * X / lx) * np.cos(5.0 * np.pi * Y / ly))
```

```{code-cell} ipython3
def gauss_seidel(p, b, tolerance, max_iter):
    
    nx, ny = b.shape
    iter = 0
    diff = 1
    tol_hist_gs = []
    
    pnew = p.copy()
    
    while (diff > tolerance):
    
        if iter > max_iter:
            print('\nSolution did not converged within the maximum'
                ' number of iterations')
            break
        
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                pnew[i, j] = ( 0.25*(pnew[i-1, j] + p[i+1, j] + pnew[i, j-1]
                                + p[i, j+1] - b[i, j]*dx**2 ))
        
        diff = l2_diff(pnew, p)
        tol_hist_gs.append(diff)

        # Show iteration progress (I would like to add iter but cannot do it)
        # Problems in numba here
        # print('\r', f'diff: {diff:5.2e}', end='')
        
        p = pnew.copy()
        iter += 1

    else:
        print('The solution converged after {:d} iterations'.format(iter))
        return p, tol_hist_gs

@njit
def gauss_seidel_numba(p, b, tolerance, max_iter):
    
    nx, ny = b.shape
    iter = 0
    diff = 1
    tol_hist_gs = []
    
    pnew = p.copy()
    
    while (diff > tolerance):
    
        if iter > max_iter:
            print('\nSolution did not converged within the maximum'
                ' number of iterations')
            break
        
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                pnew[i, j] = ( 0.25*(pnew[i-1, j] + p[i+1, j] + pnew[i, j-1]
                                + p[i, j+1] - b[i, j]*dx**2 ))
        
        diff = l2_diff(pnew, p)
        tol_hist_gs.append(diff)

        # Show iteration progress (I would like to add iter but cannot do it)
        # Problems in numba here
        # print('\r', f'diff: {diff:5.2e}', end='')
        
        p = pnew.copy()
        iter += 1

    else:
        print('The solution converged after iterations', iter)
        return p, tol_hist_gs
```

```{code-cell} ipython3
p0 = np.zeros((nx,ny))
```

```{code-cell} ipython3
# 20 seconds on my computer (7 times)
%timeit p, tol_hist_gs = gauss_seidel(p0.copy(), b, tolerance = 1e-10, max_iter = 1e6)
```

```{code-cell} ipython3
# 80ms on my computer (70 times) !!!
%timeit p, tol_hist_gs = gauss_seidel_numba(p0.copy(), b, tolerance = 1e-10, max_iter = 1e6)
```

```{code-cell} ipython3
# Compute the exact solution (and p for the plot %%timeit does not return it)
p, tol_hist_gs = gauss_seidel_numba(p0.copy(), b, tolerance = 1e-10, max_iter = 1e6)
p_e = p_exact_2d(X, Y)
```

```{code-cell} ipython3
fig, (ax_1, ax_2, ax_3) = plt.subplots(1, 3, figsize=(16,5))
# We shall now use the
# matplotlib.pyplot.contourf function.
# As X and Y, we pass the mesh data.
#
# For more info
# https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.contourf.html
#
ax_1.contourf(X, Y, p, 20)
ax_2.contourf(X, Y, p_e, 20)

# plot along the line y=0:
jc = int(ly/(2*dy))
ax_3.plot(x, p_e[:,jc], '*', color='red', markevery=2, label=r'$p_e$')
ax_3.plot(x, p[:,jc], label=r'$pnew$')

# add some labels and titles
ax_1.set_xlabel(r'$x$')
ax_1.set_ylabel(r'$y$')
ax_1.set_title('Exact solution')

ax_2.set_xlabel(r'$x$')
ax_2.set_ylabel(r'$y$')
ax_2.set_title('Numerical solution')

ax_3.set_xlabel(r'$x$')
ax_3.set_ylabel(r'$p$')
ax_3.set_title(r'$p(x,0)$')

ax_3.legend();
```

```{code-cell} ipython3
from IPython.core.display import HTML
css_file = '../styles/notebookstyle.css'
HTML(open(css_file, 'r').read())
```

```{code-cell} ipython3

```
