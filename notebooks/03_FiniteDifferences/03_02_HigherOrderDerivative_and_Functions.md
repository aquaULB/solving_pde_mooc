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

<div class="copyright" property="vk:rights">&copy;
  <span property="vk:dateCopyrighted">2020</span>
  <span property="vk:publisher">B. Knaepen & Y. Velizhanina</span>
</div>
<h1 style="text-align: center">Finite Differences II</h1>


<h2 class="nocount">Contents</h2>

1. [Introduction](#Introduction)
2. [Higher order derivative](#Second-order-derivative)
3. [Functions](#Functions)
4. [Matrix formulation](#Matrix-formulation)
5. [Summary](#Summary)


## Introduction

For convenience, we start with importing some modules needed below:

```python
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use('../styles/mainstyle.use')
```

In this notebook we extend the concept of finite differences to higher order derivatives. We also discuss the use of `functions` and finally we describe how to construct matrices corresponding to the finite difference operators. These are very useful when solving certain numerical problems like boundary value problems or eigenvalue problems.

## Higher order derivative

Using finite differences, we can construct derivatives up to any order. Before we discuss this, we first explicitly describe in detail the second-order derivative that is later used extensively in the course.


### Second-order derivative

Using Taylor's theorem we can write:

\begin{align}
& f(x+\Delta x) = f(x)+f'(x)\Delta x+\frac12 f''(x)\Delta x^2+\frac16 f'''(x)\Delta x^3+O(\Delta^4)\label{eq:leftTaylor2} \\
& f(x-\Delta x) = f(x)-f'(x)\Delta x+\frac12 f''(x)\Delta x^2-\frac16 f'''(x)\Delta x^3+O(\Delta^4)\label{eq:rightTaylor2}.
\end{align}

If we add these two equations, we can define a centered second-order accurate formula for $f''$ at grid point $x_i$:

\begin{equation}
f''(x_i)=\frac{f_{i-1}-2f_i+f_{i+1}}{\Delta^2}\label{eq:centeredDiff2}
\end{equation}

The stencil for this expression is the sequence $[-1,0,1]$ and we represent it as:

<img width="600px" src="../figures/centeredDiff2.png">

The centered second-order derivative cannot be used at the boundary nodes. Some one-sided formulas are needed at those locations.

Let us write a Python code to check that expression \ref{eq:centeredDiff2} works as expected. We use the same test function as in the previous notebook - $f(x)=e^x \sin(3\pi x)$ - and first create a fine representation of it in the interval $x\in [O, \pi]$.

```python
pi = np.pi       # 3.14...
nx = 200         # number of grid points (fine grid)
lx = pi          # length of the interval
dx = lx / (nx-1) # grid spacing
```

```python
x = np.linspace(0, lx, nx)   # coordinates in the fine grid
f = np.exp(x)*np.sin(3*pi*x) # function in the fine grid

# Let us build a numpy array for the exact repre-
# sentation of the first-order derivative of f(x).
ddf = np.exp(x)*(np.sin(3*pi*x) + 6*pi*np.cos(3*pi*x)-9*pi**2*np.sin(3*pi*x))
```

We now build a coarse grid with 80 points, and evaluate the second-order derivative using the centered finite difference formula; note how we use the slicing technique we described in the previous notebook.

```python
nx = 80 # number of grid points (coarse grid)
lx = pi # length of invertal
dx = lx / (nx-1) # grid spacing
x_c = np.linspace(0, lx, nx) # coordinates of points on the coarse grid

f_c = np.exp(x_c)*np.sin(3*pi*x_c) # function on the coarse grid

ddf_c = np.empty(nx) 
ddf_c[1:-1] = (f_c[2:] -2*f_c[1:-1] +f_c[:-2]) / dx**2 # boundary nodes are included
```

```python
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(x[1:-1], ddf[1:-1])
ax.plot(x_c[1:-1], ddf_c[1:-1], '^g')
ax.set_xlabel('$x$')
ax.set_ylabel('$f\'$')
```

```python

```

```python

```

## Functions

...


## Matrix formulation

...

## Summary


...

```python
from IPython.core.display import HTML
css_file = '../Styles/notebookstyle.css'
HTML(open(css_file, 'r').read())
```

```python

```
