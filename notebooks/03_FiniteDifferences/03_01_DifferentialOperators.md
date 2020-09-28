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
<h1 style="text-align: center">Finite Differences I</h1>


<h2 class="nocount">Contents</h2>

1. [Introduction](#Introduction)
2. [First-order derivative](#First-order-derivative)
3. [Numpy array slicing](#Numpy-array-slicing)
4. [One-sided finite differences](#One-sided-finite-differences)
5. [Summary](#Summary)


## Introduction

For convenience, we start with importing some modules needed below:

```python
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use('../styles/mainstyle.use')
```

In this part of the course we describe how to compute numerically derivatives of functions such as,

$$
f(x)=e^x \sin(3\pi x) \label{testfunc}
$$

There as several conceptually different ways to do this. Following the same approach as for time integration, we can rely on Taylor's theorem to use the value of $f(x)$ at some neighbouring points of $x$. This approach relies on what are known as finite differences. Another way to compute derivatives relies on decomposing the function $f$ on a basis of functions $T_k(x)$ and computing the derivatives of $f$ from the known derivatives of $T_k(x)$. This method is known as the spectral method and will be described later on in the course.

## First-order derivative

We first consider the first-order derivative of $f(x)$. According to Taylor's theorem, we can approximate $f(x+\Delta x)$ as,

\begin{equation}
f(x+\Delta x)= f(x)+f'(x)\Delta x+O(\Delta x^2)
\end{equation}

and then get the following expression for the derivative of $f$ at point $x$:

\begin{equation}
f'(x) = \frac{f(x+\Delta x) - f(x)}{\Delta x}+O(\Delta x) \label{forwardTaylorDiff1}
\end{equation}

This expression is the usual left derivative of $f$. 

In any numerical problem, we have to limit the number of points at which we store the values of $f$ because the random access memory (RAM) of our computers is finite. We therefore need to introduce a grid along with a set of grid points at which we evaluate all physical quantities. For simplicity we consider a uniform grid in which the $n+1$ grid points are evenly distributed. The coordinates of the grid points are therefore,

\begin{equation}
 x_i = i \Delta x, \; \; 0\leq i \leq n
\end{equation}

with the endpoints of the grid located respectively at $x_0$ and $x_n$.

From eq. \ref{forwardTaylorDiff1} we can then define the following first-oder accurate approximation of the first-order derivative of $f$ at $x_i$:

\begin{equation}
f'_{\rm f}(x_i) = \frac{f(x_{i+1}) - f(x_i)}{\Delta x}, \;\; \hbox{forward finite difference}\label{forwardDiff1}. 
\end{equation}

Schematically, we represent this expression through *a stencil* that indicates which grid points are involved in the computation:

<img width="600px" src="../figures/forwardDiff1.png">

In the above stencil, we use two grid points - indicated in red - to obtain a first-order accurate expression. The forward finite difference cannot be used at the right boundary node of the grid. In section XX, we discuss how we can handle boundary nodes when we evaluate derviatives using finite differences.

In an identical manner, we can define another first-order estimate for the first-order derivative using a backward finite difference,

\begin{equation}
f'_{\rm b}(x_i) = \frac{f(x_{i}) - f(x_{i-1})}{\Delta x}, \;\; \hbox{backward finite difference}\label{backwardDiff1}. 
\end{equation}

It is based on the right derivative of $f$ and its stencil is,

<img width="600px" src="../figures/backwardDiff1.png">

The backward finite difference cannot be used at the left boundary node of the grid. We also note that $f'_{\rm b}(x_{i+1}) = f'_{\rm f}(x_i)$.

Using two grid points, we can actually do better than first-order accuracy. Resorting again to Taylor's theorem we write:

\begin{equation}
f(x+\Delta x)= f(x)+f'(x)\Delta x+\frac12 f''(x)\Delta x^2+O(\Delta x^3) \\
f(x-\Delta x)= f(x)-f'(x)\Delta x+\frac12 f''(x)\Delta x^2+O(\Delta x^3) \\
\end{equation}

If we substract these two equations, we get:

\begin{equation}
f'(x) = \frac{f(x+\Delta x) - f(x-\Delta x)}{2\Delta x}+O(\Delta x^2) \label{centeredTaylorDiff}
\end{equation}

We can then define the following second-oder accurate approximation of the first-order derivative of $f$ at $x_i$:

\begin{equation}
f'_{\rm c}(x_i) = \frac{f(x_{i+1}) - f(x_{i-1})}{2\Delta x},\;\; \hbox{centered finite difference} \label{centeredDiff}.
\end{equation}

This expression is called the centered finite difference first-order derivative and its stencil looks like this:

<img width="600px" src="../figures/centeredDiff1.png">

Using just two grid points, it's not possible to achieve an accuracy of higher order. The centered finite difference cannot be used at the left or right boundary nodes of the grid

Let us check that our formulas work. We first create a fine grid to accuratly represent the function \eqref{testfunc} and its derivative in the interval $x\in [O\, \pi]$.

```python
pi = np.pi # 3.14...
nx = 200 # number of grid points (fine grid)
lx = pi # length of invertal
dx = lx / (nx-1) # grid spacing
x = np.linspace(0, lx, nx) # coordinates of points on the fine grid


f = np.exp(x)*np.sin(3*pi*x)
dfdx = np.exp(x)*(np.sin(3*pi*x) + 3*pi*np.cos(3*pi*x))

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(x, f)
ax[0].set_xlabel('$x$')
ax[0].set_ylabel('$f$')
ax[0].set_ylim(-25,25)

ax[1].plot(x, dfdx)
ax[1].set_xlabel('$x$')
ax[1].set_ylabel('$f\'$')
ax[1].set_ylim(-200,200)
```

The finite difference approximations, one a coarser grid, can then be evaluated as follows.

```python
nx = 40 # number of grid points (coarse grid)
lx = pi # length of invertal
dx = lx / (nx-1) # grid spacing
x_c = np.linspace(0, lx, nx) # coordinates of points on the coarse grid

f_c = np.exp(x_c)*np.sin(3*pi*x_c) # function on the coarse grid

df_forward = np.empty(nx) # forward finite difference
df_backward = np.empty(nx) # backward finite difference
df_centered = np.empty(nx) # centered finite difference

for i in range(0, nx-1): # last grid point is omitted
    df_forward[i] = (f_c[i+1] - f_c[i]) / dx
    
for i in range(1, nx): # first grid point is omitted
    df_backward[i] = (f_c[i] - f_c[i-1]) / dx

for i in range(1, nx-1): # first and last grid points are omitted
    df_centered[i] = (f_c[i+1] - f_c[i-1]) / (2*dx)
```

```python
fig, ax = plt.subplots(1,3,figsize=(12, 5))

ax[0].plot(x, dfdx)
ax[0].plot(x_c[0: nx-1], df_forward[0: nx-1], '*g')
ax[0].set_xlabel('$x$')
ax[0].set_ylabel('$f\'$')

ax[1].plot(x, dfdx)
ax[1].plot(x_c[1: nx], df_backward[1: nx], '*m')
ax[1].set_xlabel('$x$')
ax[1].set_ylabel('$f\'$')

ax[2].plot(x, dfdx)
ax[2].plot(x_c[1: nx-1], df_centered[1: nx-1], '*c')
ax[2].set_xlabel('$x$')
ax[2].set_ylabel('$f\'$')
```

What do you think about the agreement? What happens when you increase the number of points in the coarse grid?

In the above cell, we have used the slicing of numpy arrays to extract the relevant entries from our arrays. For example, for the forward finite difference, the expression is not defined at the last grid point. Therefore, the relevant grid coordinates are not the complete `x_c` array but the *slice* `x_c[0: nx-1]`. For the centered finite difference we must exclude the first and last grid points. The appropriate coordinate array slice is then `x_c[1: nx-1]`. This notions of array slicing is described in much more detail in the next section.


## Numpy array slicing


...


## One-sided finite differences

```python
from IPython.core.display import HTML
css_file = '../Styles/notebookstyle.css'
HTML(open(css_file, 'r').read())
```

```python

```
